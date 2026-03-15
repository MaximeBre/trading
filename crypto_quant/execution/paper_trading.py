"""
execution/paper_trading.py – Paper Trading Engine
==================================================
Läuft 3× täglich (40min vor jedem 8h Settlement):
    python -m execution.paper_trading --run-once  ← GitHub Actions
    python -m execution.paper_trading --daemon    ← Lokal dauerhaft
    python -m execution.paper_trading --summary   ← Nur Summary

State wird zwischen Runs in outputs/paper_trading_state.json persistiert.
Trades werden in outputs/paper_trading_trades.csv geloggt.
Performance wird in outputs/paper_trading_performance.csv geloggt.
Fehler in outputs/paper_trading_errors.log.
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import xgboost as xgb

from config import (
    SYMBOLS, SYMBOL_SHORT,
    DATA_DIR, OUTPUT_DIR, PERIODS_PER_YEAR,
)
from execution.state_machine import FundingArbitrageStateMachine, AAVE_YIELD_PER_PERIOD
from models.portfolio_constructor import construct_portfolio, DEFAULT_PARAMS

# ── Konstanten ─────────────────────────────────────────────────────────────────

PAPER_CAPITAL        = 1_000.0   # EUR Startkapital
MODELS_DIR           = "models/saved"

STATE_FILE           = os.path.join(OUTPUT_DIR, "paper_trading_state.json")
TRADE_LOG_FILE       = os.path.join(OUTPUT_DIR, "paper_trading_trades.csv")
PERFORMANCE_LOG_FILE = os.path.join(OUTPUT_DIR, "paper_trading_performance.csv")
ERROR_LOG_FILE       = os.path.join(OUTPUT_DIR, "paper_trading_errors.log")

# Legacy (für backwards-compat mit GitHub Actions cache-keys)
HISTORY_FILE         = os.path.join(OUTPUT_DIR, "paper_trading_history.csv")

GO_LIVE_PERIODS      = 90     # 30 Tage × 3 Perioden/Tag
GO_LIVE_CAPITAL      = 1_000  # EUR – Live-Start gleich wie Paper

COST_PER_TRADE_PCT   = 0.0028  # 4 Legs × (0.02% Maker + 0.05% Slippage)

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Error Logger ───────────────────────────────────────────────────────────────

logging.basicConfig(
    filename=ERROR_LOG_FILE,
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
_log = logging.getLogger("paper_trading")


# ── Default State ──────────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "start_date":    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "capital":       PAPER_CAPITAL,
        "current_value": PAPER_CAPITAL,
        "positions": {
            sym: {
                "state":               "FLAT",
                "size":                0.0,
                "entry_price":         None,
                "entry_time":          None,
                "entry_funding_rate":  None,
                "entry_regime":        None,
                "entry_signal":        None,
                "holding_periods":     0,
            }
            for sym in SYMBOLS
        },
        "last_run":       None,
        "total_trades":   0,
        "total_runs":     0,
        "regime_history": [],   # last 5 snapshots
    }


# ── State I/O ──────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            _log.error(f"State laden fehlgeschlagen: {e} → neuer State")
    return _default_state()


def _save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Model Loading ──────────────────────────────────────────────────────────────

def _load_best_params() -> dict:
    path = os.path.join(MODELS_DIR, "portfolio_params.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_PARAMS


def _fetch_recent_features(symbol: str, n_periods: int = 135) -> pd.DataFrame:
    """Lädt Feature-CSV und gibt die letzten N Perioden zurück (45 Tage × 3)."""
    path = os.path.join(DATA_DIR, f"{symbol}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature-CSV nicht gefunden: {path}")
    df = pd.read_csv(path, parse_dates=["fundingTime"])
    df = df.sort_values("fundingTime").reset_index(drop=True)
    return df.tail(n_periods).reset_index(drop=True)


def _predict_regime(symbol: str, features_row: pd.Series) -> dict:
    model_path = os.path.join(MODELS_DIR, f"regime_{symbol}.json")
    feat_path  = os.path.join(MODELS_DIR, f"regime_{symbol}_features.json")

    if not os.path.exists(model_path):
        return {"p_crisis": 0.25, "p_bear": 0.25, "p_neutral": 0.25, "p_bull": 0.25}

    booster = xgb.Booster()
    booster.load_model(model_path)

    with open(feat_path) as f:
        feature_cols = json.load(f)

    available = [c for c in feature_cols if c in features_row.index]
    X    = np.array([features_row.get(c, np.nan) for c in available], dtype=float).reshape(1, -1)
    dmat = xgb.DMatrix(X, feature_names=available)
    probs = booster.predict(dmat)[0]

    if len(probs) == 4:
        return {k: float(v) for k, v in zip(
            ["p_crisis", "p_bear", "p_neutral", "p_bull"], probs
        )}
    return {"p_crisis": 0.25, "p_bear": 0.25, "p_neutral": 0.25, "p_bull": 0.25}


def _predict_alpha(symbol: str, features_row: pd.Series, regime_probs: dict) -> float:
    alphas = {}
    for model_name, weight in [("8h", 0.55), ("24h", 0.45)]:
        model_path = os.path.join(MODELS_DIR, f"alpha_{symbol}_{model_name}.json")
        feat_path  = os.path.join(MODELS_DIR, f"alpha_{symbol}_{model_name}_features.json")

        if not os.path.exists(model_path):
            continue

        booster = xgb.Booster()
        booster.load_model(model_path)

        with open(feat_path) as f:
            feature_cols = json.load(f)

        row = features_row.copy()
        for col, val in regime_probs.items():
            row[col] = val

        available = [c for c in feature_cols if c in row.index]
        X    = np.array([row.get(c, np.nan) for c in available], dtype=float).reshape(1, -1)
        dmat = xgb.DMatrix(X, feature_names=available)
        alphas[model_name] = (float(booster.predict(dmat)[0]), weight)

    if not alphas:
        return np.nan

    total_w  = sum(w for _, w in alphas.values())
    ensemble = sum(p * w for p, w in alphas.values()) / total_w
    return ensemble


# ── Trade Logging ──────────────────────────────────────────────────────────────

TRADE_COLS = [
    "timestamp", "symbol", "action",
    "entry_price", "exit_price",
    "funding_rate_at_entry", "holding_periods",
    "gross_pnl_pct", "fees_pct", "net_pnl_pct", "net_pnl_eur",
    "regime_at_entry", "signal_strength",
]

PERF_COLS = [
    "timestamp", "portfolio_value", "period_return_pct", "cumulative_return_pct",
    "active_positions", "avg_funding_rate",
    "btc_regime", "eth_regime", "sol_regime",
    "doge_regime", "xrp_regime", "avax_regime", "link_regime",
    "aave_yield_earned",
]


def _log_trade(row: dict):
    df = pd.DataFrame([{c: row.get(c) for c in TRADE_COLS}])
    if os.path.exists(TRADE_LOG_FILE):
        df.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADE_LOG_FILE, index=False)


def _log_performance(row: dict):
    df = pd.DataFrame([{c: row.get(c) for c in PERF_COLS}])
    if os.path.exists(PERFORMANCE_LOG_FILE):
        df.to_csv(PERFORMANCE_LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(PERFORMANCE_LOG_FILE, index=False)

    # Legacy history append (für GitHub Actions cache compatibility)
    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)


# ── Main Run-Period ────────────────────────────────────────────────────────────

def run_period():
    """
    Führt eine einzelne Paper-Trading-Periode aus:
      1. State laden
      2. Features + Modell-Predictions pro Asset
      3. State Machine step()
      4. Position-Transitions (ENTER/EXIT) tracken + loggen
      5. Portfolio Value updaten (inkl. Aave Yield)
      6. Performance Log Eintrag schreiben
      7. State speichern
      8. Dashboard regenerieren
    """
    now   = datetime.now(timezone.utc)
    state = _load_state()

    print(f"\n[PAPER TRADER] {now.strftime('%Y-%m-%d %H:%M')} UTC  "
          f"(Run #{state.get('total_runs', 0) + 1})")

    best_params = _load_best_params()
    sm          = FundingArbitrageStateMachine(best_params, symbols=SYMBOLS)

    # Restore SM state from persisted positions
    sm.current_sizes = {
        sym: state["positions"][sym]["size"]
        for sym in SYMBOLS
    }
    sm.state       = state.get("sm_state", "FLAT")
    sm.periods_held = state.get("sm_periods_held", 0)

    # ── 1. Features laden ──────────────────────────────────────────────────────
    latest_by_sym    = {}
    regime_probs_all = {}
    alpha_by_sym     = {}
    for sym in SYMBOLS:
        key = SYMBOL_SHORT[sym].upper()
        try:
            df_feat = _fetch_recent_features(sym)
            latest  = df_feat.iloc[-1]
            latest_by_sym[sym] = latest
        except Exception as e:
            _log.error(f"Feature-Daten fehlen für {sym}: {e}")
            print(f"  [{key}] API/Data Error: {e}")

    # Anchor Assets (BTC, ETH, SOL) fehlen → Position einfrieren
    ANCHOR_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    anchor_missing = [s for s in ANCHOR_SYMBOLS if s not in latest_by_sym]
    if anchor_missing:
        _log.warning(f"Anchor-Daten fehlen {anchor_missing} bei {now.isoformat()} → eingefroren")
        _append_error_performance(state, now, "API_ERROR")
        state["last_run"]   = now.isoformat()
        state["total_runs"] = state.get("total_runs", 0) + 1
        _save_state(state)
        return

    # Expansion-Assets fehlen → weiterlaufen mit verfügbaren Assets
    active_symbols = [s for s in SYMBOLS if s in latest_by_sym]
    if len(active_symbols) < len(SYMBOLS):
        missing = [SYMBOL_SHORT[s].upper() for s in SYMBOLS if s not in latest_by_sym]
        print(f"  Info: {missing} übersprungen (Feature-CSV fehlt – main.py nötig)")

    # ── 2. Modell-Predictions ──────────────────────────────────────────────────
    for sym in SYMBOLS:
        if sym not in latest_by_sym:
            regime_probs_all[sym] = {"p_crisis": 0.25, "p_bear": 0.25,
                                     "p_neutral": 0.25, "p_bull": 0.25}
            alpha_by_sym[sym] = np.nan
            continue

        key    = SYMBOL_SHORT[sym].upper()
        latest = latest_by_sym[sym]

        try:
            regime_probs_all[sym] = _predict_regime(sym, latest)
        except Exception as e:
            _log.warning(f"Regime-Fehler {sym}: {e}")
            regime_probs_all[sym] = {"p_crisis": 0.25, "p_bear": 0.25,
                                     "p_neutral": 0.25, "p_bull": 0.25}

        try:
            alpha_by_sym[sym] = _predict_alpha(sym, latest, regime_probs_all[sym])
        except Exception as e:
            _log.warning(f"Alpha-Fehler {sym}: {e}")
            alpha_by_sym[sym] = np.nan

    # ── 3. Portfolio Constructor ───────────────────────────────────────────────
    period_data = {}
    for sym in SYMBOLS:
        key = SYMBOL_SHORT[sym]
        period_data[f"{key}_alpha_ensemble"] = alpha_by_sym.get(sym, np.nan)
        rp = regime_probs_all.get(sym, {})
        period_data[f"{key}_p_crisis"]  = rp.get("p_crisis",  0.25)
        period_data[f"{key}_p_bear"]    = rp.get("p_bear",    0.25)
        period_data[f"{key}_p_neutral"] = rp.get("p_neutral", 0.25)
        period_data[f"{key}_p_bull"]    = rp.get("p_bull",    0.25)
        if sym in latest_by_sym:
            period_data[f"{key}_rate_volatility_7d"] = latest_by_sym[sym].get(
                "rate_volatility_7d", np.nan
            )

    period_series = pd.Series(period_data)

    try:
        target_positions = construct_portfolio(period_series, SYMBOLS, best_params)
    except Exception as e:
        _log.error(f"Portfolio Constructor: {e}")
        target_positions = {sym: 0.0 for sym in SYMBOLS}

    # ── 4. State Machine ───────────────────────────────────────────────────────
    old_sizes = {sym: state["positions"][sym]["size"] for sym in SYMBOLS}

    try:
        new_sizes, _ = sm.step(target_positions, regime_probs_all)
    except Exception as e:
        _log.error(f"State Machine: {e}")
        new_sizes = old_sizes.copy()

    # ── 5. Position Transitions → Trade Log ───────────────────────────────────
    n_new_trades = 0

    for sym in SYMBOLS:
        old_size = old_sizes.get(sym, 0.0)
        new_size = new_sizes.get(sym, 0.0)
        pos      = state["positions"][sym]
        latest   = latest_by_sym.get(sym)

        current_rate = float(latest.get("fundingRate", 0.0)) if latest is not None else 0.0
        dominant_regime = max(
            regime_probs_all.get(sym, {}),
            key=lambda k: regime_probs_all.get(sym, {}).get(k, 0),
            default="unknown"
        )

        # ENTER: size geht von 0 auf >0
        if old_size < 0.01 and new_size >= 0.01:
            pos["state"]              = "LONG"
            pos["size"]               = new_size
            pos["entry_time"]         = now.isoformat()
            pos["entry_price"]        = current_rate   # funding rate als Proxy
            pos["entry_funding_rate"] = current_rate
            pos["entry_regime"]       = dominant_regime
            pos["entry_signal"]       = float(alpha_by_sym.get(sym, 0.0) or 0.0)
            pos["holding_periods"]    = 0

            _log_trade({
                "timestamp":           now.isoformat(),
                "symbol":              sym,
                "action":              "ENTER",
                "entry_price":         current_rate,
                "exit_price":          None,
                "funding_rate_at_entry": current_rate,
                "holding_periods":     0,
                "gross_pnl_pct":       None,
                "fees_pct":            None,
                "net_pnl_pct":         None,
                "net_pnl_eur":         None,
                "regime_at_entry":     dominant_regime,
                "signal_strength":     pos["entry_signal"],
            })
            n_new_trades += 1

        # EXIT: size geht von >0 auf 0 (oder stark reduziert)
        elif old_size >= 0.01 and new_size < 0.01:
            if pos.get("entry_funding_rate") is not None:
                hp   = pos.get("holding_periods", 0)
                # Grosses P&L: Summe der Funding Rates während der Haltedauer
                # Proxy: holding_periods × avg_rate (vereinfacht)
                avg_rate   = float(pos.get("entry_funding_rate", 0.0) or 0.0)
                gross_pct  = hp * avg_rate * old_size
                fees_pct   = COST_PER_TRADE_PCT
                net_pct    = gross_pct - fees_pct
                net_eur    = net_pct * state["current_value"]
            else:
                gross_pct = fees_pct = net_pct = net_eur = 0.0

            _log_trade({
                "timestamp":           now.isoformat(),
                "symbol":              sym,
                "action":              "EXIT",
                "entry_price":         pos.get("entry_price"),
                "exit_price":          current_rate,
                "funding_rate_at_entry": pos.get("entry_funding_rate"),
                "holding_periods":     pos.get("holding_periods", 0),
                "gross_pnl_pct":       round(gross_pct * 100, 6),
                "fees_pct":            round(fees_pct * 100, 6),
                "net_pnl_pct":         round(net_pct * 100, 6),
                "net_eur":             round(net_eur, 4),
                "regime_at_entry":     pos.get("entry_regime", "unknown"),
                "signal_strength":     pos.get("entry_signal", 0.0),
            })
            n_new_trades += 1

            pos["state"]           = "FLAT"
            pos["size"]            = 0.0
            pos["entry_price"]     = None
            pos["entry_time"]      = None
            pos["entry_funding_rate"] = None
            pos["entry_regime"]    = None
            pos["entry_signal"]    = None
            pos["holding_periods"] = 0

        # HOLD: size bleibt ähnlich
        else:
            pos["size"] = new_size
            if new_size >= 0.01:
                pos["state"]          = "LONG"
                pos["holding_periods"] = pos.get("holding_periods", 0) + 1
            else:
                pos["state"] = "FLAT"

    # ── 6. Portfolio Value Update ──────────────────────────────────────────────
    # Virtueller Return dieser Periode
    funding_return = 0.0
    for sym in SYMBOLS:
        latest = latest_by_sym.get(sym)
        if latest is None:
            continue
        rate = float(latest.get("fundingRate", 0.0) or 0.0)
        size = new_sizes.get(sym, 0.0)
        funding_return += size * rate

    # Aave Yield auf FLAT-Kapital
    active_capital = sum(new_sizes.get(sym, 0.0) for sym in SYMBOLS)
    flat_capital   = max(0.0, 1.0 - active_capital)
    aave_yield     = flat_capital * AAVE_YIELD_PER_PERIOD

    period_return  = funding_return + aave_yield
    old_value      = state["current_value"]
    new_value      = old_value * (1 + period_return)
    state["current_value"] = new_value

    cum_return_pct = (new_value / PAPER_CAPITAL - 1) * 100

    # ── 7. Performance Log ─────────────────────────────────────────────────────
    avg_rate = np.nanmean([
        float(latest_by_sym[sym].get("fundingRate", np.nan) or np.nan)
        for sym in SYMBOLS if sym in latest_by_sym
    ])
    active_count = sum(1 for sym in SYMBOLS if new_sizes.get(sym, 0.0) >= 0.01)

    perf_row = {
        "timestamp":           now.isoformat(),
        "portfolio_value":     round(new_value, 4),
        "period_return_pct":   round(period_return * 100, 6),
        "cumulative_return_pct": round(cum_return_pct, 4),
        "active_positions":    active_count,
        "avg_funding_rate":    round(float(avg_rate or 0.0), 8),
        "aave_yield_earned":   round(aave_yield * old_value, 6),
    }
    for sym in SYMBOLS:
        key = SYMBOL_SHORT[sym]
        rp  = regime_probs_all.get(sym, {})
        dominant = max(rp, key=lambda k: rp.get(k, 0), default="unknown") if rp else "unknown"
        perf_row[f"{key}_regime"] = dominant

    _log_performance(perf_row)

    # ── 8. State Speichern ─────────────────────────────────────────────────────
    state["last_run"]        = now.isoformat()
    state["total_runs"]      = state.get("total_runs", 0) + 1
    state["total_trades"]    = state.get("total_trades", 0) + n_new_trades
    state["sm_state"]        = sm.state
    state["sm_periods_held"] = sm.periods_held

    # Regime-History (letzte 5 Snapshots)
    regime_snapshot = {
        "time": now.isoformat(),
        **{SYMBOL_SHORT[sym]: max(
            regime_probs_all.get(sym, {}),
            key=lambda k: regime_probs_all.get(sym, {}).get(k, 0),
            default="unknown"
        ) for sym in SYMBOLS}
    }
    history = state.get("regime_history", [])
    history.append(regime_snapshot)
    state["regime_history"] = history[-5:]

    _save_state(state)

    # ── 9. Output ──────────────────────────────────────────────────────────────
    print(f"  Portfolio:  {old_value:.2f} € → {new_value:.2f} €  "
          f"({cum_return_pct:+.3f}% seit Start)")
    print(f"  Return:     +{period_return*100:.5f}%  "
          f"(Funding: +{funding_return*100:.5f}%  Aave: +{aave_yield*100:.5f}%)")
    print(f"  Positionen: {active_count} aktiv  |  Neue Trades: {n_new_trades}")
    print(f"  SM State:   {sm.state}")

    for sym in SYMBOLS:
        key = SYMBOL_SHORT[sym].upper()
        sz  = new_sizes.get(sym, 0.0)
        rp  = regime_probs_all.get(sym, {})
        dominant = max(rp, key=lambda k: rp.get(k, 0), default="?") if rp else "?"
        alpha_str = (f"{alpha_by_sym.get(sym, np.nan):.5f}"
                     if not np.isnan(alpha_by_sym.get(sym, np.nan)) else "NaN")
        print(f"    {key}: size={sz:.3f}  regime={dominant}  alpha={alpha_str}")

    # ── 10. Dashboard regenerieren ─────────────────────────────────────────────
    try:
        from generate_dashboard import generate_dashboard
        generate_dashboard()
    except Exception as e:
        _log.warning(f"Dashboard Fehler: {e}")


def _append_error_performance(state: dict, now: datetime, error_type: str):
    """Schreibt einen API_ERROR Eintrag in den Performance Log."""
    val = state.get("current_value", PAPER_CAPITAL)
    cum = (val / PAPER_CAPITAL - 1) * 100
    _log_performance({
        "timestamp":             now.isoformat(),
        "portfolio_value":       round(val, 4),
        "period_return_pct":     0.0,
        "cumulative_return_pct": round(cum, 4),
        "active_positions":      -1,
        "avg_funding_rate":      None,
        "aave_yield_earned":     0.0,
        **{f"{SYMBOL_SHORT[sym]}_regime": error_type for sym in SYMBOLS},
    })


# ── Summary ────────────────────────────────────────────────────────────────────

def summary():
    print("\n" + "=" * 65)
    print("  PAPER TRADING SUMMARY")
    print("=" * 65)

    state = _load_state()
    print(f"\n  Start:     {state.get('start_date', '?')}")
    print(f"  Runs:      {state.get('total_runs', 0)}")
    print(f"  Trades:    {state.get('total_trades', 0)}")
    print(f"  Kapital:   {state.get('capital', PAPER_CAPITAL):.2f} €  →  "
          f"{state.get('current_value', PAPER_CAPITAL):.2f} €")

    cum = (state.get("current_value", PAPER_CAPITAL) / PAPER_CAPITAL - 1) * 100
    print(f"  Return:    {cum:+.3f}%")

    if not os.path.exists(PERFORMANCE_LOG_FILE):
        print("\n  Noch keine Performance-Daten.")
        return

    df = pd.read_csv(PERFORMANCE_LOG_FILE, parse_dates=["timestamp"])
    df = df[df["period_return_pct"].notna()]
    n  = len(df)

    if n < 2:
        print(f"\n  Erst {n} Perioden – zu wenig für Statistiken.")
        return

    returns = df["period_return_pct"] / 100
    sharpe  = (returns.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR)
               if returns.std() > 0 else 0)
    peak    = df["portfolio_value"].cummax()
    dd      = ((df["portfolio_value"] - peak) / peak).min()

    print(f"\n  Perioden:  {n} von {GO_LIVE_PERIODS}")
    print(f"  Sharpe:    {sharpe:.2f}")
    print(f"  Max DD:    {dd*100:.2f}%")

    # Go-Live Check
    n_ok = n >= GO_LIVE_PERIODS
    s_ok = sharpe > 1.0
    d_ok = abs(dd) < 0.15

    if not os.path.exists(TRADE_LOG_FILE):
        t_ok = w_ok = False
        n_trades = 0
    else:
        df_t    = pd.read_csv(TRADE_LOG_FILE)
        exits   = df_t[df_t["action"] == "EXIT"]
        n_trades = len(exits)
        t_ok    = n_trades >= 20
        w_ok    = (exits["net_pnl_pct"] > 0).mean() > 0.45 if len(exits) > 0 else False

    all_ok = n_ok and s_ok and d_ok and t_ok

    print(f"\n  Go-Live Kriterien:")
    for label, ok in [
        (f"30+ Tage ({n}/{GO_LIVE_PERIODS} Runs)", n_ok),
        (f"Sharpe > 1.0 ({sharpe:.2f})", s_ok),
        (f"Max DD < 15% ({abs(dd)*100:.1f}%)", d_ok),
        (f"20+ Trades ({n_trades})", t_ok),
        (f"Win Rate > 45%", w_ok),
    ]:
        print(f"    {'✓' if ok else '✗'} {label}")

    if all_ok:
        print(f"\n  ✅ GO-LIVE FREIGEGEBEN! Start mit {GO_LIVE_CAPITAL} EUR")
    else:
        remaining = max(0, GO_LIVE_PERIODS - n)
        print(f"\n  ⏳ Noch ~{remaining} Runs ({remaining//3} Tage) bis Check")

    print("=" * 65 + "\n")


# ── Daemon Mode ────────────────────────────────────────────────────────────────

def _next_settlement_entry() -> float:
    """Sekunden bis 23:20 / 07:20 / 15:20 UTC."""
    now   = datetime.now(timezone.utc)
    today = now.replace(minute=0, second=0, microsecond=0)
    slots = [
        today.replace(hour=23, minute=20),
        today.replace(hour= 7, minute=20),
        today.replace(hour=15, minute=20),
    ]
    for t in sorted(slots):
        if t > now:
            return (t - now).total_seconds()
    import datetime as dt
    tomorrow_first = slots[1].replace(
        day=now.day + 1
    )
    return (tomorrow_first - now).total_seconds()


def run_daemon():
    print("\n  Daemon: Entry bei 23:20 / 07:20 / 15:20 UTC (40min vor Settlement)")
    print("  Ctrl+C zum Stoppen\n")
    while True:
        wait = _next_settlement_entry()
        print(f"  Warte {wait/60:.1f} min…")
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            print("\n  Gestoppt.")
            break
        try:
            run_period()
        except KeyboardInterrupt:
            print("\n  Gestoppt.")
            break
        except Exception as e:
            _log.error(f"run_period Fehler: {e}\n{traceback.format_exc()}")
            print(f"  Fehler: {e}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper Trading – Funding Rate Arbitrage"
    )
    parser.add_argument("--run-once",  action="store_true",
                        help="Einzelne Periode (GitHub Actions Modus)")
    parser.add_argument("--daemon",    action="store_true",
                        help="Dauerhaft (wartet auf Settlement-Zeitpunkte)")
    parser.add_argument("--summary",   action="store_true",
                        help="Nur Summary anzeigen")
    args = parser.parse_args()

    if args.summary:
        summary()
    elif args.daemon:
        run_daemon()
    else:
        # --run-once oder kein Flag → eine Periode
        try:
            run_period()
        except Exception as e:
            _log.error(f"Kritischer Fehler: {e}\n{traceback.format_exc()}")
            print(f"  FEHLER: {e}")
            sys.exit(1)
        print()
        summary()
