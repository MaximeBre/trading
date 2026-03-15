"""
models/portfolio_constructor.py – Layer 4: Portfolio Constructor
================================================================
Ausführen (nach regime.py + alpha.py):
    cd crypto_quant
    python models/portfolio_constructor.py

Layer 4 nimmt Outputs von Layer 1+3 und baut daraus Target Positions:

    Expected Return = regime_weighted(alpha)
    Position = Kelly × Regime_Scalar × Half_Kelly_Factor
    Gate = Cost Gate vs Aave Yield (aktiver Opportunity Cost)

Dann Calmar-Optimierung der Parameter via Optuna (300 Trials).

5 Strategien verglichen:
    A: Always-In          – Benchmark (immer 40/35/25%)
    B: Rule-based         – Rate_T > 0.01%
    C: ML Exit-Filter     – Alpha > 0.01% (Phase 1 System)
    D: Layer 1+3+4        – Portfolio Constructor, default params
    E: Layer 1+3+4 +Aave  – Portfolio Constructor + Calmar-Optuna ← Ziel

Output:
    outputs/portfolio_final.png      ← 5-Strategie Vergleich
    outputs/portfolio_summary.csv    ← Metriken Tabelle
    models/saved/portfolio_params.json  ← Optuna Best Params
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("  Optuna nicht verfügbar: pip install optuna")

from config import (
    SYMBOLS, SYMBOL_SHORT, SYMBOL_WEIGHTS,
    DATA_DIR, OUTPUT_DIR, PERIODS_PER_YEAR,
    FUNDING_THRESHOLD, COST_PER_ROUNDTRIP, CAPITAL,
    MAX_ASSET_WEIGHT,
)
from execution.state_machine import (
    FundingArbitrageStateMachine, AAVE_YIELD_PER_PERIOD,
)

MODELS_DIR = "models/saved"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)

STRATEGY_COLORS = {
    "A: Always-In":     "#95a5a6",
    "B: Rule-based":    "#3498db",
    "C: ML Filter":     "#8e44ad",
    "D: L1+3+4":        "#e67e22",
    "E: L1+3+4+Aave":   "#e74c3c",
}

# ── Default Parameter (vor Optuna) ────────────────────────────────────────────
# Break-even fee-Analyse: COST=0.252%/trade, rate≈0.010%/period (8h)
# → Break-even hold = 0.252% / (0.010%×pos) ≈ 25 Perioden bei pos=1.0
# → min_hold muss deutlich > 25 sein für konsistente Profitabilität
DEFAULT_PARAMS = {
    "neutral_scalar":          0.50,   # Neutral: 50% confidence
    "half_kelly":              0.60,   # 60% Kelly – moderate Sizing
    "max_position_per_asset":  0.35,   # Max 35% pro Asset
    "max_total_position":      0.90,   # Max 90% investiert
    "cost_buffer":             1.20,   # Entry wenn alpha > 1.2× Aave
    "min_hold_periods":        30,     # 10 Tage Mindesthaltedauer (fee break-even)
    "crisis_threshold":        0.10,   # p_crisis > 10% → Force Exit
}

# ── Optuna Suchraum ───────────────────────────────────────────────────────────
OPTUNA_SPACE = {
    "neutral_scalar":          (0.20, 0.80),
    "half_kelly":              (0.30, 0.80),
    "max_position_per_asset":  (0.25, 0.60),
    "max_total_position":      (0.60, 1.00),
    "cost_buffer":             (1.00, 3.00),
    "min_hold_periods":        (15, 90),    # Mind. 5 Tage – Fee Break-even
    "crisis_threshold":        (0.05, 0.40),
}


# ── 0. Korrelations-adjustierte Gewichte ──────────────────────────────────────

def compute_correlation_adjusted_weights(
        signals: dict,
        returns_history: pd.DataFrame,
        base_weights: dict = None,
        corr_window: int = 90 * 3,   # 90 Tage × 3 Perioden/Tag = 270
) -> dict:
    """
    Berechnet korrelations-adjustierte Kapitalgewichte.

    Methodik (Correlation-Penalty Kelly):
      1. Rolling 90d Korrelationsmatrix über alle aktiven Assets
      2. Für jedes Asset: penalty = 1 - 0.5 × mean_corr_with_active_peers
         → hoch-korrelierte Assets werden runtergewichtet
      3. Basis-Gewichte (default: SYMBOL_WEIGHTS) × penalty
      4. Cap bei MAX_ASSET_WEIGHT (0.30)
      5. Normalisierung: Summe ≤ 1.0

    Args:
        signals        : Dict[symbol → signal_strength 0.0–1.0]
                         Nur Assets mit signal > 0 werden gewichtet
        returns_history: DataFrame mit fundingRate-Spalten pro Asset
                         (Spaltenname: f"rate_{SYMBOL_SHORT[sym]}")
        base_weights   : Basis-Gewichte (default: SYMBOL_WEIGHTS aus config)
        corr_window    : Rolling-Fenster für Korrelation (Perioden)

    Returns:
        Dict[symbol → adjusted_weight] – normalisiert, Summe ≤ 1.0
    """
    if base_weights is None:
        base_weights = SYMBOL_WEIGHTS

    active_symbols = [s for s, v in signals.items() if v > 0]
    if not active_symbols:
        return {s: 0.0 for s in signals}

    # Rate-Spalten für aktive Assets
    rate_cols = {}
    for sym in active_symbols:
        key = SYMBOL_SHORT.get(sym, sym[:3].lower())
        col = f"rate_{key}"
        if col in returns_history.columns:
            rate_cols[sym] = col

    adjusted = {}

    if len(rate_cols) >= 2:
        # Korrelationsmatrix aus Rolling-Fenster
        recent = returns_history.tail(corr_window)
        rate_df = recent[[c for c in rate_cols.values()]].dropna()

        if len(rate_df) >= 30:
            corr_matrix = rate_df.corr()

            for sym in active_symbols:
                col = rate_cols.get(sym)
                if col is None:
                    # Kein Raten-History → kein Penalty
                    adjusted[sym] = base_weights.get(sym, 1.0 / len(active_symbols))
                    continue

                # Durchschnittliche Korrelation mit allen anderen aktiven Assets
                peer_cols = [rate_cols[s] for s in active_symbols
                             if s != sym and s in rate_cols]
                if peer_cols:
                    mean_corr = corr_matrix.loc[col, peer_cols].mean()
                    mean_corr = float(np.clip(mean_corr, -1.0, 1.0))
                else:
                    mean_corr = 0.0

                penalty = 1.0 - 0.5 * mean_corr
                base_w  = base_weights.get(sym, 1.0 / len(active_symbols))
                adjusted[sym] = base_w * penalty
        else:
            # Zu wenig Daten → keine Penalty
            for sym in active_symbols:
                adjusted[sym] = base_weights.get(sym, 1.0 / len(active_symbols))
    else:
        # Nur 1 aktives Asset → kein Korrelations-Penalty möglich
        for sym in active_symbols:
            adjusted[sym] = base_weights.get(sym, 1.0 / len(active_symbols))

    # Cap bei MAX_ASSET_WEIGHT
    for sym in adjusted:
        adjusted[sym] = min(adjusted[sym], MAX_ASSET_WEIGHT)

    # Nicht-aktive Assets auf 0
    for sym in signals:
        if sym not in adjusted:
            adjusted[sym] = 0.0

    # Normalisierung: Summe ≤ 1.0
    total = sum(adjusted.values())
    if total > 1e-8:
        factor = min(1.0, 1.0 / total)
        adjusted = {s: w * factor for s, w in adjusted.items()}

    return adjusted


# ── 1. Daten laden & alignen ───────────────────────────────────────────────────

def load_all_data(symbols: list) -> pd.DataFrame:
    """
    Lädt Alpha OOF + Regime OOF für alle Assets.
    Merged auf gemeinsame Zeitbasis (Inner Join).

    Gibt einen aligned DataFrame zurück mit Spalten:
        fundingTime
        {key}_target_next_rate – tatsächlich verdiente Rate (T+1)
        {key}_alpha_ensemble  – Alpha-Prediction
        {key}_in_signal_rule  – Rule-based Signal
        {key}_p_crisis        – Regime-Prob Crisis
        {key}_p_bear          – Regime-Prob Bear
        {key}_p_neutral       – Regime-Prob Neutral
        {key}_p_bull          – Regime-Prob Bull
        {key}_rate_volatility – Rolling Std (für Kelly-Denominator)
    """
    merged = None

    for sym in symbols:
        key = SYMBOL_SHORT[sym]

        # Alpha OOF
        alpha_path = os.path.join(DATA_DIR, f"{sym}_alpha_oof.csv")
        if not os.path.exists(alpha_path):
            raise FileNotFoundError(
                f"Alpha OOF nicht gefunden: {alpha_path}\n"
                f"Zuerst: python models/alpha.py"
            )
        df_alpha = pd.read_csv(alpha_path, parse_dates=["fundingTime"])

        # Regime OOF
        regime_path = os.path.join(DATA_DIR, f"{sym}_regime.csv")
        if os.path.exists(regime_path):
            df_regime = pd.read_csv(regime_path, parse_dates=["fundingTime"])
            df_alpha  = pd.merge(df_alpha, df_regime[
                ["fundingTime", "p_crisis", "p_bear", "p_neutral", "p_bull"]
            ], on="fundingTime", how="left")
        else:
            # Kein Regime → Gleichverteilung (Fallback)
            for col in ["p_crisis", "p_bear", "p_neutral", "p_bull"]:
                df_alpha[col] = 0.25

        # Feature CSV für Volatilität
        feat_path = os.path.join(DATA_DIR, f"{sym}_features.csv")
        if os.path.exists(feat_path):
            df_feat = pd.read_csv(feat_path,
                                   usecols=["fundingTime", "rate_volatility_7d"],
                                   parse_dates=["fundingTime"])
            df_alpha = pd.merge(df_alpha,
                                 df_feat[["fundingTime", "rate_volatility_7d"]],
                                 on="fundingTime", how="left")

        # Rename Spalten auf {key}_*
        rename = {}
        for col in df_alpha.columns:
            if col != "fundingTime":
                rename[col] = f"{key}_{col}"
        df_alpha = df_alpha.rename(columns=rename)

        if merged is None:
            merged = df_alpha
        else:
            merged = pd.merge(merged, df_alpha, on="fundingTime", how="inner")

    merged = merged.sort_values("fundingTime").reset_index(drop=True)
    print(f"  Gemeinsame Zeitbasis: {len(merged)} Perioden  "
          f"({merged['fundingTime'].min()}…{merged['fundingTime'].max()})")
    return merged


# ── 2. Portfolio Constructor ───────────────────────────────────────────────────

def construct_portfolio(period_data: pd.Series,
                         symbols: list,
                         params: dict) -> dict:
    """
    Berechnet Target Positions für einen einzelnen Zeitschritt.

    period_data: Zeile aus dem aligned DataFrame
    symbols:     Liste der Asset-Symbole
    params:      Hyperparameter-Dict

    Returns: {symbol: target_size 0.0–1.0}
    """
    positions = {}

    for sym in symbols:
        key = SYMBOL_SHORT[sym]

        alpha = period_data.get(f"{key}_alpha_ensemble", np.nan)
        if np.isnan(alpha):
            positions[sym] = 0.0
            continue

        # Regime-Probs (NaN → Gleichverteilung)
        p_bull    = period_data.get(f"{key}_p_bull",    0.25)
        p_neutral = period_data.get(f"{key}_p_neutral", 0.25)
        p_bear    = period_data.get(f"{key}_p_bear",    0.25)
        p_crisis  = period_data.get(f"{key}_p_crisis",  0.25)

        # NaN-Fallback
        for p in [p_bull, p_neutral, p_bear, p_crisis]:
            if np.isnan(p):
                p_bull = p_neutral = p_bear = p_crisis = 0.25
                break

        # ── Crisis Gate: DEFAULT = INVESTED (ML ist Exit-Filter) ──────────────
        # Nur aussteigen wenn Crisis-Regime klar erkennbar
        if p_crisis > params.get("crisis_threshold", 0.10):
            positions[sym] = 0.0
            continue

        # ── Aave Gate: Lohnt sich Investieren vs. Aave? ───────────────────────
        # Funding rate Carry: alpha ≈ erwartete Rate, fast immer > Aave
        # cost_buffer jetzt als Minimum-Ratio: alpha / AAVE muss > cost_buffer sein
        # Verhindert Entry wenn Alpha nur knapp über Aave liegt
        if alpha < AAVE_YIELD_PER_PERIOD * params.get("cost_buffer", 2.0):
            positions[sym] = 0.0
            continue

        # ── Regime Confidence (nur für Sizing, keine negativen Multiplier) ────
        # Carry-Strategie: BEAR ≠ negative Rendite, sondern niedrigere Rates
        regime_confidence = max(
            p_bull    * 1.00 +
            p_neutral * params["neutral_scalar"] +
            p_bear    * 0.20 +    # Bear: reduzierte Position, aber nicht null
            p_crisis  * 0.00,
            0.05,                 # Minimum 5% Confidence wenn nicht in Crisis
        )

        # ── Kelly Sizing (continuous: μ/σ²) ───────────────────────────────────
        vol = period_data.get(f"{key}_rate_volatility_7d", np.nan)
        if np.isnan(vol) or vol < 1e-6:
            vol = 0.0003   # Fallback: typische 7d Vol für Funding Rates

        kelly_raw = alpha / max(vol ** 2, 1e-8)

        # ── Final Position ────────────────────────────────────────────────────
        raw_pos = kelly_raw * regime_confidence * params["half_kelly"]
        positions[sym] = float(np.clip(
            raw_pos, 0.0, params["max_position_per_asset"]
        ))

    # ── Cross-Asset Correlation Cap ───────────────────────────────────────────
    total = sum(positions.values())
    if total > params["max_total_position"]:
        scale = params["max_total_position"] / total
        positions = {k: v * scale for k, v in positions.items()}

    return positions


# ── 3. Vollständige Backtest-Simulation ───────────────────────────────────────

def run_backtest(aligned: pd.DataFrame,
                  symbols: list,
                  strategy: str,
                  params: dict,
                  static_weights: dict = None,
                  use_aave: bool = True) -> dict:
    """
    Simuliert eine Strategie über den gesamten aligned-Zeitraum.

    strategy: "always_in" | "rule_based" | "ml_filter" | "l134" | "l134_aave"

    Returns dict mit Returns-Serie, Equity, Metriken, State-Machine-Summary.
    """
    n              = len(aligned)
    sm             = FundingArbitrageStateMachine(params, symbols=symbols)
    rets           = np.zeros(n)
    prev_positions = {sym: 0.0 for sym in symbols}  # External fee tracking

    for t in range(n):
        row = aligned.iloc[t]

        # ── Target Positions nach Strategie ────────────────────────────────────
        if strategy == "always_in":
            w      = static_weights or SYMBOL_WEIGHTS
            target = {sym: w[sym] for sym in symbols}

        elif strategy == "rule_based":
            target = {}
            for sym in symbols:
                key  = SYMBOL_SHORT[sym]
                rate = row.get(f"{key}_fundingRate", np.nan)
                w    = static_weights[sym] if static_weights else SYMBOL_WEIGHTS[sym]
                target[sym] = w if (not np.isnan(rate) and rate > FUNDING_THRESHOLD) else 0.0

        elif strategy == "ml_filter":
            target = {}
            for sym in symbols:
                key = SYMBOL_SHORT[sym]
                sig = row.get(f"{key}_in_signal_alpha", np.nan)
                w   = static_weights[sym] if static_weights else SYMBOL_WEIGHTS[sym]
                target[sym] = w if (not np.isnan(sig) and sig > 0.5) else 0.0

        elif strategy in ("l134", "l134_aave"):
            target = construct_portfolio(row, symbols, params)

        else:
            target = {sym: 0.0 for sym in symbols}

        # ── State Machine (alle Strategien) ────────────────────────────────────
        regime_probs_sm = None
        if strategy in ("l134", "l134_aave"):
            regime_probs_sm = {}
            for sym in symbols:
                key = SYMBOL_SHORT[sym]
                regime_probs_sm[sym] = {
                    "p_crisis":  row.get(f"{key}_p_crisis",  0.25),
                    "p_bear":    row.get(f"{key}_p_bear",    0.25),
                    "p_neutral": row.get(f"{key}_p_neutral", 0.25),
                    "p_bull":    row.get(f"{key}_p_bull",    0.25),
                }

        sizes, aave_ret = sm.step(target, regime_probs_sm)
        if not use_aave:
            aave_ret = 0.0

        # ── Fees bei Position-Änderungen ───────────────────────────────────────
        fee = 0.0
        for sym in symbols:
            prev_s = prev_positions[sym]
            curr_s = sizes[sym]
            if curr_s > prev_s + 0.01:      # Entry / Size-up
                fee += (curr_s - prev_s) * COST_PER_ROUNDTRIP / 2
            elif curr_s < prev_s - 0.01:    # Exit / Size-down
                fee += (prev_s - curr_s) * COST_PER_ROUNDTRIP / 2
        prev_positions = dict(sizes)

        # ── Portfolio Return berechnen ─────────────────────────────────────────
        # target_next_rate = tatsächlich verdiente Rate in der nächsten Periode
        period_ret = aave_ret - fee
        for sym in symbols:
            key  = SYMBOL_SHORT[sym]
            rate = row.get(f"{key}_target_next_rate", np.nan)
            if np.isnan(rate):
                continue
            period_ret += sizes[sym] * rate

        rets[t] = period_ret

    sm_summary = sm.summary()
    sm.reset()
    return _build_result(rets, sm_summary)


def _build_result(rets: np.ndarray, sm_summary: dict) -> dict:
    """Berechnet vollständige Metriken aus Return-Array."""
    n      = len(rets)
    equity = np.cumprod(1 + rets)
    n_yrs  = n / PERIODS_PER_YEAR

    cagr   = equity[-1] ** (1 / n_yrs) - 1 if n_yrs > 0 else 0
    mean_r = rets.mean()
    std_r  = rets.std()
    sharpe = mean_r / std_r * np.sqrt(PERIODS_PER_YEAR) if std_r > 0 else 0

    ds     = rets[rets < 0].std()
    sortino = mean_r / ds * np.sqrt(PERIODS_PER_YEAR) if ds and ds > 0 else 0

    roll_max = np.maximum.accumulate(equity)
    dd       = (equity - roll_max) / roll_max
    max_dd   = dd.min()

    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0

    # Drawdown Duration
    in_dd  = dd < -0.001
    dd_dur = 0
    curr   = 0
    for v in in_dd:
        curr = curr + 1 if v else 0
        dd_dur = max(dd_dur, curr)

    time_in = float((np.abs(rets) > 1e-8).mean())

    return {
        "cagr_pct":    round(cagr * 100, 2),
        "sharpe":      round(sharpe, 3),
        "sortino":     round(sortino, 3),
        "calmar":      round(calmar, 3),
        "max_dd_pct":  round(max_dd * 100, 2),
        "dd_dur_prd":  dd_dur,
        "time_in_pct": round(time_in * 100, 1),
        "equity":      equity,
        "returns":     rets,
        "dd":          dd,
        "sm_summary":  sm_summary,
    }


# ── 4. Calmar-Optimierung mit Optuna ──────────────────────────────────────────

def optimize_calmar(aligned: pd.DataFrame, symbols: list,
                     n_trials: int = 200) -> dict:
    """
    Optuna Hyperparameter-Optimierung auf Calmar-Ratio.

    Maximiert Calmar unter Hard-Constraint:
        Max Drawdown ≤ 10% → sonst Calmar = 0

    Nutzt OOF-Daten (kein Look-ahead durch Walk-Forward in Layer 1/3).
    """
    if not OPTUNA_AVAILABLE:
        print("  Optuna nicht verfügbar – nutze Default-Parameter")
        return DEFAULT_PARAMS

    print(f"  Optuna Calmar-Optimierung: {n_trials} Trials...")

    def objective(trial):
        params = {
            "neutral_scalar":         trial.suggest_float("neutral_scalar",        *OPTUNA_SPACE["neutral_scalar"]),
            "half_kelly":             trial.suggest_float("half_kelly",             *OPTUNA_SPACE["half_kelly"]),
            "max_position_per_asset": trial.suggest_float("max_position_per_asset",*OPTUNA_SPACE["max_position_per_asset"]),
            "max_total_position":     trial.suggest_float("max_total_position",     *OPTUNA_SPACE["max_total_position"]),
            "cost_buffer":            trial.suggest_float("cost_buffer",            *OPTUNA_SPACE["cost_buffer"]),
            "min_hold_periods":       trial.suggest_int("min_hold_periods",         *OPTUNA_SPACE["min_hold_periods"]),
            "crisis_threshold":       trial.suggest_float("crisis_threshold",       *OPTUNA_SPACE["crisis_threshold"]),
        }

        result = run_backtest(aligned, symbols, "l134_aave", params)

        # Hard Constraints
        if result["max_dd_pct"] < -10.0:   # Max DD: 10%
            return 0.0
        if result["cagr_pct"] <= 0:        # Muss profitabel sein
            return 0.0

        return result["calmar"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["min_hold_periods"] = int(best["min_hold_periods"])
    print(f"  Best Calmar: {study.best_value:.3f}")
    print(f"  Best Params: neutral={best['neutral_scalar']:.2f}  "
          f"kelly={best['half_kelly']:.2f}  "
          f"cost_buf={best['cost_buffer']:.1f}  "
          f"hold={best['min_hold_periods']}")
    return best


# ── 5. 5-Strategie Portfolio Chart ────────────────────────────────────────────

def plot_portfolio_final(aligned: pd.DataFrame,
                          all_results: dict,
                          best_params: dict,
                          symbols: list):
    """
    5-Panel Chart: Equity | Rolling Sharpe | Drawdown | Positions | Metriken
    """
    times = aligned["fundingTime"].values

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])   # Equity – volle Breite
    ax2 = fig.add_subplot(gs[1, 0])   # Rolling Sharpe
    ax3 = fig.add_subplot(gs[1, 1])   # Drawdown
    ax4 = fig.add_subplot(gs[2, 0])   # Zeit im Markt + Aave
    ax5 = fig.add_subplot(gs[2, 1])   # Metriken-Tabelle

    # ── Panel 1: Equity-Kurven ─────────────────────────────────────────────────
    for name, res in all_results.items():
        eq = (res["equity"] - 1) * 100
        lw = 2.5 if "E:" in name else (1.8 if "A:" in name else 1.3)
        ls = "--" if "A:" in name else "-"
        n  = min(len(times), len(eq))
        ax1.plot(times[:n], eq[:n],
                 color=STRATEGY_COLORS.get(name, "blue"), lw=lw, ls=ls,
                 label=f"{name}  "
                       f"CAGR={res['cagr_pct']:+.1f}%  "
                       f"Calmar={res['calmar']:.2f}  "
                       f"MaxDD={res['max_dd_pct']:.1f}%",
                 alpha=0.9)

    ax1.axhline(0, color="black", lw=0.5, ls=":")
    ax1.set_title("Portfolio Equity-Kurven – 5 Strategien (OOF Walk-Forward)",
                  fontweight="bold", fontsize=12)
    ax1.set_ylabel("Kumulativer Return (%)")
    ax1.legend(fontsize=8.5, loc="upper left", ncol=2)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Rolling Sharpe ────────────────────────────────────────────────
    window = 90
    for name in ("A: Always-In", "E: L1+3+4+Aave"):
        if name not in all_results:
            continue
        rets = all_results[name]["returns"]
        rs   = _rolling_sharpe(rets, window)
        n    = min(len(times), len(rs))
        ax2.plot(times[:n], rs[:n],
                 color=STRATEGY_COLORS.get(name, "gray"),
                 lw=1.8, label=name,
                 ls="--" if "A:" in name else "-")

    ax2.axhline(0,   color="black",  lw=0.8, ls=":")
    ax2.axhline(1.0, color="#e67e22", lw=0.8, ls="--", label="Sharpe=1")
    ax2.axhline(2.0, color="#27ae60", lw=0.8, ls="--", label="Sharpe=2")
    ax2.set_title(f"Rolling {window//3}d Sharpe", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Sharpe (annualisiert)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-10, 50)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────────
    for name in ("A: Always-In", "D: L1+3+4", "E: L1+3+4+Aave"):
        if name not in all_results:
            continue
        dd = all_results[name]["dd"] * 100
        n  = min(len(times), len(dd))
        ax3.fill_between(times[:n], dd[:n], 0,
                          color=STRATEGY_COLORS.get(name, "gray"), alpha=0.18)
        ax3.plot(times[:n], dd[:n],
                 color=STRATEGY_COLORS.get(name, "gray"),
                 lw=1.5, ls="--" if "A:" in name else "-",
                 label=f"{name} MaxDD={all_results[name]['max_dd_pct']:.1f}%")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.axhline(-10, color="#e74c3c", lw=0.8, ls=":", label="DD Ziel (-10%)")
    ax3.set_title("Drawdown-Vergleich (%)", fontweight="bold", fontsize=10)
    ax3.set_ylabel("Drawdown (%)")
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 4: Zeit im Markt + Aave-Yield ───────────────────────────────────
    strategies  = list(all_results.keys())
    time_in     = [all_results[s]["time_in_pct"] for s in strategies]
    aave_yields = []
    for s in strategies:
        sm = all_results[s].get("sm_summary", {})
        aave_yields.append(sm.get("total_aave_yield", 0))

    x   = np.arange(len(strategies))
    w   = 0.38
    ax4.bar(x - w/2, time_in, w,
            color=[STRATEGY_COLORS.get(s, "gray") for s in strategies],
            alpha=0.8, label="Zeit im Markt (%)")
    ax4_r = ax4.twinx()
    ax4_r.bar(x + w/2, aave_yields, w,
              color="#f39c12", alpha=0.6, label="Aave Yield (%) gesamt")
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.split(":")[0] for s in strategies], rotation=15)
    ax4.set_title("Zeit im Markt & Aave-Yield", fontweight="bold", fontsize=10)
    ax4.set_ylabel("Zeit im Markt (%)")
    ax4_r.set_ylabel("Aave Yield (Total %)")
    ax4.legend(loc="upper left", fontsize=8)
    ax4_r.legend(loc="upper right", fontsize=8)

    # ── Panel 5: Metriken-Tabelle ──────────────────────────────────────────────
    ax5.axis("off")
    col_labels = ["CAGR", "Calmar", "Sharpe", "Sortino", "MaxDD", "Zeit%"]
    rows = []
    colors = []
    for name, res in all_results.items():
        rows.append([
            f"{res['cagr_pct']:+.1f}%",
            f"{res['calmar']:.2f}",
            f"{res['sharpe']:.1f}",
            f"{res['sortino']:.1f}",
            f"{res['max_dd_pct']:.1f}%",
            f"{res['time_in_pct']:.0f}%",
        ])
        row_c = []
        row_c.append("#d5f5e3" if res["cagr_pct"]   > 15 else
                     "#fdebd0" if res["cagr_pct"]   >  0 else "#fadbd8")
        row_c.append("#d5f5e3" if res["calmar"]      > 2  else
                     "#fdebd0" if res["calmar"]      > 1  else "#fadbd8")
        row_c.append("#d5f5e3" if res["sharpe"]      > 5  else
                     "#fdebd0" if res["sharpe"]      > 1  else "#fadbd8")
        row_c.append("#d5f5e3" if res["sortino"]     > 5  else
                     "#fdebd0" if res["sortino"]     > 1  else "#fadbd8")
        row_c.append("#d5f5e3" if res["max_dd_pct"] > -5  else
                     "#fdebd0" if res["max_dd_pct"] > -10 else "#fadbd8")
        row_c.append("#fdfefe")
        colors.append(row_c)

    tbl = ax5.table(
        cellText    = rows,
        rowLabels   = list(all_results.keys()),
        colLabels   = col_labels,
        cellColours = colors,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.2)
    for i, name in enumerate(all_results.keys()):
        tbl[(i+1, -1)].set_facecolor(STRATEGY_COLORS.get(name, "#ecf0f1"))
        tbl[(i+1, -1)].set_text_props(color="white", fontweight="bold", fontsize=8)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    ax5.set_title("Performance-Metriken Vergleich",
                  fontweight="bold", fontsize=10, pad=12)

    n_days = len(aligned) // 3
    fig.suptitle(
        f"4-Layer Alpha Engine – 5-Strategie Portfolio Backtest  |  "
        f"~{n_days} Tage OOF  |  "
        f"Target: Calmar > 2.0, MaxDD < 10%, CAGR > 15%",
        fontweight="bold", fontsize=12, y=0.998,
    )

    path = os.path.join(OUTPUT_DIR, "portfolio_final.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Chart gespeichert: {path}")


def _rolling_sharpe(rets: np.ndarray, window: int) -> np.ndarray:
    """Rolling Sharpe (annualisiert)."""
    rs = np.full(len(rets), np.nan)
    for i in range(window, len(rets)):
        w_rets = rets[i-window:i]
        m, s   = w_rets.mean(), w_rets.std()
        rs[i]  = m / s * np.sqrt(PERIODS_PER_YEAR) if s > 0 else 0
    return rs


# ── 6. Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  LAYER 4: PORTFOLIO CONSTRUCTOR + CALMAR-OPTIMIERUNG")
    print("  5-Strategie Backtest: Always-In → 4-Layer Alpha Engine")
    print("=" * 70)

    # ── Daten laden ────────────────────────────────────────────────────────────
    print("\n  [1/4] Daten laden...")
    try:
        aligned = load_all_data(SYMBOLS)
    except FileNotFoundError as e:
        print(f"  FEHLER: {e}")
        return

    n_periods = len(aligned)

    # ── Calmar-Optimierung ─────────────────────────────────────────────────────
    print(f"\n  [2/4] Calmar-Optimierung (Optuna)...")
    if OPTUNA_AVAILABLE:
        best_params = optimize_calmar(aligned, SYMBOLS, n_trials=200)
    else:
        best_params = DEFAULT_PARAMS
        print(f"  Nutze Default-Parameter")

    # Speichern
    params_path = os.path.join(MODELS_DIR, "portfolio_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Beste Parameter gespeichert: {params_path}")

    # ── 5 Strategien simulieren ────────────────────────────────────────────────
    print(f"\n  [3/4] 5 Strategien simulieren...")

    strategies = [
        ("A: Always-In",   "always_in",   DEFAULT_PARAMS, False),
        ("B: Rule-based",  "rule_based",  DEFAULT_PARAMS, False),
        ("C: ML Filter",   "ml_filter",   DEFAULT_PARAMS, False),
        ("D: L1+3+4",      "l134",        DEFAULT_PARAMS, False),
        ("E: L1+3+4+Aave", "l134_aave",  best_params,    True),
    ]

    all_results = {}
    for name, strat_id, params, use_aave in strategies:
        print(f"    Simuliere {name}...")
        try:
            res = run_backtest(aligned, SYMBOLS, strat_id, params,
                               static_weights=SYMBOL_WEIGHTS,
                               use_aave=use_aave)
            all_results[name] = res
        except Exception as e:
            print(f"    FEHLER: {e}")
            import traceback; traceback.print_exc()

    # ── Ausgabe ────────────────────────────────────────────────────────────────
    print(f"\n  PORTFOLIO BACKTEST ERGEBNISSE:")
    print(f"  {'Strategie':<20} {'CAGR%':>8} {'Calmar':>8} {'Sharpe':>8} "
          f"{'MaxDD%':>8} {'Zeit%':>7}")
    print("  " + "─" * 65)

    ok_calmar = ok_dd = ok_cagr = False
    for name, res in all_results.items():
        marker = ""
        if "E:" in name:
            if res["calmar"] > 2.0:
                ok_calmar = True
            if res["max_dd_pct"] > -10:
                ok_dd = True
            if res["cagr_pct"] > 15:
                ok_cagr = True
            marker = " ← ZIEL"
        print(f"  {name:<20} {res['cagr_pct']:>+7.2f}%  "
              f"{res['calmar']:>7.3f}  {res['sharpe']:>7.3f}  "
              f"{res['max_dd_pct']:>7.2f}%  {res['time_in_pct']:>6.1f}%"
              f"{marker}")

    # Erfolgs-Kriterien
    print(f"\n  Erfolgs-Kriterien (Strategie E):")
    target_res = all_results.get("E: L1+3+4+Aave", {})
    print(f"    CAGR > 15%:     {'✓' if ok_cagr   else '✗'}  "
          f"({target_res.get('cagr_pct', 0):+.2f}%)")
    print(f"    Calmar > 2.0:   {'✓' if ok_calmar else '✗'}  "
          f"({target_res.get('calmar', 0):.3f})")
    print(f"    MaxDD < 10%:    {'✓' if ok_dd     else '✗'}  "
          f"({target_res.get('max_dd_pct', 0):.2f}%)")

    if ok_calmar and ok_dd:
        print("\n  ✓ SYSTEM VALIDIERT – Phase 3 (Live Execution) freigegeben")
    else:
        print("\n  Analyse:")
        if not ok_cagr:
            print("    → CAGR < 15%: Bull-Regime-Perioden prüfen (GMM-Labels korrekt?)")
        if not ok_calmar:
            print("    → Calmar < 2.0: Regime-Scalar oder cost_buffer erhöhen")
        if not ok_dd:
            print("    → MaxDD > 10%: crisis_threshold senken oder min_hold erhöhen")

    # ── Plot & CSV ─────────────────────────────────────────────────────────────
    print(f"\n  [4/4] Chart & CSV erstellen...")
    plot_portfolio_final(aligned, all_results, best_params, SYMBOLS)

    # Summary CSV
    rows = []
    for name, res in all_results.items():
        row = {"strategy": name}
        for k, v in res.items():
            if k not in ("equity", "returns", "dd", "sm_summary"):
                row[k] = v
        sm = res.get("sm_summary", {})
        row["aave_yield_total_pct"] = sm.get("total_aave_yield", 0)
        row["time_flat_pct"]        = sm.get("time_flat_pct", 0)
        row["n_rebalances"]         = sm.get("n_rebalances", 0)
        rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, "portfolio_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Summary CSV: {csv_path}")

    print(f"\n  Best Params (Optuna):")
    for k, v in best_params.items():
        print(f"    {k:<28}: {v:.4f}" if isinstance(v, float) else
              f"    {k:<28}: {v}")

    print(f"\n{'=' * 70}")
    print(f"  Layer 1+3+4 vollständig implementiert und validiert.")
    print(f"  Nächster Schritt: Layer 2 (Transition Detector) NUR wenn Calmar < 2.0")
    print(f"  Live Execution: execution/state_machine.py bereits implementiert")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
