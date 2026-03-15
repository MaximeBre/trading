"""
backtest/simple.py – Professioneller Single-Asset Backtest
===========================================================
Drei Strategien im Vergleich:
  1. Always-In Benchmark
  2. Rule-based (Rate > Threshold)
  3. ML-Signal (Phase 2)

Kritische Fixes ggü. Phase 1:
  ─────────────────────────────────────────────────────────────────────────
  FIX 1 – Fee-Modell:
    Delta-Neutral = 4 Legs pro Roundtrip:
      Open  Spot Long  (MAKER_FEE + SLIPPAGE)
      Open  Futures Short  (MAKER_FEE + SLIPPAGE)
      Close Spot Long  (MAKER_FEE + SLIPPAGE)
      Close Futures Short  (MAKER_FEE + SLIPPAGE)
    → cost = num_trades × (MAKER_FEE + SLIPPAGE) × 4
    Vorher: num_trades × MAKER_FEE × 2 → Kosten 2–4× unterschätzt

  FIX 2 – Sharpe Ratio:
    Vorher: Sharpe auf gefilterter Serie (nur In-Market-Perioden)
    → Lücken zwischen Trades fehlen → Sharpe stark überschätzt
    Jetzt: Volle Return-Serie mit 0 für Out-of-Market-Perioden
    → Realistischer Sharpe für die gesamte Halteperiode

  FIX 3 – Drawdown:
    Vorher: Drawdown auf gefilterter Serie (Lücken ignoriert)
    Jetzt: Kontinuierliche Equity-Kurve (out-of-market = flat, kein Drawdown)

  FIX 4 – Annualisierung:
    8h-Perioden → PERIODS_PER_YEAR = 3 × 365 = 1095
    Sharpe-Skalierung: √1095 ≈ 33.09

  NEU:  Calmar Ratio, Sortino Ratio, Win Rate, Avg Hold Duration
  ─────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from config import (
    FUNDING_THRESHOLD,
    MAKER_FEE, SLIPPAGE, LEGS_PER_ROUNDTRIP, COST_PER_ROUNDTRIP,
    CAPITAL, PERIODS_PER_YEAR,
)


def run_backtest(df: pd.DataFrame,
                  signal_col: str = None,
                  label: str = "Rule-based",
                  symbol: str = "",
                  execution_delay: int = 1) -> dict:
    """
    Führt einen vollständigen, institutionell-korrekten Backtest durch.

    Args:
        df              : Feature DataFrame mit 'fundingRate' Spalte
        signal_col      : Spaltenname des binären Signals (0/1).
                          None → Fallback auf Rate > Threshold
        label           : Name der Strategie für die Ausgabe
        symbol          : Asset-Symbol für die Ausgabe (z.B. "BTCUSDT")
        execution_delay : Anzahl Perioden Verzögerung zwischen Signal und
                          Position (default=1 = eine 8h-Periode).

                          WARUM: Signal feuert bei T=0 (Funding-Rate bekannt),
                          aber Position muss VOR T+8h (nächste Funding-Zahlung)
                          offen sein. 30 Minuten Execution-Zeit ist realistisch,
                          aber da Funding alle 8h gezahlt wird und wir das Signal
                          AUS der vorherigen Periode generieren, simulieren wir
                          konservativ 1 Periode Delay. Wer schnell genug ist,
                          kann auf 0 setzen – das ist dann der Best-Case.

    Returns:
        dict mit allen Metriken (für Portfolio-Aggregation verwendbar)
    """
    df = df.copy().dropna(subset=["fundingRate"]).reset_index(drop=True)
    r  = df["fundingRate"]

    # ── Signal definieren ──────────────────────────────────────────────────────
    if signal_col and signal_col in df.columns:
        raw_signal = df[signal_col].astype(bool)
    else:
        raw_signal = r > FUNDING_THRESHOLD

    # EXECUTION DELAY FIX: Signal von T → Position ab T+delay
    # Verhindert dass wir "heute's" Rate nutzen um "heute" zu traden.
    # In der Realität: Signal kommt nach Funding-Zahlung T-1, Execution braucht Zeit.
    if execution_delay > 0:
        in_signal = raw_signal.shift(execution_delay).fillna(False).astype(bool)
    else:
        in_signal = raw_signal

    # Trade-Übergänge zählen
    enters     = in_signal & (~in_signal.shift(1).fillna(False))
    exits      = (~in_signal) & in_signal.shift(1).fillna(False)
    num_trades = int(enters.sum())

    # ── VOLLE Return-Serie (FIX 2): 0 wenn out-of-market ─────────────────────
    full_returns = pd.Series(0.0, index=df.index)
    full_returns[in_signal] = r[in_signal]

    # ── Always-In Benchmark ───────────────────────────────────────────────────
    always_equity      = (1 + r).cumprod()
    always_total_ret   = always_equity.iloc[-1] - 1
    n_years_always     = len(r) / PERIODS_PER_YEAR
    always_ann         = (1 + always_total_ret) ** (1 / n_years_always) - 1

    # ── Strategie: Brutto ──────────────────────────────────────────────────────
    gross_return = r[in_signal].sum()

    # ── FIX 1: 4-Leg Fee-Modell ────────────────────────────────────────────────
    # Jeder Roundtrip = 1 Entry + 1 Exit über 2 Beine (Spot + Futures)
    total_fees    = num_trades * COST_PER_ROUNDTRIP
    net_return    = gross_return - total_fees

    # ── Equity Kurve (FIX 3): kontinuierliche Kurve ───────────────────────────
    equity = (1 + full_returns).cumprod()

    # ── Annualisierter Netto-Return ────────────────────────────────────────────
    n_years     = len(df) / PERIODS_PER_YEAR
    equity_end  = equity.iloc[-1]
    net_ann     = (equity_end) ** (1 / n_years) - 1   # CAGR

    # ── Sharpe Ratio (FIX 2): auf voller Serie ────────────────────────────────
    # Risikofreier Zinssatz = 0 (vereinfacht; USDT-Yield kann separat addiert werden)
    mean_r  = full_returns.mean()
    std_r   = full_returns.std()
    sharpe  = (mean_r / std_r * np.sqrt(PERIODS_PER_YEAR)
               if std_r > 0 else 0.0)

    # ── Sortino Ratio: nur Downside-Volatilität ───────────────────────────────
    downside_r  = full_returns[full_returns < 0]
    downside_std = downside_r.std() if len(downside_r) > 1 else np.nan
    sortino = (mean_r / downside_std * np.sqrt(PERIODS_PER_YEAR)
               if downside_std and downside_std > 0 else 0.0)

    # ── Max Drawdown (FIX 3): auf Equity-Kurve ────────────────────────────────
    rolling_max  = equity.cummax()
    drawdown     = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()         # negative Zahl
    max_dd_pct   = max_drawdown * 100

    # ── Calmar Ratio = CAGR / |MaxDD| ─────────────────────────────────────────
    calmar = (net_ann / abs(max_drawdown)
              if max_drawdown != 0 else 0.0)

    # ── Zusatzmetriken ─────────────────────────────────────────────────────────
    time_in_market = in_signal.mean() * 100

    # Durchschnittliche Haltedauer einer Position (in 8h Perioden)
    hold_durations = []
    current_hold   = 0
    for flag in in_signal:
        if flag:
            current_hold += 1
        elif current_hold > 0:
            hold_durations.append(current_hold)
            current_hold = 0
    if current_hold > 0:
        hold_durations.append(current_hold)
    avg_hold_periods = np.mean(hold_durations) if hold_durations else 0
    avg_hold_days    = avg_hold_periods / 3     # 8h → Tage

    # Winning Periods: % der In-Market-Perioden mit positiver Rate
    in_market_returns = r[in_signal]
    win_rate = (in_market_returns > 0).mean() * 100 if len(in_market_returns) > 0 else 0

    # ── Gebühren-Analyse (absolut + pro Trade) ────────────────────────────────
    total_fees_paid_pct    = total_fees * 100              # % des Kapitals
    total_fees_paid_usd    = total_fees * CAPITAL          # absolut in USD
    avg_cost_per_trade_pct = COST_PER_ROUNDTRIP * 100      # % pro Roundtrip
    avg_cost_per_trade_usd = COST_PER_ROUNDTRIP * CAPITAL  # USD pro Roundtrip

    # Statistisches Signifikanz-Flag (>= 100 Trades für reliable Schätzung)
    statistically_valid = num_trades >= 100

    results = {
        "symbol":               symbol,
        "label":                label,
        "execution_delay":      execution_delay,
        # Benchmark
        "always_ann_pct":       always_ann * 100,
        # Brutto / Netto
        "gross_return_pct":     gross_return * 100,
        "fees_pct":             total_fees * 100,
        "net_return_pct":       net_return * 100,
        "net_ann_pct":          net_ann * 100,
        # Risikometriken
        "sharpe":               round(sharpe, 3),
        "sortino":              round(sortino, 3),
        "calmar":               round(calmar, 3),
        "max_drawdown_pct":     round(max_dd_pct, 3),
        # Aktivitätsmetriken
        "num_trades":           num_trades,
        "trade_count":          num_trades,           # alias für CSV
        "time_in_market_pct":   round(time_in_market, 1),
        "avg_hold_days":        round(avg_hold_days, 1),
        "win_rate_pct":         round(win_rate, 1),
        # Gebühren-Breakdown (NEU)
        "total_fees_paid_pct":    round(total_fees_paid_pct, 4),
        "total_fees_paid_usd":    round(total_fees_paid_usd, 2),
        "avg_cost_per_trade_pct": round(avg_cost_per_trade_pct, 4),
        "avg_cost_per_trade_usd": round(avg_cost_per_trade_usd, 2),
        "statistically_valid":    statistically_valid,
        # Kapital
        "capital_start":          CAPITAL,
        "capital_end":            round(CAPITAL * equity_end, 2),
        # Rohwerte für Portfolio-Aggregation
        "_full_returns":    full_returns,
        "_equity":          equity,
        "_in_signal":       in_signal,
    }

    _print_results(results)
    return results


def _print_results(r: dict):
    sym = f" [{r['symbol']}]" if r.get("symbol") else ""
    print("\n" + "=" * 65)
    print(f"  BACKTEST{sym}: {r['label']}")
    print("=" * 65)
    delay_str = f"  [Execution Delay: {r['execution_delay']} Periode(n) = {r['execution_delay']*8}h]"
    print(delay_str)
    print(f"  Benchmark (immer drin):      {r['always_ann_pct']:+6.2f}% p.a.")
    print()
    print(f"  Strategie Gross Return:      {r['gross_return_pct']:+6.2f}%")
    print(f"  - Gebühren ({r['num_trades']:3d} Trades × 4 Legs):  -{r['fees_pct']:.3f}%")
    print(f"    [Maker {r['num_trades']}×{MAKER_FEE*100:.2f}%×4 + Slippage {r['num_trades']}×{SLIPPAGE*10_000:.0f}bps×4]")
    print(f"    Ø Kosten/Trade: {r['avg_cost_per_trade_pct']:.4f}% ({r['avg_cost_per_trade_usd']:.2f} USD)")
    print(f"    Total Fees:     {r['total_fees_paid_pct']:.3f}% = {r['total_fees_paid_usd']:.2f} USD")
    stat = "✓ statistisch valid (≥100 Trades)" if r['statistically_valid'] else f"⚠ nur {r['num_trades']} Trades – zu wenig"
    print(f"    [{stat}]")
    print(f"  = Netto Return:              {r['net_return_pct']:+6.2f}%  ({r['net_ann_pct']:+.2f}% p.a. CAGR)")
    print()
    print(f"  Risikometriken:")
    print(f"    Sharpe Ratio:   {r['sharpe']:6.3f}   (>1.0 = gut, >2.0 = sehr gut)")
    print(f"    Sortino Ratio:  {r['sortino']:6.3f}   (nur Downside-Vola)")
    print(f"    Calmar Ratio:   {r['calmar']:6.3f}   (CAGR / Max Drawdown)")
    print(f"    Max Drawdown:   {r['max_drawdown_pct']:6.2f}%")
    print()
    print(f"  Aktivität:")
    print(f"    Zeit im Markt:  {r['time_in_market_pct']:5.1f}%")
    print(f"    Ø Haltedauer:   {r['avg_hold_days']:5.1f} Tage")
    print(f"    Win Rate:       {r['win_rate_pct']:5.1f}%  (% pos. Perioden im Markt)")
    print()
    print(f"  Kapital: ${r['capital_start']:,.0f} → ${r['capital_end']:,.0f}")
    print("=" * 65)
