"""
backtest/portfolio.py – Multi-Asset Portfolio Backtest
=======================================================
Simuliert das vollständige Delta-Neutral Portfolio über alle 3 Assets
(BTC, ETH, SOL) gleichzeitig mit dynamischer Kapitalallokation.

Portfolio-Design:
  - Kapital aufgeteilt nach SYMBOL_WEIGHTS (40/35/25)
  - Jedes Asset hat sein eigenes Einstiegs-Signal
  - Portfolio-Metriken berücksichtigen Korrelationseffekte
  - Negative Funding-Perioden werden korrekt als Verlust gebucht

Zwei Allokations-Modi:
  1. static_weights  – feste Gewichte aus config.py (Baseline)
  2. signal_weights  – Gewichte dynamisch basierend auf ML-Signal-Stärke
                       (Phase 2: rel_score_{btc/eth/sol} aus Cross-Asset Features)

Kritische Risiko-Szenarien die korrekt abgebildet werden:
  - Rate geht negativ → Verlust wird gebucht (du zahlst als Short-Side)
  - Alle Assets fallen gleichzeitig (Crash-Korrelation) → kein Diversifikations-Schutz
  - Gebühren pro Asset separat gerechnet (nicht gemittelt)

Ausgabe:
  - Per-Asset Equity-Kurven
  - Portfolio Equity-Kurve (gewichtet)
  - Korrelationsmatrix der Asset-Returns
  - Portfolio-Level: Sharpe, Sortino, Calmar, Max Drawdown
  - Vergleich: Static vs. Signal-weighted Allokation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from config import (
    SYMBOLS, SYMBOL_WEIGHTS, SYMBOL_SHORT,
    FUNDING_THRESHOLD, COST_PER_ROUNDTRIP,
    CAPITAL, PERIODS_PER_YEAR,
    MAKER_FEE, SLIPPAGE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _asset_returns(df: pd.DataFrame,
                    signal_col: Optional[str] = None) -> pd.Series:
    """
    Berechnet die Full-Return-Serie für ein Asset (0 wenn out-of-market).

    Returns eine Series aligned auf df.index mit Rate wenn im Trade,
    0 wenn nicht im Trade, und zieht Entry-/Exit-Gebühren ab.

    Die Gebühren werden beim Einstieg (Entry) und Ausstieg (Exit) gebucht,
    nicht verteilt über die Haltedauer.
    """
    df = df.dropna(subset=["fundingRate"]).copy()
    r  = df["fundingRate"]

    if signal_col and signal_col in df.columns:
        in_signal = df[signal_col].astype(bool)
    else:
        in_signal = r > FUNDING_THRESHOLD

    enters = in_signal & (~in_signal.shift(1).fillna(False))
    exits  = (~in_signal) & in_signal.shift(1).fillna(False)

    full_returns = pd.Series(0.0, index=df.index)
    full_returns[in_signal] = r[in_signal]

    # Gebühren: Entry-Kosten am ersten Tag, Exit-Kosten am Austrittstag
    full_returns[enters] -= COST_PER_ROUNDTRIP / 2   # Einstieg: 2 Legs öffnen
    full_returns[exits]  -= COST_PER_ROUNDTRIP / 2   # Ausstieg: 2 Legs schließen

    return full_returns, in_signal


def _compute_metrics(full_returns: pd.Series,
                      n_periods: int) -> dict:
    """Berechnet alle Risikoemetriken auf einer Return-Serie."""
    n_years    = n_periods / PERIODS_PER_YEAR
    equity     = (1 + full_returns).cumprod()
    equity_end = equity.iloc[-1]

    net_ann    = (equity_end ** (1 / n_years) - 1) if n_years > 0 else 0.0

    mean_r = full_returns.mean()
    std_r  = full_returns.std()
    sharpe = (mean_r / std_r * np.sqrt(PERIODS_PER_YEAR)
              if std_r > 0 else 0.0)

    downside = full_returns[full_returns < 0].std()
    sortino  = (mean_r / downside * np.sqrt(PERIODS_PER_YEAR)
                if downside and downside > 0 else 0.0)

    rolling_max  = equity.cummax()
    dd           = (equity - rolling_max) / rolling_max
    max_dd       = dd.min()

    calmar = (net_ann / abs(max_dd) if max_dd != 0 else 0.0)

    return {
        "net_total_pct":  (equity_end - 1) * 100,
        "net_ann_pct":    net_ann * 100,
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "calmar":         round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "capital_end":    round(CAPITAL * equity_end, 2),
        "_equity":        equity,
        "_full_returns":  full_returns,
    }


# ── Portfolio Backtest ─────────────────────────────────────────────────────────

def run_portfolio_backtest(
        dfs: Dict[str, pd.DataFrame],
        symbols: list = SYMBOLS,
        weights: Dict[str, float] = SYMBOL_WEIGHTS,
        signal_col: Optional[str] = None,
        use_signal_weights: bool = False,
) -> dict:
    """
    Vollständiger Portfolio-Backtest über alle Assets.

    Args:
        dfs                : Dict[symbol → Feature-DataFrame]
        symbols            : Symbole in der Verarbeitungsreihenfolge
        weights            : Statische Kapitalgewichte (wird genutzt wenn
                             use_signal_weights=False oder kein rel_score vorhanden)
        signal_col         : Spaltenname für binäres Einstiegssignal (0/1)
        use_signal_weights : Wenn True: Gewichte dynamisch aus rel_score_* Features
                             (Phase 2: ML-basierte Allokation)

    Returns:
        dict mit per-Asset- und Portfolio-Metriken + Equity-Kurven
    """
    print("\n" + "=" * 65)
    print("  PORTFOLIO BACKTEST – Multi-Asset Delta-Neutral")
    print("=" * 65)
    print(f"  Assets:       {', '.join([SYMBOL_SHORT[s].upper() for s in symbols])}")
    print(f"  Gewichte:     {', '.join([f'{SYMBOL_SHORT[s].upper()} {weights[s]*100:.0f}%' for s in symbols])}")
    print(f"  Kapital:      ${CAPITAL:,.0f}")
    mode = "Signal-gewichtet (ML)" if use_signal_weights else "Statisch"
    print(f"  Allokation:   {mode}")
    print()

    # ── Per-Asset Returns berechnen ────────────────────────────────────────────
    asset_returns: Dict[str, pd.Series] = {}
    asset_signals: Dict[str, pd.Series] = {}
    asset_dfs:     Dict[str, pd.DataFrame] = {}

    # Zeitachse alignen (schneidende Zeitbasis)
    common_idx = None
    for sym in symbols:
        df = dfs[sym].dropna(subset=["fundingRate"]).copy()
        df["_fundingTime_floor"] = df["fundingTime"].dt.floor("8h")
        asset_dfs[sym] = df

        ts_set = set(df["_fundingTime_floor"])
        common_idx = ts_set if common_idx is None else common_idx & ts_set

    # Auf gemeinsame Zeitpunkte filtern
    for sym in symbols:
        df = asset_dfs[sym]
        df = df[df["_fundingTime_floor"].isin(common_idx)].copy()
        df = df.sort_values("_fundingTime_floor").reset_index(drop=True)
        asset_dfs[sym] = df

    # Returns pro Asset auf gemeinsamer Zeitbasis
    for sym in symbols:
        df = asset_dfs[sym]
        ret_series, sig_series = _asset_returns(df, signal_col)
        asset_returns[sym] = ret_series.values   # numpy für alignment
        asset_signals[sym] = sig_series.values

    n_periods = len(asset_dfs[symbols[0]])

    # ── Portfolio-Returns kombinieren ──────────────────────────────────────────
    # Statische Gewichte
    static_port_returns = np.zeros(n_periods)
    for sym in symbols:
        static_port_returns += weights[sym] * asset_returns[sym]

    static_metrics = _compute_metrics(
        pd.Series(static_port_returns), n_periods
    )

    # Signal-gewichtete Allokation (dynamisch, wenn rel_score vorhanden)
    signal_metrics = None
    if use_signal_weights:
        signal_metrics = _run_signal_weighted(
            asset_dfs, asset_returns, symbols, n_periods, weights
        )

    # ── Per-Asset Metriken ─────────────────────────────────────────────────────
    per_asset = {}
    for sym in symbols:
        key       = SYMBOL_SHORT[sym]
        metrics   = _compute_metrics(pd.Series(asset_returns[sym]), n_periods)
        num_in    = int(asset_signals[sym].sum())
        num_trades = int(
            np.sum(np.diff(asset_signals[sym].astype(int), prepend=0) == 1)
        )
        per_asset[sym] = {
            **metrics,
            "weight":       weights[sym],
            "num_trades":   num_trades,
            "time_in_pct":  round(num_in / n_periods * 100, 1),
        }

    # ── Korrelationsmatrix der Asset-Returns ───────────────────────────────────
    ret_df = pd.DataFrame(
        {SYMBOL_SHORT[s]: asset_returns[s] for s in symbols}
    )
    corr_matrix = ret_df.corr()

    # ── Ausgabe ────────────────────────────────────────────────────────────────
    _print_portfolio_results(per_asset, static_metrics, corr_matrix,
                              symbols, signal_metrics)

    return {
        "per_asset":       per_asset,
        "static":          static_metrics,
        "signal_weighted": signal_metrics,
        "correlation":     corr_matrix,
        "n_periods":       n_periods,
        "asset_returns":   ret_df,
    }


def _run_signal_weighted(asset_dfs, asset_returns, symbols,
                          n_periods, fallback_weights) -> dict:
    """
    Dynamische Gewichtung basierend auf rel_score_* aus Cross-Asset Features.
    Falls nicht verfügbar: Fallback auf statische Gewichte.
    """
    keys = [SYMBOL_SHORT[s] for s in symbols]
    score_cols = [f"rel_score_{k}" for k in keys]

    # Prüfen ob rel_scores vorhanden
    first_df = asset_dfs[symbols[0]]
    if not all(c in first_df.columns for c in score_cols[:1]):
        print("  [Hinweis] rel_score Features nicht gefunden – "
              "führe Cross-Asset-Feature-Step zuerst aus.")
        return None

    dyn_returns = np.zeros(n_periods)

    for i in range(n_periods):
        period_weights = {}
        score_sum = 0.0
        for sym in symbols:
            key = SYMBOL_SHORT[sym]
            col = f"rel_score_{key}"
            if col in asset_dfs[sym].columns:
                score = float(asset_dfs[sym][col].iloc[i])
                score = max(score, 0.0)   # Keine negativen Gewichte
            else:
                score = fallback_weights[sym]
            period_weights[sym] = score
            score_sum += score

        if score_sum > 0:
            for sym in symbols:
                period_weights[sym] /= score_sum
        else:
            period_weights = fallback_weights

        for sym in symbols:
            dyn_returns[i] += period_weights[sym] * asset_returns[sym][i]

    return _compute_metrics(pd.Series(dyn_returns), n_periods)


def _print_portfolio_results(per_asset, static_metrics, corr_matrix,
                               symbols, signal_metrics):
    """Gibt Portfolio-Ergebnisse formatiert aus."""

    print(f"  {'Asset':<8} {'Gewicht':>7} {'Netto%':>8} {'p.a.%':>8} "
          f"{'Sharpe':>7} {'MaxDD%':>8} {'Trades':>7} {'Zeit%':>7}")
    print("  " + "-" * 62)

    for sym in symbols:
        m   = per_asset[sym]
        key = SYMBOL_SHORT[sym].upper()
        print(f"  {key:<8} {m['weight']*100:>6.0f}%  "
              f"{m['net_total_pct']:>+7.2f}%  "
              f"{m['net_ann_pct']:>+7.2f}%  "
              f"{m['sharpe']:>7.3f}  "
              f"{m['max_drawdown_pct']:>7.2f}%  "
              f"{m['num_trades']:>6}  "
              f"{m['time_in_pct']:>6.1f}%")

    print("  " + "─" * 62)

    sm = static_metrics
    print(f"  {'PORTFOLIO':<8} {'100%':>7}  "
          f"{sm['net_total_pct']:>+7.2f}%  "
          f"{sm['net_ann_pct']:>+7.2f}%  "
          f"{sm['sharpe']:>7.3f}  "
          f"{sm['max_drawdown_pct']:>7.2f}%")

    if signal_metrics:
        swm = signal_metrics
        print(f"  {'PORT(ML)':<8} {'100%':>7}  "
              f"{swm['net_total_pct']:>+7.2f}%  "
              f"{swm['net_ann_pct']:>+7.2f}%  "
              f"{swm['sharpe']:>7.3f}  "
              f"{swm['max_drawdown_pct']:>7.2f}%")

    print()
    print(f"  Sortino:  {sm['sortino']:.3f}   "
          f"Calmar: {sm['calmar']:.3f}   "
          f"Kapital: ${sm['capital_end']:,.0f}")

    print("\n  Return-Korrelationen (Periode-by-Periode):")
    keys = [SYMBOL_SHORT[s].upper() for s in symbols]
    header = f"  {'':>6} " + "  ".join(f"{k:>6}" for k in keys)
    print(header)
    for i, s1 in enumerate(symbols):
        row = f"  {SYMBOL_SHORT[s1].upper():>6} "
        for s2 in symbols:
            val = corr_matrix.loc[SYMBOL_SHORT[s1], SYMBOL_SHORT[s2]]
            row += f"  {val:>6.3f}"
        print(row)

    print()
    print("  Hinweis: Hohe Korrelation (>0.7) in Crash-Szenarien erwartet.")
    print("  Das Portfolio bietet keine echte Diversifikation in Extremphasen.")
    print("=" * 65)
