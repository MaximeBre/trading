"""
main.py – Entry Point: Multi-Asset Funding Rate Arbitrage System
=================================================================
Phase 1 + 2 Pipeline:

  1. Daten laden       – Paginierte 3-Jahres-Historie pro Asset
  2. Features bauen    – Standard-Features pro Asset (mit Bugfixes)
  3. Cross-Asset       – Proprietäre Multi-Asset Features (neu)
  4. Statistiken       – Pro Asset + Cross-Asset Summary
  5. Backtests         – Pro Asset (korrigiertes Fee-Modell)
  6. Portfolio         – Gewichtetes Portfolio über alle Assets
  7. Plots             – Pro Asset + Portfolio-Übersicht
  8. Export            – CSV pro Asset

Ausführen:
    cd crypto_quant
    python main.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SYMBOLS, SYMBOL_WEIGHTS, SYMBOL_SHORT,
    FUNDING_DAYS_FULL, OUTPUT_DIR, DATA_DIR,
    COST_PER_ROUNDTRIP, FUNDING_THRESHOLDS,
)
from data import (
    get_funding_rates_paginated,
    get_funding_rates,
    get_basis_history,
    get_bybit_funding_rates,
    get_predicted_funding_rate_history,
    get_btc_dominance_history,
)
from data.okx import get_okx_funding_rates
from data.stablecoins import get_combined_stablecoin_supply
from features import build_all_features
from features.engineering import build_cross_asset_features, build_labels
from analysis import print_stats
from analysis.stats import print_cross_asset_summary
from analysis.plots import plot_portfolio, plot_single_asset, plot_ic_heatmap
from analysis.ic_analysis import run_ic_report
from backtest import run_backtest
from backtest.portfolio import run_portfolio_backtest
from generate_dashboard import generate_dashboard


def main():
    print("\n" + "=" * 65)
    print("  CRYPTO QUANT SYSTEM – Multi-Asset Funding Rate Arbitrage")
    print("=" * 65)
    print(f"  Assets:    {', '.join(SYMBOLS)}")
    print(f"  Gewichte:  {', '.join([f'{SYMBOL_SHORT[s].upper()} {v*100:.0f}%' for s, v in SYMBOL_WEIGHTS.items()])}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 1: Daten laden (alle Assets)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 1: Daten laden")
    print(f"{'─'*65}")

    dfs_rates     = {}
    dfs_bybit     = {}
    dfs_basis     = {}
    dfs_predicted = {}
    dfs_okx       = {}

    for symbol in SYMBOLS:
        key = SYMBOL_SHORT[symbol].upper()
        print(f"\n  [{key}] Lade Daten...")

        try:
            dfs_rates[symbol] = get_funding_rates_paginated(symbol, days=FUNDING_DAYS_FULL)
        except Exception as e:
            print(f"  Paginated fetch fehlgeschlagen ({e}), Fallback auf 1000 Einträge")
            dfs_rates[symbol] = get_funding_rates(symbol, limit=1000)

        try:
            dfs_bybit[symbol] = get_bybit_funding_rates(symbol, limit=200)
        except Exception as e:
            print(f"  Bybit nicht verfügbar: {e}")

        try:
            dfs_basis[symbol] = get_basis_history(symbol, interval="8h", limit=500)
        except Exception as e:
            print(f"  Basis History nicht verfügbar: {e}")

        try:
            dfs_predicted[symbol] = get_predicted_funding_rate_history(symbol, limit=500)
        except Exception as e:
            print(f"  Predicted Funding nicht verfügbar: {e}")

        try:
            dfs_okx[symbol] = get_okx_funding_rates(symbol, limit=500)
        except Exception as e:
            print(f"  OKX nicht verfügbar: {e}")

    df_stable = None
    try:
        df_stable = get_combined_stablecoin_supply()
    except Exception as e:
        print(f"\n  Stablecoin Daten nicht verfügbar: {e}")

    df_dominance = None
    try:
        df_dominance = get_btc_dominance_history(symbols=SYMBOLS)
    except Exception as e:
        print(f"\n  BTC OI Dominanz nicht verfügbar: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 2: Standard-Features pro Asset
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 2: Feature Engineering (Single-Asset)")
    print(f"{'─'*65}")

    dfs_features = {}
    for symbol in SYMBOLS:
        key = SYMBOL_SHORT[symbol].upper()
        print(f"\n  [{key}]")
        dfs_features[symbol] = build_all_features(
            df_rates     = dfs_rates[symbol],
            df_bybit     = dfs_bybit.get(symbol),
            df_basis     = dfs_basis.get(symbol),
            df_stable    = df_stable,
            df_predicted = dfs_predicted.get(symbol),
            df_dominance = df_dominance,
            df_okx       = dfs_okx.get(symbol),
            add_labels   = False,   # Labels erst nach Cross-Asset Features
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 3: Cross-Asset Features
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 3: Cross-Asset Feature Engineering")
    print(f"{'─'*65}")

    try:
        dfs_features = build_cross_asset_features(
            dfs     = dfs_features,
            symbols = SYMBOLS,
            weights = SYMBOL_WEIGHTS,
        )
        sample_cols = dfs_features[SYMBOLS[0]].columns
        n_cross = sum(1 for c in sample_cols if any(
            c.startswith(p) for p in [
                "btc_eth_spread", "btc_sol_spread", "eth_sol_spread",
                "hierarchy_", "all_above_zscore", "sync_score",
                "btc_falling_", "rotation_direction", "portfolio_rate_",
                "rel_score_",
            ]
        ))
        print(f"  + {n_cross} Cross-Asset Features hinzugefügt")
    except Exception as e:
        print(f"  Cross-Asset Features fehlgeschlagen: {e}")

    for symbol in SYMBOLS:
        threshold = FUNDING_THRESHOLDS.get(symbol, 0.0001)
        dfs_features[symbol] = build_labels(dfs_features[symbol],
                                            threshold=threshold)

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 4: Statistiken
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 4: Statistiken")
    print(f"{'─'*65}")

    for symbol in SYMBOLS:
        print_stats(dfs_features[symbol], symbol=symbol)

    try:
        print_cross_asset_summary(dfs_features, SYMBOLS)
    except Exception as e:
        print(f"  Cross-Asset Summary fehlgeschlagen: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 4.5: IC-Analyse (Feature Quality & Stability)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 4.5: IC-Analyse – Feature Quality & Stability")
    print(f"{'─'*65}")

    ic_reports = {}
    for symbol in SYMBOLS:
        try:
            ic_reports[symbol] = run_ic_report(
                symbol      = symbol,
                features_df = dfs_features[symbol],
                target_col  = "target_next_rate",
            )
        except Exception as e:
            print(f"  [{SYMBOL_SHORT[symbol].upper()}] IC-Analyse fehlgeschlagen: {e}")
            ic_reports[symbol] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 5: Single-Asset Backtests
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 5: Backtests – Pro Asset")
    print("  [4 Legs × (Maker + Slippage) pro Roundtrip]")
    print(f"{'─'*65}")

    backtest_results = {}
    for symbol in SYMBOLS:
        backtest_results[symbol] = run_backtest(
            df     = dfs_features[symbol],
            label  = "Rule-based (Rate > 0.01%)",
            symbol = symbol,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 6: Portfolio-Backtest
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 6: Portfolio-Backtest (Multi-Asset)")
    print(f"{'─'*65}")

    portfolio_result = None
    try:
        portfolio_result = run_portfolio_backtest(
            dfs     = dfs_features,
            symbols = SYMBOLS,
            weights = SYMBOL_WEIGHTS,
        )
    except Exception as e:
        print(f"  Portfolio-Backtest fehlgeschlagen: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 7: Plots
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 7: Visualisierungen")
    print(f"{'─'*65}")

    for symbol in SYMBOLS:
        plot_single_asset(dfs_features[symbol], symbol=symbol, save=True)

    if portfolio_result is not None:
        try:
            plot_portfolio(
                dfs              = dfs_features,
                portfolio_result = portfolio_result,
                symbols          = SYMBOLS,
                save             = True,
            )
        except Exception as e:
            print(f"  Portfolio-Plot fehlgeschlagen: {e}")

    # IC-Heatmaps (nutzt gespeicherte ic_report CSVs)
    for symbol in SYMBOLS:
        try:
            ic_series = ic_reports.get(symbol, {}).get("ic_series")
            plot_ic_heatmap(symbol, ic_df=ic_series, save=True)
        except Exception as e:
            print(f"  IC-Heatmap [{SYMBOL_SHORT[symbol].upper()}] fehlgeschlagen: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 8: Export
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  SCHRITT 8: Export")
    print(f"{'─'*65}")

    for symbol in SYMBOLS:
        key  = SYMBOL_SHORT[symbol]
        path = os.path.join(DATA_DIR, f"{symbol}_features.csv")
        dfs_features[symbol].to_csv(path, index=False)
        n = len(dfs_features[symbol])
        f = len([c for c in dfs_features[symbol].columns
                 if c not in {"fundingTime", "target_next_positive",
                               "target_next_3", "target_next_rate",
                               "target_label_ordinal", "rate_annualized_pct"}])
        print(f"  {key.upper()}: {path}  ({n} Zeilen, {f} Features)")

    # ── backtest_results.csv (inkl. total_fees_paid, avg_cost_per_trade) ──────
    skip_raw = {"_full_returns", "_equity", "_in_signal"}
    bt_rows  = []
    for symbol, res in backtest_results.items():
        row = {k: v for k, v in res.items() if k not in skip_raw}
        bt_rows.append(row)

    if bt_rows:
        bt_csv_path = os.path.join(OUTPUT_DIR, "backtest_results.csv")
        pd.DataFrame(bt_rows).to_csv(bt_csv_path, index=False)
        print(f"  backtest_results.csv: {bt_csv_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCHRITT 9: ABSCHLIESSENDE VALIDIERUNG
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print("  ABSCHLIESSENDE VALIDIERUNG")
    print(f"{'═'*65}")

    # ── 9a. Trade Count & Statistische Validität ───────────────────────────
    print("\n  Trade Count Validität:")
    for symbol, res in backtest_results.items():
        key    = SYMBOL_SHORT[symbol].upper()
        n_tr   = res.get("num_trades", 0)
        valid  = res.get("statistically_valid", False)
        marker = "✓" if valid else "⚠"
        print(f"    {marker} {key}: {n_tr} Trades  "
              f"({'statistisch valid' if valid else 'zu wenig – Ergebnis unsicher'})")

    # ── 9b. Effektive Gesamtkosten pro Trade ──────────────────────────────
    print(f"\n  Effektive Kosten pro Trade (4-Leg Delta-Neutral):")
    print(f"    COST_PER_ROUNDTRIP = {COST_PER_ROUNDTRIP*100:.4f}%")
    for symbol, res in backtest_results.items():
        key = SYMBOL_SHORT[symbol].upper()
        print(f"    {key}: Ø {res.get('avg_cost_per_trade_pct', 0):.4f}%  "
              f"= {res.get('avg_cost_per_trade_usd', 0):.2f} USD  |  "
              f"Total: {res.get('total_fees_paid_usd', 0):.2f} USD "
              f"({res.get('total_fees_paid_pct', 0):.3f}% des Kapitals)")

    # ── 9c. Top-3 Features nach ICIR ──────────────────────────────────────
    print(f"\n  Top-3 Features nach ICIR:")
    for symbol in SYMBOLS:
        key     = SYMBOL_SHORT[symbol].upper()
        report  = ic_reports.get(symbol, {})
        icir_df = report.get("icir_df", pd.DataFrame())
        if len(icir_df) == 0:
            print(f"    {key}: IC-Report nicht verfügbar")
            continue
        top3 = icir_df[icir_df["icir"].notna()].head(3)
        print(f"    {key}:")
        for _, row in top3.iterrows():
            hl_str = (f"  HL={row['decay_halflife_hours']:.0f}h"
                      if pd.notna(row.get("decay_halflife_hours")) else "")
            print(f"      {row['feature']:<32}  ICIR={row['icir']:+.3f}  "
                  f"[{row.get('quality', '?')}]{hl_str}")

    # ── 9d. Output-Dateien prüfen ─────────────────────────────────────────
    print(f"\n  Output-Dateien:")
    check_files = (
        [os.path.join(OUTPUT_DIR, "backtest_results.csv")]
        + [os.path.join(OUTPUT_DIR, f"ic_summary_{s}.csv") for s in SYMBOLS]
        + [os.path.join(OUTPUT_DIR, f"ic_heatmap_{SYMBOL_SHORT[s]}.png") for s in SYMBOLS]
    )
    for fpath in check_files:
        exists = os.path.exists(fpath)
        print(f"    {'✓' if exists else '✗'} {os.path.basename(fpath)}")

    # ── SCHRITT 10: Dashboard generieren ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  SCHRITT 10: Dashboard")
    print(f"{'─'*65}")
    try:
        generate_dashboard()
    except Exception as e:
        print(f"  Dashboard fehlgeschlagen: {e}")

    print(f"\n{'═'*65}")
    print("  PIPELINE ABGESCHLOSSEN")
    print(f"{'═'*65}")
    print()
    print("  Nächste Schritte (Phase 2):")
    print("    python models/train.py           ← XGBoost Walk-Forward Training")
    print("    python models/regime.py          ← GMM Regime Classifier")
    print("    python models/alpha.py           ← 3-Modell Alpha Ensemble")
    print("    python models/portfolio_constructor.py  ← Layer 4 + Optuna")
    print("    python execution/scenarios.py   ← Pre-Live Scenario Tests")
    print("    python execution/paper_trading.py ← Paper Trading starten")
    print()


if __name__ == "__main__":
    main()
