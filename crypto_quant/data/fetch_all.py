"""
data/fetch_all.py – Vollständiger Daten-Download für alle Assets
================================================================
Standalone-Script: lädt alles und speichert als CSV.
Muss VOR models/train.py ausgeführt werden.

Ausführen:
    cd crypto_quant
    python data/fetch_all.py

Was passiert:
  1. Paginierte Funding Rates (3 Jahre) für BTC, ETH, SOL
  2. Bybit Funding Rates (Cross-Exchange Divergenz)
  3. Basis History (Mark - Index Price)
  4. Open Interest History (inkl. Veränderungs-Features)
  5. Stablecoin Supply (USDT + USDC kombiniert, mit Look-ahead Fix)
  6. Vollständige Feature Pipeline pro Asset
  7. Cross-Asset Features
  8. Speichern nach data/raw/{symbol}_features.csv

Output Summary zeigt an ob genug Daten für Training vorhanden.
Minimum für Walk-Forward: 1500 Perioden (~500 Tage) pro Asset.
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    SYMBOLS, SYMBOL_WEIGHTS, SYMBOL_SHORT,
    EXPANSION_SYMBOLS, EXPANSION_WEIGHTS,
    FUNDING_DAYS_FULL, DATA_DIR,
)
from data.binance import (
    get_funding_rates_paginated,
    get_basis_history,
    get_open_interest_history,
)
from data.bybit import get_bybit_funding_rates
from data.stablecoins import get_combined_stablecoin_supply
from features.engineering import (
    build_all_features,
    build_cross_asset_features,
    build_labels,
)

os.makedirs(DATA_DIR, exist_ok=True)

MIN_PERIODS_FOR_TRAINING = 1500   # ~500 Tage – Minimum für Walk-Forward


def fetch_and_save(symbols=None, weights=None):
    """
    Lädt Daten für alle angegebenen Assets.

    symbols: Liste der Symbols (Standard: SYMBOLS aus config)
    weights: Dict {symbol: weight} (Standard: SYMBOL_WEIGHTS)
    """
    if symbols is None:
        symbols = SYMBOLS
    if weights is None:
        weights = SYMBOL_WEIGHTS

    print("\n" + "=" * 65)
    print("  DATA FETCH – Multi-Asset Funding Rate Arbitrage")
    print(f"  Assets: {[SYMBOL_SHORT.get(s, s) for s in symbols]}")
    print(f"  Ziel: {FUNDING_DAYS_FULL} Tage pro Asset ({FUNDING_DAYS_FULL * 3} Perioden)")
    print("=" * 65)

    dfs_rates = {}
    dfs_bybit = {}
    dfs_basis = {}
    dfs_oi    = {}

    # ── Pro Asset: alle Daten laden ───────────────────────────────────────────
    for symbol in symbols:
        key = SYMBOL_SHORT[symbol].upper()
        print(f"\n{'─'*55}")
        print(f"  [{key}] Lade alle Datenquellen...")
        print(f"{'─'*55}")

        # 1. Funding Rates (paginated – das Herzstück)
        dfs_rates[symbol] = get_funding_rates_paginated(symbol, days=FUNDING_DAYS_FULL)
        time.sleep(0.5)

        # 2. Bybit Cross-Exchange
        try:
            dfs_bybit[symbol] = get_bybit_funding_rates(symbol, limit=200)
            time.sleep(0.3)
        except Exception as e:
            print(f"  Bybit fehlgeschlagen: {e}")

        # 3. Basis (Mark - Index Price)
        try:
            dfs_basis[symbol] = get_basis_history(symbol, interval="8h", limit=500)
            time.sleep(0.3)
        except Exception as e:
            print(f"  Basis fehlgeschlagen: {e}")

        # 4. Open Interest
        try:
            dfs_oi[symbol] = get_open_interest_history(symbol, limit=500)
            time.sleep(0.3)
        except Exception as e:
            print(f"  OI fehlgeschlagen: {e}")

    # ── Stablecoin (einmal für alle Assets) ───────────────────────────────────
    print(f"\n{'─'*55}")
    print("  [STABLE] Lade Stablecoin Supply (USDT + USDC)...")
    print(f"{'─'*55}")

    df_stable = None
    try:
        df_stable = get_combined_stablecoin_supply()
    except Exception as e:
        print(f"  Stablecoin fehlgeschlagen: {e}")

    # ── Feature Engineering pro Asset ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Feature Engineering...")
    print(f"{'─'*55}")

    dfs_features = {}
    for symbol in symbols:
        key = SYMBOL_SHORT.get(symbol, symbol[:3]).upper()
        print(f"\n  [{key}]")
        dfs_features[symbol] = build_all_features(
            df_rates   = dfs_rates[symbol],
            df_bybit   = dfs_bybit.get(symbol),
            df_basis   = dfs_basis.get(symbol),
            df_stable  = df_stable,
            df_oi      = dfs_oi.get(symbol),
            add_labels = False,
        )

    # ── Cross-Asset Features ───────────────────────────────────────────────────
    try:
        dfs_features = build_cross_asset_features(
            dfs=dfs_features, symbols=symbols, weights=weights
        )
        print(f"\n  Cross-Asset Features hinzugefügt")
    except Exception as e:
        print(f"  Cross-Asset Features fehlgeschlagen: {e}")

    # ── Labels (immer zuletzt) ─────────────────────────────────────────────────
    for symbol in symbols:
        dfs_features[symbol] = build_labels(dfs_features[symbol])

    # ── Speichern ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Speichern...")
    print(f"{'─'*55}")

    for symbol in symbols:
        path = os.path.join(DATA_DIR, f"{symbol}_features.csv")
        dfs_features[symbol].to_csv(path, index=False)
        print(f"  Gespeichert: {path}")

    # ── Summary & Validierung ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FETCH SUMMARY")
    print(f"{'='*65}")

    all_ready = True
    for symbol in symbols:
        df   = dfs_features[symbol]
        key  = SYMBOL_SHORT.get(symbol, symbol[:3]).upper()
        n    = len(df)
        days = n / 3
        feature_cols = [c for c in df.columns
                        if c not in {"fundingTime", "target_next_rate",
                                     "target_next_positive", "target_next_3",
                                     "target_label_ordinal", "rate_annualized_pct"}]
        n_features = len(feature_cols)

        # NaN-Check: Wie viele Zeilen haben alle Features?
        n_complete = df[feature_cols].dropna().shape[0]

        ready = n_complete >= MIN_PERIODS_FOR_TRAINING
        if not ready:
            all_ready = False

        status = "READY" if ready else f"ZU WENIG ({n_complete} < {MIN_PERIODS_FOR_TRAINING})"
        print(f"\n  {key}:")
        print(f"    Perioden gesamt:      {n:>5}  (~{days:.0f} Tage)")
        print(f"    Vollständige Zeilen:  {n_complete:>5}  [{status}]")
        print(f"    Features:             {n_features:>5}")
        print(f"    Zeitraum:             {df['fundingTime'].min().date()} → "
              f"{df['fundingTime'].max().date()}")

        # Feature-Completeness pro Kategorie
        basis_cols    = [c for c in feature_cols if "basis"   in c]
        oi_cols       = [c for c in feature_cols if "oi_"     in c]
        stable_cols   = [c for c in feature_cols if "usdt"    in c or "stablecoin" in c]
        cross_cols    = [c for c in feature_cols if any(x in c for x in
                         ["spread", "sync", "hierarchy", "rotation", "portfolio_rate",
                          "rel_score", "btc_falling"])]
        bybit_cols    = [c for c in feature_cols if "divergence" in c or "premium" in c]

        print(f"    Feature-Gruppen:")
        for group, cols in [("Basis", basis_cols), ("OI", oi_cols),
                             ("Stablecoin", stable_cols), ("Cross-Asset", cross_cols),
                             ("Cross-Exchange", bybit_cols)]:
            if cols:
                nan_pct = df[cols].isna().mean().mean() * 100
                print(f"      {group:<15}: {len(cols):>2} Features  "
                      f"({nan_pct:.0f}% NaN)")
            else:
                print(f"      {group:<15}: NICHT VERFÜGBAR")

    print(f"\n{'─'*65}")
    if all_ready:
        print("  ALLE ASSETS READY → python models/train.py ausführen")
    else:
        print("  WARNUNG: Nicht alle Assets haben genug Daten.")
        print("  Prüfe API-Erreichbarkeit und erhöhe FUNDING_DAYS_FULL in config.py")
    print("=" * 65)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch funding rate data für alle Assets."
    )
    parser.add_argument(
        "--expand", action="store_true",
        help="Expansion Assets laden (BNB, AVAX, LINK, DOT, MATIC, LTC, ADA)"
    )
    parser.add_argument(
        "--all", dest="all_assets", action="store_true",
        help="Core + Expansion Assets laden (alle 10 Assets)"
    )
    args = parser.parse_args()

    if args.all_assets:
        print("  Modus: ALLE Assets (Core + Expansion)")
        combined = SYMBOLS + [s for s in EXPANSION_SYMBOLS if s not in SYMBOLS]
        combined_weights = {**SYMBOL_WEIGHTS, **EXPANSION_WEIGHTS}
        # Normalisiere Weights
        total_w = sum(combined_weights.values())
        combined_weights = {k: v / total_w for k, v in combined_weights.items()}
        fetch_and_save(symbols=combined, weights=combined_weights)
    elif args.expand:
        print("  Modus: EXPANSION Assets (BNB, AVAX, LINK, DOT, MATIC, LTC, ADA)")
        fetch_and_save(symbols=EXPANSION_SYMBOLS, weights=EXPANSION_WEIGHTS)
    else:
        # Standard: Core Portfolio (BTC, ETH, SOL)
        fetch_and_save()
