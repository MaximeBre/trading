"""
analysis/stats.py – Statistische Auswertungen
===============================================
Gibt detaillierte Kennzahlen aus – pro Asset oder über alle Assets.

Neu:
  - print_stats() akzeptiert jetzt symbol-Parameter (multi-asset kompatibel)
  - print_regime_analysis(): Erkennt aktuelle Marktphase (Bull/Bear/Sideways)
  - print_cross_asset_summary(): Gibt Cross-Asset Signals zusammen aus
"""

import pandas as pd
import numpy as np
from config import SYMBOL_BINANCE, FUNDING_THRESHOLD, SYMBOL_SHORT


def print_stats(df: pd.DataFrame, symbol: str = None):
    """
    Gibt alle wichtigen Kennzahlen für ein einzelnes Asset aus.

    Args:
        df     : Feature DataFrame mit fundingRate Spalte
        symbol : Symbol-Name für die Ausgabe (z.B. "BTCUSDT")
    """
    sym = symbol or SYMBOL_BINANCE
    key = SYMBOL_SHORT.get(sym, sym[:3].upper())
    r   = df["fundingRate"]
    ann = df["rate_annualized_pct"]

    print("\n" + "=" * 65)
    print(f"  FUNDING RATE STATISTIKEN – {sym} [{key.upper()}]")
    print("=" * 65)
    print(f"  Zeitraum:            {df['fundingTime'].min().date()} → "
          f"{df['fundingTime'].max().date()}")
    print(f"  Datenpunkte:         {len(df)}  "
          f"({len(df) / (3 * 365):.1f} Jahre)")
    print()
    print(f"  Ø Rate pro Periode:  {r.mean()*100:+.4f}%")
    print(f"  Ø Rate annualisiert: {ann.mean():+.1f}% p.a.")
    print(f"  Median Rate:         {r.median()*100:+.4f}%")
    print(f"  Max Rate:            {r.max()*100:+.4f}%  ({ann.max():+.0f}% p.a.)")
    print(f"  Min Rate:            {r.min()*100:+.4f}%  ({ann.min():+.0f}% p.a.)")
    print(f"  Std Rate:            {r.std()*100:.4f}%")
    print()
    print(f"  % Perioden positiv:  {(r > 0).mean()*100:.1f}%")
    print(f"  % Perioden > 0.01%:  {(r > FUNDING_THRESHOLD).mean()*100:.1f}%  ← lohnend")
    print(f"  % Perioden negativ:  {(r < 0).mean()*100:.1f}%")
    print()

    cumret_always = (1 + r).cumprod().iloc[-1] - 1
    only_above    = r[r > FUNDING_THRESHOLD].sum()
    print(f"  Kum. Return (immer drin):         {cumret_always*100:+.1f}%")
    print(f"  Kum. Return (nur > 0.01%):        {only_above*100:+.1f}%")

    print_regime_analysis(df)

    if "cross_divergence" in df.columns:
        div = df["cross_divergence"].dropna()
        print(f"\n  Cross-Exchange Divergenz (Binance − Bybit):")
        print(f"    Ø Divergenz:       {div.mean()*100:+.4f}%")
        print(f"    Max Divergenz:     {div.max()*100:+.4f}%")
        print(f"    % Zeit > ±0.01%:  {(div.abs() > 0.0001).mean()*100:.0f}%")

    if "basis_pct" in df.columns:
        basis     = df["basis_pct"].dropna()
        corr      = df["fundingRate"].corr(df["basis_pct"])
        corr_lag1 = df["fundingRate"].shift(-1).corr(df["basis_pct"])
        print(f"\n  Basis (Future − Spot):")
        print(f"    Ø Basis:              {basis.mean():+.4f}%")
        print(f"    Korr. mit Rate:       {corr:.3f}")
        lead_str = "← Leading!" if abs(corr_lag1) > abs(corr) else ""
        print(f"    Korr. nächste Rate:   {corr_lag1:.3f}  {lead_str}")

    if "usdt_inflow_7d_pct" in df.columns:
        sc = df["usdt_inflow_7d_pct"].dropna()
        corr = df["fundingRate"].corr(df["usdt_inflow_7d_pct"])
        lead_corrs = {
            lag: df["fundingRate"].shift(-lag).corr(df["usdt_inflow_7d_pct"])
            for lag in [3, 7, 14, 21]
        }
        best_lag = max(lead_corrs, key=lambda k: abs(lead_corrs[k]))
        print(f"\n  Stablecoin Inflows (USDT 7d):")
        print(f"    Ø Wachstum p.W.:      {sc.mean():+.2f}%")
        print(f"    Korr. mit Rate:       {corr:.3f}")
        print(f"    Stärkster Lead:       {best_lag} Perioden  "
              f"(ρ={lead_corrs[best_lag]:.3f})")

    print("=" * 65)


def print_regime_analysis(df: pd.DataFrame):
    """
    Erkennt das aktuelle Markt-Regime basierend auf der Rate-Historie.

    Regime-Typen:
      Bull:     Rate > 0.03% konsistent → Spekulationshoch
      Normal:   Rate 0.01-0.03% → gesundes Interesse
      Flat:     Rate 0-0.01% → wenig Spekulation
      Bear:     Rate < 0% → Short-Druck überwiegt
    """
    r = df["fundingRate"]
    recent = r.tail(90)   # letzte 30 Tage

    bull_pct   = (recent > 0.0003).mean() * 100
    normal_pct = ((recent >= 0.0001) & (recent <= 0.0003)).mean() * 100
    flat_pct   = ((recent >= 0) & (recent < 0.0001)).mean() * 100
    bear_pct   = (recent < 0).mean() * 100

    if bull_pct > 50:
        regime = "BULL (hohe Spekulation)"
    elif normal_pct > 50:
        regime = "NORMAL (gesundes Funding)"
    elif bear_pct > 30:
        regime = "BEAR (Short-Druck)"
    else:
        regime = "SIDEWAYS (Low Funding)"

    print(f"\n  Regime (letzte 30 Tage):")
    print(f"    Bull  (>0.03%):         {bull_pct:5.1f}%")
    print(f"    Normal (0.01%-0.03%):   {normal_pct:5.1f}%")
    print(f"    Flat  (0%-0.01%):       {flat_pct:5.1f}%")
    print(f"    Bear  (<0%):            {bear_pct:5.1f}%")
    print(f"    → Aktuelles Regime: {regime}")


def print_cross_asset_summary(dfs: dict, symbols: list):
    """
    Gibt einen Überblick über alle Cross-Asset Signals aus.
    Wird nach build_cross_asset_features() aufgerufen.
    """
    print("\n" + "=" * 65)
    print("  CROSS-ASSET SIGNAL SUMMARY")
    print("=" * 65)

    print(f"\n  {'Asset':<8} {'Rate%':>7} {'ZScore':>8} {'Ann.Rate':>9}")
    print("  " + "-" * 35)
    for sym in symbols:
        df  = dfs[sym]
        key = SYMBOL_SHORT[sym].upper()
        r   = df["fundingRate"].iloc[-1]
        z   = df["rate_zscore"].iloc[-1] if "rate_zscore" in df.columns else float("nan")
        ann = r * 3 * 365 * 100
        print(f"  {key:<8} {r*100:>6.4f}%  {z:>7.2f}σ  {ann:>8.1f}%p.a.")

    df_ref = dfs[symbols[0]]
    last   = df_ref.iloc[-1]

    if "sync_score" in df_ref.columns:
        sync   = int(last.get("sync_score", 0))
        hier   = int(last.get("hierarchy_normal", 0))
        all_hi = int(last.get("all_above_zscore_1", 0))
        rot    = last.get("rotation_direction", 0)
        port_r = last.get("portfolio_rate_weighted", float("nan"))

        print(f"\n  Cross-Asset Signale (aktuell):")
        print(f"    Sync Score:            {sync}/3 Assets lohnend")
        print(f"    Hierarchie normal:     {'JA (SOL>ETH>BTC)' if hier else 'NEIN (invertiert!)'}")
        print(f"    Alle über +1σ:         {'JA (stabiles Regime)' if all_hi else 'NEIN'}")
        print(f"    Rotation Direction:    {'+1 (→ SOL/risk-on)' if rot > 0 else '-1 (→ BTC/risk-off)' if rot < 0 else '0 (neutral)'}")
        if not pd.isna(port_r):
            print(f"    Portfolio Rate (gew.): {port_r*100*3*365:.1f}% p.a.")

    if "btc_eth_spread" in df_ref.columns:
        print(f"\n  Rate Spreads (aktuell):")
        for col in ["btc_eth_spread", "btc_sol_spread", "eth_sol_spread"]:
            val  = last.get(col, float("nan"))
            zval = last.get(f"{col}_zscore", float("nan"))
            if not pd.isna(val):
                z_str = f"  (zscore: {zval:+.2f}σ)" if not pd.isna(zval) else ""
                print(f"    {col:<22}: {val*100:+.4f}%{z_str}")

    print("=" * 65)


def print_feature_summary(df: pd.DataFrame):
    """Listet alle Features mit NaN-Anteil auf."""
    skip = {"fundingTime", "target_next_positive", "target_next_3",
            "target_next_rate", "target_label_ordinal", "rate_annualized_pct"}
    cols = [c for c in df.columns if c not in skip]

    print("\n  Feature Übersicht:")
    print(f"  {'#':<4} {'Feature':<40} {'NaN%':>6}  {'Min':>10}  {'Max':>10}")
    print("  " + "-" * 75)
    for i, col in enumerate(cols, 1):
        nan_pct = df[col].isna().mean() * 100
        try:
            mn = f"{df[col].min():.5f}"
            mx = f"{df[col].max():.5f}"
        except Exception:
            mn = mx = "n/a"
        print(f"  {i:<4} {col:<40} {nan_pct:>5.0f}%  {mn:>10}  {mx:>10}")
