"""
data/stablecoins.py – Stablecoin Supply Daten via DeFiLlama
=============================================================
Komplett kostenlos, kein API-Key nötig.

WARUM STABLECOIN INFLOWS EIN LEADING INDICATOR SIND:
Stablecoins (USDT, USDC) sind das "Trockenpulver" des Crypto-Markts.
Jemand der Crypto kaufen will, wandelt zuerst Fiat in Stablecoins um –
BEVOR er BTC oder andere Assets kauft.

Wachsende Stablecoin Supply = frisches Kapital kommt rein
→ Dieses Kapital wird bald in Longs investiert
→ Funding Rates steigen

Schrumpfende Supply = Kapital verlässt den Markt
→ Long-Positionen werden abgebaut
→ Funding Rates fallen

Das Zeitfenster zwischen Stablecoin-Inflow und Funding Rate Anstieg
beträgt typischerweise 3-14 Tage – das ist unser Edge.
"""

import requests
import pandas as pd
from config import DEFILLAMA_URL


def get_stablecoin_inflows(stablecoin_id: int = 1) -> pd.DataFrame:
    """
    Historische Stablecoin Supply von DeFiLlama.

    Stablecoin IDs (die wichtigsten):
        1  = USDT (Tether)       – größte, wichtigste
        2  = USDC (Circle)       – zweitgrößte
        3  = BUSD (Binance USD)  – deprecated
        6  = DAI                 – dezentral

    Features die berechnet werden:
        usdt_mcap               : Absolute Supply in USD
        usdt_inflow_7d          : Absolute Veränderung über 7 Tage
        usdt_inflow_7d_pct      : % Veränderung über 7 Tage ← Hauptsignal
        usdt_inflow_3d_pct      : % Veränderung über 3 Tage ← schnelleres Signal
        stablecoin_momentum     : Beschleunigt der Inflow gerade?
        stablecoin_inflow_zscore: Wie ungewöhnlich hoch ist der Inflow?

    Returns:
        DataFrame mit täglichen Datenpunkten, wird später auf 8h interpoliert.
    """
    print(f"  [DeFiLlama] Stablecoin Supply (ID={stablecoin_id})...")
    r = requests.get(
        f"{DEFILLAMA_URL}/stablecoincharts/all",
        params={"stablecoin": stablecoin_id},
        timeout=15
    )
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"].astype(int), unit="s", utc=True)
    df["totalCirculating"] = df["totalCirculating"].apply(
        lambda x: float(x.get("peggedUSD", 0)) if isinstance(x, dict) else 0.0
    )
    df = df.rename(columns={"date": "fundingTime", "totalCirculating": "usdt_mcap"})
    df = df.sort_values("fundingTime").reset_index(drop=True)

    # Inflow Features
    df["usdt_inflow_7d"]      = df["usdt_mcap"].diff(7)
    df["usdt_inflow_3d"]      = df["usdt_mcap"].diff(3)
    df["usdt_inflow_7d_pct"]  = df["usdt_mcap"].pct_change(7) * 100
    df["usdt_inflow_3d_pct"]  = df["usdt_mcap"].pct_change(3) * 100

    # Beschleunigung: Inflow schneller oder langsamer als letzte Woche?
    df["stablecoin_momentum"] = df["usdt_inflow_7d"].diff(3)

    # Z-Score
    roll_mean = df["usdt_inflow_7d_pct"].rolling(30).mean()
    roll_std  = df["usdt_inflow_7d_pct"].rolling(30).std()
    df["stablecoin_inflow_zscore"] = (df["usdt_inflow_7d_pct"] - roll_mean) / roll_std

    print(f"  ✓ {len(df)} Tages-Datenpunkte "
          f"({df['fundingTime'].min().date()} → {df['fundingTime'].max().date()})")
    return df[[
        "fundingTime", "usdt_mcap",
        "usdt_inflow_7d", "usdt_inflow_3d",
        "usdt_inflow_7d_pct", "usdt_inflow_3d_pct",
        "stablecoin_momentum", "stablecoin_inflow_zscore"
    ]]


def get_combined_stablecoin_supply() -> pd.DataFrame:
    """
    Lädt USDT + USDC Supply und kombiniert sie zu einem Gesamt-Signal.
    Robuster als nur USDT, weil manchmal Kapital zwischen den beiden rotiert.
    """
    print("  [DeFiLlama] Kombinierte Stablecoin Supply (USDT + USDC)...")

    # USDT
    df_usdt = get_stablecoin_inflows(stablecoin_id=1)
    df_usdt = df_usdt.rename(columns={
        "usdt_mcap":               "usdt_supply",
        "usdt_inflow_7d_pct":      "usdt_inflow_7d_pct",
    })[["fundingTime", "usdt_supply", "usdt_inflow_7d_pct"]]

    # USDC
    try:
        df_usdc = get_stablecoin_inflows(stablecoin_id=2)
        df_usdc = df_usdc.rename(columns={
            "usdt_mcap":           "usdc_supply",
            "usdt_inflow_7d_pct":  "usdc_inflow_7d_pct",
        })[["fundingTime", "usdc_supply", "usdc_inflow_7d_pct"]]

        df = pd.merge(df_usdt, df_usdc, on="fundingTime", how="outer").sort_values("fundingTime")
        df["total_stablecoin_supply"] = df["usdt_supply"].fillna(0) + df["usdc_supply"].fillna(0)
        df["total_inflow_7d_pct"]     = (
            df["usdt_inflow_7d_pct"].fillna(0) + df["usdc_inflow_7d_pct"].fillna(0)
        ) / 2

    except Exception:
        # Fallback: nur USDT
        df = df_usdt.copy()
        df["total_stablecoin_supply"] = df["usdt_supply"]
        df["total_inflow_7d_pct"]     = df["usdt_inflow_7d_pct"]

    print(f"  ✓ Kombinierte Supply: {len(df)} Datenpunkte")
    return df
