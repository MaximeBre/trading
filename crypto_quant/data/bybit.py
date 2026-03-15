"""
data/bybit.py – Bybit API Calls
================================
Aktuell: Funding Rates für Cross-Exchange Divergenz.
"""

import requests
import pandas as pd
from config import BYBIT_URL


def get_bybit_funding_rates(symbol: str = "BTCUSDT",
                             limit: int = 200) -> pd.DataFrame:
    """
    Historische Funding Rates von Bybit.

    PROPRIETÄRES FEATURE – Cross-Exchange Divergenz:
    Wenn Binance deutlich mehr zahlt als Bybit (oder umgekehrt), ist das
    ein Signal für ungleiche Marktdynamiken. Große Divergenzen konvergieren
    meist innerhalb von 1-3 Perioden.

    Signal-Logik:
      divergence > +0.02%  → Binance Rate könnte bald fallen (Exit-Signal)
      divergence < -0.02%  → Bybit Rate ist ungewöhnlich hoch, könnte arbitriert werden

    Returns:
        DataFrame: fundingTime (UTC), fundingRate_bybit
    """
    print(f"  [Bybit] Funding Rates für {symbol}...")
    r = requests.get(
        f"{BYBIT_URL}/v5/market/funding/history",
        params={"category": "linear", "symbol": symbol, "limit": limit},
        timeout=10
    )
    r.raise_for_status()
    records = r.json()["result"]["list"]

    df = pd.DataFrame(records)
    df["fundingRateTimestamp"] = pd.to_datetime(
        df["fundingRateTimestamp"].astype(int), unit="ms", utc=True
    )
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.sort_values("fundingRateTimestamp").reset_index(drop=True)
    df = df.rename(columns={
        "fundingRateTimestamp": "fundingTime",
        "fundingRate":          "fundingRate_bybit"
    })
    print(f"  ✓ {len(df)} Einträge ({df['fundingTime'].min().date()} → {df['fundingTime'].max().date()})")
    return df[["fundingTime", "fundingRate_bybit"]]
