"""
data/market_context.py – Markt-Kontext Features
=================================================
BTC Dominanz via Open-Interest-Proxy.

Warum OI-Dominanz statt Marktcap-Dominanz?
  - CoinGecko Marktcap ist tagesweise (1×/Tag) → schlechte Granularität
  - Binance OI ist 4h/8h verfügbar → passt zu unserer Funding-Rate-Granularität
  - OI-Dominanz misst Futures-Kapital-Flüsse direkt → direkterer Signal für
    Funding Rates als Spot-Marktcap

Alt-Season-Definition:
  btc_oi_dominance < 0.40 → Kapital rotiert von BTC zu Alts
  → höhere Alt-Funding-Rates wahrscheinlicher
"""

import pandas as pd
import numpy as np
from typing import List

from config import BINANCE_FUTURES_URL, SYMBOLS, SYMBOL_SHORT
from data.binance import get_open_interest_history


def get_btc_dominance_history(
        symbols: List[str] = None,
        period: str = "4h",
        limit: int = 500,
) -> pd.DataFrame:
    """
    Berechnet BTC OI-Dominanz aus Binance Open Interest History.

    Methodik:
      btc_oi_dominance = btc_oi / (btc_oi + eth_oi + sol_oi + ... alle Assets)

    Returns DataFrame mit:
      - timestamp       : 8h-aligned UTC
      - btc_oi_dominance: float [0,1] – wie viel % des gesamten OI ist BTC
      - alt_season_signal: bool – True wenn BTC OI < 40% des Gesamt-OI
      - dominance_change_24h: float – Veränderung in 24h (3 Perioden bei 8h)
    """
    if symbols is None:
        symbols = SYMBOLS

    print("  [MarketContext] Lade OI für alle Assets (Dominanz-Berechnung)...")

    oi_data = {}
    for sym in symbols:
        try:
            df_oi = get_open_interest_history(sym, period=period, limit=limit)
            # Bereits auf 8h aggregiert von get_open_interest_history()
            oi_data[sym] = df_oi[["timestamp", "sumOpenInterestValue"]].copy()
            oi_data[sym] = oi_data[sym].rename(
                columns={"sumOpenInterestValue": f"oi_{SYMBOL_SHORT[sym]}"}
            )
        except Exception as e:
            print(f"  [{SYMBOL_SHORT[sym].upper()}] OI nicht verfügbar: {e}")

    if "BTCUSDT" not in oi_data:
        raise RuntimeError("BTC OI nicht verfügbar – Dominanz kann nicht berechnet werden")

    # Inner Join aller verfügbaren Assets
    combined = oi_data["BTCUSDT"].copy()
    for sym in symbols:
        if sym == "BTCUSDT" or sym not in oi_data:
            continue
        combined = pd.merge(
            combined, oi_data[sym],
            on="timestamp", how="inner"
        )

    # OI-Spalten identifizieren
    oi_cols = [c for c in combined.columns if c.startswith("oi_")]

    # Gesamt-OI und BTC-Anteil
    combined["total_oi"] = combined[oi_cols].sum(axis=1)
    combined["btc_oi_dominance"] = (
        combined["oi_btc"] / combined["total_oi"]
    ).clip(0.0, 1.0)

    # Alt-Season: BTC OI < 40% → Kapital in Alts
    combined["alt_season_signal"] = (combined["btc_oi_dominance"] < 0.40).astype(int)

    # 24h-Veränderung (3 Perioden bei 8h-Granularität)
    combined["dominance_change_24h"] = combined["btc_oi_dominance"].diff(3)

    result_cols = ["timestamp", "btc_oi_dominance", "alt_season_signal", "dominance_change_24h"]
    df_result = combined[result_cols].sort_values("timestamp").reset_index(drop=True)

    print(f"  ✓ {len(df_result)} Dominanz-Datenpunkte "
          f"({df_result['timestamp'].min().date()} → {df_result['timestamp'].max().date()})")
    print(f"  ✓ Aktuelle BTC OI Dominanz: "
          f"{df_result['btc_oi_dominance'].iloc[-1]:.1%}  "
          f"Alt-Season: {'Ja' if df_result['alt_season_signal'].iloc[-1] else 'Nein'}")

    return df_result
