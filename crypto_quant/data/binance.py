"""
data/binance.py – Alle Binance API Calls
=========================================
Alle Funktionen geben saubere, typisierte DataFrames zurück.
Keine Logik hier – nur Daten holen und normalisieren.

Neu in Phase 2:
  - get_funding_rates_paginated(): Bis zu 3 Jahre Geschichte via Pagination
  - get_all_assets_data(): Lädt alle SYMBOLS auf einmal
  - Retry-Logik auf allen Calls
"""

import time
import requests
import pandas as pd
from typing import Dict, List, Optional
from config import (
    BINANCE_FUTURES_URL, BINANCE_SPOT_URL,
    SYMBOLS, SYMBOL_SHORT,
    FUNDING_DAYS_FULL,
)


# ── HTTP Helpers ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict, timeout: int = 10, retries: int = 3) -> dict:
    """GET mit exponentiellem Backoff. Wirft erst nach `retries` Fehlversuchen."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 1.5 ** attempt          # 1s, 1.5s, 2.25s ...
            time.sleep(wait)


# ── Funding Rates ──────────────────────────────────────────────────────────────

def get_funding_rates(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Historische Funding Rates von Binance Futures (letzter Request, max 1000 Einträge).

    Für volle Geschichte: get_funding_rates_paginated() verwenden.

    Returns:
        DataFrame: fundingTime (UTC), fundingRate (float)
    """
    print(f"  [Binance] Funding Rates für {symbol} ({limit} Einträge)...")
    data = _get(
        f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
        {"symbol": symbol, "limit": limit}
    )
    return _parse_funding_rates(data, symbol)


def get_funding_rates_paginated(symbol: str,
                                 days: int = FUNDING_DAYS_FULL) -> pd.DataFrame:
    """
    Zieht die vollständige Funding Rate Historie via Pagination.

    Binance liefert max 1000 Einträge pro Request. Diese Funktion
    paginiert rückwärts bis `days` Tage in der Vergangenheit.

    Bei 8h-Intervall und 1000 Records pro Request = ~333 Tage.
    → Für 3 Jahre brauchen wir ~10 Requests.

    Args:
        symbol : z.B. "BTCUSDT"
        days   : Wie viele Tage zurück (default 1095 = 3 Jahre)

    Returns:
        DataFrame: fundingTime (UTC), fundingRate (float) – sortiert aufsteigend
    """
    print(f"  [Binance] Vollständige Funding Rate Historie für {symbol} ({days} Tage)...")

    now_ms    = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms  = now_ms - days * 24 * 3_600_000

    all_records: list = []
    current_start = start_ms

    while True:
        data = _get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": current_start, "limit": 1000}
        )
        if not data:
            break

        all_records.extend(data)

        last_ts = int(data[-1]["fundingTime"])

        # Abbruchbedingungen
        if last_ts >= now_ms or len(data) < 1000:
            break

        # Nächste Seite: 1ms nach letztem Eintrag
        current_start = last_ts + 1
        time.sleep(0.12)   # Rate limiting courtesy (Binance: 2400 weight/min)

    df = _parse_funding_rates(all_records, symbol)
    print(f"  ✓ {len(df)} Perioden "
          f"({df['fundingTime'].min().date()} → {df['fundingTime'].max().date()})")
    return df


def _parse_funding_rates(data: list, symbol: str) -> pd.DataFrame:
    """Rohe API-Daten → sauberer DataFrame."""
    df = pd.DataFrame(data)
    df["fundingTime"] = pd.to_datetime(
        df["fundingTime"].astype(int), unit="ms", utc=True
    )
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = (df.drop_duplicates("fundingTime")
            .sort_values("fundingTime")
            .reset_index(drop=True))
    return df[["fundingTime", "fundingRate"]]


def get_all_assets_funding_rates(symbols: List[str] = SYMBOLS,
                                  days: int = FUNDING_DAYS_FULL) -> Dict[str, pd.DataFrame]:
    """
    Lädt paginierte Funding Rates für alle Assets.

    Returns:
        Dict[symbol → DataFrame]
    """
    print(f"\n[Binance] Lade Funding Rates für {len(symbols)} Assets...")
    result = {}
    for sym in symbols:
        result[sym] = get_funding_rates_paginated(sym, days=days)
    return result


# ── Open Interest ──────────────────────────────────────────────────────────────

def get_open_interest_history(symbol: str,
                               period: str = "4h",
                               limit: int = 500) -> pd.DataFrame:
    """
    Historisches Open Interest von Binance.

    HINWEIS: Binance akzeptiert KEIN "8h" für openInterestHist.
    Valide Perioden: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
    Default "4h" → wird in merge_oi_features() auf 8h aggregiert.

    Interpretation:
      Steigendes OI + steigende Rates  → gesunder Bull-Trend (Entry erlaubt)
      Fallendes OI + steigende Rates   → Short-Covering / LUNA-Muster (kein Entry!)
      Fallendes OI                     → Positionen werden abgebaut → Rate fällt bald

    Returns:
        DataFrame: timestamp (4h Granularität), sumOpenInterest, sumOpenInterestValue
    """
    print(f"  [Binance] Open Interest für {symbol} ({period})...")
    data = _get(
        f"{BINANCE_FUTURES_URL}/futures/data/openInterestHist",
        {"symbol": symbol, "period": period, "limit": limit}
    )
    df = pd.DataFrame(data)
    df["timestamp"]            = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["sumOpenInterest"]      = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Auf 8h aggregieren (letzte Wert pro 8h-Fenster, wie Funding-Rate-Timestamps)
    df["_8h"] = df["timestamp"].dt.floor("8h")
    df = (df.groupby("_8h")[["sumOpenInterest", "sumOpenInterestValue"]]
            .last().reset_index().rename(columns={"_8h": "timestamp"}))

    # OI-Veränderungs-Features: pct_change auf 8h-aggr. Serie
    # 3 Perioden = 1 Tag (3×8h), 21 Perioden = 7 Tage
    df["oi_change_pct"]  = df["sumOpenInterestValue"].pct_change(3)
    df["oi_change_7d"]   = df["sumOpenInterestValue"].pct_change(21)

    print(f"  ✓ {len(df)} Einträge geladen")
    return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue",
               "oi_change_pct", "oi_change_7d"]]


# ── Long/Short Ratio ───────────────────────────────────────────────────────────

def get_long_short_ratio(symbol: str,
                          period: str = "8h",
                          limit: int = 500) -> pd.DataFrame:
    """
    Long/Short Ratio der Top Trader.

    Ratio > 1 → Mehr Longs (bullish Sentiment) – aber oft Contrarian-Signal!
    Extreme Werte (>1.5 oder <0.7) markieren oft Wendepunkte.

    Returns:
        DataFrame: timestamp, longShortRatio, ls_zscore (wie extrem ist der Wert?)
    """
    print(f"  [Binance] Long/Short Ratio für {symbol}...")
    data = _get(
        f"{BINANCE_FUTURES_URL}/futures/data/topLongShortPositionRatio",
        {"symbol": symbol, "period": period, "limit": limit}
    )
    df = pd.DataFrame(data)
    df["timestamp"]     = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["longShortRatio"] = df["longShortRatio"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Z-Score: Wie extrem ist der aktuelle Ratio?
    roll_mean = df["longShortRatio"].rolling(90).mean()
    roll_std  = df["longShortRatio"].rolling(90).std()
    df["ls_zscore"] = (df["longShortRatio"] - roll_mean) / roll_std

    print(f"  ✓ {len(df)} Einträge geladen")
    return df[["timestamp", "longShortRatio", "ls_zscore"]]


# ── Basis (Spot vs. Future) ────────────────────────────────────────────────────

def get_basis_history(symbol: str,
                       interval: str = "8h",
                       limit: int = 500) -> pd.DataFrame:
    """
    Historischer Basis = Mark Price - Index Price.

    PROPRIETÄRES FEATURE: Leading Indicator für Funding Rate.

    Große positiver Basis → Long-Futures sehr gefragt
      → Funding Rate wird hoch bleiben
    Schrumpfender Basis → Longs werden abgebaut
      → Rate fällt bald (Exit-Signal!)

    Regel: Wenn basis_momentum dreht (von positiv auf negativ),
    ist das oft ein 1-3 Perioden Leading-Signal vor dem Rate-Rückgang.

    Returns:
        DataFrame: fundingTime, basis_abs, basis_pct, basis_7d_mean,
                   basis_momentum, basis_zscore
    """
    print(f"  [Binance] Historischer Basis für {symbol}...")

    mark_data = _get(
        f"{BINANCE_FUTURES_URL}/fapi/v1/markPriceKlines",
        {"symbol": symbol, "interval": interval, "limit": limit}
    )
    idx_data = _get(
        f"{BINANCE_FUTURES_URL}/fapi/v1/indexPriceKlines",
        {"pair": symbol, "interval": interval, "limit": limit}
    )

    cols = ["openTime", "open", "high", "low", "close", "volume",
            "closeTime", "x1", "x2", "x3", "x4", "x5"]
    df_mark = pd.DataFrame(mark_data, columns=cols)
    df_idx  = pd.DataFrame(idx_data,  columns=cols)

    df_mark["openTime"]   = pd.to_datetime(df_mark["openTime"], unit="ms", utc=True)
    df_idx["openTime"]    = pd.to_datetime(df_idx["openTime"],  unit="ms", utc=True)
    df_mark["mark_close"] = df_mark["close"].astype(float)
    df_idx["index_close"] = df_idx["close"].astype(float)

    df = pd.merge(
        df_mark[["openTime", "mark_close"]],
        df_idx[["openTime",  "index_close"]],
        on="openTime"
    ).rename(columns={"openTime": "fundingTime"})

    df["basis_abs"] = df["mark_close"] - df["index_close"]
    df["basis_pct"] = (df["basis_abs"] / df["index_close"]) * 100

    # Rolling Features (30d Fenster für konsistente Z-Scores)
    df["basis_7d_mean"]  = df["basis_pct"].rolling(21).mean()
    df["basis_momentum"] = df["basis_pct"].diff(3)

    roll_mean = df["basis_pct"].rolling(90).mean()
    roll_std  = df["basis_pct"].rolling(90).std()
    df["basis_zscore"] = (df["basis_pct"] - roll_mean) / roll_std

    print(f"  ✓ {len(df)} Einträge geladen")
    return df[["fundingTime", "basis_abs", "basis_pct",
               "basis_7d_mean", "basis_momentum", "basis_zscore"]]


def get_spot_price(symbol: str = "BTCUSDT") -> float:
    """Aktueller Spot-Preis."""
    data = _get(f"{BINANCE_SPOT_URL}/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])


# ── Predicted Funding Rate ─────────────────────────────────────────────────────

def get_predicted_funding_rate(symbol: str) -> float:
    """
    Nächste geschätzte Funding Rate von Binance premiumIndex.

    Binance veröffentlicht `nextFundingRate` in /fapi/v1/premiumIndex.
    Das ist die aktuell akkumulierte Premium-Rate, die in den nächsten
    Settlement-Zeitpunkt einfliesst.

    Returns:
        float: nextFundingRate (z.B. 0.0001 = 0.01%)
    """
    data = _get(
        f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex",
        {"symbol": symbol}
    )
    return float(data.get("nextFundingRate", 0.0))


def get_predicted_funding_rate_history(symbol: str,
                                        limit: int = 500) -> pd.DataFrame:
    """
    Proxy für historische Predicted Funding Rate via Basis-History.

    Binance stellt keine History von nextFundingRate bereit.
    Proxy-Formel: predicted_proxy = basis_pct / 0.375
    (8h-Normalisierung: basis/fundingRate-Verhältnis empirisch ~0.375)

    Args:
        symbol : z.B. "BTCUSDT"
        limit  : Anzahl Einträge (max 500, 8h-Granularität)

    Returns:
        DataFrame: fundingTime, predicted_funding_proxy
    """
    print(f"  [Binance] Predicted Funding Proxy für {symbol}...")
    df_basis = get_basis_history(symbol, interval="8h", limit=limit)

    df = df_basis[["fundingTime", "basis_pct"]].copy()
    # Proxy: basis_pct ist in %, Rate ist als Dezimal → / 100 / 0.375
    df["predicted_funding_proxy"] = df["basis_pct"] / 100.0 / 0.375
    df = df[["fundingTime", "predicted_funding_proxy"]]

    print(f"  ✓ {len(df)} Predicted-Proxy Einträge")
    return df
