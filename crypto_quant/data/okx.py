"""
data/okx.py – OKX API Calls
============================
Funding Rates für Cross-Exchange Tri-Divergenz Features.
Kein API Key nötig – öffentliche Endpoints.

OKX Symbol Mapping: BTCUSDT → BTC-USDT-SWAP
"""

import time
import requests
import pandas as pd

OKX_BASE_URL = "https://www.okx.com"

# Binance Symbol → OKX instId Mapping
OKX_SYMBOL_MAP = {
    "BTCUSDT":  "BTC-USDT-SWAP",
    "ETHUSDT":  "ETH-USDT-SWAP",
    "SOLUSDT":  "SOL-USDT-SWAP",
    "DOGEUSDT": "DOGE-USDT-SWAP",
    "XRPUSDT":  "XRP-USDT-SWAP",
    "AVAXUSDT": "AVAX-USDT-SWAP",
    "LINKUSDT": "LINK-USDT-SWAP",
}

_EMPTY_DF = pd.DataFrame(columns=["fundingTime", "okx_funding_rate"])


def _get(url: str, params: dict, timeout: int = 10, retries: int = 3) -> dict:
    """GET mit exponentiellem Backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)


def get_okx_funding_rates(symbol: str, limit: int = 500) -> pd.DataFrame:
    """
    Historische Funding Rates von OKX.

    Endpoint: GET /api/v5/public/funding-rate-history
    Parameter: instId={OKX_SYMBOL}, limit={limit} (max 100 per request)

    Für Cross-Exchange Tri-Divergenz Features:
      binance_rate - okx_rate → ungleiche Liquidität → Reversion Signal

    Args:
        symbol : Binance-Symbol, z.B. "BTCUSDT"
        limit  : Anzahl Einträge (wird intern auf 100er Pages aufgeteilt)

    Returns:
        DataFrame: fundingTime (UTC), okx_funding_rate
        Bei Fehler: leerer DataFrame mit korrekten Spalten
    """
    inst_id = OKX_SYMBOL_MAP.get(symbol)
    if inst_id is None:
        return _EMPTY_DF.copy()

    print(f"  [OKX] Funding Rates für {symbol} ({inst_id})...")

    # OKX liefert max 100 Records pro Request → paginieren
    all_records = []
    after = None          # Cursor: letzte fundingTime (ältester Zeitstempel)
    page_limit = 100

    pages_needed = (limit + page_limit - 1) // page_limit

    try:
        for _ in range(pages_needed):
            params = {"instId": inst_id, "limit": str(page_limit)}
            if after is not None:
                params["after"] = str(after)

            data = _get(f"{OKX_BASE_URL}/api/v5/public/funding-rate-history", params)

            if data.get("code") != "0":
                break

            records = data.get("data", [])
            if not records:
                break

            all_records.extend(records)

            if len(records) < page_limit:
                break

            # OKX paginiert mit `after` = ältestem Timestamp der Seite
            after = int(records[-1]["fundingTime"])
            time.sleep(0.1)

    except Exception as e:
        print(f"  [OKX] Fehler für {symbol}: {e}")
        return _EMPTY_DF.copy()

    if not all_records:
        print(f"  [OKX] Keine Daten für {symbol}")
        return _EMPTY_DF.copy()

    df = pd.DataFrame(all_records)
    df["fundingTime"] = pd.to_datetime(
        df["fundingTime"].astype(int), unit="ms", utc=True
    )
    # Bevorzuge realizedRate (tatsächlich gezahlt) wenn verfügbar, sonst fundingRate
    rate_col = "realizedRate" if "realizedRate" in df.columns else "fundingRate"
    df["okx_funding_rate"] = df[rate_col].astype(float)

    df = (df.drop_duplicates("fundingTime")
            .sort_values("fundingTime")
            .reset_index(drop=True))

    df = df[["fundingTime", "okx_funding_rate"]].head(limit)
    print(f"  ✓ {len(df)} OKX Einträge "
          f"({df['fundingTime'].min().date()} → {df['fundingTime'].max().date()})")
    return df
