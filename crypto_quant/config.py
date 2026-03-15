"""
config.py – Zentrale Konfiguration für das Crypto Quant System
==============================================================
Alle Parameter an einem Ort. Hier anpassen, nirgendwo sonst.
"""

# ── Core Portfolio (7 Assets) ─────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT",
]

# Static capital weights (equal-risk baseline – ML optimiert in Phase 2)
SYMBOL_WEIGHTS = {
    "BTCUSDT":  0.25,   # BTC: Anker, stabile Persistenz
    "ETHUSDT":  0.20,   # ETH: Mittleres Risiko/Return
    "SOLUSDT":  0.15,   # SOL: Höchste Rates, schnellste Regime-Wechsel
    "DOGEUSDT": 0.10,   # DOGE: Meme-Coin, höheres Threshold nötig
    "XRPUSDT":  0.10,   # XRP: Liquid, regulatorisches Risiko beachten
    "AVAXUSDT": 0.10,   # AVAX: Stabile Alt-Coin Rates
    "LINKUSDT": 0.10,   # LINK: Oracle Token, moderate Rates
}

# Short keys for column naming: "BTCUSDT" → "btc"
SYMBOL_SHORT = {
    "BTCUSDT":  "btc",
    "ETHUSDT":  "eth",
    "SOLUSDT":  "sol",
    "DOGEUSDT": "doge",
    "XRPUSDT":  "xrp",
    "AVAXUSDT": "avax",
    "LINKUSDT": "link",
    # Legacy/unused assets
    "BNBUSDT":  "bnb",
    "DOTUSDT":  "dot",
    "MATICUSDT": "matic",
    "LTCUSDT":  "ltc",
    "ADAUSDT":  "ada",
}

# ── Asset-Spezifische Thresholds ───────────────────────────────────────────────
# Kalibriert auf std(fundingRate) × 0.5 pro Asset
# Kleinere Coins haben höhere Volatilität → höhere Mindestrate für profitablen Entry
FUNDING_THRESHOLDS = {
    "BTCUSDT":  0.0001,   # 0.01% – Standard (niedrige Volatilität)
    "ETHUSDT":  0.0001,   # 0.01% – Standard
    "SOLUSDT":  0.00015,  # 0.015% – höher, schnelle Regime-Wechsel
    "DOGEUSDT": 0.0002,   # 0.02% – Meme-Coin Prämie
    "XRPUSDT":  0.00018,  # 0.018% – leicht erhöht
    "AVAXUSDT": 0.00015,  # 0.015% – moderate Volatilität
    "LINKUSDT": 0.00015,  # 0.015% – moderate Volatilität
}

# ── Asset-Spezifische Bid/Ask-Spreads ─────────────────────────────────────────
# Realistischer Market Impact pro Leg (BTC sehr liquid → enger Spread)
BID_ASK_SPREADS = {
    "BTCUSDT":  0.0001,   # 0.01% – tiefstes Orderbuch
    "ETHUSDT":  0.00015,  # 0.015%
    "SOLUSDT":  0.0003,   # 0.03%
    "DOGEUSDT": 0.0004,   # 0.04% – größter Spread
    "XRPUSDT":  0.0003,   # 0.03%
    "AVAXUSDT": 0.0004,   # 0.04%
    "LINKUSDT": 0.0004,   # 0.04%
}

# ── Portfolio-Constraints ──────────────────────────────────────────────────────
MAX_ASSET_WEIGHT = 0.30   # Max 30% in einem einzelnen Asset (Korrelations-Diversifikation)

# Legacy single-asset compat (stats/plots default)
SYMBOL_BINANCE = "BTCUSDT"
SYMBOL_BYBIT   = "BTCUSDT"

# ── Daten-Parameter ────────────────────────────────────────────────────────────
FUNDING_LIMIT     = 1000       # Einträge pro API-Request (Binance-Max)
FUNDING_DAYS_FULL = 1095       # 3 Jahre für paginiertes Fetch (Walk-Forward braucht Volumen)
OI_LIMIT          = 500        # Open Interest History Einträge
KLINE_INTERVAL    = "8h"       # Basis-Granularität

# ── Strategie-Parameter ────────────────────────────────────────────────────────
FUNDING_THRESHOLD = 0.0001     # 0.01% – Mindestrate um profitabel zu sein

# Delta-Neutral: 4 Legs pro Roundtrip (Spot Long + Futures Short, jeweils rein + raus)
MAKER_FEE         = 0.0002     # 0.02% Binance Maker Fee (pro Leg)
LEGS_PER_ROUNDTRIP = 4         # 2 Legs öffnen + 2 Legs schließen
SLIPPAGE_BPS      = 5          # 5 Basispunkte = 0.05% Market Impact pro Leg (konservativ)
SLIPPAGE          = SLIPPAGE_BPS / 10_000

# Gesamtkosten pro Roundtrip (realistisch für Delta-Neutral):
# (MAKER_FEE + SLIPPAGE) × 4 = ~0.1% für ein vollständiges Öffnen+Schließen
COST_PER_ROUNDTRIP = (MAKER_FEE + SLIPPAGE) * LEGS_PER_ROUNDTRIP

CAPITAL           = 10_000     # Startkapital in USD
MARGIN_BUFFER     = 0.20       # 20% als Margin-Reserve halten (nie 100% als Collateral)
EFFECTIVE_CAPITAL = CAPITAL * (1 - MARGIN_BUFFER)  # 8.000 USD effektiv einsetzbar

# ── Feature Engineering ────────────────────────────────────────────────────────
ROLLING_7D   = 21              # 7 Tage in 8h Perioden
ROLLING_30D  = 90              # 30 Tage in 8h Perioden
PERIODS_PER_YEAR = 3 * 365     # 1095 – für korrekte Annualisierung

# ── ML Labels ──────────────────────────────────────────────────────────────────
# Ordinale Label-Schwellen: 0=unter Threshold, 1=marginal, 2=gut, 3=excellent
LABEL_THRESHOLDS = [
    FUNDING_THRESHOLD,          # 0.01% → Label >= 1
    FUNDING_THRESHOLD * 3,      # 0.03% → Label >= 2
    FUNDING_THRESHOLD * 8,      # 0.08% → Label = 3 (Bull-Regime)
]

# ── Pfade ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/raw"
OUTPUT_DIR  = "outputs"

# ── API Endpoints ──────────────────────────────────────────────────────────────
BINANCE_FUTURES_URL    = "https://fapi.binance.com"
BINANCE_SPOT_URL       = "https://api.binance.com"
BYBIT_URL              = "https://api.bybit.com"
DEFILLAMA_URL          = "https://stablecoins.llama.fi"
COINGECKO_DOMINANCE_URL = "https://api.coingecko.com/api/v3/global"
