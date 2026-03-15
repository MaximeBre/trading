"""
features/engineering.py – Feature Engineering Pipeline
========================================================
Nimmt rohe DataFrames und baut daraus ML-ready Features.

Feature-Gruppen:
  1. Standard Rate Features          (immer verfügbar, pro Asset)
  2. Cross-Exchange Features         (braucht Bybit Daten)
  3. Basis Features                  (braucht Mark/Index Price)
  4. Stablecoin Features             (braucht DeFiLlama Daten)
  5. Cross-Asset Features            (braucht alle 3 Assets gleichzeitig) ← NEU
  6. ML Labels                       (Target-Variablen, immer zuletzt)

Wichtige Korrekturen ggü. Phase 1:
  - rate_zscore: konsistentes 30d-Fenster für Mean UND Std (war: 30d/7d-Mix)
  - fillna(method=...) ersetzt durch ffill()/bfill() (pandas deprecation)
  - Ordinale Labels statt nur binärer (0/1/2/3 Qualitätsstufen)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from config import (
    ROLLING_7D, ROLLING_30D, FUNDING_THRESHOLD,
    SYMBOLS, SYMBOL_SHORT, SYMBOL_WEIGHTS,
    LABEL_THRESHOLDS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Konsistenter Z-Score: mean und std aus demselben Fenster."""
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)


def _sym(symbol: str) -> str:
    """'BTCUSDT' → 'btc'"""
    return SYMBOL_SHORT.get(symbol, symbol[:3].lower())


# ── 1. Standard Rate Features ──────────────────────────────────────────────────

def build_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baut Features aus der Funding Rate selbst.
    Keine externen Daten nötig.

    FIX: rate_zscore nutzt jetzt einheitliches 30d-Fenster für Mean + Std.
    Vorher: 30d-Mean / 7d-Std → verzerrte Z-Scores in Hochvolatilitätsphasen.
    """
    df = df.copy()

    # Annualisierte Rate (3 Zahlungen/Tag × 365)
    df["rate_annualized_pct"] = df["fundingRate"] * 3 * 365 * 100

    # Rolling Statistics
    df["rate_7d_mean"]  = df["fundingRate"].rolling(ROLLING_7D).mean()
    df["rate_7d_std"]   = df["fundingRate"].rolling(ROLLING_7D).std()
    df["rate_30d_mean"] = df["fundingRate"].rolling(ROLLING_30D).mean()
    df["rate_30d_std"]  = df["fundingRate"].rolling(ROLLING_30D).std()

    # FIX: Einheitliches Fenster – 30d Mean + 30d Std (kein Mischfenster mehr)
    # > +2: Rate sehr hoch → oft Contrarian Exit-Signal
    # < -2: Rate sehr niedrig → potenzielle Einstiegsgelegenheit
    df["rate_zscore"] = _zscore(df["fundingRate"], ROLLING_30D)

    # Kurzfristiger Z-Score (7d) als separates schnelleres Signal
    df["rate_zscore_7d"] = _zscore(df["fundingRate"], ROLLING_7D)

    # Momentum: Steigt oder fällt die Rate?
    df["rate_momentum_1d"] = df["fundingRate"].diff(3)    # vs. vor 1 Tag
    df["rate_momentum_3d"] = df["fundingRate"].diff(9)    # vs. vor 3 Tagen
    df["rate_momentum_7d"] = df["fundingRate"].diff(21)   # vs. vor 7 Tagen

    # Acceleration: Beschleunigt der Anstieg?
    # Explosiv steigende Rates kollabieren oft schneller
    df["rate_acceleration"] = df["rate_momentum_1d"].diff(3)

    # Regime: Wie viele der letzten N Perioden waren oberhalb Threshold?
    df["pct_positive_7d"]  = (df["fundingRate"] > FUNDING_THRESHOLD).rolling(ROLLING_7D).mean()
    df["pct_positive_30d"] = (df["fundingRate"] > FUNDING_THRESHOLD).rolling(ROLLING_30D).mean()

    # Volatilität der Rate selbst (nicht des Preises!)
    df["rate_volatility_7d"] = df["fundingRate"].rolling(ROLLING_7D).std()

    # Zyklische Zeitfeatures (besser als rohe Zahlen für Baummodelle)
    df["hour"]    = df["fundingTime"].dt.hour
    df["weekday"] = df["fundingTime"].dt.weekday
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]    / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]    / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    return df


# ── 2. Cross-Exchange Features ─────────────────────────────────────────────────

def build_cross_exchange_features(df: pd.DataFrame,
                                   df_bybit: pd.DataFrame) -> pd.DataFrame:
    """
    Merged Bybit Rates und berechnet Divergenz-Features.

    Große Divergenz = ungleiche Liquidität zwischen Exchanges.
    Diese konvergiert meist innerhalb 1-3 Perioden → Timing-Signal.

    Signal-Logik:
      cross_divergence_zscore > +2 → Binance überhitzt, Rate fällt bald
      cross_divergence_zscore < -2 → Bybit überhitzt, Binance könnte folgen
    """
    df_b = df.copy()
    df_y = df_bybit.copy()

    df_b["_key"] = df_b["fundingTime"].dt.floor("8h")
    df_y["_key"] = df_y["fundingTime"].dt.floor("8h")

    df_merged = pd.merge(
        df_b,
        df_y[["_key", "fundingRate_bybit"]],
        on="_key", how="left"
    ).drop(columns=["_key"])

    # Divergenz (positiv = Binance zahlt mehr)
    df_merged["cross_divergence"]     = (
        df_merged["fundingRate"] - df_merged["fundingRate_bybit"]
    )
    df_merged["cross_divergence_abs"] = df_merged["cross_divergence"].abs()

    # FIX: 30d Z-Score für Divergenz (konsistentes Fenster)
    df_merged["cross_divergence_zscore"] = _zscore(
        df_merged["cross_divergence"], ROLLING_30D
    )

    # Binance-Premium-Signal (Dummy: 1 wenn Binance > 1 Basispunkt teurer)
    df_merged["binance_premium"] = (
        df_merged["cross_divergence"] > 0.0001
    ).astype(int)

    return df_merged


# ── 3. Stablecoin Features ─────────────────────────────────────────────────────

def merge_stablecoin_features(df: pd.DataFrame,
                               df_stable: pd.DataFrame) -> pd.DataFrame:
    """
    Merged tägliche Stablecoin-Daten auf 8h Funding Rate Timestamps.

    LOOK-AHEAD FIX:
    DeFiLlama rapportiert Supply am Ende des Tages – exakter Zeitpunkt
    variiert, aber Daten von Tag D sind frühestens am nächsten Morgen
    zuverlässig verfügbar.

    Fix: Stablecoin-Datum um 1 Tag nach vorne schieben vor dem Merge.
    → "Tag D"-Supply wird erst ab Tag D+1 als Feature verwendet.
    Kosten: 1 Tag Lead-Zeit verloren – akzeptabel da Signal ohnehin
    3-14 Tage vorlaufend ist.
    """
    df_f = df.copy()
    df_s = df_stable.copy()

    df_f["_date"] = df_f["fundingTime"].dt.normalize()

    # FIX Look-ahead: Supply von Tag D erst ab Tag D+1 verwenden
    df_s["_date"] = df_s["fundingTime"].dt.normalize() + pd.Timedelta(days=1)

    stablecoin_cols = [c for c in df_s.columns if c not in {"fundingTime", "_date"}]
    df_merged = pd.merge(
        df_f,
        df_s[["_date"] + stablecoin_cols],
        on="_date",
        how="left"
    ).drop(columns=["_date"])

    # ffill: tägliche Daten auf 3 × 8h-Slots aufteilen
    df_merged[stablecoin_cols] = df_merged[stablecoin_cols].ffill()

    return df_merged


# ── 4. Open Interest Features ──────────────────────────────────────────────────

def merge_oi_features(df: pd.DataFrame,
                       df_oi: pd.DataFrame) -> pd.DataFrame:
    """
    Merged Open Interest Features in den Haupt-DataFrame.

    OI-Daten kommen von get_open_interest_history() und enthalten
    oi_change_pct (1-Tages-Veränderung) und oi_change_7d (7-Tage-Veränderung).

    Proprietäres Feature: oi_health_flag
    Erkennt LUNA-ähnliche Muster:
      Rate steigt stark (zscore > 2) ABER OI fällt gleichzeitig
      → Kein echtes Interesse, nur Short-Covering oder Manipulation
      → kein Entry-Signal!

    In normalen Bull-Märkten: Rate steigt + OI steigt = gesunder Trend.
    Warnsignal: Rate über +2σ + OI 7d-Change < -15% = potenzielle Falle.
    """
    df_f = df.copy()
    df_o = df_oi.copy()

    # OI timestamp auf 8h flooren für Merge
    df_f["_key"] = df_f["fundingTime"].dt.floor("8h")
    df_o["_key"] = df_o["timestamp"].dt.floor("8h")

    oi_cols = [c for c in df_o.columns if c not in {"timestamp", "_key"}]
    df_merged = pd.merge(
        df_f,
        df_o[["_key"] + oi_cols],
        on="_key", how="left"
    ).drop(columns=["_key"])

    # OI Health Flag: 1 = gesund, 0 = Warnsignal
    # Warnsignal: Rate überhitzt (zscore > 2) OHNE OI-Backing
    if "rate_zscore" in df_merged.columns and "oi_change_7d" in df_merged.columns:
        df_merged["oi_health_flag"] = ~(
            (df_merged["rate_zscore"] > 2.0) &
            (df_merged["oi_change_7d"] < -0.15)
        )
        df_merged["oi_health_flag"] = df_merged["oi_health_flag"].astype(int)
    elif "oi_change_7d" in df_merged.columns:
        df_merged["oi_health_flag"] = (df_merged["oi_change_7d"] > -0.20).astype(int)

    return df_merged


# ── 5. Basis Features ──────────────────────────────────────────────────────────

def merge_basis_features(df: pd.DataFrame,
                          df_basis: pd.DataFrame) -> pd.DataFrame:
    """
    Merged Basis-Features (aus binance.py get_basis_history) in den Haupt-DataFrame.
    """
    df_f = df.copy()
    df_b = df_basis.copy()

    # basis_cols VOR dem Hinzufügen von _key berechnen – sonst Duplikat
    basis_cols = [c for c in df_b.columns if c not in {"fundingTime", "_key"}]

    df_f["_key"] = df_f["fundingTime"].dt.floor("8h")
    df_b["_key"] = df_b["fundingTime"].dt.floor("8h")

    df_merged = pd.merge(
        df_f,
        df_b[["_key"] + basis_cols],
        on="_key", how="left"
    ).drop(columns=["_key"])

    return df_merged


# ── 2b. Tri-Exchange Features (Binance + Bybit + OKX) ─────────────────────────

def build_tri_exchange_features(df: pd.DataFrame,
                                  df_bybit: pd.DataFrame = None,
                                  df_okx: pd.DataFrame = None) -> pd.DataFrame:
    """
    Erweitert Cross-Exchange Features um OKX als dritte Datenquelle.

    Neue Features:
      okx_funding_rate          : OKX Rate auf Binance Timestamps aligniert
      binance_okx_spread        : binance_rate - okx_rate
      binance_okx_zscore        : Rolling 30-Period Z-Score des B-OKX Spreads
      tri_exchange_mean         : Mittelwert der verfügbaren Exchange-Rates
      tri_exchange_std          : Std der 3 Rates → Markt-Uneinigkeit
      tri_exchange_outlier      : abs(binance - tri_mean) > 1.5 × tri_std (bool)
      max_exchange_spread       : max - min über alle verfügbaren Exchanges

    Look-ahead Schutz: OKX-Daten werden um 1 Period geshiftet vor Merge.

    Args:
        df       : Haupt-Feature-DataFrame (mit fundingRate)
        df_bybit : Bybit Rates (optional, hat fundingRate_bybit)
        df_okx   : OKX Rates (optional, hat okx_funding_rate)
    """
    df_f = df.copy()

    if df_okx is None or df_okx.empty:
        # Kein OKX → Features auf 0/neutral setzen
        for col in ["okx_funding_rate", "binance_okx_spread", "binance_okx_zscore",
                    "tri_exchange_mean", "tri_exchange_std",
                    "tri_exchange_outlier", "max_exchange_spread"]:
            df_f[col] = 0.0
        df_f["tri_exchange_outlier"] = 0
        return df_f

    df_o = df_okx.copy()

    # Look-ahead Schutz: OKX Rate von vorheriger Periode verwenden
    df_o = df_o.sort_values("fundingTime").copy()
    df_o["okx_funding_rate"] = df_o["okx_funding_rate"].shift(1)

    df_f["_key"] = df_f["fundingTime"].dt.floor("8h")
    df_o["_key"] = df_o["fundingTime"].dt.floor("8h")

    df_merged = pd.merge(
        df_f,
        df_o[["_key", "okx_funding_rate"]],
        on="_key", how="left"
    ).drop(columns=["_key"])

    # Binance – OKX Spread
    df_merged["binance_okx_spread"] = (
        df_merged["fundingRate"] - df_merged["okx_funding_rate"]
    )
    df_merged["binance_okx_zscore"] = _zscore(
        df_merged["binance_okx_spread"], ROLLING_30D
    )

    # Tri-Exchange Statistiken
    rate_cols = ["fundingRate"]
    if df_bybit is not None and "fundingRate_bybit" in df_merged.columns:
        rate_cols.append("fundingRate_bybit")
    rate_cols.append("okx_funding_rate")

    rate_matrix = df_merged[rate_cols].copy()
    df_merged["tri_exchange_mean"] = rate_matrix.mean(axis=1)
    df_merged["tri_exchange_std"]  = rate_matrix.std(axis=1).fillna(0.0)

    # Outlier: Binance weicht stark vom Konsens ab → Reversion Signal
    outlier_mask = (
        (df_merged["fundingRate"] - df_merged["tri_exchange_mean"]).abs()
        > 1.5 * df_merged["tri_exchange_std"]
    )
    df_merged["tri_exchange_outlier"] = outlier_mask.astype(int)

    # Maximale Spread-Breite
    df_merged["max_exchange_spread"] = (
        rate_matrix.max(axis=1) - rate_matrix.min(axis=1)
    )

    return df_merged


# ── 5a. Predicted Funding Features ────────────────────────────────────────────

def merge_predicted_funding_features(df: pd.DataFrame,
                                      df_predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Merged Predicted Funding Rate Proxy-Features.

    Features:
      predicted_funding_proxy     : Proxy für nextFundingRate (basis_pct / 0.375 / 100)
      pred_vs_actual_spread       : predicted - actual (positiv = Markt erwartet höhere Rate)
      pred_direction_matches_actual: 1 wenn predicted und actual gleiche Richtung (über Threshold)

    Args:
        df          : Haupt-Feature-DataFrame (mit fundingRate)
        df_predicted: Von get_predicted_funding_rate_history() – hat fundingTime + predicted_funding_proxy
    """
    df_f = df.copy()
    df_p = df_predicted.copy()

    df_f["_key"] = df_f["fundingTime"].dt.floor("8h")
    df_p["_key"] = df_p["fundingTime"].dt.floor("8h")

    df_merged = pd.merge(
        df_f,
        df_p[["_key", "predicted_funding_proxy"]],
        on="_key", how="left"
    ).drop(columns=["_key"])

    df_merged["pred_vs_actual_spread"] = (
        df_merged["predicted_funding_proxy"] - df_merged["fundingRate"]
    )

    # Richtungs-Übereinstimmung: 1 wenn beide über Threshold oder beide darunter
    threshold = FUNDING_THRESHOLD
    df_merged["pred_direction_matches_actual"] = (
        (df_merged["predicted_funding_proxy"] > threshold) ==
        (df_merged["fundingRate"] > threshold)
    ).astype(int)

    return df_merged


# ── 5b. BTC Dominance Features ─────────────────────────────────────────────────

def merge_btc_dominance_features(df: pd.DataFrame,
                                  df_dominance: pd.DataFrame) -> pd.DataFrame:
    """
    Merged BTC OI Dominanz-Features in den Haupt-DataFrame.

    Features:
      btc_oi_dominance   : float [0,1] – BTC-Anteil am Gesamt-OI
      alt_season_signal  : int 0/1 – 1 wenn BTC OI < 40%
      dominance_change_24h: float – 24h Veränderung der Dominanz

    Args:
        df            : Haupt-Feature-DataFrame
        df_dominance  : Von get_btc_dominance_history() – hat timestamp + features
    """
    df_f = df.copy()
    df_d = df_dominance.copy()

    df_f["_key"] = df_f["fundingTime"].dt.floor("8h")
    df_d["_key"] = df_d["timestamp"].dt.floor("8h")

    dominance_cols = [c for c in df_d.columns if c not in {"timestamp", "_key"}]

    df_merged = pd.merge(
        df_f,
        df_d[["_key"] + dominance_cols],
        on="_key", how="left"
    ).drop(columns=["_key"])

    # ffill für lücklose Zeitreihe
    df_merged[dominance_cols] = df_merged[dominance_cols].ffill()

    return df_merged


# ── 5. Cross-Asset Features ────────────────────────────────────────────────────

def build_cross_asset_features(
        dfs: Dict[str, pd.DataFrame],
        symbols: List[str] = SYMBOLS,
        weights: Dict[str, float] = SYMBOL_WEIGHTS,
) -> Dict[str, pd.DataFrame]:
    """
    Berechnet alle proprietären Cross-Asset Features und merged sie
    in jeden einzelnen Asset-DataFrame zurück.

    Erfordert: dfs hat bereits Standard-Rate-Features (rate_zscore etc.)

    Cross-Asset Features:
      Spreads:
        btc_eth_spread, btc_sol_spread, eth_sol_spread
        + jeweiliger 30d Z-Score (wie ungewöhnlich ist der Spread?)
      Hierarchy Signal:
        hierarchy_normal    – SOL > ETH > BTC? (Bull-Regime)
        hierarchy_inversion – Umkehrung = Contrarian-Signal
      Synchronized High:
        all_above_zscore_1  – Alle 3 Assets gleichzeitig über +1 Sigma?
        sync_score          – Wie viele Assets lohnen sich gerade? (0-3)
      Capital Rotation:
        btc_falling_sol_rising – Rotation-Indikator
        rotation_direction     – Wohin fließt Kapital? (-1/0/+1)
      Portfolio Rate:
        portfolio_rate_weighted – Gewichtete Gesamt-Rate (3 Assets)
        rel_score_{btc/eth/sol} – Relative Z-Score-Stärke pro Asset

    Args:
        dfs     : Dict[symbol → Feature-DataFrame] (mit rate_zscore)
        symbols : Liste der Symbole in der Reihenfolge [BTC, ETH, SOL]
        weights : Kapitalgewichte pro Asset

    Returns:
        Dict[symbol → DataFrame] mit hinzugefügten Cross-Asset Spalten
    """
    # ── Align: gemeinsame Zeitachse aller Rates ────────────────────────────────
    rate_parts = []
    for sym in symbols:
        key = _sym(sym)
        sub = dfs[sym][["fundingTime", "fundingRate", "rate_zscore",
                         "rate_momentum_1d"]].copy()
        sub = sub.rename(columns={
            "fundingRate":     f"rate_{key}",
            "rate_zscore":     f"zscore_{key}",
            "rate_momentum_1d": f"mom_{key}",
        })
        rate_parts.append(sub)

    # Inner Join: nur Timestamps die in ALLEN Assets vorhanden sind
    combined = rate_parts[0]
    for rp in rate_parts[1:]:
        combined = pd.merge(combined, rp, on="fundingTime", how="inner")

    # Kurz-Keys: btc, eth, sol
    keys = [_sym(s) for s in symbols]
    r = {k: combined[f"rate_{k}"] for k in keys}
    z = {k: combined[f"zscore_{k}"] for k in keys}
    m = {k: combined[f"mom_{k}"] for k in keys}

    # ── Cross-Asset Rate Spreads ───────────────────────────────────────────────
    if len(keys) >= 2:
        combined["btc_eth_spread"] = r["btc"] - r["eth"]
        combined["btc_sol_spread"] = r["btc"] - r["sol"]
        combined["eth_sol_spread"] = r["eth"] - r["sol"]

        for col in ["btc_eth_spread", "btc_sol_spread", "eth_sol_spread"]:
            combined[f"{col}_zscore"] = _zscore(combined[col], ROLLING_30D)

    # ── Rate Hierarchy Signal ──────────────────────────────────────────────────
    # In Bull-Märkten: SOL zahlt am meisten, dann ETH, dann BTC
    # Inversion = Regime-Wechsel im Anmarsch
    combined["hierarchy_normal"] = (
        (r["sol"] > r["eth"]) & (r["eth"] > r["btc"])
    ).astype(int)

    combined["hierarchy_inversion"] = (
        (r["btc"] > r["eth"]) | (r["eth"] > r["sol"])
    ).astype(int)

    # ── Synchronized High Signal ───────────────────────────────────────────────
    # Alle drei über +1 Sigma gleichzeitig → stabiles Regime, Rate bleibt hoch
    combined["all_above_zscore_1"] = (
        (z["btc"] > 1) & (z["eth"] > 1) & (z["sol"] > 1)
    ).astype(int)

    # Sync Score: Wie viele Assets lohnen sich gerade? (0 = keiner, 3 = alle)
    combined["sync_score"] = sum(
        (r[k] > FUNDING_THRESHOLD).astype(int) for k in keys
    )

    # ── Capital Rotation Signal ────────────────────────────────────────────────
    # BTC fällt + SOL steigt = Rotation von Large-Cap zu High-Beta
    combined["btc_falling_sol_rising"] = (
        (m["btc"] < 0) & (m["sol"] > 0)
    ).astype(int)

    # rotation_direction: +1 = Kapital rotiert Richtung SOL (risk-on)
    #                      0 = neutral
    #                     -1 = Kapital rotiert Richtung BTC (risk-off)
    combined["rotation_direction"] = np.sign(m["sol"] - m["btc"])

    # ── Weighted Portfolio Rate ────────────────────────────────────────────────
    combined["portfolio_rate_weighted"] = sum(
        r[_sym(sym)] * weights[sym] for sym in symbols
    )

    # ── Relative Z-Score Stärke pro Asset ─────────────────────────────────────
    # Normalisiert auf [0,1] über alle 3 Assets: welcher Asset ist am stärksten?
    # Basis für optimale ML-Allokations-Signal in Phase 2
    z_sum = sum(z[k].abs() for k in keys) + 1e-8
    for k in keys:
        combined[f"rel_score_{k}"] = z[k] / z_sum

    # ── Merge zurück in jeden Asset-DataFrame ─────────────────────────────────
    cross_cols = [c for c in combined.columns if c != "fundingTime"]

    result_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        merged = pd.merge(
            dfs[sym],
            combined[["fundingTime"] + cross_cols],
            on="fundingTime",
            how="left"
        )
        result_dfs[sym] = merged

    return result_dfs


# ── 6. ML Labels ───────────────────────────────────────────────────────────────

def build_labels(df: pd.DataFrame,
                  threshold: float = FUNDING_THRESHOLD) -> pd.DataFrame:
    """
    Baut Target-Variablen für das ML-Modell.

    Labels:
        target_next_positive : Rate in nächster Periode > threshold? (0/1)
        target_next_3        : Rate in ALLEN nächsten 3 Perioden > threshold? (0/1)
        target_next_rate     : Exakte Rate in nächster Periode (Regression)

        target_label_ordinal : 0=unter Threshold, 1=marginal, 2=gut, 3=excellent
                               Besser als binary für Modell-Qualität – eine Rate
                               von 0.08% ist wertvoller als 0.011%, beide wären
                               im binary-Label "1".

    WICHTIG: Labels immer ZULETZT nach allen Feature-Schritten bauen.
    Kein Lookahead-Bias da shift(-1) nur den nächsten bekannten Wert nutzt.
    """
    df = df.copy()

    next_rate = df["fundingRate"].shift(-1)

    df["target_next_positive"] = (next_rate > threshold).astype(int)

    df["target_next_3"] = (
        (df["fundingRate"].shift(-1) > threshold) &
        (df["fundingRate"].shift(-2) > threshold) &
        (df["fundingRate"].shift(-3) > threshold)
    ).astype(int)

    df["target_next_rate"] = next_rate

    # Ordinale Labels: robusteres Ziel für XGBoost
    thresholds = LABEL_THRESHOLDS
    label = pd.Series(0, index=df.index)
    label[next_rate > thresholds[0]] = 1   # 0.01%
    label[next_rate > thresholds[1]] = 2   # 0.03%
    label[next_rate > thresholds[2]] = 3   # 0.08%
    df["target_label_ordinal"] = label.astype(int)

    return df


# ── Master Pipeline (Single-Asset) ─────────────────────────────────────────────

def build_all_features(df_rates: pd.DataFrame,
                        df_bybit: pd.DataFrame = None,
                        df_basis: pd.DataFrame = None,
                        df_stable: pd.DataFrame = None,
                        df_oi: pd.DataFrame = None,
                        df_predicted: pd.DataFrame = None,
                        df_dominance: pd.DataFrame = None,
                        df_okx: pd.DataFrame = None,
                        add_labels: bool = True) -> pd.DataFrame:
    """
    Führt alle Single-Asset Feature-Engineering Schritte aus.

    In der Multi-Asset Pipeline wird diese Funktion pro Asset aufgerufen,
    danach folgt build_cross_asset_features() über alle Assets.

    Args:
        df_rates   : Binance Funding Rates (Pflicht)
        df_bybit   : Bybit Funding Rates (optional)
        df_basis   : Basis History (optional)
        df_stable  : Stablecoin Supply (optional)
        add_labels : Labels bauen? In Multi-Asset Pipeline erst nach
                     Cross-Asset-Features setzen (default: True)

    Returns:
        Feature DataFrame, ML-ready
    """
    print("\n  Feature Engineering Pipeline...")

    df = build_rate_features(df_rates)
    n_new = len(df.columns) - len(df_rates.columns)
    print(f"  + {n_new:2d} Standard Rate Features")

    if df_bybit is not None:
        df = build_cross_exchange_features(df, df_bybit)
        print("  +  4 Cross-Exchange Divergenz Features")

    if df_okx is not None or df_bybit is not None:
        df = build_tri_exchange_features(df, df_bybit=df_bybit, df_okx=df_okx)
        n_tri = 7 if df_okx is not None and not df_okx.empty else 0
        if n_tri:
            print("  +  7 Tri-Exchange Features (Binance/Bybit/OKX)")

    if df_basis is not None:
        df = merge_basis_features(df, df_basis)
        print("  +  5 Basis (Spot vs. Future) Features")

    if df_stable is not None:
        df = merge_stablecoin_features(df, df_stable)
        print("  +  6 Stablecoin Inflow Features")

    if df_oi is not None:
        df = merge_oi_features(df, df_oi)
        print("  +  3 Open Interest Features (inkl. OI Health Flag)")

    if df_predicted is not None:
        df = merge_predicted_funding_features(df, df_predicted)
        print("  +  3 Predicted Funding Features")

    if df_dominance is not None:
        df = merge_btc_dominance_features(df, df_dominance)
        print("  +  3 BTC OI Dominance Features")

    if add_labels:
        df = build_labels(df)
        print("  +  4 ML Labels")

    n_features = len([c for c in df.columns
                      if c not in {"fundingTime", "target_next_positive",
                                   "target_next_3", "target_next_rate",
                                   "target_label_ordinal", "rate_annualized_pct"}])
    print(f"  → {n_features} Features total | {len(df)} Datenpunkte")

    return df


def merge_all_data(df_rates, df_bybit=None, df_basis=None, df_stable=None):
    """Alias für build_all_features – backwards compat."""
    return build_all_features(df_rates, df_bybit, df_basis, df_stable)
