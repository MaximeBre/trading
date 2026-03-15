"""
analysis/ic_analysis.py – Information Coefficient & Feature Stability
======================================================================
Quant-Standard Monitoring für Alpha-Signal-Qualität.

Warum IC-Analyse?
  Ein Feature mit hohem Korrelationskoeffizienten ist gut – aber:
  - Ist die Korrelation über Zeit stabil? (IC-Stabilität)
  - Wie schnell zerfällt das Signal? (IC-Decay → optimale Haltedauer)
  - Driften Features? (FSI → warnt vor Modell-Degeneration)

Metriken:
  IC     = Spearman-Korrelation Feature → Forward Return (Rolling)
  ICIR   = mean(IC) / std(IC) → Risk-Adjusted Signal Quality
           > 0.5: gut | > 1.0: sehr gut | < 0.3: schwach
  IC Decay = IC bei Lag 1..N → zeigt wie lange das Signal hält
  FSI    = Population Stability Index (PSI) → Feature-Drift-Detektor
           < 0.10: stabil | 0.10–0.25: leichte Drift | > 0.25: instabil

Ausführen:
    cd crypto_quant
    python analysis/ic_analysis.py

Output:
    outputs/ic_report_{symbol}.csv      – Rolling IC pro Feature
    outputs/ic_summary_{symbol}.csv     – ICIR, mean IC, Decay-Halbwertszeit
    outputs/fsi_report_{symbol}.csv     – Feature Stability pro Window
    outputs/ic_heatmap_{symbol}.png     – Heatmap IC vs. Zeit
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import SYMBOLS, SYMBOL_SHORT, DATA_DIR, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features die nie als Inputs verwendet werden dürfen (Leakage)
EXCLUDE_COLS = {
    "fundingTime", "rate_annualized_pct",
    "target_next_rate", "target_next_positive",
    "target_next_3", "target_label_ordinal",
    "_fundingTime_floor",
}

# Minimum gültige Beobachtungen pro Fenster
MIN_VALID_OBS = 15


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list:
    """Gibt numerische Features zurück, die nicht excluded sind und < 70% NaN haben."""
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        nan_pct = df[c].isna().mean()
        if nan_pct >= 0.70:
            continue
        cols.append(c)
    return cols


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman-Korrelation mit NaN-Handling. Gibt NaN bei zu wenig Daten zurück."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < MIN_VALID_OBS:
        return np.nan
    try:
        corr, _ = spearmanr(x[mask], y[mask])
        return float(corr) if np.isfinite(corr) else np.nan
    except Exception:
        return np.nan


# ── 1. IC-Zeitreihe (Rolling Spearman) ───────────────────────────────────────

def compute_ic_series(features_df: pd.DataFrame,
                       target_col: str,
                       feature_cols: list,
                       window: int = 30) -> pd.DataFrame:
    """
    Information Coefficient: Spearman-Korrelation zwischen Feature-Wert und
    Forward-Return über rollendes Fenster.

    Gibt DataFrame zurück mit IC pro Feature pro Zeitperiode.
    Index entspricht features_df.index; fundingTime als erste Spalte.

    Interpretation:
      IC > 0.05:  schwaches, aber vorhandenes Signal
      IC > 0.10:  moderates Signal (profitabel bei ausreichend Trades)
      IC > 0.20:  starkes Signal (Quant-Goldstandard)
    """
    n           = len(features_df)
    target_vals = features_df[target_col].values.astype(float)
    ic_dict     = {}

    for feat in feature_cols:
        feat_vals = features_df[feat].values.astype(float)
        ic_vals   = np.full(n, np.nan)

        for t in range(window, n):
            w_feat = feat_vals[t - window: t]
            w_tgt  = target_vals[t - window: t]
            ic_vals[t] = _spearman_safe(w_feat, w_tgt)

        ic_dict[feat] = ic_vals

    ic_df = pd.DataFrame(ic_dict, index=features_df.index)
    if "fundingTime" in features_df.columns:
        ic_df.insert(0, "fundingTime", features_df["fundingTime"].values)

    return ic_df


# ── 2. ICIR (IC Information Ratio) ────────────────────────────────────────────

def compute_icir(ic_series: pd.DataFrame) -> pd.DataFrame:
    """
    IC Information Ratio = mean(IC) / std(IC)

    Gibt DataFrame mit einer Zeile pro Feature zurück:
      feature | ic_mean | ic_std | icir | ic_positive_pct

    ICIR > 0.5  = gutes Feature
    ICIR > 1.0  = sehr gutes Feature
    ICIR > 2.0  = Ausnahme-Feature (selten)
    """
    feat_cols = [c for c in ic_series.columns if c != "fundingTime"]
    rows      = []

    for feat in feat_cols:
        vals    = ic_series[feat].dropna()
        if len(vals) < 5:
            continue
        ic_mean    = vals.mean()
        ic_std     = vals.std()
        icir       = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_pos_pct = (vals > 0).mean() * 100

        rows.append({
            "feature":       feat,
            "ic_mean":       round(ic_mean, 5),
            "ic_std":        round(ic_std, 5),
            "icir":          round(icir, 4) if np.isfinite(icir) else np.nan,
            "ic_positive_pct": round(ic_pos_pct, 1),
            "ic_obs":        len(vals),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("icir", ascending=False, na_position="last")
        df = df.reset_index(drop=True)
    return df


# ── 3. IC-Decay ───────────────────────────────────────────────────────────────

def compute_ic_decay(features_df: pd.DataFrame,
                      target_col: str,
                      feature_cols: list,
                      max_lag: int = 10) -> pd.DataFrame:
    """
    IC bei Lag 1, 2, ..., max_lag.

    Für jeden Lag L:
      IC_L = Spearman(feature[t], return[t+L])
    Zeigt wie schnell das Alpha-Signal zerfällt.

    Interpretation:
      IC bei Lag 1 >> IC bei Lag 5 → kurzlebiges Signal → kurze Haltedauer
      IC bei Lag 5 ähnlich wie Lag 1 → persistentes Signal → längere Haltedauer

    Gibt DataFrame zurück: index=Lag (1..max_lag), columns=Feature-Namen.
    """
    target_vals = features_df[target_col].values.astype(float)
    n           = len(features_df)
    lag_results = {}

    for lag in range(1, max_lag + 1):
        lag_corrs = {}
        for feat in feature_cols:
            feat_vals = features_df[feat].values.astype(float)
            # feature[:-lag] vs target[lag:]
            if n - lag < MIN_VALID_OBS:
                lag_corrs[feat] = np.nan
                continue
            lag_corrs[feat] = _spearman_safe(feat_vals[:-lag], target_vals[lag:])
        lag_results[lag] = lag_corrs

    decay_df        = pd.DataFrame(lag_results).T   # index=lag, columns=features
    decay_df.index.name = "lag_periods"
    return decay_df


def _compute_decay_halflife(decay_series: pd.Series) -> float:
    """
    Berechnet die Halbwertszeit des IC-Zerfalls.
    Halbwertszeit = Lag bei dem IC auf 50% des initialen Werts fällt.
    Gibt nan zurück wenn kein klarer Abfall erkennbar.
    """
    ic0 = abs(decay_series.iloc[0]) if len(decay_series) > 0 else np.nan
    if np.isnan(ic0) or ic0 < 0.01:
        return np.nan

    half_ic = ic0 * 0.5
    for lag, val in enumerate(decay_series.values, start=1):
        if not np.isnan(val) and abs(val) <= half_ic:
            return float(lag)
    return float(len(decay_series))   # Hält die gesamte Periode


# ── 4. Feature Stability Index (PSI-basiert) ──────────────────────────────────

def compute_feature_stability_index(features_df: pd.DataFrame,
                                     feature_cols: list = None,
                                     window: int = 90,
                                     step: int = 30) -> pd.DataFrame:
    """
    Feature Stability Index (FSI) = PSI-basiert.

    Vergleicht Feature-Verteilung in aufeinanderfolgenden Fenstern.
    PSI = Σ_bins (actual% − expected%) × ln(actual% / expected%)

    FSI < 0.10:  stabil (kein Handlungsbedarf)
    FSI 0.10–0.25: leichte Drift (beobachten)
    FSI > 0.25:  instabil (Retraining nötig!)

    Gibt DataFrame zurück: eine Zeile pro Zeitfenster, Spalten = Features.
    """
    if feature_cols is None:
        feature_cols = _get_feature_cols(features_df)

    n       = len(features_df)
    rows    = []
    times   = features_df["fundingTime"].values if "fundingTime" in features_df.columns else None

    for start in range(0, n - 2 * window + 1, step):
        ref_slice = features_df.iloc[start: start + window]
        cur_slice = features_df.iloc[start + window: start + 2 * window]

        window_start = times[start]              if times is not None else start
        window_mid   = times[start + window]     if times is not None else start + window

        row = {
            "window_start": window_start,
            "window_current_start": window_mid,
        }

        for feat in feature_cols:
            ref_vals = ref_slice[feat].dropna().values.astype(float)
            cur_vals = cur_slice[feat].dropna().values.astype(float)

            if len(ref_vals) < MIN_VALID_OBS or len(cur_vals) < MIN_VALID_OBS:
                row[feat] = np.nan
                continue

            try:
                # Bins von der Referenz-Verteilung
                _, bins = np.histogram(ref_vals, bins=10)
                bins[0]  -= 1e-9   # include minimum
                bins[-1] += 1e-9   # include maximum

                ref_hist, _ = np.histogram(ref_vals, bins=bins)
                cur_hist, _ = np.histogram(cur_vals, bins=bins)

                ref_pct = (ref_hist / ref_hist.sum()).clip(1e-4)
                cur_pct = (cur_hist / cur_hist.sum()).clip(1e-4)

                psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
                row[feat] = round(psi, 5)

            except Exception:
                row[feat] = np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).reset_index(drop=True)
    return result


# ── 5. Vollständiger IC-Report ─────────────────────────────────────────────────

def run_ic_report(symbol: str,
                   features_df: pd.DataFrame,
                   target_col: str = "target_next_rate") -> dict:
    """
    Führt alle IC-Analysen aus und speichert Ergebnisse.

    Outputs:
        outputs/ic_report_{symbol}.csv     – Rolling IC-Zeitreihe
        outputs/ic_summary_{symbol}.csv    – ICIR, Decay-Halbwertszeit
        outputs/fsi_report_{symbol}.csv    – Feature Stability per Window

    Returns:
        dict mit ic_summary DataFrame und top/bottom Features nach ICIR
    """
    key = SYMBOL_SHORT.get(symbol, symbol[:3]).upper()
    print(f"\n  [{key}] IC-Analyse...")

    # Feature-Spalten auswählen
    feature_cols = _get_feature_cols(features_df)
    if target_col not in features_df.columns:
        print(f"    ⚠ target_col '{target_col}' nicht gefunden – übersprungen")
        return {}

    # Nur Zeilen mit gültigem Target
    df_clean = features_df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df_clean) < 60:
        print(f"    ⚠ Zu wenig Daten ({len(df_clean)} Zeilen)")
        return {}

    print(f"    Features analysiert: {len(feature_cols)}")
    print(f"    Perioden:            {len(df_clean)}")

    # ── IC-Zeitreihe (Rolling 30 Perioden = 10 Tage) ──────────────────────────
    print(f"    [1/4] Rolling IC (window=30)...")
    ic_series = compute_ic_series(df_clean, target_col, feature_cols, window=30)
    ic_path   = os.path.join(OUTPUT_DIR, f"ic_report_{symbol}.csv")
    ic_series.to_csv(ic_path, index=False)

    # ── ICIR ──────────────────────────────────────────────────────────────────
    print(f"    [2/4] ICIR berechnen...")
    icir_df = compute_icir(ic_series)

    # ── IC-Decay ──────────────────────────────────────────────────────────────
    print(f"    [3/4] IC-Decay (Lag 1–10)...")
    decay_df = compute_ic_decay(df_clean, target_col, feature_cols, max_lag=10)

    # Halbwertszeit pro Feature
    halflives = {}
    for feat in feature_cols:
        if feat in decay_df.columns:
            halflives[feat] = _compute_decay_halflife(decay_df[feat].dropna())

    # IC-Summary: ICIR + Decay-Halbwertszeit zusammenführen
    if len(icir_df) > 0:
        icir_df["decay_halflife_periods"] = icir_df["feature"].map(halflives)
        icir_df["decay_halflife_hours"]   = icir_df["decay_halflife_periods"] * 8
        icir_df["quality"] = icir_df["icir"].apply(
            lambda x: "sehr gut" if x >= 1.0
            else ("gut"    if x >= 0.5
            else ("schwach" if x >= 0.3
            else  "unbrauchbar"))
            if pd.notna(x) else "n/a"
        )

    summary_path = os.path.join(OUTPUT_DIR, f"ic_summary_{symbol}.csv")
    icir_df.to_csv(summary_path, index=False)
    print(f"    Gespeichert: {summary_path}")

    # ── Feature Stability Index ───────────────────────────────────────────────
    print(f"    [4/4] Feature Stability Index (PSI)...")
    fsi_df   = compute_feature_stability_index(df_clean, feature_cols,
                                                window=90, step=30)
    fsi_path = os.path.join(OUTPUT_DIR, f"fsi_report_{symbol}.csv")
    fsi_df.to_csv(fsi_path, index=False)
    print(f"    Gespeichert: {fsi_path}")

    # ── Ausgabe Top/Bottom Features ───────────────────────────────────────────
    if len(icir_df) > 0:
        top5 = icir_df[icir_df["icir"].notna()].head(5)
        bot5 = icir_df[icir_df["icir"].notna()].tail(5)

        print(f"\n    TOP-5 Features nach ICIR [{key}]:")
        for _, row in top5.iterrows():
            hl = f"  HL={row['decay_halflife_hours']:.0f}h" if pd.notna(row.get("decay_halflife_hours")) else ""
            print(f"      {row['feature']:<35}  ICIR={row['icir']:+.3f}  IC={row['ic_mean']:+.4f}  [{row['quality']}]{hl}")

        print(f"\n    BOTTOM-5 Features nach ICIR [{key}]:")
        for _, row in bot5.iterrows():
            print(f"      {row['feature']:<35}  ICIR={row['icir']:+.3f}  IC={row['ic_mean']:+.4f}  [{row['quality']}]")

        # Instabile Features warnen (FSI)
        if len(fsi_df) > 0:
            feat_fsi_cols = [c for c in fsi_df.columns
                             if c not in {"window_start", "window_current_start"}]
            if feat_fsi_cols:
                avg_fsi = fsi_df[feat_fsi_cols].mean()
                unstable = avg_fsi[avg_fsi > 0.25].index.tolist()
                if unstable:
                    print(f"\n    ⚠ INSTABILE FEATURES (FSI > 0.25, Retraining erwägen):")
                    for f in unstable[:5]:
                        print(f"      {f}: FSI={avg_fsi[f]:.3f}")

    return {
        "ic_series":    ic_series,
        "icir_df":      icir_df,
        "decay_df":     decay_df,
        "fsi_df":       fsi_df,
        "top3_features": icir_df["feature"].head(3).tolist() if len(icir_df) > 0 else [],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  IC-ANALYSE – Feature Quality & Stability Report")
    print("=" * 65)

    for symbol in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{symbol}_features.csv")
        if not os.path.exists(path):
            print(f"\n  [{SYMBOL_SHORT[symbol].upper()}] Feature-CSV nicht gefunden: {path}")
            print("  Zuerst ausführen: python data/fetch_all.py && python main.py")
            continue

        df = pd.read_csv(path, parse_dates=["fundingTime"])
        run_ic_report(symbol, df)

    print(f"\n{'='*65}")
    print("  IC-Reports gespeichert in outputs/")
    print(f"{'='*65}\n")
