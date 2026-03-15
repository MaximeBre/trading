"""
models/regime.py – Layer 1: Regime Classifier
===============================================
Ausführen:
    cd crypto_quant
    python models/regime.py

2-stufige Regime-Erkennung:

  Stufe 1 (Unsupervised):
    GaussianMixture mit 4 Komponenten auf normalisierten Regime-Features.
    Auto-Labeling: Cluster sortiert nach rate_zscore-Mean:
      Niedrigster = CRISIS (-1), dann BEAR (0), NEUTRAL (1), BULL (2)
    Warum rate_zscore und nicht fundingRate: Zscore ist stationär,
    fundingRate hat Bull-Market-Bias der GMM verzerrt.

  Stufe 2 (Supervised, Walk-Forward):
    XGBClassifier(objective="multi:softprob") auf GMM-Labels trainiert.
    Output pro Periode: [p_crisis, p_bear, p_neutral, p_bull]
    Walk-Forward Expanding Window (identisch zu models/train.py).

    WICHTIG: In Layer 3 und Layer 4 IMMER die Walk-Forward OOF Probs
    verwenden, nie die finalen Modell-Predictions auf Trainingsdaten.
    Das verhindert Look-ahead Bias im downstream Pipeline.

Outputs:
    data/raw/{symbol}_regime.csv         ← OOF Regime Probs (für Layer 2/3/4)
    models/saved/regime_{symbol}.json    ← Finales Modell
    outputs/regime_history_{symbol}.png  ← Regime Timeline
    outputs/regime_transition_matrix.png ← Wechsel-Häufigkeit Heatmap

Regime Bedeutung:
    CRISIS  (-1): Sehr niedrige/negative Rates, OI-Kollaps Muster
    BEAR    ( 0): Negative bis neutrale Rates, abnehmendes Kapital
    NEUTRAL ( 1): Rates in normalem Bereich, gemischtes Signal
    BULL    ( 2): Hohe positive Rates, starkes Kapitalwachstum
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import (
    SYMBOLS, SYMBOL_SHORT, SYMBOL_WEIGHTS,
    DATA_DIR, OUTPUT_DIR, PERIODS_PER_YEAR,
)

MODELS_DIR = "models/saved"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Regime Konfiguration ───────────────────────────────────────────────────────

N_REGIMES   = 4
REGIME_MAP  = {-1: "CRISIS", 0: "BEAR", 1: "NEUTRAL", 2: "BULL"}
REGIME_COLS = ["p_crisis", "p_bear", "p_neutral", "p_bull"]

# Farben: Rot=Crisis, Orange=Bear, Grau=Neutral, Grün=Bull
REGIME_COLORS = {
    -1: "#e74c3c",   # CRISIS – Rot
     0: "#e67e22",   # BEAR   – Orange
     1: "#95a5a6",   # NEUTRAL– Grau
     2: "#27ae60",   # BULL   – Grün
}

# Features für GMM-Labeling: nur Spalten mit < 10% NaN
# Basis (85% NaN) und OI (65% NaN) bewusst ausgelassen
REGIME_FEATURES = [
    # Rate Dynamics – der Kern des Signals
    "rate_zscore",            # Stationär: wie extrem ist die Rate gerade?
    "rate_7d_mean",           # Kurzfristiger Trend
    "rate_momentum_7d",       # Beschleunigung des Trends
    "rate_acceleration",      # Beschleunigung der Beschleunigung
    "rate_volatility_7d",     # Wie instabil ist die Rate?
    "pct_positive_7d",        # Anteil positiver Perioden (0-1)
    # Kapitalfluss
    "stablecoin_inflow_zscore",  # Wie ungewöhnlich sind Inflows?
    "total_inflow_7d_pct",       # Richtung der Inflows
    # Cross-Asset Kontext (nur für Assets die diese haben)
    "sync_score",             # Bewegen sich alle Assets gleich?
    "portfolio_rate_weighted",# Durchschnittliche Market-Rate
]

# Walk-Forward Parameter (identisch zu models/train.py)
MIN_TRAIN    = 540   # 180 Tage Minimum
TEST_PERIODS = 90    # 30 Tage Test-Fenster
STEP_PERIODS = 90    # 30 Tage vorwärts


# ── 1. Daten laden ─────────────────────────────────────────────────────────────

def load_data(symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """Lädt Feature-CSV und gibt DataFrame + verfügbare Regime-Features zurück."""
    path = os.path.join(DATA_DIR, f"{symbol}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Features nicht gefunden: {path}\n"
            f"Zuerst ausführen: python data/fetch_all.py"
        )
    df = pd.read_csv(path, parse_dates=["fundingTime"])
    df = df.sort_values("fundingTime").reset_index(drop=True)

    # Nur Features die tatsächlich vorhanden und nicht zu sparse sind
    available = []
    for feat in REGIME_FEATURES:
        if feat in df.columns:
            nan_pct = df[feat].isna().mean()
            if nan_pct <= 0.20:   # Max 20% NaN erlaubt für Regime-Features
                available.append(feat)

    key = SYMBOL_SHORT[symbol].upper()
    print(f"  {key}: {len(df)} Perioden  |  "
          f"{len(available)}/{len(REGIME_FEATURES)} Regime-Features verfügbar")
    if len(available) < len(REGIME_FEATURES):
        missing = [f for f in REGIME_FEATURES if f not in available]
        print(f"    Nicht verfügbar: {', '.join(missing)}")

    return df, available


# ── 2. GMM Labeling (Stufe 1) ─────────────────────────────────────────────────

def generate_gmm_labels(df: pd.DataFrame,
                         feature_cols: list,
                         symbol: str,
                         n_components: int = N_REGIMES,
                         n_init: int = 15) -> pd.Series:
    """
    Unsupervised Regime-Labeling via Gaussian Mixture Model.

    Strategie zur Umgehung von Label-Switching (GMM-Kernproblem):
      1. Mehrere Starts (n_init) → stabilstes Ergebnis
      2. Ordering nach rate_zscore-Mean statt fundingRate
         (rate_zscore ist stationär → keine Bull-Market-Bias)
      3. Mapping: niedrigster zscore-Mean = CRISIS, höchster = BULL

    Gibt die Regime-Labels als pd.Series (index = df.index) zurück.
    NaN wo Features fehlen.
    """
    df_fit = df[feature_cols].copy()

    # Zeilen für GMM-Training: nur vollständige Zeilen
    mask_valid = df_fit.notna().all(axis=1)
    X_valid    = df_fit[mask_valid].values

    if len(X_valid) < 200:
        raise ValueError(
            f"Zu wenig vollständige Zeilen für GMM: {len(X_valid)}. "
            f"Min: 200. Prüfe Feature-Verfügbarkeit."
        )

    # Normalisieren (GMM ist skalen-sensitiv)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # GMM mit vielen Starts für Stabilität
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        max_iter=200,
        random_state=42,
        reg_covar=1e-5,   # Numerische Stabilität
    )
    gmm.fit(X_scaled)

    # Labels für alle gültigen Zeilen
    raw_labels = gmm.predict(X_scaled)

    # ── Cluster → Regime Mapping ───────────────────────────────────────────────
    # Verwende rate_zscore für Ordering (stationär, kein Bias)
    if "rate_zscore" in df.columns:
        order_col = "rate_zscore"
    else:
        order_col = feature_cols[0]

    # Berechne Mean der Ordering-Spalte pro Cluster
    order_values = df[order_col][mask_valid].values
    cluster_means = {}
    for c in range(n_components):
        mask_c = raw_labels == c
        cluster_means[c] = order_values[mask_c].mean() if mask_c.sum() > 0 else 0.0

    # Sortiere Cluster nach aufsteigendem rate_zscore-Mean
    sorted_clusters = sorted(cluster_means.keys(),
                              key=lambda c: cluster_means[c])
    # Mapping: Index in sorted_clusters → Regime-Wert
    regime_values = [-1, 0, 1, 2]   # CRISIS, BEAR, NEUTRAL, BULL
    label_mapping  = {sorted_clusters[i]: regime_values[i]
                      for i in range(n_components)}

    mapped_labels = pd.Series(raw_labels, dtype=int).map(label_mapping).values

    # Ergebnis zurück auf Original-Index
    regime_labels = pd.Series(np.nan, index=df.index, dtype=float)
    regime_labels[mask_valid] = mapped_labels

    # Statistik
    key = SYMBOL_SHORT.get(
        next((s for s in SYMBOLS if s in (symbol if symbol else "")), ""),
        symbol[:3] if len(symbol) >= 3 else symbol
    )
    counts = pd.Series(mapped_labels).value_counts().sort_index()
    total  = len(mapped_labels)
    print(f"    GMM Regime-Verteilung (auf {total} gültigen Zeilen):")
    for rv, rn in REGIME_MAP.items():
        n = int(counts.get(rv, 0))
        print(f"      {rn:<8}: {n:>5} Perioden  ({n/total*100:.1f}%)")

    return regime_labels


# ── 3. Supervised Regime Classifier – Walk-Forward (Stufe 2) ──────────────────

def _fit_fold_gmm(X_train_scaled: np.ndarray,
                   order_vals: np.ndarray,
                   n_components: int = 4) -> tuple[np.ndarray, dict]:
    """
    Fitted GMM auf Trainingsdaten, gibt Labels + Mapping zurück.
    Ordering nach rate_zscore-Mean (stationär, kein Look-ahead).
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=10,
        max_iter=200,
        random_state=42,
        reg_covar=1e-5,
    )
    gmm.fit(X_train_scaled)
    raw_labels = gmm.predict(X_train_scaled)

    cluster_means = {}
    for c in range(n_components):
        mask_c = raw_labels == c
        cluster_means[c] = order_vals[mask_c].mean() if mask_c.sum() > 0 else 0.0

    sorted_clusters = sorted(cluster_means.keys(), key=lambda c: cluster_means[c])
    regime_values  = [-1, 0, 1, 2]   # CRISIS, BEAR, NEUTRAL, BULL
    label_mapping  = {sorted_clusters[i]: regime_values[i] for i in range(n_components)}

    mapped = np.array([label_mapping[c] for c in raw_labels])
    return mapped, gmm, label_mapping


def train_regime_walk_forward(df: pd.DataFrame,
                               feature_cols: list,
                               symbol: str) -> pd.DataFrame:
    """
    Walk-Forward Expanding Window Regime Classifier.

    FIX (GMM Look-ahead Bias): GMM und Scaler werden per Fold NUR auf
    Trainingsdaten gefittet. Die Regime-Labels für t wurden ausschliesslich
    aus Daten t-1 und früher erzeugt.

    WARUM Supervised nach GMM?
      GMM-Labels sind "Pseudo-Labels" mit Unsicherheit. Das XGBoost-Modell
      lernt daraus ein smootheres, kalibrierteres Signal. Außerdem kann es
      zeitliche Muster erkennen (GMM ist zeitlos, XGB kennt Sequenzen via
      Lag-Features).
    """
    gmm_to_xgb = {-1: 0, 0: 1, 1: 2, 2: 3}

    # Nur vollständige Zeilen (Features + fundingTime)
    df_valid = df.dropna(subset=feature_cols).reset_index(drop=True)

    n   = len(df_valid)
    key = SYMBOL_SHORT[symbol].upper() if symbol in SYMBOL_SHORT else symbol[:3].upper()
    print(f"    Walk-Forward Regime [{key}]: {n} Perioden (GMM per Fold)")

    if n < MIN_TRAIN + TEST_PERIODS:
        raise ValueError(
            f"Zu wenig valide Perioden: {n}. Min: {MIN_TRAIN + TEST_PERIODS}"
        )

    xgb_params = {
        "objective":        "multi:softprob",
        "num_class":        4,
        "n_estimators":     300,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "tree_method":      "hist",
        "verbosity":        0,
        "eval_metric":      "mlogloss",
        "random_state":     42,
    }

    oof_records = []
    train_end   = MIN_TRAIN
    fold        = 0

    # Index der rate_zscore-Spalte (für Cluster-Ordering pro Fold)
    if "rate_zscore" in feature_cols:
        zscore_idx = feature_cols.index("rate_zscore")
    elif "rate_zscore" in df_valid.columns:
        zscore_idx = None   # aus Originaldaten holen
    else:
        zscore_idx = 0      # Fallback: erste Feature-Spalte

    while train_end + TEST_PERIODS <= n:
        test_end = min(train_end + TEST_PERIODS, n)
        fold    += 1

        df_train = df_valid.iloc[:train_end]
        df_test  = df_valid.iloc[train_end:test_end]

        X_train  = df_train[feature_cols].values
        X_test   = df_test[feature_cols].values

        # ── Scaler: NUR auf Train-Daten fitten ────────────────────────────────
        scaler          = StandardScaler()
        X_train_scaled  = scaler.fit_transform(X_train)
        X_test_scaled   = scaler.transform(X_test)

        # ── GMM: NUR auf Train-Daten fitten ───────────────────────────────────
        if zscore_idx is not None:
            order_vals_train = X_train[:, zscore_idx]
        else:
            order_vals_train = df_train["rate_zscore"].values

        gmm_labels_train, gmm_obj, label_mapping = _fit_fold_gmm(
            X_train_scaled, order_vals_train
        )

        # GMM-Labels → XGBoost-Targets (0-3)
        y_train = np.array([gmm_to_xgb[lbl] for lbl in gmm_labels_train])

        # Sicherstellen dass alle 4 Klassen vertreten
        missing = set(range(4)) - set(np.unique(y_train))
        if missing:
            mean_x = X_train_scaled.mean(axis=0, keepdims=True)
            for mc in missing:
                X_train_scaled = np.vstack([X_train_scaled, mean_x])
                y_train        = np.append(y_train, mc)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train_scaled, y_train, verbose=False)

        # ── OOF Predictions auf Test-Daten ────────────────────────────────────
        probs = model.predict_proba(X_test_scaled)

        # Fold-spezifische GMM-Labels für Test (für Accuracy-Reporting)
        if zscore_idx is not None:
            order_vals_test = X_test[:, zscore_idx]
        else:
            order_vals_test = df_test["rate_zscore"].values

        raw_test = gmm_obj.predict(X_test_scaled)
        gmm_test = np.array([label_mapping.get(c, 0) for c in raw_test])

        inv_map = {v: k for k, v in gmm_to_xgb.items()}
        for i, (_, row) in enumerate(df_test.iterrows()):
            pred_xgb = int(np.argmax(probs[i]))
            record = {
                "fundingTime":     row["fundingTime"],
                "regime_gmm":      float(gmm_test[i]),   # fold-spezifisch, kein Look-ahead
                "regime_label":    REGIME_MAP.get(int(gmm_test[i]), "UNKNOWN"),
                "fold":            fold,
                "regime_predicted": inv_map[pred_xgb],
            }
            for j, col in enumerate(REGIME_COLS):
                record[col] = probs[i, j]

            oof_records.append(record)

        train_end += STEP_PERIODS

        if fold % 5 == 0 or train_end + TEST_PERIODS > n:
            pct = min(train_end / n * 100, 100)
            print(f"    Fold {fold:2d}  Train: {len(df_train):4d}  "
                  f"Test: {len(df_test):3d}  ({pct:.0f}%)")

    oof_df = pd.DataFrame(oof_records)
    print(f"    Walk-Forward: {fold} Folds, {len(oof_df)} OOF Predictions")

    valid_preds = oof_df.dropna(subset=["regime_predicted"])
    if len(valid_preds) > 0:
        acc = (valid_preds["regime_gmm"] == valid_preds["regime_predicted"]).mean()
        print(f"    OOF Accuracy (vs Fold-GMM Labels, kein Look-ahead): {acc*100:.1f}%")

    return oof_df


# ── 4. Finales Regime-Modell speichern ────────────────────────────────────────

def train_final_regime_model(df: pd.DataFrame,
                              feature_cols: list,
                              symbol: str) -> xgb.XGBClassifier:
    """Trainiert finales Regime-Modell auf ALLEN Daten und speichert es."""
    gmm_to_xgb = {-1: 0, 0: 1, 1: 2, 2: 3}

    df_clean = df.dropna(subset=feature_cols + ["regime_gmm"]).copy()
    df_clean["_target"] = df_clean["regime_gmm"].map(gmm_to_xgb)

    X = df_clean[feature_cols].values
    y = df_clean["_target"].astype(int).values

    params = {
        "objective":        "multi:softprob",
        "num_class":        4,
        "n_estimators":     300,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "tree_method":      "hist",
        "verbosity":        0,
        "random_state":     42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    model_path = os.path.join(MODELS_DIR, f"regime_{symbol}.json")
    feat_path  = os.path.join(MODELS_DIR, f"regime_{symbol}_features.json")

    model.save_model(model_path)
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)

    print(f"    Finales Regime-Modell gespeichert: {model_path}")
    return model


# ── 5. Visualisierungen ────────────────────────────────────────────────────────

def plot_regime_history(df: pd.DataFrame,
                         oof_df: pd.DataFrame,
                         symbol: str):
    """
    3-Panel Regime Timeline:
      1. Funding Rate mit Regime-Farben im Hintergrund
      2. Regime-Wahrscheinlichkeiten über Zeit (gestapelt)
      3. Predicted vs. GMM-Label Vergleich (Diskrepanz-Plot)
    """
    key = SYMBOL_SHORT[symbol].upper() if symbol in SYMBOL_SHORT else symbol[:3].upper()

    # Merge OOF auf Haupt-DataFrame
    df_plot = df.merge(
        oof_df[["fundingTime"] + REGIME_COLS + ["regime_predicted"]],
        on="fundingTime", how="left"
    )
    df_plot = df_plot.sort_values("fundingTime")

    times = df_plot["fundingTime"]

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Regime History – {key}  (OOF Walk-Forward)",
                 fontweight="bold", fontsize=13)

    # ── Panel 1: Funding Rate mit Regime-Background ────────────────────────────
    ax1 = axes[0]
    rate = df_plot["fundingRate"] * 100

    # Regime-Hintergrund: farbige Bereiche
    if "regime_predicted" in df_plot.columns:
        regime_series = df_plot["regime_predicted"]
        for rv, color in REGIME_COLORS.items():
            mask = regime_series == rv
            ax1.fill_between(times, rate.min() * 1.2, rate.max() * 1.2,
                              where=mask, alpha=0.18, color=color,
                              label=f"{REGIME_MAP[rv]}")

    ax1.plot(times, rate, color="#2c3e50", lw=0.8, alpha=0.9)
    ax1.axhline(0, color="black", lw=0.5, ls=":")
    ax1.set_title("Funding Rate (%) mit Regime-Hintergrund")
    ax1.set_ylabel("Rate (%)")
    ax1.legend(ncol=4, fontsize=8, loc="upper left")

    # ── Panel 2: Regime Probabilities (gestapelt) ─────────────────────────────
    ax2 = axes[1]
    colors_stack = [REGIME_COLORS[-1], REGIME_COLORS[0],
                    REGIME_COLORS[1], REGIME_COLORS[2]]
    labels_stack = ["P(Crisis)", "P(Bear)", "P(Neutral)", "P(Bull)"]

    valid_mask = df_plot[REGIME_COLS[0]].notna()
    if valid_mask.sum() > 0:
        t_valid = times[valid_mask]
        probs   = df_plot.loc[valid_mask, REGIME_COLS].values.T
        ax2.stackplot(t_valid, probs,
                      labels=labels_stack,
                      colors=colors_stack,
                      alpha=0.75)
    ax2.set_ylabel("Regime-Wahrscheinlichkeit")
    ax2.set_title("Regime-Probs [p_crisis, p_bear, p_neutral, p_bull]")
    ax2.legend(ncol=4, fontsize=8, loc="upper left")
    ax2.set_ylim(0, 1)

    # ── Panel 3: Diskrepanz (GMM vs. Predicted) ───────────────────────────────
    ax3 = axes[2]
    if "regime_predicted" in df_plot.columns:
        gmm_valid = df_plot["regime_gmm"]
        pred_valid = df_plot["regime_predicted"]
        wrong = (gmm_valid != pred_valid) & gmm_valid.notna() & pred_valid.notna()

        # Predicted Regime als colored line
        for rv, color in REGIME_COLORS.items():
            mask = pred_valid == rv
            ax3.scatter(times[mask], pred_valid[mask],
                        c=color, s=2, alpha=0.5, label=f"Pred: {REGIME_MAP[rv]}")

        # Fehler markieren
        ax3.scatter(times[wrong], pred_valid[wrong],
                    c="black", s=15, alpha=0.6,
                    marker="x", label="Fehlklassifikation", zorder=5)

        total_valid = int((gmm_valid.notna() & pred_valid.notna()).sum())
        n_wrong     = int(wrong.sum())
        acc_pct     = (1 - n_wrong / max(total_valid, 1)) * 100
        ax3.set_title(f"Predicted Regime (OOF)  |  Accuracy vs GMM: {acc_pct:.1f}%")
        ax3.legend(ncol=5, fontsize=7, loc="upper left")

    ax3.set_ylabel("Regime")
    ax3.set_yticks([-1, 0, 1, 2])
    ax3.set_yticklabels(["CRISIS", "BEAR", "NEUTRAL", "BULL"])
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"regime_history_{SYMBOL_SHORT.get(symbol, symbol[:3])}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Regime History Plot gespeichert: {path}")


def plot_transition_matrix(all_oof: dict, symbols: list):
    """
    Regime-Übergangsmatrix für alle Assets.
    Zeigt: Wie oft wechselt BTC von Bull → Neutral, Neutral → Bear, etc.?
    """
    fig, axes = plt.subplots(1, len(symbols), figsize=(6 * len(symbols), 5))
    if len(symbols) == 1:
        axes = [axes]
    fig.suptitle("Regime Transition Matrix (OOF Predictions)",
                 fontweight="bold", fontsize=12)

    for i, sym in enumerate(symbols):
        ax  = axes[i]
        key = SYMBOL_SHORT[sym].upper() if sym in SYMBOL_SHORT else sym[:3].upper()
        oof = all_oof[sym].dropna(subset=["regime_predicted"])

        if len(oof) < 10:
            ax.text(0.5, 0.5, "Nicht genug Daten", ha="center", va="center")
            ax.set_title(key)
            continue

        regime_seq = oof["regime_predicted"].astype(int).values
        labels     = ["CRISIS", "BEAR", "NEUTRAL", "BULL"]
        rv         = [-1, 0, 1, 2]
        n_reg      = len(rv)

        # Übergangsmatrix zählen
        trans = np.zeros((n_reg, n_reg), dtype=int)
        for t in range(len(regime_seq) - 1):
            r_from = regime_seq[t]
            r_to   = regime_seq[t + 1]
            i_from = rv.index(r_from) if r_from in rv else -1
            i_to   = rv.index(r_to)   if r_to   in rv else -1
            if i_from >= 0 and i_to >= 0:
                trans[i_from, i_to] += 1

        # Normalisieren: Zeilen-weise (bedingte Wahrscheinlichkeit)
        row_sums = trans.sum(axis=1, keepdims=True)
        trans_pct = np.where(row_sums > 0, trans / row_sums * 100, 0)

        sns.heatmap(trans_pct, annot=True, fmt=".0f",
                    xticklabels=labels, yticklabels=labels,
                    cmap="RdYlGn", vmin=0, vmax=100,
                    ax=ax, linewidths=0.5, cbar=False,
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{key}  (Zeile → Von, Spalte → Nach)", fontsize=10)
        ax.set_xlabel("→ Nach Regime")
        ax.set_ylabel("Von Regime →")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "regime_transition_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Transition Matrix gespeichert: {path}")


def plot_regime_summary(all_oof: dict, symbols: list):
    """
    Kompakter Summary: Regime-Zeitanteile + Average Rate pro Regime.
    """
    fig, axes = plt.subplots(2, len(symbols),
                              figsize=(6 * len(symbols), 8))
    if len(symbols) == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Regime Summary – Zeitanteile & Durchschnitts-Rate",
                 fontweight="bold", fontsize=12)

    for i, sym in enumerate(symbols):
        key = SYMBOL_SHORT[sym].upper() if sym in SYMBOL_SHORT else sym[:3].upper()
        oof = all_oof[sym].dropna(subset=["regime_predicted"])

        # Panel oben: Zeitanteil pro Regime
        ax_top = axes[0, i]
        counts  = oof["regime_predicted"].value_counts()
        labels  = [REGIME_MAP.get(rv, str(rv)) for rv in [-1, 0, 1, 2]]
        values  = [counts.get(rv, 0) for rv in [-1, 0, 1, 2]]
        colors  = [REGIME_COLORS[rv] for rv in [-1, 0, 1, 2]]
        bars = ax_top.bar(labels, [v / max(sum(values), 1) * 100 for v in values],
                          color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
        ax_top.set_title(f"{key} – Zeitanteil (%)")
        ax_top.set_ylabel("% der OOF Perioden")
        for bar, val in zip(bars, values):
            pct = val / max(sum(values), 1) * 100
            ax_top.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.1f}%", ha="center", fontsize=9)

        # Panel unten: Durchschnittsrate pro Regime
        ax_bot = axes[1, i]
        if "regime_predicted" in oof.columns and "regime_gmm" in oof.columns:
            # Wir haben keine fundingRate in OOF – nutze regime_gmm als Proxy
            # OOF hat: fundingTime, regime_gmm, p_crisis..p_bull, regime_predicted
            # Lade Rates aus Feature-CSV
            rate_path = os.path.join(DATA_DIR, f"{sym}_features.csv")
            if os.path.exists(rate_path):
                df_rates = pd.read_csv(rate_path,
                                        usecols=["fundingTime", "fundingRate"],
                                        parse_dates=["fundingTime"])
                merged = oof.merge(df_rates, on="fundingTime", how="left")
                avg_rates = {}
                for rv in [-1, 0, 1, 2]:
                    mask = merged["regime_predicted"] == rv
                    r = merged.loc[mask, "fundingRate"]
                    avg_rates[rv] = r.mean() * 100 if len(r) > 0 else 0.0

                bars2 = ax_bot.bar(
                    labels,
                    [avg_rates.get(rv, 0) for rv in [-1, 0, 1, 2]],
                    color=colors, alpha=0.85, edgecolor="white", linewidth=1.5
                )
                ax_bot.axhline(0, color="black", lw=0.8)
                ax_bot.set_title(f"{key} – Ø Rate pro Regime")
                ax_bot.set_ylabel("Durchschnittliche Funding Rate (%)")
                for bar, rv in zip(bars2, [-1, 0, 1, 2]):
                    val = avg_rates.get(rv, 0)
                    ax_bot.text(bar.get_x() + bar.get_width() / 2,
                                val + (0.001 if val >= 0 else -0.002),
                                f"{val:.4f}%", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "regime_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Regime Summary Plot gespeichert: {path}")


# ── 6. Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 68)
    print("  LAYER 1: REGIME CLASSIFIER")
    print("=" * 68)

    all_oof = {}

    for symbol in SYMBOLS:
        key = SYMBOL_SHORT[symbol].upper()
        print(f"\n{'─' * 55}")
        print(f"  [{key}] Regime Pipeline")
        print(f"{'─' * 55}")

        # 1. Laden
        df, feature_cols = load_data(symbol)

        if len(feature_cols) < 3:
            print(f"  SKIP {key}: zu wenige Regime-Features ({len(feature_cols)})")
            continue

        # 2. Walk-Forward Supervised Classifier (GMM per Fold – kein Look-ahead)
        print(f"\n  [1/4] Walk-Forward Regime Classifier (GMM per Fold)...")
        oof_df = train_regime_walk_forward(df, feature_cols, symbol)

        # 3. Finales Modell: GMM auf allen Daten (für Live-Inference, kein Backtest-Bias)
        print(f"\n  [2/4] Finales Modell trainieren (GMM auf Gesamtdaten – nur für Live)...")
        regime_labels = generate_gmm_labels(df, feature_cols, symbol)
        df["regime_gmm"] = regime_labels
        train_final_regime_model(df, feature_cols, symbol)

        # 4. OOF speichern (für Layer 2/3/4)
        save_path = os.path.join(DATA_DIR, f"{symbol}_regime.csv")
        oof_df.to_csv(save_path, index=False)
        print(f"\n  [3/4] OOF Regime Probs gespeichert: {save_path}")

        all_oof[symbol] = oof_df

    # ── Portfolio-Level Regime (nutzt BTC-Datei mit Cross-Asset Features) ──────
    print(f"\n{'─' * 55}")
    print("  [PORTFOLIO] Portfolio-Level Regime")
    print(f"{'─' * 55}")

    btc_sym = "BTCUSDT"
    if btc_sym in [s for s in SYMBOLS if s in all_oof]:
        try:
            df_btc, feat_btc = load_data(btc_sym)
            oof_port = train_regime_walk_forward(df_btc, feat_btc, btc_sym + "_portfolio")
            # Finales Modell für Live-Inference
            df_btc["regime_gmm"] = generate_gmm_labels(df_btc, feat_btc, btc_sym)
            train_final_regime_model(df_btc, feat_btc, "PORTFOLIO")
            port_path = os.path.join(DATA_DIR, "PORTFOLIO_regime.csv")
            oof_port.to_csv(port_path, index=False)
            print(f"  Portfolio-Regime gespeichert: {port_path}")
            all_oof["PORTFOLIO"] = oof_port
        except Exception as e:
            print(f"  Portfolio-Regime fehlgeschlagen: {e}")

    # ── Visualisierungen ────────────────────────────────────────────────────────
    if all_oof:
        print(f"\n{'─' * 55}")
        print("  Visualisierungen erstellen...")
        print(f"{'─' * 55}")

        valid_symbols = [s for s in SYMBOLS if s in all_oof]
        for sym in valid_symbols:
            df_sym, feat_sym = load_data(sym)
            # regime_gmm für Visualisierung: OOF-Labels aus Walk-Forward nutzen
            oof_sym = all_oof[sym][["fundingTime", "regime_gmm"]].dropna()
            df_sym  = df_sym.merge(oof_sym, on="fundingTime", how="left")
            plot_regime_history(df_sym, all_oof[sym], sym)

        plot_transition_matrix(all_oof, valid_symbols)
        plot_regime_summary(all_oof, valid_symbols)

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("  REGIME CLASSIFIER – SUMMARY")
    print(f"{'=' * 68}")
    print(f"\n  {'Asset':<12} {'OOF Perioden':>14} {'Accuracy':>10} "
          f"{'Bull%':>8} {'Bear%':>8} {'Crisis%':>10}")
    print("  " + "─" * 62)

    for sym in SYMBOLS:
        if sym not in all_oof:
            continue
        key = SYMBOL_SHORT[sym].upper()
        oof = all_oof[sym].dropna(subset=["regime_predicted"])
        n   = len(oof)
        acc = (oof["regime_gmm"] == oof["regime_predicted"]).mean() * 100
        bull_pct    = (oof["regime_predicted"] == 2).mean()  * 100
        bear_pct    = (oof["regime_predicted"] == 0).mean()  * 100
        crisis_pct  = (oof["regime_predicted"] == -1).mean() * 100
        print(f"  {key:<12} {n:>14} {acc:>9.1f}% "
              f"{bull_pct:>7.1f}% {bear_pct:>7.1f}% {crisis_pct:>9.1f}%")

    print(f"\n  Erfolgskriterium: Accuracy > 80% OOF (vs. GMM Pseudo-Labels)")
    has_oof = [s for s in SYMBOLS if s in all_oof]
    if has_oof:
        avg_acc = np.mean([
            (all_oof[s].dropna(subset=["regime_predicted"])["regime_gmm"]
             == all_oof[s].dropna(subset=["regime_predicted"])["regime_predicted"]
            ).mean() * 100
            for s in has_oof
            if len(all_oof[s].dropna(subset=["regime_predicted"])) > 0
        ])
        print(f"  Ergebnis: {avg_acc:.1f}% Ø Accuracy über alle Assets  "
              f"{'✓ ERFÜLLT' if avg_acc >= 80 else '✗ NICHT ERFÜLLT'}")

    print(f"\n  Outputs:")
    print(f"    data/raw/{{symbol}}_regime.csv     ← für Layer 2/3/4")
    print(f"    models/saved/regime_*.json        ← finale Modelle")
    print(f"    outputs/regime_history_*.png      ← Timeline")
    print(f"    outputs/regime_transition_*.png   ← Übergangsmatrix")
    print(f"\n  Nächster Schritt:")
    print(f"    python models/alpha.py  (Layer 3: 3 Alpha Modelle)")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
