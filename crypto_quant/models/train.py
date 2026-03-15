"""
models/train.py – XGBoost Training + Walk-Forward Validation
=============================================================
Ausführen (nach data/fetch_all.py):
    cd crypto_quant
    python models/train.py

Pipeline pro Asset:
  1. Daten laden + Feature-Selektion (korrelierte Features prunen)
  2. Optuna Hyperparameter-Tuning (TimeSeriesSplit, Expanding Window)
  3. Walk-Forward Out-of-Sample Predictions (OOF)
  4. Finales Modell auf allen Daten trainieren → speichern
  5. Feature Importance Plot (farbcodiert nach Feature-Kategorie)
  6. OOF-Backtest: Vergleich Rule-based vs. ML-Signal

Ziel-Variable: target_next_rate (exakte Rate der nächsten Periode)
Modell: XGBRegressor mit Huber Loss
  → Robust gegen Outlier-Spikes, behält volle numerische Information
  → Threshold erst bei Inference (flexibel ohne Retraining änderbar)

Walk-Forward Design:
  - Expanding Window (NICHT Rolling) – älteres Wissen bleibt erhalten
  - Min Train: 540 Perioden (180 Tage)
  - Test Window: 90 Perioden (30 Tage)
  - Step: 90 Perioden vorwärts
  - Mit 3 Jahren Daten: ~28 Folds → robuste OOF-Performance
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost nicht installiert: pip install xgboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("  [Hinweis] Optuna nicht installiert – Default-Parameter werden genutzt.")
    print("  Installieren: pip install optuna")

from sklearn.model_selection import TimeSeriesSplit

from config import (
    SYMBOLS, SYMBOL_SHORT, SYMBOL_WEIGHTS,
    DATA_DIR, OUTPUT_DIR, FUNDING_THRESHOLD,
    COST_PER_ROUNDTRIP, CAPITAL, PERIODS_PER_YEAR,
)

MODELS_DIR = "models/saved"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)

# ── Walk-Forward Parameter ─────────────────────────────────────────────────────
MIN_TRAIN_PERIODS = 540    # 180 Tage – Minimum für erstes Training
TEST_PERIODS      = 90     # 30 Tage pro Test-Fenster
STEP_PERIODS      = 90     # 30 Tage vorwärts

# ── Optuna ─────────────────────────────────────────────────────────────────────
N_OPTUNA_TRIALS   = 60     # Pro Asset; ~2-5 Min auf modernem Laptop
N_CV_SPLITS       = 5      # Inner TimeSeriesSplit für Optuna

# ── Feature-Kategorien (für farbcodierten Importance-Plot) ────────────────────
FEATURE_CATEGORIES = {
    "standard":       {
        "color": "#95a5a6",  # grau
        "keywords": ["rate_zscore", "rate_zscore_7d", "rate_momentum", "rate_acceleration",
                     "rate_7d", "rate_30d", "rate_volatility", "pct_positive",
                     "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                     "fundingRate"],
    },
    "cross_exchange": {
        "color": "#e67e22",  # orange
        "keywords": ["cross_divergence", "binance_premium"],
    },
    "basis":          {
        "color": "#1abc9c",  # türkis
        "keywords": ["basis_pct", "basis_abs", "basis_7d_mean", "basis_momentum",
                     "basis_zscore"],
    },
    "stablecoin":     {
        "color": "#27ae60",  # grün
        "keywords": ["usdt_inflow", "usdt_mcap", "usdc", "stablecoin",
                     "total_stablecoin", "total_inflow"],
    },
    "open_interest":  {
        "color": "#2980b9",  # blau
        "keywords": ["oi_change", "oi_health", "sumOpenInterest",
                     "longShortRatio", "ls_zscore"],
    },
    "cross_asset":    {
        "color": "#8e44ad",  # lila
        "keywords": ["btc_eth_spread", "btc_sol_spread", "eth_sol_spread",
                     "hierarchy", "sync_score", "rotation", "portfolio_rate",
                     "rel_score", "btc_falling", "all_above_zscore"],
    },
}

# ── Feature-Spalten die NIEMALS als Input verwendet werden ────────────────────
EXCLUDE_COLS = {
    "fundingTime",
    "rate_annualized_pct",       # redundant: fundingRate × Konstante
    "fundingRate_bybit",         # roh, bereits via cross_divergence erfasst
    "hour", "weekday",           # bereits via sin/cos encodiert
    "target_next_rate",          # Ziel-Variable (kein Leakage!)
    "target_next_positive",      # weitere Targets
    "target_next_3",
    "target_label_ordinal",
    "_fundingTime_floor",        # interner Merge-Key
}

TARGET_COL = "target_next_rate"


# ── 1. Daten laden ─────────────────────────────────────────────────────────────

def load_data(symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Lädt Feature-CSV, wählt Trainings-Features aus und pruned Kollineare.

    Feature-Selektion:
      1. Alle Spalten außer EXCLUDE_COLS
      2. Spalten mit > 50% NaN werden gedroppt (kein ausreichendes Signal)
      3. Kollineare Features (Pearson-Korrelation > 0.95) werden gedroppt:
         Von zwei hochkorrelierten Features wird der representativere behalten
         (höhere Korrelation mit Target). Das verhindert Splitting von Feature
         Importance auf redundante Spalten.

    Returns:
        (df, feature_cols) – bereinigter DataFrame und finale Feature-Liste
    """
    path = os.path.join(DATA_DIR, f"{symbol}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Keine Daten gefunden: {path}\n"
            f"Zuerst ausführen: python data/fetch_all.py"
        )

    df = pd.read_csv(path, parse_dates=["fundingTime"])
    df = df.sort_values("fundingTime").reset_index(drop=True)

    # Nur Zeilen mit gültigem Target (letzte Zeilen durch shift(-1) NaN)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Feature-Kandidaten
    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and not c.startswith("target_")]

    # Spalten mit zu vielen NaN droppen
    nan_fractions = df[feature_cols].isna().mean()
    feature_cols  = [c for c in feature_cols if nan_fractions[c] <= 0.50]

    # Zeilen droppen wo Target NaN ist
    df = df.dropna(subset=[TARGET_COL])

    # Kollineare Features prunen (Pearson > 0.95)
    feature_cols = _prune_collinear(df, feature_cols, threshold=0.95)

    n_valid = len(df.dropna(subset=feature_cols))
    print(f"  Geladen: {len(df)} Perioden | {len(feature_cols)} Features | "
          f"{n_valid} vollständige Zeilen")

    return df, feature_cols


def _prune_collinear(df: pd.DataFrame,
                      feature_cols: list,
                      threshold: float = 0.95) -> list:
    """
    Entfernt kollineare Features (Korrelation > threshold).

    Strategie: Bei zwei hochkorrelierten Features wird das Feature
    mit niedrigerer Korrelation zum Target gedroppt.
    Das behält das informativere Feature und macht Importance-Plots lesbarer.
    """
    df_feat = df[feature_cols + [TARGET_COL]].dropna()
    if len(df_feat) < 100:
        return feature_cols

    corr_matrix = df_feat[feature_cols].corr().abs()
    target_corr = df_feat[feature_cols].corrwith(df_feat[TARGET_COL]).abs()

    to_drop = set()
    for i in range(len(feature_cols)):
        if feature_cols[i] in to_drop:
            continue
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > threshold:
                # Drop das Feature mit niedrigerer Target-Korrelation
                fi, fj = feature_cols[i], feature_cols[j]
                drop = fi if target_corr.get(fi, 0) < target_corr.get(fj, 0) else fj
                to_drop.add(drop)

    pruned = [c for c in feature_cols if c not in to_drop]
    if to_drop:
        print(f"  Kollinearitäts-Pruning: {len(to_drop)} Features gedroppt "
              f"(Threshold: {threshold})")
    return pruned


# ── 2. Hyperparameter-Tuning ───────────────────────────────────────────────────

def tune_hyperparams(df: pd.DataFrame,
                      feature_cols: list,
                      symbol: str,
                      n_trials: int = N_OPTUNA_TRIALS) -> dict:
    """
    Optuna-basiertes Hyperparameter-Tuning mit TimeSeriesSplit.

    Tuning-Raum:
      n_estimators    : 200–1000
      max_depth       : 3–7
      learning_rate   : 0.01–0.15
      subsample       : 0.6–1.0
      colsample_bytree: 0.5–1.0
      min_child_weight: 1–10
      huber_slope     : 0.0001–0.005  ← KORRIGIERT: Rates sind Dezimalzahlen!
                                         0.01% = 0.0001 → Slope muss in gleicher Skala sein

    Optimiert auf: mittlerer MAE über alle CV-Folds (Time-Series-aware).
    """
    if not OPTUNA_AVAILABLE:
        print("  Optuna nicht verfügbar – verwende Defaults")
        return _default_params()

    print(f"  Optuna Tuning [{symbol}]: {n_trials} Trials × {N_CV_SPLITS} Folds...")

    # Trainings-Daten: erst 80% (zeitlich) für Tuning verwenden
    df_tune  = df.iloc[:int(len(df) * 0.8)].copy()
    df_tune  = df_tune.dropna(subset=feature_cols + [TARGET_COL])
    X_tune   = df_tune[feature_cols].values
    y_tune   = df_tune[TARGET_COL].values

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    def objective(trial):
        params = {
            "objective":        "reg:pseudohubererror",
            "huber_slope":      trial.suggest_float("huber_slope", 0.0001, 0.005, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "tree_method":      "hist",
            "verbosity":        0,
        }

        maes = []
        for train_idx, val_idx in tscv.split(X_tune):
            X_tr, X_val = X_tune[train_idx], X_tune[val_idx]
            y_tr, y_val = y_tune[train_idx], y_tune[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            preds = model.predict(X_val)
            mae   = np.mean(np.abs(preds - y_val))
            maes.append(mae)

        return np.mean(maes)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["objective"]   = "reg:pseudohubererror"
    best["tree_method"] = "hist"
    best["verbosity"]   = 0

    print(f"  Best MAE: {study.best_value:.6f}  "
          f"(huber_slope={best['huber_slope']:.5f}, "
          f"depth={best['max_depth']}, lr={best['learning_rate']:.4f})")

    return best


def _default_params() -> dict:
    """Solide Default-Parameter wenn Optuna nicht verfügbar."""
    return {
        "objective":         "reg:pseudohubererror",
        "huber_slope":       0.0005,   # ~5 Basispunkte Outlier-Grenze
        "n_estimators":      500,
        "max_depth":         5,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  3,
        "tree_method":       "hist",
        "verbosity":         0,
        "early_stopping_rounds": 50,
    }


# ── 3. Walk-Forward Out-of-Sample Predictions ─────────────────────────────────

def walk_forward_train(df: pd.DataFrame,
                        feature_cols: list,
                        best_params: dict,
                        symbol: str) -> pd.DataFrame:
    """
    Expanding Window Walk-Forward Validation.

    EXPANDING (nicht Rolling): Das Modell vergisst keine historischen Daten.
    Wenn das Modell in einem Bear-Markt trainiert wurde, behält es dieses
    Wissen auch wenn es danach auf Bull-Markt-Daten trifft.

    Ablauf:
      Fold 1: Train[0:540]    → Predict[540:630]
      Fold 2: Train[0:630]    → Predict[630:720]
      Fold 3: Train[0:720]    → Predict[720:810]
      ...

    Für jeden Fold:
      - Zeilen mit NaN in Feature-Cols werden NICHT für Training genutzt
      - Predictions werden für alle Test-Zeilen gemacht (auch NaN-Feature-Zeilen)
        um Zeitlücken zu vermeiden; fehlende Feature-Werte → Prediction = NaN

    Returns:
        DataFrame mit Spalten: fundingTime, actual_rate, predicted_rate,
                                fold, in_signal_rule, in_signal_ml
    """
    df_clean = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    n        = len(df_clean)

    if n < MIN_TRAIN_PERIODS + TEST_PERIODS:
        raise ValueError(
            f"Zu wenig Daten: {n} Perioden. "
            f"Minimum: {MIN_TRAIN_PERIODS + TEST_PERIODS}"
        )

    oof_records = []
    fold = 0

    train_end = MIN_TRAIN_PERIODS

    while train_end + TEST_PERIODS <= n:
        test_end = min(train_end + TEST_PERIODS, n)
        fold    += 1

        df_train = df_clean.iloc[:train_end].dropna(subset=feature_cols)
        df_test  = df_clean.iloc[train_end:test_end]

        X_train = df_train[feature_cols].values
        y_train = df_train[TARGET_COL].values

        # Modell trainieren (Früh-Stopping braucht Val-Set, hier weglassen
        # damit wir alle Train-Daten nutzen – n_estimators aus Optuna ist genug)
        params = {k: v for k, v in best_params.items()
                  if k != "early_stopping_rounds"}
        model  = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

        # Predictions (NaN-Zeilen bekommen NaN)
        X_test  = df_test[feature_cols].copy()
        has_nan = X_test.isna().any(axis=1)
        preds   = np.full(len(df_test), np.nan)

        if (~has_nan).sum() > 0:
            preds[~has_nan.values] = model.predict(X_test[~has_nan].values)

        for i, (_, row) in enumerate(df_test.iterrows()):
            pred = preds[i]
            oof_records.append({
                "fundingTime":      row["fundingTime"],
                "actual_rate":      row[TARGET_COL],
                "predicted_rate":   pred,
                "fold":             fold,
                # Rule-based: Rate T > Threshold → invest in T+1
                # (execution_delay=1 bereits im raw Signal)
                "in_signal_rule":   float(row["fundingRate"] > FUNDING_THRESHOLD),
                # ML-Signal: Predicted Rate T+1 > Threshold
                "in_signal_ml":     float(pred > FUNDING_THRESHOLD) if not np.isnan(pred) else 0.0,
            })

        train_end += STEP_PERIODS

        if fold % 5 == 0 or train_end + TEST_PERIODS > n:
            pct_done = min(train_end / n * 100, 100)
            print(f"  Walk-Forward: Fold {fold:2d}  "
                  f"Train: {len(df_train):4d}  "
                  f"Test: {len(df_test):3d}  "
                  f"({pct_done:.0f}% Daten verarbeitet)")

    oof_df = pd.DataFrame(oof_records)
    print(f"  Walk-Forward abgeschlossen: {fold} Folds, "
          f"{len(oof_df)} OOF Predictions")

    return oof_df


# ── 4. Finales Modell auf allen Daten ─────────────────────────────────────────

def train_final_model(df: pd.DataFrame,
                       feature_cols: list,
                       best_params: dict,
                       symbol: str) -> xgb.XGBRegressor:
    """
    Trainiert das finale Modell auf ALLEN verfügbaren Daten.

    Kein Validation-Set – wir wollen das gesamte historische Wissen
    ins Modell einbacken. Die OOF-Performance ist unsere realistische
    Performance-Schätzung.

    Speichert:
      models/saved/{symbol}_model.json
      models/saved/{symbol}_feature_cols.json
      models/saved/{symbol}_best_params.json
    """
    df_clean = df.dropna(subset=feature_cols + [TARGET_COL])
    X = df_clean[feature_cols].values
    y = df_clean[TARGET_COL].values

    params = {k: v for k, v in best_params.items()
              if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)

    # Speichern
    model_path  = os.path.join(MODELS_DIR, f"{symbol}_model.json")
    feat_path   = os.path.join(MODELS_DIR, f"{symbol}_feature_cols.json")
    params_path = os.path.join(MODELS_DIR, f"{symbol}_best_params.json")

    model.save_model(model_path)
    with open(feat_path,   "w") as f:
        json.dump(feature_cols, f)
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"  Finales Modell gespeichert: {model_path}")
    return model


# ── 5. Feature Importance ──────────────────────────────────────────────────────

def _get_feature_category(feature_name: str) -> str:
    """Ordnet ein Feature seiner Kategorie zu."""
    for cat, info in FEATURE_CATEGORIES.items():
        if any(kw in feature_name for kw in info["keywords"]):
            return cat
    return "standard"


def plot_feature_importance(model: xgb.XGBRegressor,
                              feature_cols: list,
                              symbol: str,
                              top_n: int = 25):
    """
    Feature Importance Plot farbcodiert nach Kategorie.

    Farben:
      Grau   = Standard Rate Features
      Orange = Cross-Exchange (Bybit)
      Türkis = Basis (Mark - Index)
      Grün   = Stablecoin Inflows
      Blau   = Open Interest
      Lila   = Cross-Asset (proprietär, neue Features)

    Lila-Features weit oben = unsere Cross-Asset-Hypothese ist valide.
    """
    key = SYMBOL_SHORT[symbol].upper()

    importance = model.feature_importances_
    feat_imp   = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).head(top_n)

    # Kategorie + Farbe zuweisen
    feat_imp["category"] = feat_imp["feature"].apply(_get_feature_category)
    feat_imp["color"]    = feat_imp["category"].map(
        {cat: info["color"] for cat, info in FEATURE_CATEGORIES.items()}
    ).fillna("#95a5a6")

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.35)))
    bars = ax.barh(
        feat_imp["feature"][::-1],
        feat_imp["importance"][::-1],
        color=feat_imp["color"][::-1].values,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title(f"Feature Importance – {key}\n"
                 f"(XGBoost Gain, Top {top_n} von {len(feature_cols)})",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Importance (Gain)", fontsize=10)

    # Legende
    legend_patches = [
        mpatches.Patch(color=info["color"], label=cat.replace("_", " ").title())
        for cat, info in FEATURE_CATEGORIES.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"feature_importance_{SYMBOL_SHORT[symbol]}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Feature Importance Plot gespeichert: {path}")

    # Top 5 pro Kategorie ausgeben
    print(f"\n  Top Features [{key}] – nach Kategorie:")
    for cat in FEATURE_CATEGORIES:
        top = feat_imp[feat_imp["category"] == cat].head(3)
        if not top.empty:
            names = ", ".join(top["feature"].tolist())
            print(f"    {cat:<16}: {names}")

    return feat_imp


# ── 6. OOF Analyse & Backtest-Vergleich ───────────────────────────────────────

def analyze_oof(oof_df: pd.DataFrame, symbol: str) -> dict:
    """
    Analysiert OOF-Predictions und vergleicht drei Strategien:
      A: Always-In Benchmark
      B: Rule-based (Rate T > Threshold → In bei T+1)
      C: ML-Signal (Predicted Rate T+1 > Threshold)

    Gibt Metriken aus und erstellt Equity-Vergleichs-Plot.
    """
    key = SYMBOL_SHORT[symbol].upper()
    df  = oof_df.dropna(subset=["actual_rate", "predicted_rate"]).copy()

    # Prediction Quality
    mae  = np.mean(np.abs(df["predicted_rate"] - df["actual_rate"]))
    corr = df["actual_rate"].corr(df["predicted_rate"])
    sign_acc = (np.sign(df["predicted_rate"]) == np.sign(df["actual_rate"])).mean()

    print(f"\n  OOF Prediction Quality [{key}]:")
    print(f"    MAE:                 {mae*100:.5f}%  ({mae*100*3*365:.2f}% p.a. equiv.)")
    print(f"    Korrelation:         {corr:.4f}")
    print(f"    Vorzeichengenauigkeit: {sign_acc*100:.1f}%")

    # Backtest-Vergleich
    r       = df["actual_rate"].values
    sig_rul = df["in_signal_rule"].astype(bool).values
    sig_ml  = df["in_signal_ml"].astype(bool).values

    results = {}
    for label, signal in [("Always-In", np.ones(len(r), dtype=bool)),
                           ("Rule-based", sig_rul),
                           ("ML-Signal",  sig_ml)]:
        full_ret = np.where(signal, r, 0.0)
        n_trades = int(np.sum(np.diff(signal.astype(int), prepend=0) == 1))
        total_fees = n_trades * COST_PER_ROUNDTRIP
        full_ret_net = full_ret.copy()

        # Fees bei Entries abziehen
        enters = np.diff(signal.astype(int), prepend=0) == 1
        exits  = np.diff(signal.astype(int), prepend=0) == -1
        full_ret_net[enters] -= COST_PER_ROUNDTRIP / 2
        full_ret_net[exits]  -= COST_PER_ROUNDTRIP / 2

        equity     = np.cumprod(1 + full_ret_net)
        n_years    = len(r) / PERIODS_PER_YEAR
        cagr       = equity[-1] ** (1 / n_years) - 1
        mean_r     = np.mean(full_ret_net)
        std_r      = np.std(full_ret_net)
        sharpe     = mean_r / std_r * np.sqrt(PERIODS_PER_YEAR) if std_r > 0 else 0
        roll_max   = np.maximum.accumulate(equity)
        max_dd     = np.min((equity - roll_max) / roll_max)

        results[label] = {
            "cagr_pct":    round(cagr * 100, 2),
            "sharpe":      round(sharpe, 3),
            "max_dd_pct":  round(max_dd * 100, 2),
            "n_trades":    n_trades,
            "equity":      equity,
        }

    # Ausgabe
    print(f"\n  OOF Backtest-Vergleich [{key}]:")
    print(f"  {'Strategie':<15} {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>8}")
    print("  " + "─" * 52)
    for label, m in results.items():
        marker = " ←" if label == "ML-Signal" else ""
        print(f"  {label:<15} {m['cagr_pct']:>+7.2f}%  "
              f"{m['sharpe']:>7.3f}  "
              f"{m['max_dd_pct']:>7.2f}%  "
              f"{m['n_trades']:>7}{marker}")

    ml_beats_rule = results["ML-Signal"]["sharpe"] > results["Rule-based"]["sharpe"]
    print(f"\n  ML vs. Rule-based Sharpe: "
          f"{'ML BESSER ✓' if ml_beats_rule else 'ML SCHLECHTER – Modell überprüfen'}")

    # Equity-Kurven Plot
    _plot_oof_equity(df, results, symbol)
    _plot_oof_scatter(df, symbol)

    return {
        "mae":        mae,
        "corr":       corr,
        "sign_acc":   sign_acc,
        "backtest":   results,
        "ml_beats":   ml_beats_rule,
    }


def _plot_oof_equity(df: pd.DataFrame, results: dict, symbol: str):
    """Equity-Kurven: Always-In vs. Rule-based vs. ML."""
    key = SYMBOL_SHORT[symbol].upper()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"OOF Backtest – {key}", fontweight="bold", fontsize=13)

    # Equity-Kurven
    ax = axes[0]
    colors = {"Always-In": "gray", "Rule-based": "steelblue", "ML-Signal": "#8e44ad"}
    for label, m in results.items():
        ax.plot(df["fundingTime"].values,
                m["equity"] * 100 - 100,
                label=f"{label} ({m['cagr_pct']:+.1f}% p.a., Sharpe {m['sharpe']:.2f})",
                color=colors[label],
                lw=2.0 if label == "ML-Signal" else 1.2,
                linestyle="--" if label == "Always-In" else "-",
                alpha=0.9)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Equity-Kurven (OOF, kumulativer Return %)")
    ax.set_ylabel("Kumulativer Return (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Residuals über Zeit
    ax = axes[1]
    residuals = (df["predicted_rate"] - df["actual_rate"]) * 100
    ax.scatter(df["fundingTime"], residuals, s=3, alpha=0.4, color="#8e44ad")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Prediction Residuals über Zeit (%)\n(0 = perfekte Vorhersage)")
    ax.set_ylabel("Predicted − Actual (%)")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"oof_backtest_{SYMBOL_SHORT[symbol]}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  OOF-Backtest Plot gespeichert: {path}")


def _plot_oof_scatter(df: pd.DataFrame, symbol: str):
    """Actual vs. Predicted Rate Scatter."""
    key = SYMBOL_SHORT[symbol].upper()
    fig, ax = plt.subplots(figsize=(7, 7))

    # Punkte einfärben nach Qualitätslabel
    colors = df["actual_rate"].apply(
        lambda r: "#e74c3c" if r < 0 else
                  "#95a5a6" if r < FUNDING_THRESHOLD else
                  "#27ae60" if r < FUNDING_THRESHOLD * 3 else
                  "#8e44ad"
    )

    ax.scatter(df["actual_rate"] * 100, df["predicted_rate"] * 100,
               s=5, alpha=0.35, c=colors)
    lim = max(abs(df["actual_rate"].max()), abs(df["actual_rate"].min())) * 100 * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="Perfekte Vorhersage")
    ax.axhline(FUNDING_THRESHOLD * 100, color="orange", lw=0.8, linestyle="--",
               label=f"Threshold ({FUNDING_THRESHOLD*100:.2f}%)")
    ax.axvline(FUNDING_THRESHOLD * 100, color="orange", lw=0.8, linestyle="--")
    corr = df["actual_rate"].corr(df["predicted_rate"])
    ax.set_title(f"Actual vs. Predicted Rate [{key}]\n(OOF, ρ={corr:.3f})",
                 fontweight="bold")
    ax.set_xlabel("Actual Rate (%)")
    ax.set_ylabel("Predicted Rate (%)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"oof_scatter_{SYMBOL_SHORT[symbol]}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  MODEL TRAINING – XGBoost Walk-Forward")
    print("=" * 65)
    print(f"  Assets:          {', '.join(SYMBOLS)}")
    print(f"  Walk-Forward:    Expanding Window")
    print(f"  Min Train:       {MIN_TRAIN_PERIODS} Perioden ({MIN_TRAIN_PERIODS//3} Tage)")
    print(f"  Test Window:     {TEST_PERIODS} Perioden ({TEST_PERIODS//3} Tage)")
    print(f"  Optuna Trials:   {N_OPTUNA_TRIALS if OPTUNA_AVAILABLE else 'DISABLED'}")
    print()

    all_results = {}

    for symbol in SYMBOLS:
        key = SYMBOL_SHORT[symbol].upper()
        print(f"\n{'═'*65}")
        print(f"  [{key}] Training Pipeline")
        print(f"{'═'*65}")

        try:
            # 1. Daten laden
            print(f"\n  [1/4] Daten laden...")
            df, feature_cols = load_data(symbol)

            # 2. Hyperparameter Tuning
            print(f"\n  [2/4] Hyperparameter Tuning...")
            best_params = tune_hyperparams(df, feature_cols, symbol,
                                            n_trials=N_OPTUNA_TRIALS)

            # 3. Walk-Forward OOF Predictions
            print(f"\n  [3/4] Walk-Forward Validation...")
            oof_df = walk_forward_train(df, feature_cols, best_params, symbol)

            # OOF speichern
            oof_path = os.path.join(DATA_DIR, f"{symbol}_oof_predictions.csv")
            oof_df.to_csv(oof_path, index=False)
            print(f"  OOF Predictions gespeichert: {oof_path}")

            # 4. Finales Modell
            print(f"\n  [4/4] Finales Modell trainieren...")
            final_model = train_final_model(df, feature_cols, best_params, symbol)

            # Feature Importance
            imp_df = plot_feature_importance(final_model, feature_cols, symbol)

            # OOF Analyse
            oof_metrics = analyze_oof(oof_df, symbol)

            all_results[symbol] = {
                "feature_cols":  feature_cols,
                "best_params":   best_params,
                "oof_metrics":   oof_metrics,
                "n_features":    len(feature_cols),
                "n_train":       len(df),
            }

        except FileNotFoundError as e:
            print(f"  FEHLER: {e}")
            continue
        except Exception as e:
            import traceback
            print(f"  FEHLER bei [{key}]: {e}")
            traceback.print_exc()
            continue

    # ── Gesamt-Summary ─────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*65}")
        print("  TRAINING ABGESCHLOSSEN – Summary")
        print(f"{'='*65}")
        print(f"\n  {'Asset':<8} {'Features':>9} {'Corr':>8} {'ML Sharpe':>10} {'Rule Sharpe':>12} {'ML besser?':>12}")
        print("  " + "─" * 62)
        for sym, res in all_results.items():
            key  = SYMBOL_SHORT[sym].upper()
            m    = res["oof_metrics"]
            bt   = m["backtest"]
            ml_s = bt.get("ML-Signal",  {}).get("sharpe", 0)
            rb_s = bt.get("Rule-based", {}).get("sharpe", 0)
            wins = "JA ✓" if m["ml_beats"] else "NEIN ✗"
            print(f"  {key:<8} {res['n_features']:>9}  "
                  f"{m['corr']:>7.4f}  "
                  f"{ml_s:>9.3f}  "
                  f"{rb_s:>11.3f}  "
                  f"{wins:>12}")

        print(f"\n  Gespeicherte Modelle: {MODELS_DIR}/")
        print(f"  Outputs: {OUTPUT_DIR}/")
        print()
        print("  Nächster Schritt:")
        print("  → Erfolg: Sharpe ML > Sharpe Rule-based auf ALLEN Assets")
        print("  → Dann:   backtest/portfolio.py mit ML-Signal + Kelly laufen lassen")
        print()


if __name__ == "__main__":
    main()
