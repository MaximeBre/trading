"""
models/alpha.py – Layer 3: Alpha Models (3 spezialisierte Modelle)
===================================================================
Ausführen (nach regime.py):
    cd crypto_quant
    python models/alpha.py

3 Alpha-Modelle pro Asset, Walk-Forward:
    3A: 8h-Horizon  – Exit/Entry Timing (target_next_rate)
    3B: 24h-Horizon – Hold/Reduce Entscheidung (3-Perioden Ø Rate)
    3C: Regime-Conditional Ensemble – spezialisiert pro Regime

Ensemble-Prediction (Gewichte per Optuna optimierbar):
    final_alpha = 0.40 * alpha_8h + 0.35 * alpha_24h + 0.25 * alpha_regime

Regime-Probs aus Layer 1 fließen als Features in 3A und 3B ein.
Für 3C: Separates Modell pro Regime (CRISIS/BEAR/NEUTRAL/BULL),
         dann gewichteter Ensemble via Regime-Probs.

Look-ahead Prüfung:
    - Regime OOF Probs (nur Out-of-Sample Perioden ab Fold 1)
    - GMM-Labels: globale Fit, aber nur auf aktuellen Features → kein LA Bias
    - 3C nutzt NUR OOF Regime-Probs für Gewichtung → korrekt

Output:
    data/raw/{symbol}_alpha_oof.csv   ← Ensemble + Einzelpredictions
    models/saved/alpha_{symbol}_*.json
    outputs/alpha_performance_{symbol}.png
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

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import xgboost as xgb

from config import (
    SYMBOLS, SYMBOL_SHORT, SYMBOL_WEIGHTS,
    DATA_DIR, OUTPUT_DIR, PERIODS_PER_YEAR,
    FUNDING_THRESHOLD, COST_PER_ROUNDTRIP,
)
from models.regime import REGIME_COLS   # ["p_crisis", "p_bear", "p_neutral", "p_bull"]

MODELS_DIR = "models/saved"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Walk-Forward Parameter (identisch zu train.py und regime.py) ──────────────
MIN_TRAIN    = 540
TEST_PERIODS = 90
STEP_PERIODS = 90

# ── Ensemble Gewichte ─────────────────────────────────────────────────────────
W_8H     = 0.40
W_24H    = 0.35
W_REGIME = 0.25

# ── Features die NIEMALS als Input (Leakage-Prüfung) ─────────────────────────
EXCLUDE_ALWAYS = {
    "fundingTime", "rate_annualized_pct", "hour", "weekday",
    "fundingRate_bybit",
    "target_next_rate", "target_next_positive", "target_next_3",
    "target_label_ordinal", "_fundingTime_floor",
}


# ── 1. Daten laden ─────────────────────────────────────────────────────────────

def load_features_with_regime(symbol: str) -> tuple[pd.DataFrame, list, list]:
    """
    Lädt Feature-CSV + Regime-OOF-Probs und merged sie.

    Regime-Probs sind nur für OOF-Perioden (ab Fold 1, ~Periode 540) verfügbar.
    Für frühere Perioden: NaN → XGBoost behandelt als fehlenden Wert (korrekt).

    Returns:
        df           : Gesamt-DataFrame mit Regime-Probs
        feature_cols : Alle Features für 3A/3B
        regime_cols  : Nur die verfügbaren Regime-Prob-Spalten
    """
    feat_path   = os.path.join(DATA_DIR, f"{symbol}_features.csv")
    regime_path = os.path.join(DATA_DIR, f"{symbol}_regime.csv")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features nicht gefunden: {feat_path}")

    df = pd.read_csv(feat_path, parse_dates=["fundingTime"])
    df = df.sort_values("fundingTime").reset_index(drop=True)

    # Regime-Probs mergen (OOF, daher NaN in frühen Perioden)
    avail_regime_cols = []
    if os.path.exists(regime_path):
        df_reg = pd.read_csv(regime_path, parse_dates=["fundingTime"])
        df_reg = df_reg[["fundingTime"] + [c for c in REGIME_COLS
                                            if c in df_reg.columns]].copy()
        df = pd.merge(df, df_reg, on="fundingTime", how="left")
        avail_regime_cols = [c for c in REGIME_COLS if c in df.columns]
        nan_pct = df[avail_regime_cols].isna().mean().mean() * 100
        print(f"    Regime-Probs: {len(avail_regime_cols)} Features  "
              f"({nan_pct:.0f}% NaN – OOF coverage ab Fold 1)")
    else:
        print(f"    Regime-Probs: nicht gefunden  "
              f"(Zuerst python models/regime.py ausführen)")

    # Feature-Selektion: < 50% NaN, kein Leakage
    candidate_cols = [c for c in df.columns
                      if c not in EXCLUDE_ALWAYS
                      and not c.startswith("target_")
                      and df[c].isna().mean() <= 0.50]

    key = SYMBOL_SHORT[symbol].upper()
    print(f"    {key}: {len(df)} Perioden  |  "
          f"{len(candidate_cols)} Basis-Features  |  "
          f"{len(avail_regime_cols)} Regime-Features")

    return df, candidate_cols, avail_regime_cols


def _prune_collinear(df: pd.DataFrame, feature_cols: list,
                     target_col: str, threshold: float = 0.95) -> list:
    """Entfernt Features mit Pearson-Korrelation > threshold."""
    feature_cols = list(dict.fromkeys(feature_cols))   # Duplikate entfernen
    df_fit = df[feature_cols + [target_col]].dropna()
    if len(df_fit) < 100:
        return feature_cols
    corr   = df_fit[feature_cols].corr().abs()
    t_corr = df_fit[feature_cols].corrwith(df_fit[target_col]).abs()
    # t_corr als dict für O(1) scalar lookup (verhindert Series-Vergleich Bug)
    t_corr_dict = t_corr.to_dict()
    to_drop = set()
    for i in range(len(feature_cols)):
        if feature_cols[i] in to_drop:
            continue
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in to_drop:
                continue
            if corr.iloc[i, j] > threshold:
                fi, fj = feature_cols[i], feature_cols[j]
                drop = fi if t_corr_dict.get(fi, 0) < t_corr_dict.get(fj, 0) else fj
                to_drop.add(drop)
    pruned = [c for c in feature_cols if c not in to_drop]
    if to_drop:
        print(f"    Kollinearitäts-Pruning: {len(to_drop)} Features gedroppt")
    return pruned


# ── 2. Alpha 8h (Modell 3A) ───────────────────────────────────────────────────

BASE_PARAMS_8H = {
    "objective":        "reg:pseudohubererror",
    "huber_slope":      0.0005,
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "tree_method":      "hist",
    "verbosity":        0,
}

BASE_PARAMS_24H = {
    **BASE_PARAMS_8H,
    "huber_slope": 0.001,    # Toleranter bei 24h Horizon
    "max_depth":   4,        # Weniger Overfit auf Kurzfrist-Noise
}


def _walk_forward_regressor(df_clean: pd.DataFrame,
                              feature_cols: list,
                              target_col: str,
                              params: dict,
                              model_label: str) -> np.ndarray:
    """
    Generisches Walk-Forward für einen Regressor.
    Gibt OOF-Predictions als Array zurück (aligned auf df_clean.index).
    Perioden ohne Prediction bekommen NaN.
    """
    n    = len(df_clean)
    oof  = np.full(n, np.nan)

    train_end = MIN_TRAIN
    fold      = 0

    while train_end + TEST_PERIODS <= n:
        test_end = min(train_end + TEST_PERIODS, n)
        fold    += 1

        df_tr = df_clean.iloc[:train_end].dropna(subset=feature_cols)
        df_te = df_clean.iloc[train_end:test_end]

        if len(df_tr) < 100:
            train_end += STEP_PERIODS
            continue

        X_tr = df_tr[feature_cols].values
        y_tr = df_tr[target_col].dropna().values
        # Stelle sicher X und y aligned sind (target könnte NaN haben)
        tr_valid = df_tr[target_col].notna()
        X_tr = df_tr[feature_cols][tr_valid].values
        y_tr = df_tr[target_col][tr_valid].values

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)

        X_te    = df_te[feature_cols].copy()
        has_nan = X_te.isna().any(axis=1)
        preds   = np.full(len(df_te), np.nan)
        if (~has_nan).sum() > 0:
            preds[~has_nan.values] = model.predict(X_te[~has_nan].values)

        start_idx = train_end
        oof[start_idx:test_end] = preds

        train_end += STEP_PERIODS

    return oof, fold


# ── 3. Alpha 24h (Modell 3B) ──────────────────────────────────────────────────

def _build_target_24h(df: pd.DataFrame) -> pd.Series:
    """
    24h Durchschnitts-Rate: Mittel der nächsten 3 × 8h Perioden.
    target_24h_avg = mean(rate[T+1], rate[T+2], rate[T+3])
    Keine Normalisierung – direktes Return-Ziel.
    """
    return df["fundingRate"].shift(-1).rolling(3, min_periods=1).mean().shift(-2)


# ── 4. Regime-Conditional (Modell 3C) ─────────────────────────────────────────

REGIME_MAP_INT = {0: "crisis", 1: "bear", 2: "neutral", 3: "bull"}

def _walk_forward_regime_conditional(
        df_clean: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        regime_prob_cols: list,
) -> np.ndarray:
    """
    Regime-Conditional Ensemble:
      Für jeden Test-Zeitpunkt:
        1. Trainiere 4 Modelle auf Perioden pro Regime (innerhalb Trainings-Split)
        2. Ensemble-Prediction = gewichteter Mix via Regime-Probs
           alpha_regime = p_crisis * M_crisis(X) + p_bear * M_bear(X) + ...
        3. Falls Regime-Probs NaN: Fallback auf einfaches Modell (wie 3A)

    KEIN Look-ahead Bias: Regime-Probs für Test-Daten = OOF aus Layer 1.
    Für Training (Perioden vor erstem OOF-Fold): GMM-Label direkt nutzen.
    """
    n    = len(df_clean)
    oof  = np.full(n, np.nan)

    # Für Training: GMM-Label (als Integer 0-3 direkt)
    has_gmm = "regime_gmm" in df_clean.columns

    params_regime = {
        **BASE_PARAMS_8H,
        "n_estimators": 300,   # Weniger – seltene Regime haben weniger Daten
        "max_depth":    4,
    }

    train_end = MIN_TRAIN
    fold      = 0

    while train_end + TEST_PERIODS <= n:
        test_end = min(train_end + TEST_PERIODS, n)
        fold    += 1

        df_tr = df_clean.iloc[:train_end].copy()
        df_te = df_clean.iloc[train_end:test_end].copy()

        X_te    = df_te[feature_cols].copy()
        has_nan = X_te.isna().any(axis=1)
        preds   = np.full(len(df_te), np.nan)

        # Regime-conditional Modelle pro Regime
        regime_models = {}
        for reg_int, reg_name in REGIME_MAP_INT.items():
            if has_gmm and "regime_gmm" in df_tr.columns:
                # Trainiere auf Perioden in diesem Regime
                mask_reg = (df_tr["regime_gmm"] == (reg_int - 2)).fillna(False)
                # regime_gmm ist -1,0,1,2 → reg_int ist 0,1,2,3 → offset -2
                # CRISIS(-1)=0, BEAR(0)=1, NEUTRAL(1)=2, BULL(2)=3
                # Also: mask_reg = df_tr["regime_gmm"] == (reg_int - 1 - 1)?
                # Let me be explicit:
                gmm_to_int = {-1: 0, 0: 1, 1: 2, 2: 3}
                int_to_gmm = {v: k for k, v in gmm_to_int.items()}
                gmm_val    = int_to_gmm[reg_int]
                mask_reg   = df_tr["regime_gmm"] == gmm_val
            else:
                mask_reg = pd.Series(True, index=df_tr.index)  # Alle

            df_reg = df_tr[mask_reg].dropna(subset=feature_cols + [target_col])

            if len(df_reg) < 50:
                continue   # Zu wenig Daten für dieses Regime

            X_r = df_reg[feature_cols].values
            y_r = df_reg[target_col].values

            m = xgb.XGBRegressor(**params_regime)
            m.fit(X_r, y_r, verbose=False)
            regime_models[reg_int] = m

        # Fallback-Modell falls kein Regime-Modell trainiert
        if not regime_models:
            df_full = df_tr.dropna(subset=feature_cols + [target_col])
            if len(df_full) > 100:
                fallback = xgb.XGBRegressor(**BASE_PARAMS_8H)
                fallback.fit(df_full[feature_cols].values,
                             df_full[target_col].values, verbose=False)
                regime_models = {i: fallback for i in range(4)}
            else:
                train_end += STEP_PERIODS
                continue

        # Predictions für Test-Periode
        valid_rows = ~has_nan.values

        if valid_rows.sum() > 0:
            X_te_valid = X_te[~has_nan].values

            # Regime-Probs aus OOF für Test-Daten
            # Falls verfügbar: Ensemble; sonst: einfacher Mittelwert
            prob_cols_avail = [c for c in regime_prob_cols if c in df_te.columns]

            if prob_cols_avail:
                te_probs = df_te[prob_cols_avail][~has_nan].values
                # Fallback: NaN-Probs → gleichgewichtet
                row_nan  = np.isnan(te_probs).any(axis=1)
                te_probs[row_nan] = 0.25
                # Normalisieren (sollten bereits summen zu 1, aber sicherheitshalber)
                row_sums = te_probs.sum(axis=1, keepdims=True)
                te_probs = te_probs / np.where(row_sums > 0, row_sums, 1)
            else:
                n_valid  = valid_rows.sum()
                te_probs = np.full((n_valid, 4), 0.25)

            row_preds = np.zeros(valid_rows.sum())
            for reg_int, m in regime_models.items():
                if reg_int < te_probs.shape[1]:
                    w = te_probs[:, reg_int]
                    row_preds += w * m.predict(X_te_valid)

            preds[valid_rows] = row_preds

        oof[train_end:test_end] = preds
        train_end += STEP_PERIODS

    return oof, fold


# ── 5. Gesamte Alpha Pipeline pro Asset ───────────────────────────────────────

def run_alpha_pipeline(symbol: str) -> pd.DataFrame:
    """
    Lädt Daten, trainiert 3 Alpha-Modelle in Walk-Forward,
    erstellt Ensemble, bewertet Performance, speichert OOF CSV.
    """
    key = SYMBOL_SHORT[symbol].upper() if symbol in SYMBOL_SHORT else symbol[:3].upper()
    print(f"\n  [{key}] Alpha Pipeline (3 Modelle + Ensemble)...")

    # Daten + Features
    df, base_feature_cols, regime_prob_cols = load_features_with_regime(symbol)

    # ── Target 24h ─────────────────────────────────────────────────────────────
    df["target_24h_avg"] = _build_target_24h(df)

    # ── Feature Sets ───────────────────────────────────────────────────────────
    # 3A: Standard Features + Regime-Probs
    feature_cols_8h = _prune_collinear(
        df, base_feature_cols + regime_prob_cols,
        target_col="target_next_rate"
    )

    # 3B: Gleiche Features, anderes Target
    feature_cols_24h = _prune_collinear(
        df.dropna(subset=["target_24h_avg"]),
        base_feature_cols + regime_prob_cols,
        target_col="target_24h_avg"
    )

    # Clean-ups: dropna auf Target
    df_8h  = df.dropna(subset=["target_next_rate"]).reset_index(drop=True)
    df_24h = df.dropna(subset=["target_24h_avg"]).reset_index(drop=True)
    # df_24h braucht auch regime_gmm für 3C
    if "regime_gmm" in df.columns:
        df_24h["regime_gmm"] = df.dropna(subset=["target_24h_avg"])["regime_gmm"].values

    print(f"\n    [3A] 8h Alpha (target_next_rate)...")
    oof_8h, folds_8h = _walk_forward_regressor(
        df_8h, feature_cols_8h, "target_next_rate", BASE_PARAMS_8H, "8h"
    )
    print(f"    → {folds_8h} Folds, "
          f"{(~np.isnan(oof_8h)).sum()} OOF Predictions")

    print(f"\n    [3B] 24h Alpha (target_24h_avg)...")
    oof_24h, folds_24h = _walk_forward_regressor(
        df_24h, feature_cols_24h, "target_24h_avg", BASE_PARAMS_24H, "24h"
    )
    print(f"    → {folds_24h} Folds, "
          f"{(~np.isnan(oof_24h)).sum()} OOF Predictions")

    print(f"\n    [3C] Regime-Conditional Alpha...")
    # 3C läuft auf df_8h (identisches Target wie 3A)
    oof_regime, folds_reg = _walk_forward_regime_conditional(
        df_8h, feature_cols_8h, "target_next_rate", regime_prob_cols
    )
    print(f"    → {folds_reg} Folds, "
          f"{(~np.isnan(oof_regime)).sum()} OOF Predictions")

    # ── Ensemble ────────────────────────────────────────────────────────────────
    # Alignment: 3A und 3C sind auf df_8h; 3B auf df_24h (fast gleich)
    # Für Ensemble: merge auf gemeinsame fundingTime
    result_df = df_8h[["fundingTime", "target_next_rate",
                         "fundingRate"]].copy()
    result_df["alpha_8h"]     = oof_8h
    result_df["alpha_regime"] = oof_regime

    # 24h alignment: df_24h hat andere Länge potentiell
    df_24h_merge = df_24h[["fundingTime", "target_24h_avg"]].copy()
    df_24h_merge["alpha_24h"] = oof_24h
    result_df = pd.merge(result_df, df_24h_merge, on="fundingTime", how="left")

    # Gewichtetes Ensemble
    valid_mask = (result_df["alpha_8h"].notna() &
                  result_df["alpha_24h"].notna() &
                  result_df["alpha_regime"].notna())

    result_df["alpha_ensemble"] = np.nan
    if valid_mask.sum() > 0:
        result_df.loc[valid_mask, "alpha_ensemble"] = (
            W_8H     * result_df.loc[valid_mask, "alpha_8h"]
            + W_24H    * result_df.loc[valid_mask, "alpha_24h"]
            + W_REGIME * result_df.loc[valid_mask, "alpha_regime"]
        )

    # Fallback: nur 3A wenn 3B/3C NaN
    only_8h = result_df["alpha_8h"].notna() & result_df["alpha_ensemble"].isna()
    result_df.loc[only_8h, "alpha_ensemble"] = result_df.loc[only_8h, "alpha_8h"]

    # Signale: Exit-Filter Logik
    # In > 0 wenn predicted alpha > threshold
    result_df["in_signal_alpha"] = (
        result_df["alpha_ensemble"] > FUNDING_THRESHOLD
    ).astype(float)
    result_df.loc[result_df["alpha_ensemble"].isna(), "in_signal_alpha"] = np.nan

    return result_df, {
        "feature_cols_8h":  feature_cols_8h,
        "feature_cols_24h": feature_cols_24h,
    }


# ── 6. Finale Modelle speichern ────────────────────────────────────────────────

def train_and_save_final_models(symbol: str,
                                  df: pd.DataFrame,
                                  feature_cols: list):
    """Trainiert finale Modelle auf ALLEN Daten und speichert sie."""
    key = SYMBOL_SHORT[symbol].upper() if symbol in SYMBOL_SHORT else symbol[:3].upper()

    for model_name, target, params in [
        ("8h",  "target_next_rate", BASE_PARAMS_8H),
        ("24h", "target_24h_avg",   BASE_PARAMS_24H),
    ]:
        df_clean = df.dropna(subset=feature_cols + [target])
        if len(df_clean) < 100:
            continue
        X = df_clean[feature_cols].values
        y = df_clean[target].values
        model = xgb.XGBRegressor(**params)
        model.fit(X, y, verbose=False)

        path = os.path.join(MODELS_DIR, f"alpha_{symbol}_{model_name}.json")
        feat_path = os.path.join(MODELS_DIR,
                                  f"alpha_{symbol}_{model_name}_features.json")
        model.save_model(path)
        with open(feat_path, "w") as f:
            json.dump(feature_cols, f)

    print(f"    Finale Modelle gespeichert: alpha_{symbol}_8h.json, _24h.json")


# ── 7. Performance Analysis ────────────────────────────────────────────────────

def analyze_alpha_performance(result_df: pd.DataFrame, symbol: str) -> dict:
    """
    Bewertet Alpha-Performance für jedes der 3 Modelle + Ensemble.
    Vergleicht mit Always-In Benchmark.
    """
    key = SYMBOL_SHORT[symbol].upper() if symbol in SYMBOL_SHORT else symbol[:3].upper()

    df = result_df.dropna(subset=["target_next_rate", "alpha_ensemble"]).copy()
    r  = df["target_next_rate"].values

    metrics = {}
    print(f"\n    Alpha Performance [{key}]:")
    print(f"    {'Modell':<16} {'Corr':>7} {'Sign%':>7} {'MAE (bps)':>10}")
    print("    " + "─" * 44)

    for model_col, label in [
        ("alpha_8h",       "3A: 8h"),
        ("alpha_24h",      "3B: 24h"),
        ("alpha_regime",   "3C: Regime"),
        ("alpha_ensemble", "Ensemble"),
    ]:
        if model_col not in df.columns:
            continue
        sub = df.dropna(subset=[model_col])
        if len(sub) < 100:
            continue

        pred = sub[model_col].values
        act  = sub["target_next_rate"].values
        corr = np.corrcoef(pred, act)[0, 1]
        sign = (np.sign(pred) == np.sign(act)).mean()
        mae  = np.mean(np.abs(pred - act)) * 10000   # in Basispunkten

        print(f"    {label:<16} {corr:>7.4f} {sign*100:>6.1f}% {mae:>9.2f}bp")
        metrics[model_col] = {"corr": corr, "sign_acc": sign, "mae_bps": mae}

    # ── OOF Backtest: Always-In vs Ensemble ───────────────────────────────────
    sig_ens = df["in_signal_alpha"].astype(bool).values

    results = {}
    print(f"\n    OOF Backtest [{key}]:")
    print(f"    {'Strategie':<18} {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}"
          f" {'Trades':>8}")
    print("    " + "─" * 56)

    for label, signal in [
        ("Always-In",   np.ones(len(r), dtype=bool)),
        ("Ensemble",    sig_ens),
    ]:
        full_ret = np.where(signal, r, 0.0)
        enters   = np.diff(signal.astype(int), prepend=0) == 1
        exits    = np.diff(signal.astype(int), prepend=0) == -1
        full_ret_net = full_ret.copy()
        full_ret_net[enters] -= COST_PER_ROUNDTRIP / 2
        full_ret_net[exits]  -= COST_PER_ROUNDTRIP / 2

        equity  = np.cumprod(1 + full_ret_net)
        n_yrs   = len(r) / PERIODS_PER_YEAR
        cagr    = equity[-1] ** (1 / n_yrs) - 1
        mean_r  = np.mean(full_ret_net)
        std_r   = np.std(full_ret_net)
        sharpe  = mean_r / std_r * np.sqrt(PERIODS_PER_YEAR) if std_r > 0 else 0
        dd      = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
        max_dd  = dd.min()
        n_trd   = int(np.sum(enters))

        results[label] = {
            "cagr_pct": round(cagr * 100, 2),
            "sharpe":   round(sharpe, 3),
            "max_dd_pct": round(max_dd * 100, 2),
            "n_trades": n_trd,
            "equity":   equity,
        }
        marker = " ←" if label == "Ensemble" else ""
        print(f"    {label:<18} {cagr*100:>+7.2f}%  {sharpe:>7.3f}  "
              f"{max_dd*100:>7.2f}%  {n_trd:>7}{marker}")

    metrics["backtest"] = results
    return metrics


# ── 8. Plot ────────────────────────────────────────────────────────────────────

def plot_alpha_performance(result_df: pd.DataFrame,
                            metrics: dict,
                            symbol: str):
    """2-Panel: Equity-Kurven + Scatter Actual vs Ensemble."""
    key = SYMBOL_SHORT[symbol].upper() if symbol in SYMBOL_SHORT else symbol[:3].upper()

    df = result_df.dropna(subset=["target_next_rate", "alpha_ensemble"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Alpha Model Performance – {key}  (OOF Walk-Forward)",
                 fontweight="bold", fontsize=12)

    # Panel 1: Equity-Kurven
    ax = axes[0]
    bt = metrics.get("backtest", {})
    colors = {"Always-In": "#95a5a6", "Ensemble": "#8e44ad"}
    for label, m in bt.items():
        eq = (m["equity"] - 1) * 100
        ax.plot(df["fundingTime"].values[:len(eq)], eq,
                color=colors.get(label, "blue"),
                lw=2.0 if label == "Ensemble" else 1.2,
                ls="--" if label == "Always-In" else "-",
                label=f"{label}  Sharpe={m['sharpe']:.1f}  "
                      f"CAGR={m['cagr_pct']:+.1f}%")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Equity-Kurven (OOF)")
    ax.set_ylabel("Kumulativer Return (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Panel 2: Actual vs Ensemble Scatter
    ax = axes[1]
    corr_info = metrics.get("alpha_ensemble", {})
    ax.scatter(df["target_next_rate"] * 100,
               df["alpha_ensemble"] * 100,
               s=4, alpha=0.35, color="#8e44ad")
    lim = df["target_next_rate"].abs().max() * 100 * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.axhline(FUNDING_THRESHOLD * 100, color="#e67e22", lw=0.8, ls="--")
    ax.axvline(FUNDING_THRESHOLD * 100, color="#e67e22", lw=0.8, ls="--")
    corr = corr_info.get("corr", 0)
    sign = corr_info.get("sign_acc", 0)
    ax.set_title(f"Actual vs. Ensemble Prediction  ρ={corr:.3f}, Sign={sign:.1%}")
    ax.set_xlabel("Actual Rate (%)")
    ax.set_ylabel("Predicted Rate (%)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                         f"alpha_performance_{SYMBOL_SHORT.get(symbol, symbol[:3])}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Plot gespeichert: {path}")


# ── 9. Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 68)
    print("  LAYER 3: ALPHA MODELS (3A: 8h | 3B: 24h | 3C: Regime-Conditional)")
    print("=" * 68)

    all_metrics = {}

    for symbol in SYMBOLS:
        key = SYMBOL_SHORT[symbol].upper()
        print(f"\n{'─' * 55}")
        print(f"  [{key}] Alpha Pipeline")
        print(f"{'─' * 55}")

        try:
            result_df, feature_info = run_alpha_pipeline(symbol)

            # Performance analysieren
            metrics = analyze_alpha_performance(result_df, symbol)
            all_metrics[symbol] = metrics

            # Plots
            plot_alpha_performance(result_df, metrics, symbol)

            # OOF speichern
            save_path = os.path.join(DATA_DIR, f"{symbol}_alpha_oof.csv")
            result_df.to_csv(save_path, index=False)
            print(f"\n    OOF Alpha Predictions gespeichert: {save_path}")

            # Finale Modelle speichern
            df_full, _, _ = load_features_with_regime(symbol)
            df_full["target_24h_avg"] = _build_target_24h(df_full)
            train_and_save_final_models(
                symbol, df_full, feature_info["feature_cols_8h"]
            )

        except Exception as e:
            print(f"  FEHLER [{key}]: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("  ALPHA MODELS – SUMMARY")
    print(f"{'=' * 68}")
    print(f"\n  {'Asset':<8} {'Corr 8h':>9} {'Corr 24h':>10} "
          f"{'Corr Ens':>10} {'Sign Ens':>10}")
    print("  " + "─" * 52)
    for sym in SYMBOLS:
        if sym not in all_metrics:
            continue
        m   = all_metrics[sym]
        key = SYMBOL_SHORT[sym].upper()
        c8h  = m.get("alpha_8h",       {}).get("corr", 0)
        c24h = m.get("alpha_24h",      {}).get("corr", 0)
        cens = m.get("alpha_ensemble", {}).get("corr", 0)
        sign = m.get("alpha_ensemble", {}).get("sign_acc", 0)
        print(f"  {key:<8} {c8h:>9.4f} {c24h:>10.4f} "
              f"{cens:>10.4f} {sign*100:>9.1f}%")

    print(f"\n  Erfolgskriterium: Ensemble Korrelation > 0.60 auf OOF")
    ok_assets = [
        s for s in SYMBOLS
        if all_metrics.get(s, {}).get("alpha_ensemble", {}).get("corr", 0) > 0.60
    ]
    print(f"  Erfüllt: {len(ok_assets)}/{len(SYMBOLS)} Assets")

    print(f"\n  Nächster Schritt:")
    print(f"    python models/portfolio_constructor.py  (Layer 4)")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
