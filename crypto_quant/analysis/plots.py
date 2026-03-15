"""
analysis/plots.py – Alle Visualisierungen
==========================================

Single-Asset Plot:
  plot_single_asset()  – 8 Subplots für ein Asset (Rate, Annualisiert,
                          Verteilung, Z-Score, Divergenz, Basis, Stablecoins,
                          Korrelations-Heatmap)

Multi-Asset Plot:
  plot_portfolio()     – Portfolio-Vergleich über alle Assets
                          (Equity-Kurven, Korrelationen, Spreads, Sync-Score)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, Optional

from config import OUTPUT_DIR, SYMBOL_SHORT


# ── Style ──────────────────────────────────────────────────────────────────────

COLORS = {
    "btc": "#F7931A",
    "eth": "#627EEA",
    "sol": "#9945FF",
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "steelblue",
}


def _fmt_axis(ax, df=None, col="fundingTime"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def _label(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontweight="bold", fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


# ── Single-Asset Plots ─────────────────────────────────────────────────────────

def plot_single_asset(df: pd.DataFrame,
                       symbol: str = "BTCUSDT",
                       save: bool = True):
    """Erstellt 8-Panel Analyse-Plot für ein Asset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    key        = SYMBOL_SHORT.get(symbol, symbol[:3].lower())
    has_cross  = "cross_divergence"   in df.columns
    has_basis  = "basis_pct"          in df.columns
    has_stable = "usdt_inflow_7d_pct" in df.columns

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f"Funding Rate Analysis – {symbol}",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.subplots_adjust(hspace=0.45, wspace=0.3)

    # ── 1. Funding Rate History ────────────────────────────────────────────────
    ax = axes[0, 0]
    colors = [COLORS["positive"] if x > 0 else COLORS["negative"]
              for x in df["fundingRate"]]
    ax.bar(df["fundingTime"], df["fundingRate"] * 100,
           color=colors, alpha=0.7, width=0.25)
    ax.axhline(0.01,  color="orange", linestyle="--", lw=1, label="0.01% Threshold")
    ax.axhline(0,     color="black",  lw=0.5)
    _label(ax, "Funding Rate History (%)", ylabel="Rate (%)")
    ax.legend(fontsize=8)
    _fmt_axis(ax)

    # ── 2. Annualisierte Rate (30d Rolling Mean) ───────────────────────────────
    ax = axes[0, 1]
    ax.plot(df["fundingTime"], df["rate_annualized_pct"],
            alpha=0.25, color=COLORS["neutral"], lw=0.8)
    ax.plot(df["fundingTime"], df["rate_annualized_pct"].rolling(90).mean(),
            color=COLORS["neutral"], lw=2, label="30d Mean")
    ax.axhline(0,  color="black",  lw=0.5)
    ax.axhline(10, color="orange", linestyle="--", lw=1, label="10% p.a.")
    _label(ax, "Annualisierte Rate (% p.a.)", ylabel="% p.a.")
    ax.legend(fontsize=8)
    _fmt_axis(ax)

    # ── 3. Verteilung ──────────────────────────────────────────────────────────
    ax = axes[1, 0]
    pos = df[df["fundingRate"] > 0]["fundingRate"] * 100
    neg = df[df["fundingRate"] <= 0]["fundingRate"] * 100
    ax.hist(pos, bins=60, color=COLORS["positive"], alpha=0.7,
            label=f"Positiv ({len(pos)})")
    ax.hist(neg, bins=30, color=COLORS["negative"], alpha=0.7,
            label=f"Negativ ({len(neg)})")
    ax.axvline(0.01, color="orange", linestyle="--", label="0.01%")
    _label(ax, "Verteilung der Funding Rates",
           xlabel="Rate (%)", ylabel="Häufigkeit")
    ax.legend(fontsize=8)

    # ── 4. Z-Score History ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    zs = df["rate_zscore"].dropna()
    zt = df["fundingTime"][zs.index]
    ax.plot(zt, zs, lw=0.8, color="purple", alpha=0.7)
    ax.axhline( 2, color="red",  linestyle="--", lw=1, label="+2σ")
    ax.axhline(-2, color="blue", linestyle="--", lw=1, label="-2σ")
    ax.axhline( 0, color="black", lw=0.5)
    ax.fill_between(zt, zs, 0, where=(zs > 2),  alpha=0.2, color="red")
    ax.fill_between(zt, zs, 0, where=(zs < -2), alpha=0.2, color="blue")
    _label(ax, "Z-Score (30d konsistentes Fenster)", ylabel="Z-Score")
    ax.legend(fontsize=8)
    _fmt_axis(ax)

    # ── 5. Cross-Exchange Divergenz ────────────────────────────────────────────
    ax = axes[2, 0]
    if has_cross:
        div = df["cross_divergence"] * 100
        ax.plot(df["fundingTime"], div, lw=0.8, color="darkorange", alpha=0.8)
        ax.axhline(0,     color="black", lw=0.5)
        ax.axhline( 0.01, color="red",  linestyle="--", lw=1, label="+0.01%")
        ax.axhline(-0.01, color="blue", linestyle="--", lw=1, label="-0.01%")
        ax.fill_between(df["fundingTime"], div, 0,
                        where=(div > 0.01),  alpha=0.2, color="red")
        ax.fill_between(df["fundingTime"], div, 0,
                        where=(div < -0.01), alpha=0.2, color="blue")
        _label(ax, "Cross-Exchange Divergenz (Binance − Bybit)",
               ylabel="Differenz (%)")
        ax.legend(fontsize=7)
        _fmt_axis(ax)
    else:
        ax.text(0.5, 0.5, "Nicht geladen\n(Bybit-Daten fehlen)",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=12)
        _label(ax, "Cross-Exchange Divergenz")

    # ── 6. Basis ───────────────────────────────────────────────────────────────
    ax = axes[2, 1]
    if has_basis:
        ax.plot(df["fundingTime"], df["basis_pct"],
                lw=0.8, color="teal", alpha=0.5, label="Basis %")
        ax.plot(df["fundingTime"], df["basis_7d_mean"],
                lw=2,   color="teal", label="7d Mean")
        ax.axhline(0, color="black", lw=0.5)
        _label(ax, "Basis: Future − Spot (%)\n← Leading Indicator",
               ylabel="Basis (%)")
        ax.legend(fontsize=8)
        _fmt_axis(ax)
    else:
        ax.text(0.5, 0.5, "Nicht geladen\n(Mark/Index Price fehlen)",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=12)
        _label(ax, "Basis: Future − Spot")

    # ── 7. Stablecoin Inflows ──────────────────────────────────────────────────
    ax = axes[3, 0]
    if has_stable:
        sc = df[["fundingTime", "usdt_inflow_7d_pct"]].dropna()
        c  = [COLORS["positive"] if x > 0 else COLORS["negative"]
              for x in sc["usdt_inflow_7d_pct"]]
        ax.bar(sc["fundingTime"], sc["usdt_inflow_7d_pct"],
               color=c, alpha=0.6, width=0.3)
        ax.axhline(0, color="black", lw=0.5)
        _label(ax, "USDT Supply Wachstum (7d %)\n← Leading Indicator (3-14 Tage)",
               ylabel="Wachstum (%)")
        _fmt_axis(ax)
    else:
        ax.text(0.5, 0.5, "Nicht geladen\n(DeFiLlama-Daten fehlen)",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=12)
        _label(ax, "USDT Stablecoin Inflows")

    # ── 8. Feature Korrelations-Heatmap ───────────────────────────────────────
    ax = axes[3, 1]
    feature_cols = ["fundingRate", "rate_zscore", "rate_momentum_1d",
                    "rate_acceleration"]
    if has_cross:   feature_cols.append("cross_divergence")
    if has_basis:   feature_cols.append("basis_pct")
    if has_stable:  feature_cols.append("usdt_inflow_7d_pct")
    if "target_next_positive" in df.columns:
        feature_cols.append("target_next_positive")

    corr = df[feature_cols].dropna().corr()
    sns.heatmap(corr, ax=ax, cmap="RdYlGn", center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    _label(ax, f"Feature Korrelationen – {key.upper()}\n(target = nächste Rate > 0.01%)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=7)

    if save:
        path = os.path.join(OUTPUT_DIR, f"analysis_{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Plot gespeichert: {path}")
    else:
        plt.show()
    plt.close()


def plot_portfolio(dfs: Dict[str, pd.DataFrame],
                    portfolio_result: dict,
                    symbols: list,
                    save: bool = True):
    """
    Multi-Asset Portfolio Visualisierung: 4 Panels.

      1. Equity-Kurven aller Assets + gewichtetes Portfolio
      2. Rolling 30d Annualisierte Rate im Vergleich
      3. Cross-Asset Spreads + Sync-Score
      4. Return-Korrelationsmatrix
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Portfolio – Multi-Asset Funding Rate Arbitrage",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # ── 1. Equity-Kurven ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    for sym in symbols:
        key = SYMBOL_SHORT[sym]
        eq  = portfolio_result["per_asset"][sym]["_equity"]
        df_sym = dfs[sym].dropna(subset=["fundingRate"]).reset_index(drop=True)
        n   = min(len(eq), len(df_sym))
        ann = portfolio_result["per_asset"][sym]["net_ann_pct"]
        ax.plot(df_sym["fundingTime"].iloc[:n],
                eq.values[:n] * 100 - 100,
                color=COLORS[key], lw=1.8,
                label=f"{key.upper()} ({ann:+.1f}% p.a.)")

    port_eq = portfolio_result["static"]["_equity"]
    df_ref  = dfs[symbols[0]].dropna(subset=["fundingRate"]).reset_index(drop=True)
    n       = min(len(port_eq), len(df_ref))
    port_ann = portfolio_result["static"]["net_ann_pct"]
    ax.plot(df_ref["fundingTime"].iloc[:n],
            port_eq.values[:n] * 100 - 100,
            color="black", lw=2.5, linestyle="--",
            label=f"Portfolio ({port_ann:+.1f}% p.a.)")

    ax.axhline(0, color="gray", lw=0.5)
    _label(ax, "Equity-Kurven: Rule-based Strategy\n(Kumulativer Return %)",
           ylabel="Kumulativer Return (%)")
    ax.legend(fontsize=9)
    _fmt_axis(ax)

    # ── 2. Rolling 30d Rate Vergleich ─────────────────────────────────────────
    ax = axes[0, 1]
    for sym in symbols:
        key    = SYMBOL_SHORT[sym]
        df_sym = dfs[sym]
        roll_ann = df_sym["fundingRate"].rolling(90).mean() * 3 * 365 * 100
        ax.plot(df_sym["fundingTime"], roll_ann,
                color=COLORS[key], lw=1.5, alpha=0.9, label=key.upper())

    ax.axhline(0,  color="black", lw=0.5)
    ax.axhline(10, color="orange", linestyle="--", lw=1, label="10% p.a.")
    _label(ax, "Rolling 30d Annualisierte Rate (% p.a.)",
           ylabel="% p.a.")
    ax.legend(fontsize=9)
    _fmt_axis(ax)

    # ── 3. Cross-Asset Spreads + Sync-Score ───────────────────────────────────
    ax   = axes[1, 0]
    df_ref = dfs[symbols[0]]

    if "btc_eth_spread" in df_ref.columns:
        ax.plot(df_ref["fundingTime"], df_ref["btc_eth_spread"] * 100,
                lw=1, color=COLORS["eth"], alpha=0.85, label="BTC−ETH Spread")
        ax.plot(df_ref["fundingTime"], df_ref["btc_sol_spread"] * 100,
                lw=1, color=COLORS["sol"], alpha=0.85, label="BTC−SOL Spread")
        ax.axhline(0, color="black", lw=0.5)
        _label(ax, "Cross-Asset Rate Spreads (%)\n(negativ = Alt zahlt mehr als BTC)",
               ylabel="Spread (%)")
        ax.legend(fontsize=9, loc="upper left")
        _fmt_axis(ax)

        if "sync_score" in df_ref.columns:
            ax2 = ax.twinx()
            ax2.fill_between(df_ref["fundingTime"], df_ref["sync_score"],
                             alpha=0.15, color="gold")
            ax2.set_ylim(0, 4)
            ax2.set_ylabel("Sync Score (0–3)", fontsize=8, color="goldenrod")
            ax2.tick_params(axis="y", labelcolor="goldenrod")
    else:
        ax.text(0.5, 0.5,
                "Cross-Asset Features nicht berechnet\n"
                "(build_cross_asset_features() aufrufen)",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=11)
        _label(ax, "Cross-Asset Spreads")

    # ── 4. Return-Korrelationsmatrix ───────────────────────────────────────────
    ax   = axes[1, 1]
    corr = portfolio_result["correlation"].copy()
    labels = [SYMBOL_SHORT[s].upper() for s in symbols]
    corr.index   = labels
    corr.columns = labels

    sns.heatmap(corr, ax=ax, cmap="RdYlGn", center=0,
                annot=True, fmt=".3f", linewidths=1.0,
                annot_kws={"size": 12, "weight": "bold"},
                vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    _label(ax, "Return-Korrelationen\n(Period-by-Period, In-Market Returns)")
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=10)

    if save:
        path = os.path.join(OUTPUT_DIR, "portfolio_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Plot gespeichert: {path}")
    else:
        plt.show()
    plt.close()


def plot_all(df: pd.DataFrame,
             symbol: str = "BTCUSDT",
             save: bool = True):
    """Backwards-kompatibler Wrapper für Single-Asset Plot."""
    plot_single_asset(df, symbol=symbol, save=save)


def plot_ic_heatmap(symbol: str,
                    ic_df: Optional[pd.DataFrame] = None,
                    top_n: int = 20,
                    save: bool = True):
    """
    IC-Heatmap: Features als Zeilen, Zeit als Spalten, IC-Wert als Farbe.

    Grün = positiver IC (Feature predicts returns korrekt)
    Rot  = negativer IC (inverse Beziehung)
    Weiß = kein Signal

    ic_df: DataFrame aus compute_ic_series() – wenn None, wird aus CSV geladen.
    top_n: Anzahl Features nach |mean IC| (zeigt die informativsten).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    key = SYMBOL_SHORT.get(symbol, symbol[:3].lower())

    # IC-Daten laden
    if ic_df is None:
        ic_path = os.path.join(OUTPUT_DIR, f"ic_report_{symbol}.csv")
        if not os.path.exists(ic_path):
            print(f"  IC-Report nicht gefunden: {ic_path}")
            print("  Zuerst ausführen: python analysis/ic_analysis.py")
            return
        ic_df = pd.read_csv(ic_path, parse_dates=["fundingTime"])

    # Zeit-Spalte separieren
    time_col = "fundingTime"
    feat_cols = [c for c in ic_df.columns if c != time_col]

    if len(feat_cols) == 0:
        print("  Keine Feature-Spalten in IC-DataFrame gefunden.")
        return

    # Top-N Features nach |mean IC| auswählen
    ic_vals   = ic_df[feat_cols]
    mean_abs  = ic_vals.abs().mean(skipna=True).sort_values(ascending=False)
    top_feats = mean_abs.head(top_n).index.tolist()

    if not top_feats:
        print("  Keine Features mit IC-Werten verfügbar.")
        return

    # Downsampling: monatliche Stichproben für übersichtliche x-Achse
    if time_col in ic_df.columns:
        ic_df = ic_df.set_index(time_col)
    else:
        ic_df.index = pd.RangeIndex(len(ic_df))

    # Resample auf wöchentliche Perioden (Mittelwert)
    if hasattr(ic_df.index, "to_pydatetime"):
        sampled = ic_df[top_feats].resample("2W").mean()
    else:
        # Wenn kein Datetime-Index: gleichmäßige Stichproben
        step = max(1, len(ic_df) // 60)
        sampled = ic_df[top_feats].iloc[::step]

    # Transponieren: Features als Zeilen, Zeit als Spalten
    heatmap_data = sampled.T   # shape: (n_features, n_timepoints)

    # Kurze Feature-Namen für bessere Lesbarkeit
    short_names = {f: f[:28] + "…" if len(f) > 30 else f for f in heatmap_data.index}
    heatmap_data.index = [short_names[f] for f in heatmap_data.index]

    # X-Achsenbeschriftungen
    if hasattr(sampled.index, "strftime"):
        x_labels = [t.strftime("%b '%y") for t in sampled.index]
    else:
        x_labels = [str(i) for i in sampled.index]

    # Heatmap zeichnen
    fig_height = max(8, len(top_feats) * 0.42)
    fig, ax    = plt.subplots(figsize=(18, fig_height))
    fig.patch.set_facecolor("white")

    im = ax.imshow(
        heatmap_data.values,
        cmap="RdYlGn",
        vmin=-0.25, vmax=0.25,
        aspect="auto",
        interpolation="nearest",
    )

    # Achsenbeschriftungen
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8)

    n_ticks = min(20, len(x_labels))
    tick_idx = np.linspace(0, len(x_labels) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x_labels[i] for i in tick_idx],
                       rotation=35, ha="right", fontsize=8)

    # Farbskala
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("IC (Spearman)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Nulllinie-Markierung (weiß → kein Signal)
    ax.axvline(-0.5, color="white", lw=0.3)   # cosmetic

    # ICIR-Werte als Annotation (rechts)
    mean_ic = ic_vals[top_feats].mean(skipna=True)
    for i, feat in enumerate(mean_abs.head(top_n).index):
        ic_val = mean_ic.get(feat, np.nan)
        if pd.notna(ic_val):
            color = "#1a7a1a" if ic_val > 0 else "#a01a1a"
            ax.text(len(x_labels) + 0.5, i,
                    f"μIC={ic_val:+.3f}",
                    va="center", ha="left", fontsize=7, color=color)

    ax.set_xlabel("Zeit (Monatlich)", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.set_title(
        f"IC-Heatmap – {symbol}  |  Top-{top_n} Features nach |mean IC|\n"
        f"Grün = positives Signal, Rot = inverses Signal, Weiß = kein Signal",
        fontweight="bold", fontsize=11, pad=10,
    )

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, f"ic_heatmap_{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  IC-Heatmap gespeichert: {path}")
    else:
        plt.show()
    plt.close()
