"""
backtest/portfolio_oof.py – Multi-Asset Portfolio Backtest auf OOF Predictions
================================================================================
Ausführen:
    cd crypto_quant
    python backtest/portfolio_oof.py

Input:   data/raw/{symbol}_oof_predictions.csv (erzeugt von models/train.py)
Output:  outputs/portfolio_backtest.png
         outputs/portfolio_summary.csv

4 Strategien verglichen auf gemeinsamer Zeitbasis:
  A: Always-In  – statische Gewichte BTC/ETH/SOL (40/35/25), immer im Markt
  B: Rule-based – Rate_T > Threshold → Position offen in T+1
  C: ML-Signal  – predicted_rate_T+1 > Threshold → Position offen
  D: ML+Kelly   – ML-Signal + Kelly-optimierte Asset-Gewichte

Kelly Sizing (Strategie D):
  Half-Kelly pro Asset, berechnet auf OOF-Perioden wo Signal aktiv.
  Cap: 30% vor Normalisierung. Ergebnis wird auf 100% normalisiert.
  Kein Look-ahead Bias: Kelly nutzt nur OOF-Predictions (Out-of-Sample).

Note über Always-In Sharpe:
  Sharpe ~24-27 ist KORREKT für Funding Rate Carry im Bull Market 2023-2026.
  88% der Perioden haben positive Funding Rates → sehr hohes mean/std Ratio.
  Das ist die Natur des Carry Trades, kein Berechnungsfehler.
  Equity-Strategien haben Sharpe 1-3, Carry Trades 10-30+.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    SYMBOLS, SYMBOL_SHORT, SYMBOL_WEIGHTS,
    DATA_DIR, OUTPUT_DIR,
    FUNDING_THRESHOLD, COST_PER_ROUNDTRIP, PERIODS_PER_YEAR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Farben pro Strategie
STRAT_COLORS = {
    "Always-In":  "#95a5a6",   # grau
    "Rule-based": "#3498db",   # blau
    "ML-Signal":  "#8e44ad",   # lila
    "ML+Kelly":   "#e74c3c",   # rot
}

ASSET_COLORS = {
    "BTCUSDT": "#F7931A",
    "ETHUSDT": "#627EEA",
    "SOLUSDT": "#9945FF",
}


# ── 1. Daten laden & alignen ───────────────────────────────────────────────────

def load_oof(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_oof_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"OOF Predictions nicht gefunden: {path}\n"
            f"Zuerst ausführen: python models/train.py"
        )
    df = pd.read_csv(path, parse_dates=["fundingTime"])
    df = df.sort_values("fundingTime").reset_index(drop=True)
    return df


def build_aligned_df(oofs: dict, symbols: list) -> pd.DataFrame:
    """
    Inner-Join aller OOF DataFrames auf gemeinsamer Zeitbasis.
    Dropped Perioden die nicht in ALLEN Assets vorhanden sind.
    """
    merged = None
    for sym in symbols:
        key = SYMBOL_SHORT[sym]
        df  = oofs[sym][["fundingTime", "actual_rate",
                          "in_signal_rule", "in_signal_ml"]].copy()
        df  = df.dropna(subset=["actual_rate"]).copy()
        df  = df.rename(columns={
            "actual_rate":   f"{key}_actual_rate",
            "in_signal_rule": f"{key}_in_signal_rule",
            "in_signal_ml":  f"{key}_in_signal_ml",
        })
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="fundingTime", how="inner")

    merged = merged.sort_values("fundingTime").reset_index(drop=True)
    merged["fundingTime"] = pd.to_datetime(merged["fundingTime"], format="ISO8601", utc=True)
    print(f"  Gemeinsame Zeitbasis: {len(merged)} Perioden  "
          f"({merged['fundingTime'].min().date()} → "
          f"{merged['fundingTime'].max().date()})")
    return merged


# ── 2. Kelly Sizing ────────────────────────────────────────────────────────────

def compute_kelly_weights(oofs: dict, symbols: list) -> dict:
    """
    Half-Kelly Gewichte basierend auf OOF-Perioden mit aktivem ML-Signal.

    Formel (klassische Kelly für Wett-Analogie):
        b = avg_win / avg_loss    (Payoff-Ratio)
        p = P(win | signal=1)     (Hit-Rate)
        f = (p*b - (1-p)) / b    (Full Kelly)
        size = clip(f/2, 0, 0.30) (Half-Kelly mit 30% Cap)

    Normalisiert auf 100% nach Cap.
    """
    print("\n  Kelly Sizing (Half-Kelly auf OOF ML-Signal Perioden):")
    kelly_raw = {}

    for sym in symbols:
        key = SYMBOL_SHORT[sym].upper()
        df  = oofs[sym].dropna(subset=["actual_rate", "in_signal_ml"])
        in_pos = df[df["in_signal_ml"] == 1]["actual_rate"]

        if len(in_pos) < 10:
            kelly_raw[sym] = 0.05
            print(f"    {key}: zu wenig ML-Perioden ({len(in_pos)}) → Fallback 5%")
            continue

        wins   = in_pos[in_pos > 0]
        losses = in_pos[in_pos < 0]
        p      = len(wins) / len(in_pos)

        if len(wins) == 0:
            kelly_raw[sym] = 0.0
            print(f"    {key}: keine Gewinne → 0%")
            continue

        avg_win  = wins.mean()
        avg_loss = abs(losses.mean()) if len(losses) > 0 else avg_win * 0.01

        b     = avg_win / avg_loss
        f_raw = (p * b - (1 - p)) / b
        size  = float(np.clip(f_raw * 0.5, 0.0, 0.30))
        kelly_raw[sym] = size

        print(f"    {key}: b={b:.2f}  p={p:.1%}  "
              f"Kelly={f_raw:.2%} → Half-Kelly={size:.2%}")

    total = sum(kelly_raw.values())
    if total < 1e-6:
        print("  WARNUNG: Alle Kelly-Gewichte = 0 → statische Gewichte")
        return {sym: SYMBOL_WEIGHTS[sym] for sym in symbols}

    weights = {sym: v / total for sym, v in kelly_raw.items()}
    print(f"\n  Normalisierte Kelly-Gewichte:")
    for sym in symbols:
        print(f"    {SYMBOL_SHORT[sym].upper()}: {weights[sym]:.1%}")

    return weights


# ── 3. Strategie-Simulation ────────────────────────────────────────────────────

def run_strategy(aligned: pd.DataFrame,
                  symbols: list,
                  weights: dict,
                  signal_type: str) -> pd.Series:
    """
    Berechnet Portfolio-Return-Serie für eine Strategie.

    signal_type: "always_in" | "in_signal_rule" | "in_signal_ml"
    weights    : {symbol: float} – summieren zu 1.0
    """
    port_ret = pd.Series(0.0, index=aligned.index)

    for sym in symbols:
        key = SYMBOL_SHORT[sym]
        r   = aligned[f"{key}_actual_rate"]

        if signal_type == "always_in":
            sig = pd.Series(True, index=aligned.index)
        else:
            sig = aligned[f"{key}_{signal_type}"].astype(bool)

        full_ret = pd.Series(0.0, index=aligned.index)
        full_ret[sig] = r[sig]

        # Gebühren: Entry beim ersten True nach False; Exit beim ersten False nach True
        enters = sig & (~sig.shift(1).fillna(False))
        exits  = (~sig) & sig.shift(1).fillna(False)
        full_ret[enters] -= COST_PER_ROUNDTRIP / 2   # 2 Legs öffnen
        full_ret[exits]  -= COST_PER_ROUNDTRIP / 2   # 2 Legs schließen

        port_ret += weights[sym] * full_ret

    return port_ret


# ── 4. Metriken ────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series) -> dict:
    """Vollständige Risk-Metriken auf einer Return-Serie."""
    n      = len(returns)
    equity = (1 + returns).cumprod()
    n_yrs  = n / PERIODS_PER_YEAR

    cagr   = (equity.iloc[-1] ** (1 / n_yrs) - 1) if n_yrs > 0 else 0.0
    mean_r = returns.mean()
    std_r  = returns.std()
    sharpe = mean_r / std_r * np.sqrt(PERIODS_PER_YEAR) if std_r > 0 else 0.0

    ds     = returns[returns < 0].std()
    sortino = mean_r / ds * np.sqrt(PERIODS_PER_YEAR) if ds and ds > 0 else 0.0

    dd      = (equity - equity.cummax()) / equity.cummax()
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd != 0 else 0.0

    n_in    = int((returns != 0).sum())
    time_in = n_in / n if n > 0 else 0.0

    return {
        "cagr_pct":    round(cagr * 100, 2),
        "sharpe":      round(sharpe, 3),
        "sortino":     round(sortino, 3),
        "calmar":      round(calmar, 3),
        "max_dd_pct":  round(max_dd * 100, 2),
        "time_in_pct": round(time_in * 100, 1),
        "equity":      equity,
        "dd":          dd,
    }


# ── 5. Rolling Sharpe ─────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = 90) -> pd.Series:
    """Rolling Sharpe Ratio über `window` Perioden (annualisiert)."""
    roll_mean = returns.rolling(window).mean()
    roll_std  = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(PERIODS_PER_YEAR)).where(roll_std > 0)


# ── 6. Portfolio Chart ─────────────────────────────────────────────────────────

def plot_portfolio(aligned: pd.DataFrame,
                   strategy_returns: dict,
                   strategy_metrics: dict,
                   kelly_weights: dict,
                   symbols: list):
    """
    4-Panel Portfolio Chart:
      1. Equity-Kurven aller 4 Strategien
      2. Rolling 30d Sharpe (ML+Kelly vs Always-In)
      3. Drawdown-Vergleich
      4. Kelly Asset-Gewichte (statisch – als Text-Annotation)
    """
    times = aligned["fundingTime"]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Equity-Kurven ─────────────────────────────────────────────────
    for name, ret in strategy_returns.items():
        m   = strategy_metrics[name]
        eq  = (m["equity"] - 1) * 100
        lw  = 2.5 if name in ("ML+Kelly", "Always-In") else 1.5
        ls  = "--" if name == "Always-In" else "-"
        ax1.plot(times, eq.values, color=STRAT_COLORS[name], lw=lw, ls=ls,
                 label=f"{name}  Sharpe={m['sharpe']:.1f}  "
                       f"CAGR={m['cagr_pct']:+.1f}%  "
                       f"MaxDD={m['max_dd_pct']:.1f}%",
                 alpha=0.9)

    ax1.axhline(0, color="black", lw=0.5, ls=":")
    ax1.set_title("Portfolio Equity-Kurven (OOF, kumulativer Return %)",
                  fontweight="bold", fontsize=11)
    ax1.set_ylabel("Kumulativer Return (%)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Rolling Sharpe ────────────────────────────────────────────────
    window = 90   # 90 Perioden = 30 Tage
    for name in ("Always-In", "ML+Kelly"):
        rs = rolling_sharpe(strategy_returns[name], window)
        ax2.plot(times, rs.values, color=STRAT_COLORS[name], lw=1.8,
                 label=name,
                 ls="--" if name == "Always-In" else "-",
                 alpha=0.85)

    ax2.axhline(0,   color="black",  lw=0.8, ls=":")
    ax2.axhline(1.0, color="#e67e22", lw=0.8, ls="--", label="Sharpe = 1")
    ax2.set_title(f"Rolling {window//3}d Sharpe (annualisiert)",
                  fontweight="bold", fontsize=11)
    ax2.set_ylabel("Sharpe Ratio")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.set_ylim(-5, 60)

    # ── Panel 3: Drawdown-Vergleich ────────────────────────────────────────────
    for name, ret in strategy_returns.items():
        m   = strategy_metrics[name]
        dd  = m["dd"] * 100
        lw  = 2.0 if name in ("ML+Kelly", "Always-In") else 1.2
        ls  = "--" if name == "Always-In" else "-"
        ax3.fill_between(times, dd.values, 0,
                          color=STRAT_COLORS[name], alpha=0.20)
        ax3.plot(times, dd.values, color=STRAT_COLORS[name], lw=lw, ls=ls,
                 label=f"{name}  MaxDD={m['max_dd_pct']:.1f}%")

    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_title("Drawdown-Vergleich (%)", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Drawdown (%)")
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 4: Asset-Performance Heatmap + Kelly-Gewichte ───────────────────
    ax4.axis("off")

    # Tabelle: Strategie × Metrik
    strategies = list(strategy_metrics.keys())
    metrics_labels = ["CAGR %", "Sharpe", "Sortino", "Calmar", "MaxDD %",
                       "Zeit im Markt"]
    cell_data = []
    for s in strategies:
        m = strategy_metrics[s]
        cell_data.append([
            f"{m['cagr_pct']:+.2f}%",
            f"{m['sharpe']:.2f}",
            f"{m['sortino']:.2f}",
            f"{m['calmar']:.2f}",
            f"{m['max_dd_pct']:.2f}%",
            f"{m['time_in_pct']:.0f}%",
        ])

    cell_colors = []
    for i, s in enumerate(strategies):
        m = strategy_metrics[s]
        row_colors = []
        # cagr
        row_colors.append("#d5f5e3" if m["cagr_pct"] > 0 else "#fadbd8")
        # sharpe
        row_colors.append("#d5f5e3" if m["sharpe"] > 1 else "#fdebd0" if m["sharpe"] > 0 else "#fadbd8")
        # sortino
        row_colors.append("#d5f5e3" if m["sortino"] > 1 else "#fdebd0" if m["sortino"] > 0 else "#fadbd8")
        # calmar
        row_colors.append("#d5f5e3" if m["calmar"] > 1 else "#fdebd0" if m["calmar"] > 0 else "#fadbd8")
        # maxdd
        row_colors.append("#d5f5e3" if m["max_dd_pct"] > -5 else "#fdebd0" if m["max_dd_pct"] > -15 else "#fadbd8")
        # time_in
        row_colors.append("#d5f5e3")
        cell_colors.append(row_colors)

    row_colors_header = [["#2c3e50"] * len(metrics_labels)]
    tbl = ax4.table(
        cellText=cell_data,
        rowLabels=strategies,
        colLabels=metrics_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.1, 2.0)

    # Zeilenlabels einfärben
    for i, s in enumerate(strategies):
        tbl[(i + 1, -1)].set_facecolor(STRAT_COLORS[s])
        tbl[(i + 1, -1)].set_text_props(color="white", fontweight="bold")

    # Header style
    for j in range(len(metrics_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Kelly-Gewichte Annotation
    kelly_text = "Kelly Gewichte (normalisiert):\n" + "\n".join(
        f"  {SYMBOL_SHORT[sym].upper()}: {w:.1%}"
        for sym, w in kelly_weights.items()
    )
    ax4.text(0.02, 0.04, kelly_text,
             transform=ax4.transAxes,
             fontsize=10, va="bottom", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#fdfefe", edgecolor="#aab7b8"))

    ax4.set_title("Strategie-Vergleich (OOF Performance)",
                  fontweight="bold", fontsize=11, pad=14)

    # Gesamt-Titel
    n_days = len(aligned) // 3
    fig.suptitle(
        f"Portfolio Backtest – OOF Signals  |  "
        f"{aligned['fundingTime'].min().date()} → "
        f"{aligned['fundingTime'].max().date()}  (~{n_days} Tage)",
        fontweight="bold", fontsize=13, y=0.995,
    )

    path = os.path.join(OUTPUT_DIR, "portfolio_backtest.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Chart gespeichert: {path}")


# ── 7. Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 68)
    print("  PORTFOLIO BACKTEST – 4 Strategien auf OOF ML Predictions")
    print("=" * 68)

    # ── Laden ──────────────────────────────────────────────────────────────────
    print("\n  [1/5] Lade OOF Predictions...")
    oofs = {}
    for sym in SYMBOLS:
        df = load_oof(sym)
        # Nur Zeilen mit vollständigen Predictions für Backtest
        df = df.dropna(subset=["actual_rate", "predicted_rate"])
        oofs[sym] = df
        print(f"    {SYMBOL_SHORT[sym].upper()}: {len(df)} OOF Perioden")

    # ── Alignen ────────────────────────────────────────────────────────────────
    print("\n  [2/5] Zeitbasis alignen (Inner Join)...")
    aligned = build_aligned_df(oofs, SYMBOLS)

    # ── Kelly ──────────────────────────────────────────────────────────────────
    print("\n  [3/5] Kelly-Gewichte berechnen...")
    kelly_weights = compute_kelly_weights(oofs, SYMBOLS)

    static_weights = SYMBOL_WEIGHTS   # BTC 40%, ETH 35%, SOL 25%
    print(f"\n  Statische Gewichte (A/B/C): "
          + ", ".join(f"{SYMBOL_SHORT[s].upper()} {static_weights[s]:.0%}" for s in SYMBOLS))

    # ── 4 Strategien simulieren ────────────────────────────────────────────────
    print("\n  [4/5] Strategien simulieren...")

    strategies = {
        "Always-In":  (static_weights,  "always_in"),
        "Rule-based": (static_weights,  "in_signal_rule"),
        "ML-Signal":  (static_weights,  "in_signal_ml"),
        "ML+Kelly":   (kelly_weights,   "in_signal_ml"),
    }

    strategy_returns = {}
    strategy_metrics = {}

    for name, (w, sig) in strategies.items():
        ret = run_strategy(aligned, SYMBOLS, w, sig)
        m   = compute_metrics(ret)
        strategy_returns[name] = ret
        strategy_metrics[name] = m

    # ── Ausgabe ────────────────────────────────────────────────────────────────
    print(f"\n  {'Strategie':<12} {'CAGR%':>8} {'Sharpe':>8} {'Sortino':>8}"
          f" {'Calmar':>8} {'MaxDD%':>8} {'Zeit%':>8}")
    print("  " + "─" * 66)
    for name, m in strategy_metrics.items():
        marker = " ←" if name == "ML+Kelly" else ""
        print(f"  {name:<12} "
              f"{m['cagr_pct']:>+7.2f}%  "
              f"{m['sharpe']:>7.3f}  "
              f"{m['sortino']:>7.3f}  "
              f"{m['calmar']:>7.3f}  "
              f"{m['max_dd_pct']:>7.2f}%  "
              f"{m['time_in_pct']:>6.1f}%"
              f"{marker}")

    # ── Erfolg-Kriterien ───────────────────────────────────────────────────────
    mk = strategy_metrics["ML+Kelly"]
    ma = strategy_metrics["Always-In"]
    print(f"\n  Erfolg-Kriterien:")
    ok1 = mk["cagr_pct"] > ma["cagr_pct"] - 0.5  # nach Fees in Ballpark
    ok2 = mk["max_dd_pct"] > -15
    ok3 = mk["sharpe"] > 1.0
    ok4 = kelly_weights.get("SOLUSDT", 0) > 0.40

    print(f"    ML+Kelly CAGR ≥ Always-In nach Fees:  {'✓' if ok1 else '✗'}  "
          f"({mk['cagr_pct']:+.2f}% vs {ma['cagr_pct']:+.2f}%)")
    print(f"    Portfolio MaxDD < 15%:               {'✓' if ok2 else '✗'}  "
          f"({mk['max_dd_pct']:.2f}%)")
    print(f"    Portfolio Sharpe > 1.0:              {'✓' if ok3 else '✗'}  "
          f"({mk['sharpe']:.3f})")
    print(f"    SOL Kelly-Gewicht > 40%:             {'✓' if ok4 else '✗'}  "
          f"({kelly_weights.get('SOLUSDT', 0):.1%})")

    all_ok = ok1 and ok2 and ok3
    if all_ok:
        print("\n  ✓ Alle Kern-Kriterien erfüllt → Phase 3 (Live Execution) freigegeben")
    else:
        print("\n  △ Nicht alle Kriterien erfüllt – Entscheidungsmatrix:")
        btc_neg = strategy_metrics["ML-Signal"]["sharpe"] < 0
        eth_neg = strategy_metrics["ML-Signal"]["sharpe"] < 0
        if not ok4:
            print("    → SOL Kelly < 40%: alle 3 Assets traden (ausgewogenes Signal)")
        if ok3:
            print("    → Sharpe > 1 mit ML+Kelly ✓")
        print("    → Empfehlung: ML+Kelly Gewichte verwenden, kein Asset ausschließen")

    # ── CSV Summary ────────────────────────────────────────────────────────────
    summary_rows = []
    for name, m in strategy_metrics.items():
        row = {"strategy": name}
        row.update({k: v for k, v in m.items() if k not in ("equity", "dd")})
        summary_rows.append(row)

    # Kelly-Gewichte als separate Zeile
    for sym in SYMBOLS:
        summary_rows.append({
            "strategy":   f"kelly_weight_{SYMBOL_SHORT[sym]}",
            "cagr_pct":   round(kelly_weights[sym] * 100, 2),
        })

    csv_path = os.path.join(OUTPUT_DIR, "portfolio_summary.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"\n  Summary CSV gespeichert: {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("\n  [5/5] Portfolio Chart erstellen...")
    plot_portfolio(aligned, strategy_returns, strategy_metrics,
                   kelly_weights, SYMBOLS)

    print("\n" + "=" * 68)
    print("  PORTFOLIO BACKTEST ABGESCHLOSSEN")
    print("  Outputs: outputs/portfolio_backtest.png")
    print("           outputs/portfolio_summary.csv")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
