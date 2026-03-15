"""
Funding Rate Explorer – Binance
================================
Phase 1: Daten laden & erste Exploration

Was dieses Script macht:
1. Historische Funding Rates von Binance laden (kein API-Key nötig)
2. Open Interest Daten laden
3. Erste statistische Analyse
4. Visualisierungen speichern

Setup:
    pip install requests pandas matplotlib seaborn numpy

Ausführen:
    python funding_rate_explorer.py
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timezone
import time
import os

# ── Konfiguration ──────────────────────────────────────────────────────────────

SYMBOL = "BTCUSDT"          # Welches Pair analysieren
LIMIT  = 1000               # Max Einträge pro Request (Binance-Limit)
OUTPUT_DIR = "."            # Wo die Plots gespeichert werden

# ── Binance API Funktionen ─────────────────────────────────────────────────────

BASE_URL = "https://fapi.binance.com"   # Futures API


def get_funding_rates(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Lädt historische Funding Rates von Binance Futures.
    Kein API-Key nötig – öffentliche Endpoint.
    Gibt bis zu 1000 Einträge zurück (~333 Tage bei 8h Intervall).
    """
    print(f"📥 Lade Funding Rates für {symbol}...")
    url = f"{BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.sort_values("fundingTime").reset_index(drop=True)

    print(f"   ✓ {len(df)} Einträge geladen ({df['fundingTime'].min().date()} → {df['fundingTime'].max().date()})")
    return df[["fundingTime", "fundingRate"]]


def get_open_interest_history(symbol: str, period: str = "8h", limit: int = 500) -> pd.DataFrame:
    """
    Lädt historisches Open Interest (OI).
    Period: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
    """
    print(f"📥 Lade Open Interest für {symbol}...")
    url = f"{BASE_URL}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"   ✓ {len(df)} Einträge geladen")
    return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]]


def get_long_short_ratio(symbol: str, period: str = "8h", limit: int = 500) -> pd.DataFrame:
    """Lädt Long/Short Ratio der Top Trader."""
    print(f"📥 Lade Long/Short Ratio für {symbol}...")
    url = f"{BASE_URL}/futures/data/topLongShortPositionRatio"
    params = {"symbol": symbol, "period": period, "limit": limit}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["longShortRatio"] = df["longShortRatio"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"   ✓ {len(df)} Einträge geladen")
    return df[["timestamp", "longShortRatio"]]


# ── Feature Engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baut Features die wir später fürs ML-Modell brauchen.
    """
    df = df.copy()

    # Annualisierte Rate (3x täglich = 3*365)
    df["rate_annualized_pct"] = df["fundingRate"] * 3 * 365 * 100

    # Rolling Stats (Fenster in Perioden, 1 Periode = 8h)
    df["rate_7d_mean"]   = df["fundingRate"].rolling(21).mean()   # 7 Tage
    df["rate_7d_std"]    = df["fundingRate"].rolling(21).std()
    df["rate_30d_mean"]  = df["fundingRate"].rolling(90).mean()   # 30 Tage

    # Z-Score: Wie ungewöhnlich hoch ist die aktuelle Rate?
    df["rate_zscore"]    = (df["fundingRate"] - df["rate_30d_mean"]) / df["rate_7d_std"]

    # Momentum: Steigt oder fällt die Rate?
    df["rate_momentum_1d"]  = df["fundingRate"].diff(3)    # vs. vor 1 Tag
    df["rate_momentum_3d"]  = df["fundingRate"].diff(9)    # vs. vor 3 Tagen

    # Tageszeit & Wochentag (zyklische Muster)
    df["hour"]       = df["fundingTime"].dt.hour
    df["weekday"]    = df["fundingTime"].dt.weekday   # 0=Montag, 6=Sonntag

    # Label für ML: Ist die Rate in 1 Periode (8h) noch positiv & > Threshold?
    threshold = 0.0001   # 0.01% – mindestens das wollen wir verdienen
    df["target_next_positive"] = (df["fundingRate"].shift(-1) > threshold).astype(int)

    return df


# ── Analyse & Plots ────────────────────────────────────────────────────────────

def print_stats(df: pd.DataFrame):
    """Gibt wichtige Kennzahlen aus."""
    r = df["fundingRate"]
    ann = df["rate_annualized_pct"]

    print("\n" + "="*55)
    print(f"  FUNDING RATE STATISTIKEN – {SYMBOL}")
    print("="*55)
    print(f"  Zeitraum:           {df['fundingTime'].min().date()} → {df['fundingTime'].max().date()}")
    print(f"  Anzahl Perioden:    {len(df)}")
    print()
    print(f"  Ø Rate pro Periode: {r.mean()*100:.4f}%")
    print(f"  Ø Rate annualisiert:{ann.mean():.1f}%")
    print(f"  Max Rate:           {r.max()*100:.4f}%  ({ann.max():.0f}% p.a.)")
    print(f"  Min Rate:           {r.min()*100:.4f}%  ({ann.min():.0f}% p.a.)")
    print()
    print(f"  % Perioden positiv: {(r > 0).mean()*100:.1f}%")
    print(f"  % Perioden > 0.01%: {(r > 0.0001).mean()*100:.1f}%  ← lohnend")
    print(f"  % Perioden negativ: {(r < 0).mean()*100:.1f}%")
    print()

    # Kumulativer Return wenn immer drin
    cumulative = (1 + r).cumprod().iloc[-1] - 1
    print(f"  Kum. Return (immer drin): {cumulative*100:.1f}%")

    # Nur positive Perioden
    only_positive = r[r > 0].sum()
    print(f"  Return (nur pos. Perioden): {only_positive*100:.1f}%")
    print("="*55)


def plot_all(df: pd.DataFrame):
    """Erstellt alle wichtigen Visualisierungen."""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"Funding Rate Explorer – {SYMBOL}", fontsize=16, fontweight="bold", y=0.98)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # ── Plot 1: Funding Rate History ──────────────────────────
    ax1 = axes[0, 0]
    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in df["fundingRate"]]
    ax1.bar(df["fundingTime"], df["fundingRate"] * 100, color=colors, alpha=0.7, width=0.25)
    ax1.axhline(0.01, color="orange", linestyle="--", linewidth=1, label="0.01% Threshold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Funding Rate History (%)", fontweight="bold")
    ax1.set_ylabel("Rate (%)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

    # ── Plot 2: Annualisierte Rate (30d Rolling Mean) ──────────
    ax2 = axes[0, 1]
    ax2.plot(df["fundingTime"], df["rate_annualized_pct"], alpha=0.3, color="steelblue", linewidth=0.8)
    ax2.plot(df["fundingTime"], df["rate_annualized_pct"].rolling(90).mean(),
             color="steelblue", linewidth=2, label="30d Rolling Mean")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(10, color="orange", linestyle="--", linewidth=1, label="10% p.a.")
    ax2.set_title("Annualisierte Rate (% p.a.)", fontweight="bold")
    ax2.set_ylabel("% p.a.")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    # ── Plot 3: Verteilung der Rates ───────────────────────────
    ax3 = axes[1, 0]
    positive = df[df["fundingRate"] > 0]["fundingRate"] * 100
    negative = df[df["fundingRate"] <= 0]["fundingRate"] * 100
    ax3.hist(positive, bins=60, color="#2ecc71", alpha=0.7, label=f"Positiv ({len(positive)})")
    ax3.hist(negative, bins=30, color="#e74c3c", alpha=0.7, label=f"Negativ ({len(negative)})")
    ax3.axvline(0.01, color="orange", linestyle="--", label="0.01% Threshold")
    ax3.set_title("Verteilung der Funding Rates", fontweight="bold")
    ax3.set_xlabel("Rate (%)")
    ax3.set_ylabel("Häufigkeit")
    ax3.legend(fontsize=8)

    # ── Plot 4: Muster nach Tageszeit ─────────────────────────
    ax4 = axes[1, 1]
    hourly = df.groupby("hour")["fundingRate"].agg(["mean", "std"]) * 100
    ax4.bar(hourly.index, hourly["mean"], yerr=hourly["std"],
            color="steelblue", alpha=0.7, capsize=3)
    ax4.axhline(0, color="black", linewidth=0.5)
    ax4.set_title("Ø Rate nach Tageszeit (UTC)", fontweight="bold")
    ax4.set_xlabel("Stunde (UTC)")
    ax4.set_ylabel("Ø Rate (%)")
    ax4.set_xticks([0, 8, 16])
    ax4.set_xticklabels(["00:00\n(Zahlung)", "08:00\n(Zahlung)", "16:00\n(Zahlung)"])

    # ── Plot 5: Muster nach Wochentag ─────────────────────────
    ax5 = axes[2, 0]
    days = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    weekly = df.groupby("weekday")["fundingRate"].mean() * 100
    bars = ax5.bar(range(7), weekly.values, color="steelblue", alpha=0.7)
    for i, v in enumerate(weekly.values):
        if v > 0:
            bars[i].set_color("#2ecc71")
        else:
            bars[i].set_color("#e74c3c")
    ax5.axhline(0, color="black", linewidth=0.5)
    ax5.set_title("Ø Rate nach Wochentag", fontweight="bold")
    ax5.set_xticks(range(7))
    ax5.set_xticklabels(days)
    ax5.set_ylabel("Ø Rate (%)")

    # ── Plot 6: Z-Score History ────────────────────────────────
    ax6 = axes[2, 1]
    zscore = df["rate_zscore"].dropna()
    ztime  = df["fundingTime"][zscore.index]
    ax6.plot(ztime, zscore, linewidth=0.8, color="purple", alpha=0.7)
    ax6.axhline(2,  color="red",    linestyle="--", linewidth=1, label="+2σ (sehr hoch)")
    ax6.axhline(-2, color="blue",   linestyle="--", linewidth=1, label="-2σ (sehr niedrig)")
    ax6.axhline(0,  color="black",  linewidth=0.5)
    ax6.fill_between(ztime, zscore, 0, where=(zscore > 2), alpha=0.2, color="red")
    ax6.fill_between(ztime, zscore, 0, where=(zscore < -2), alpha=0.2, color="blue")
    ax6.set_title("Z-Score der Funding Rate (30d Basis)", fontweight="bold")
    ax6.set_ylabel("Z-Score")
    ax6.legend(fontsize=8)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=30)

    # Speichern
    output_path = os.path.join(OUTPUT_DIR, "funding_rate_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n📊 Plot gespeichert: {output_path}")
    plt.close()


def simple_backtest(df: pd.DataFrame):
    """
    Einfacher Backtest: Vergleich 'immer drin' vs. 'nur wenn Rate > Threshold'.
    Zeigt wie viel Gebühren und schlechte Perioden kosten.
    """
    threshold = 0.0001   # 0.01%
    fee       = 0.0002   # 0.02% Maker Fee pro Trade (rein + raus)

    # Strategie 1: Immer drin
    always_in = df["fundingRate"].sum()

    # Strategie 2: Nur wenn Rate > Threshold
    in_signal  = df["fundingRate"] > threshold
    enters     = in_signal & (~in_signal.shift(1).fillna(False))
    exits      = (~in_signal) & in_signal.shift(1).fillna(False)
    num_trades = enters.sum()

    smart_rate = df["fundingRate"][in_signal].sum()
    smart_fee  = num_trades * fee * 2   # rein + raus
    smart_net  = smart_rate - smart_fee

    print("\n" + "="*55)
    print("  EINFACHER BACKTEST")
    print("="*55)
    print(f"  Strategie 1 (immer drin):   {always_in*100:+.2f}%  kumulativ")
    print(f"  Strategie 2 (nur > 0.01%):  {smart_rate*100:+.2f}%  brutto")
    print(f"    - Gebühren ({num_trades} Trades):   {-smart_fee*100:.2f}%")
    print(f"    = Netto:                  {smart_net*100:+.2f}%  kumulativ")
    print()
    print(f"  → Aktiv in Markt:  {in_signal.mean()*100:.0f}% der Zeit")
    print(f"  → Trades gesamt:   {num_trades}")
    print("="*55)
    print("\n  Das ist die Baseline – das ML-Modell muss das schlagen.\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Funding Rate Explorer startet...\n")

    # 1. Daten laden
    df_rates = get_funding_rates(SYMBOL, limit=LIMIT)

    # Optional: Weitere Daten (auskommentiert um Fehler zu vermeiden falls API-Limits)
    # df_oi = get_open_interest_history(SYMBOL)
    # df_ls = get_long_short_ratio(SYMBOL)

    # 2. Features bauen
    print("\n⚙️  Feature Engineering...")
    df = build_features(df_rates)

    # 3. Statistiken ausgeben
    print_stats(df)

    # 4. Einfacher Backtest
    simple_backtest(df)

    # 5. Plots erstellen
    print("📊 Erstelle Visualisierungen...")
    plot_all(df)

    # 6. Daten als CSV speichern (für nächste Schritte)
    csv_path = os.path.join(OUTPUT_DIR, "funding_rates_btc.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Daten gespeichert: {csv_path}")

    print("\n✅ Fertig! Nächste Schritte:")
    print("   1. funding_rate_analysis.png anschauen")
    print("   2. Die Statistiken verstehen – wann lohnt es sich wirklich?")
    print("   3. Open Interest als Feature hinzufügen")
    print("   4. Erstes XGBoost-Modell trainieren")
    print()