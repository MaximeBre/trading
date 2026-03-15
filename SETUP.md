# Crypto Quant – Paper Trading Setup Guide

## Übersicht

Das System läuft vollautomatisch per GitHub Actions und führt 3× täglich
(40 Minuten vor jedem Binance 8h Settlement) eine Paper-Trading-Periode aus.

**Zeitpunkte:** 23:20 / 07:20 / 15:20 UTC

---

## 1. GitHub Repository Setup

### Schritt 1: Repository erstellen

```bash
git init
git add .
git commit -m "Initial commit – Crypto Quant Paper Trading"
git remote add origin https://github.com/DEIN-USERNAME/DEIN-REPO.git
git push -u origin main
```

### Schritt 2: GitHub Secrets setzen

Gehe zu **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name          | Wert                    | Wozu?                                         |
|----------------------|-------------------------|-----------------------------------------------|
| `BINANCE_API_KEY`    | Dein Binance API Key    | Nur für Live-Modus nötig (Paper: optional)    |
| `BINANCE_API_SECRET` | Dein Binance API Secret | Nur für Live-Modus nötig (Paper: optional)    |

> **Wichtig für Paper Trading:** Im Paper-Modus werden keine echten Orders ausgeführt.
> Binance API Credentials werden nur für Live-Preisdaten benötigt (falls aktiviert).
> Das System funktioniert im Paper-Modus auch ohne API Keys.

### Schritt 3: API Key Permissions (wenn API Keys gesetzt)

Binance API Key Einstellungen:
- ✅ **Read** (Enable Reading) — erlaubt
- ❌ **Spot & Margin Trading** — NICHT für Paper Mode
- ❌ **Futures Trading** — NICHT für Paper Mode
- ❌ **Withdrawals** — NIEMALS aktivieren

### Schritt 4: GitHub Pages aktivieren

1. **Settings → Pages**
2. **Source:** Deploy from a branch
3. **Branch:** `gh-pages` / `/ (root)`
4. Speichern

Nach dem ersten erfolgreichen Run ist das Dashboard erreichbar unter:
```
https://DEIN-USERNAME.github.io/DEIN-REPO/dashboard.html
```

---

## 2. Ersten Run starten

### Option A: Manueller Trigger (empfohlen für Test)

1. Gehe zu **Actions → Paper Trading - 3x Daily**
2. Klicke **Run workflow**
3. Klicke **Run workflow** (bestätigen)
4. Warte ~5 Minuten
5. Prüfe unter **Artifacts** die Log-Dateien

**Erwartetes Output nach erstem Run:**
```
[PAPER TRADER] 2026-03-15 23:20 UTC  (Run #1)
  Portfolio:  1000.00 € → 1000.XX €  (+0.00X% seit Start)
  Return:     +0.00XXX%
  Positionen: X aktiv  |  Neue Trades: X
  SM State:   FLAT / LONG
  [Dashboard] Generiere outputs/dashboard.html...
  ✓ Dashboard: outputs/dashboard.html
```

### Option B: Lokal testen

```bash
cd crypto_quant
python -m execution.paper_trading --run-once
```

**Voraussetzung:** Feature-CSVs müssen vorhanden sein:
```bash
python main.py   # Dauert ~10min (lädt 3 Jahre Daten, trainiert Modelle)
```

---

## 3. Nach 30 Tagen: Go-Live Entscheidung

### Dashboard prüfen

Öffne `https://DEIN-USERNAME.github.io/DEIN-REPO/dashboard.html`

Der **Go-Live Readiness Score** zeigt:
- 🟢 **> 85%** → Go-Live möglich
- 🟡 **70–85%** → Fast bereit (noch etwas warten)
- 🔴 **< 70%** → Noch nicht bereit

### Go-Live Checkliste

- [ ] Dashboard Go-Live Score > 85%
- [ ] Sharpe Ratio > 1.0 im Paper Trading
- [ ] Max Drawdown < 15%
- [ ] Mindestens 20 abgeschlossene Trades
- [ ] Win Rate > 45%
- [ ] < 5% Error Rate in `paper_trading_errors.log`

### Live-Modus aktivieren

1. **Binance API Permissions erweitern:**
   - ✅ Spot & Margin Trading
   - ✅ Futures Trading (für Delta-Neutral)

2. **Startkapital vorbereiten:**
   - 1.000 € auf Binance Spot Wallet
   - 500 € auf Binance Futures Wallet (als Margin)

3. **config.py anpassen (wenn Live-System implementiert):**
   ```python
   PAPER_TRADING = False
   CAPITAL       = 1_000   # EUR
   ```

4. **Live-System starten** (nach separater Implementierung):
   ```bash
   python execution/live_trading.py
   ```

---

## 4. Monitoring & Troubleshooting

### Logs prüfen

```bash
# Fehler-Log
cat outputs/paper_trading_errors.log

# Performance
tail -20 outputs/paper_trading_performance.csv

# Trades
tail -20 outputs/paper_trading_trades.csv
```

### Häufige Probleme

| Problem | Ursache | Lösung |
|---------|---------|--------|
| `Feature-CSV nicht gefunden` | `main.py` noch nicht gelaufen | `python main.py` ausführen |
| `Modell nicht gefunden` | Training noch nicht gestartet | `python models/train.py` |
| `API Error` | Binance Timeout | System friert Position automatisch ein (safe) |
| Dashboard leer | Noch keine Runs | Ersten Run manuell starten |

### GitHub Actions Kosten

GitHub Actions ist **kostenlos** für Public Repositories (unlimitiert Minuten).
Für Private Repos: 2.000 Minuten/Monat kostenlos, danach ~$0.008/Minute.

3 Runs × ~3 Minuten × 30 Tage = ~270 Minuten/Monat → weit unter dem Limit.

---

## 5. Systemarchitektur (Kurzübersicht)

```
main.py                     ← Phase 1+2 Pipeline (Feature Engineering)
  ├── data/binance.py        ← Binance Funding Rates, OI, Basis
  ├── data/bybit.py          ← Bybit Cross-Exchange Daten
  ├── data/okx.py            ← OKX Tri-Exchange Daten
  ├── data/market_context.py ← BTC OI Dominanz
  ├── features/engineering.py← Feature Engineering (40+ Features)
  ├── models/train.py        ← XGBoost Walk-Forward Training
  ├── models/regime.py       ← GMM Regime Classifier
  ├── models/alpha.py        ← 3-Modell Alpha Ensemble
  └── models/portfolio_constructor.py ← Layer 4 + Optuna

execution/paper_trading.py  ← Paper Trading Engine (3x täglich)
  ├── execution/state_machine.py ← Circuit Breaker
  └── generate_dashboard.py  ← HTML Dashboard

.github/workflows/
  └── paper_trading.yml      ← GitHub Actions (3 Cron Triggers)

outputs/
  ├── dashboard.html          ← Live Dashboard (GitHub Pages)
  ├── paper_trading_state.json← Persistenter State
  ├── paper_trading_trades.csv← Trade Log
  ├── paper_trading_performance.csv ← Performance Log
  └── paper_trading_errors.log ← Error Log
```

---

*Generiert für: Crypto Quant – Funding Rate Arbitrage System*
*Startkapital: 1.000 EUR Paper Trading → Go-Live nach 30 Tagen*
