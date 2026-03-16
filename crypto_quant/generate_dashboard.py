"""
generate_dashboard.py – Lokales HTML-Dashboard
===============================================
Liest alle CSVs aus outputs/ und generiert ein standalone HTML-Dashboard.

Features:
  - Kein Server nötig – öffne outputs/dashboard.html direkt im Browser
  - Dark Theme, 3-Spalten Layout
  - Chart.js von cdnjs (einzige externe Abhängigkeit)
  - Daten inline als JavaScript Arrays (kein fetch() nötig)

Sektionen:
  0. Live Status Bar (Portfolio Value, Return, nächster Run)
  1. P&L Chart (Paper Trading vs Aave Benchmark)
  2. Aktuelle Positionen
  3. Trade History (letzte 20 Trades)
  4. Asset Performance Table (Backtest Metriken)
  5. IC-Analyse (Top Features ICIR)
  6. FSI Bar Chart
  7. Go-Live Readiness Score

Ausführen:
    cd crypto_quant
    python generate_dashboard.py
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from config import SYMBOLS, SYMBOL_SHORT, OUTPUT_DIR

PAPER_CAPITAL = 1_000.0
AAVE_APR      = 0.05
PERIODS_PER_YEAR = 3 * 365


# ── Daten laden ────────────────────────────────────────────────────────────────

def _load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_json(filename: str) -> dict:
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _to_js(values) -> str:
    def _fmt(v):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(round(float(v), 8))
        return json.dumps(str(v))
    return "[" + ", ".join(_fmt(v) for v in values) + "]"


def _minutes_to_next_run() -> int:
    now = datetime.now(timezone.utc)
    slots = [
        now.replace(hour=23, minute=20, second=0, microsecond=0),
        now.replace(hour= 7, minute=20, second=0, microsecond=0),
        now.replace(hour=15, minute=20, second=0, microsecond=0),
    ]
    for t in sorted(slots):
        if t > now:
            return int((t - now).total_seconds() / 60)
    # Nächster Slot morgen
    import datetime as dt_mod
    nxt = now.replace(hour=7, minute=20, second=0, microsecond=0) + dt_mod.timedelta(days=1)
    return int((nxt - now).total_seconds() / 60)


def compute_golive_score(df_perf: pd.DataFrame, df_trades: pd.DataFrame) -> tuple:
    """Berechnet Go-Live Readiness Score (0–100)."""
    checks = {}

    n_runs = len(df_perf)
    checks["30+ Tage gelaufen"] = n_runs >= 90

    if n_runs >= 2:
        returns = df_perf["period_return_pct"].dropna() / 100
        sharpe  = (returns.mean() / returns.std() * math.sqrt(PERIODS_PER_YEAR)
                   if returns.std() > 0 else 0)
        checks["Sharpe > 1.0"] = sharpe > 1.0

        peak  = df_perf["portfolio_value"].cummax()
        dd    = ((df_perf["portfolio_value"] - peak) / peak).min()
        checks["Max Drawdown < 15%"] = abs(dd) < 0.15

        # vs Aave Benchmark
        n_days    = n_runs / 3
        aave_ret  = (1 + AAVE_APR) ** (n_days / 365) - 1
        paper_ret = (df_perf["portfolio_value"].iloc[-1] / PAPER_CAPITAL) - 1
        checks["Paper > Aave Yield"] = paper_ret > aave_ret
    else:
        for k in ["Sharpe > 1.0", "Max Drawdown < 15%", "Paper > Aave Yield"]:
            checks[k] = False

    exits = df_trades[df_trades["action"] == "EXIT"] if not df_trades.empty else pd.DataFrame()
    checks["20+ Trades"] = len(exits) >= 20

    if len(exits) > 0:
        wr = (pd.to_numeric(exits["net_pnl_pct"], errors="coerce") > 0).mean()
        checks["Win Rate > 45%"] = float(wr) > 0.45
    else:
        checks["Win Rate > 45%"] = False

    if os.path.exists(os.path.join(OUTPUT_DIR, "paper_trading_errors.log")):
        try:
            with open(os.path.join(OUTPUT_DIR, "paper_trading_errors.log")) as f:
                n_errors = sum(1 for line in f if "ERROR" in line)
            checks["Error Rate < 5%"] = (n_errors / max(n_runs, 1)) < 0.05
        except Exception:
            checks["Error Rate < 5%"] = True
    else:
        checks["Error Rate < 5%"] = True

    score = sum(checks.values()) / len(checks) * 100
    return round(score, 1), checks


def load_dashboard_data() -> dict:
    data = {}

    # ── State ─────────────────────────────────────────────────────────────────
    state = _load_json("paper_trading_state.json")
    data["state"]          = state
    data["current_value"]  = state.get("current_value", PAPER_CAPITAL)
    data["start_date"]     = state.get("start_date", "")
    data["last_run"]       = state.get("last_run", "")
    data["total_trades"]   = state.get("total_trades", 0)
    data["total_runs"]     = state.get("total_runs", 0)

    cum_ret = (data["current_value"] / PAPER_CAPITAL - 1) * 100
    data["cum_return_pct"] = round(cum_ret, 3)

    # Running days
    if data["start_date"]:
        try:
            start = datetime.strptime(data["start_date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            data["running_days"] = (datetime.now(timezone.utc) - start).days
        except Exception:
            data["running_days"] = 0
    else:
        data["running_days"] = 0

    data["minutes_to_next_run"] = _minutes_to_next_run()

    # ── Performance Log ────────────────────────────────────────────────────────
    df_perf = _load_csv("paper_trading_performance.csv")
    if not df_perf.empty and "period_return_pct" in df_perf.columns:
        df_perf = df_perf[df_perf["period_return_pct"].notna()]
        data["perf_timestamps"]  = list(df_perf.get("timestamp", pd.Series()).astype(str))
        data["perf_values"]      = list(df_perf.get("portfolio_value", pd.Series()).fillna(PAPER_CAPITAL))
        # Aave Benchmark: linear growth at 5% APR
        n = len(df_perf)
        data["aave_values"]      = [
            round(PAPER_CAPITAL * (1 + AAVE_APR) ** (i / PERIODS_PER_YEAR), 4)
            for i in range(n)
        ]
        data["n_perf_rows"] = n
    else:
        data["perf_timestamps"]  = []
        data["perf_values"]      = []
        data["aave_values"]      = []
        data["n_perf_rows"]      = 0
        df_perf = pd.DataFrame()

    # ── Trade Log ─────────────────────────────────────────────────────────────
    df_trades = _load_csv("paper_trading_trades.csv")
    if not df_trades.empty:
        data["recent_trades"] = (
            df_trades.tail(20)
            .fillna("")
            .to_dict(orient="records")
        )
    else:
        data["recent_trades"] = []

    # ── Current Positions ─────────────────────────────────────────────────────
    positions = state.get("positions", {})
    pos_list  = []
    for sym in SYMBOLS:
        p   = positions.get(sym, {})
        key = SYMBOL_SHORT[sym].upper()
        pos_list.append({
            "symbol":     key,
            "state":      p.get("state", "FLAT"),
            "size":       p.get("size", 0.0),
            "entry_time": p.get("entry_time", ""),
            "entry_rate": p.get("entry_funding_rate", None),
            "holding":    p.get("holding_periods", 0),
        })
    data["positions"] = pos_list

    # ── Backtest Results ───────────────────────────────────────────────────────
    df_bt = _load_csv("backtest_results.csv")
    data["backtest"] = df_bt.to_dict(orient="records") if not df_bt.empty else []

    # ── IC Summaries ───────────────────────────────────────────────────────────
    ic_summaries = {}
    for sym in SYMBOLS:
        key    = SYMBOL_SHORT[sym]
        df_ic  = _load_csv(f"ic_summary_{sym}.csv")
        if not df_ic.empty:
            ic_summaries[key] = df_ic.head(15).to_dict(orient="records")
    data["ic_summaries"] = ic_summaries

    # ── FSI ────────────────────────────────────────────────────────────────────
    fsi_data = {}
    for sym in SYMBOLS:
        key   = SYMBOL_SHORT[sym]
        df_f  = _load_csv(f"fsi_report_{sym}.csv")
        if not df_f.empty:
            fsi_data[key] = df_f.head(10).to_dict(orient="records")
    data["fsi"] = fsi_data

    # ── Go-Live Score ──────────────────────────────────────────────────────────
    score, checks = compute_golive_score(df_perf, df_trades)
    data["golive_score"]  = score
    data["golive_checks"] = [
        {"label": k, "ok": bool(v)} for k, v in checks.items()
    ]

    data["generated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    data["assets"]        = [SYMBOL_SHORT[s].upper() for s in SYMBOLS]

    return data


# ── HTML Template ──────────────────────────────────────────────────────────────

def generate_html(data: dict) -> str:
    assets_js    = json.dumps(data["assets"])
    backtest_js  = json.dumps(data["backtest"])
    ic_js        = json.dumps(data["ic_summaries"])
    fsi_js       = json.dumps(data["fsi"])
    perf_ts_js   = _to_js(data["perf_timestamps"])
    perf_val_js  = _to_js(data["perf_values"])
    aave_val_js  = _to_js(data["aave_values"])
    trades_js    = json.dumps(data["recent_trades"])
    positions_js = json.dumps(data["positions"])
    checks_js    = json.dumps(data["golive_checks"])

    cur_val       = data["current_value"]
    cum_ret       = data["cum_return_pct"]
    running_days  = data["running_days"]
    total_runs    = data["total_runs"]
    total_trades  = data["total_trades"]
    score         = data["golive_score"]
    next_run_min  = data["minutes_to_next_run"]
    generated_at  = data["generated_at"]
    last_run      = data["last_run"][:16].replace("T", " ") if data["last_run"] else "—"

    # Score colour
    if score >= 85:
        score_color = "#3fb950"
        score_label = "Go-Live möglich"
    elif score >= 70:
        score_color = "#d29922"
        score_label = "Fast bereit"
    else:
        score_color = "#f85149"
        score_label = "Noch nicht bereit"

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crypto Quant – Paper Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {{
    --bg:#0d1117; --bg2:#161b22; --bg3:#21262d;
    --border:#30363d; --text:#e6edf3; --text2:#8b949e;
    --green:#3fb950; --red:#f85149; --blue:#58a6ff;
    --yellow:#d29922; --purple:#bc8cff; --orange:#ffa657;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text);
          font-family:-apple-system,'Segoe UI',sans-serif; font-size:13px; }}

  /* Status bar */
  .status-bar {{ background:var(--bg2); border-bottom:1px solid var(--border);
                 padding:10px 24px; display:flex; gap:32px; align-items:center;
                 flex-wrap:wrap; }}
  .status-bar .brand {{ font-size:14px; font-weight:700; color:var(--blue); margin-right:8px; }}
  .stat {{ display:flex; flex-direction:column; }}
  .stat .lbl {{ color:var(--text2); font-size:10px; text-transform:uppercase; letter-spacing:.5px; }}
  .stat .val {{ font-size:16px; font-weight:700; }}
  .stat .val.pos {{ color:var(--green); }}
  .stat .val.neg {{ color:var(--red); }}
  .status-bar .ts {{ margin-left:auto; color:var(--text2); font-size:11px; }}

  /* Grid */
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; padding:14px; }}
  .card {{ background:var(--bg2); border:1px solid var(--border); border-radius:8px; padding:14px; }}
  .card.wide {{ grid-column:span 2; }}
  .card.full {{ grid-column:span 3; }}
  .card h2 {{ font-size:11px; font-weight:600; color:var(--text2);
              text-transform:uppercase; letter-spacing:.5px; margin-bottom:10px; }}
  canvas {{ max-height:240px; }}

  /* Tables */
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ color:var(--text2); font-weight:500; text-align:left;
        padding:5px 8px; border-bottom:1px solid var(--border); }}
  td {{ padding:4px 8px; border-bottom:1px solid var(--bg3); }}
  tr:last-child td {{ border-bottom:none; }}
  .pos {{ color:var(--green); }} .neg {{ color:var(--red); }}
  .badge {{ display:inline-block; padding:2px 6px; border-radius:4px;
            font-size:10px; font-weight:600; }}
  .badge-g {{ background:rgba(63,185,80,.15); color:var(--green); }}
  .badge-r {{ background:rgba(248,81,73,.15); color:var(--red); }}
  .badge-b {{ background:rgba(88,166,255,.15); color:var(--blue); }}
  .badge-y {{ background:rgba(210,153,34,.15); color:var(--yellow); }}

  /* Score circle */
  .score-wrap {{ display:flex; align-items:center; gap:24px; }}
  .score-circle {{ position:relative; width:100px; height:100px; flex-shrink:0; }}
  .score-circle svg {{ transform:rotate(-90deg); }}
  .score-text {{ position:absolute; inset:0; display:flex; flex-direction:column;
                 align-items:center; justify-content:center; }}
  .score-text .num {{ font-size:22px; font-weight:700; }}
  .score-text .sub {{ font-size:10px; color:var(--text2); }}
  .checklist {{ flex:1; }}
  .check-item {{ display:flex; align-items:center; gap:8px; padding:3px 0;
                 font-size:12px; }}
  .check-item .icon {{ font-size:14px; }}

  /* Tabs */
  .tab-bar {{ display:flex; gap:6px; margin-bottom:10px; flex-wrap:wrap; }}
  .tab {{ padding:3px 10px; border-radius:4px; cursor:pointer; font-size:11px;
          border:1px solid var(--border); color:var(--text2); background:var(--bg3); }}
  .tab.active {{ background:var(--blue); color:#fff; border-color:var(--blue); }}
  .tab-panel {{ display:none; }}
  .tab-panel.active {{ display:block; }}
  .empty {{ color:var(--text2); font-style:italic; padding:16px 0;
            text-align:center; font-size:12px; }}
</style>
</head>
<body>

<!-- ── Status Bar ──────────────────────────────────────────────────────────── -->
<div class="status-bar">
  <span class="brand">Crypto Quant</span>
  <div class="stat">
    <span class="lbl">Portfolio</span>
    <span class="val">{cur_val:.2f} €</span>
  </div>
  <div class="stat">
    <span class="lbl">Return</span>
    <span class="val {'pos' if cum_ret >= 0 else 'neg'}">{'+' if cum_ret >= 0 else ''}{cum_ret:.3f}%</span>
  </div>
  <div class="stat">
    <span class="lbl">Laufzeit</span>
    <span class="val">{running_days} Tage</span>
  </div>
  <div class="stat">
    <span class="lbl">Runs / Trades</span>
    <span class="val">{total_runs} / {total_trades}</span>
  </div>
  <div class="stat">
    <span class="lbl">Nächster Run</span>
    <span class="val">in {next_run_min // 60}h {next_run_min % 60}min</span>
  </div>
  <div class="stat">
    <span class="lbl">Letzter Run</span>
    <span class="val" style="font-size:13px">{last_run}</span>
  </div>
  <span class="ts">Generiert: {generated_at}</span>
</div>

<div class="grid" id="dashboard"></div>

<script>
const ASSETS     = {assets_js};
const BACKTEST   = {backtest_js};
const IC_DATA    = {ic_js};
const FSI_DATA   = {fsi_js};
const PERF_TS    = {perf_ts_js};
const PERF_VAL   = {perf_val_js};
const AAVE_VAL   = {aave_val_js};
const TRADES     = {trades_js};
const POSITIONS  = {positions_js};
const CHECKS     = {checks_js};
const SCORE      = {score};
const SCORE_COLOR= "{score_color}";
const SCORE_LABEL= "{score_label}";
const PAPER_CAP  = 1000.0;

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "-apple-system,'Segoe UI',sans-serif";
Chart.defaults.font.size = 11;

function fmt(v,d=2) {{
  if (v===null||v===undefined||isNaN(v)) return '—';
  return Number(v).toFixed(d);
}}
function pct(v,d=3) {{
  if (v===null||v===undefined||isNaN(v)) return '—';
  return (Number(v)>=0?'+':'')+Number(v).toFixed(d)+'%';
}}
function colorCls(v) {{ return Number(v)>=0?'pos':'neg'; }}

// ── 0. P&L Chart ──────────────────────────────────────────────────────────────
function renderPnLChart(canvasId) {{
  const ctx = document.getElementById(canvasId);
  if (!ctx || !PERF_VAL.length) return;
  // Trim timestamps to HH:MM
  const labels = PERF_TS.map(t => t ? t.substring(11,16) : '');
  new Chart(ctx, {{
    type:'line',
    data:{{
      labels,
      datasets:[
        {{label:'Paper Trading',data:PERF_VAL,
          borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,.08)',
          borderWidth:1.5,pointRadius:0,tension:.2}},
        {{label:'Aave 5% APR',data:AAVE_VAL,
          borderColor:'#3fb950',backgroundColor:'transparent',
          borderWidth:1,pointRadius:0,borderDash:[4,3],tension:.2}},
      ]
    }},
    options:{{
      responsive:true,
      plugins:{{
        legend:{{position:'top',labels:{{font:{{size:11}},boxWidth:12}}}},
        tooltip:{{callbacks:{{label:ctx=>` ${{ctx.dataset.label}}: ${{fmt(ctx.raw,2)}} €`}}}}
      }},
      scales:{{
        x:{{ticks:{{maxTicksLimit:8}}}},
        y:{{ticks:{{callback:v=>v+' €'}}}}
      }}
    }}
  }});
}}

// ── 1. Current Positions ───────────────────────────────────────────────────────
function renderPositions() {{
  if (!POSITIONS.length) return '<p class="empty">Kein State geladen.</p>';
  let html = '<table><thead><tr><th>Asset</th><th>Status</th><th>Size</th><th>Entry Rate</th><th>Hold (×8h)</th></tr></thead><tbody>';
  for (const p of POSITIONS) {{
    const isLong = p.state === 'LONG';
    const badge  = isLong
      ? '<span class="badge badge-b">LONG</span>'
      : '<span class="badge badge-y">FLAT → Aave</span>';
    const rate   = p.entry_rate != null ? (Number(p.entry_rate)*100).toFixed(4)+'%' : '—';
    html += `<tr>
      <td><strong>${{p.symbol}}</strong></td>
      <td>${{badge}}</td>
      <td>${{fmt(p.size,3)}}</td>
      <td>${{rate}}</td>
      <td>${{p.holding}}</td>
    </tr>`;
  }}
  return html + '</tbody></table>';
}}

// ── 2. Trade History ───────────────────────────────────────────────────────────
function renderTrades() {{
  const exits = TRADES.filter(t => t.action === 'EXIT').slice(-20).reverse();
  if (!exits.length) return '<p class="empty">Noch keine abgeschlossenen Trades.</p>';
  let html = '<table><thead><tr><th>Zeit</th><th>Symbol</th><th>Aktion</th><th>Rate (Entry)</th><th>Perioden</th><th>Net P&L %</th><th>Net P&L €</th><th>Regime</th></tr></thead><tbody>';
  for (const t of exits) {{
    const pnl  = Number(t.net_pnl_pct||0);
    const cls  = colorCls(pnl);
    const time = (t.timestamp||'').substring(0,16).replace('T',' ');
    const sym  = (t.symbol||'').replace('USDT','');
    const rate = t.funding_rate_at_entry != null ? (Number(t.funding_rate_at_entry||0)*100).toFixed(4)+'%' : '—';
    html += `<tr>
      <td style="font-size:11px;color:var(--text2)">${{time}}</td>
      <td><strong>${{sym}}</strong></td>
      <td><span class="badge badge-b">EXIT</span></td>
      <td>${{rate}}</td>
      <td>${{t.holding_periods||'—'}}</td>
      <td class="${{cls}}">${{pct(pnl)}}</td>
      <td class="${{cls}}">${{fmt(t.net_eur||t.net_pnl_eur,3)}}</td>
      <td style="font-size:11px">${{t.regime_at_entry||'—'}}</td>
    </tr>`;
  }}
  return html + '</tbody></table>';
}}

// ── 3. Backtest Table ──────────────────────────────────────────────────────────
function renderPerformanceTable() {{
  if (!BACKTEST.length) return '<p class="empty">Keine Backtest-Daten.</p>';
  let html = '<table><thead><tr><th>Asset</th><th>Trades</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Kosten</th><th>Valid</th></tr></thead><tbody>';
  for (const r of BACKTEST) {{
    const sym   = (r.symbol||'').replace('USDT','');
    const valid = r.statistically_valid;
    const badge = valid ? '<span class="badge badge-g">✓</span>' : '<span class="badge badge-r">⚠</span>';
    html += `<tr>
      <td><strong>${{sym}}</strong></td>
      <td>${{r.num_trades||'—'}}</td>
      <td class="${{colorCls(r.sharpe_ratio)}}">${{fmt(r.sharpe_ratio)}}</td>
      <td class="${{colorCls(r.cagr_pct)}}">${{fmt(r.cagr_pct)}}%</td>
      <td class="neg">${{fmt(r.max_drawdown_pct)}}%</td>
      <td>${{fmt(r.avg_cost_per_trade_pct,4)}}%</td>
      <td>${{badge}}</td>
    </tr>`;
  }}
  return html + '</tbody></table>';
}}

// ── 4. Tabbed IC/FSI Charts ────────────────────────────────────────────────────
function makeChartConfig(asset, dataMap, color1, color2) {{
  const rows = dataMap[asset.toLowerCase()] || [];
  if (!rows.length) return null;
  const labels  = rows.map(r => r.feature || r.Feature || '');
  const values  = rows.map(r => r.icir || r.fsi || 0);
  return {{
    type:'bar',
    data:{{
      labels,
      datasets:[{{ label:'',data:values,borderWidth:0,
        backgroundColor:values.map(v=>v>=0?color1:color2) }}]
    }},
    options:{{
      indexAxis:'y',responsive:true,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>` ${{ctx.raw.toFixed(3)}}`}}}}}},
      scales:{{x:{{grid:{{color:'#30363d'}}}},y:{{ticks:{{font:{{size:10}}}}}}}}
    }}
  }};
}}

function renderTabbedSection(prefix, dataFn) {{
  const tabs   = ASSETS.map((a,i) => `<span class="tab ${{i===0?'active':''}}" onclick="switchTab('${{prefix}}','${{a}}',this)">${{a}}</span>`).join('');
  const panels = ASSETS.map((a,i) => `
    <div class="tab-panel ${{i===0?'active':''}}" id="${{prefix}}-${{a}}">
      <canvas id="${{prefix}}-canvas-${{a}}"></canvas>
    </div>`).join('');
  return `<div class="tab-bar">${{tabs}}</div>${{panels}}`;
}}

function switchTab(prefix, asset, el) {{
  document.querySelectorAll(`#${{prefix}}-tab-wrap .tab-panel`).forEach(p => p.classList.remove('active'));
  el.closest('.tab-bar').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(`${{prefix}}-${{asset}}`).classList.add('active');
  el.classList.add('active');
}}

// ── 5. Go-Live Score ───────────────────────────────────────────────────────────
function renderGoLive() {{
  const r      = 40, cx = 50, cy = 50;
  const circ   = 2 * Math.PI * r;
  const filled = circ * SCORE / 100;
  const svgCirc = `<svg width="100" height="100" viewBox="0 0 100 100">
    <circle cx="${{cx}}" cy="${{cy}}" r="${{r}}" fill="none" stroke="#21262d" stroke-width="10"/>
    <circle cx="${{cx}}" cy="${{cy}}" r="${{r}}" fill="none" stroke="${{SCORE_COLOR}}"
            stroke-width="10" stroke-dasharray="${{filled}} ${{circ}}" stroke-linecap="round"/>
  </svg>`;

  const checks = CHECKS.map(c =>
    `<div class="check-item">
       <span class="icon">${{c.ok ? '✅' : '❌'}}</span>
       <span style="color:${{c.ok?'var(--text)':'var(--text2)'}}">${{c.label}}</span>
     </div>`
  ).join('');

  return `<div class="score-wrap">
    <div class="score-circle">
      ${{svgCirc}}
      <div class="score-text">
        <span class="num" style="color:${{SCORE_COLOR}}">${{SCORE}}%</span>
        <span class="sub">Score</span>
      </div>
    </div>
    <div>
      <div style="font-size:14px;font-weight:700;color:${{SCORE_COLOR}};margin-bottom:8px">
        ${{SCORE_LABEL}}
      </div>
      <div class="checklist">${{checks}}</div>
    </div>
  </div>`;
}}

// ── Build Dashboard ────────────────────────────────────────────────────────────
const dash = document.getElementById('dashboard');

function card(cls, title, content) {{
  const el = document.createElement('div');
  el.className = `card ${{cls}}`;
  el.innerHTML = `<h2>${{title}}</h2>${{content}}`;
  return el;
}}

// Row 1: P&L Chart (wide) + Go-Live Score (narrow)
const c_pnl = document.createElement('div');
c_pnl.className = 'card wide';
c_pnl.innerHTML = '<h2>Portfolio P&L – Paper Trading vs Aave 5% APR</h2>'
  + (PERF_VAL.length
     ? '<canvas id="pnl-chart"></canvas>'
     : '<p class="empty">Noch keine Performance-Daten. Führe erst --run-once aus.</p>');
dash.appendChild(c_pnl);

const c_golive = card('', 'Go-Live Readiness Score', renderGoLive());
dash.appendChild(c_golive);

// Row 2: Positionen (narrow) + Trade History (wide)
dash.appendChild(card('', 'Aktuelle Positionen', renderPositions()));
dash.appendChild(card('wide', 'Trade History (letzte 20 Exits)', renderTrades()));

// Row 3: Asset Performance (full)
dash.appendChild(card('full', 'Asset Performance – Backtest', renderPerformanceTable()));

// Row 4: IC (full)
const c_ic = document.createElement('div');
c_ic.className = 'card full';
c_ic.id = 'ic-tab-wrap';
c_ic.innerHTML = '<h2>IC-Analyse – Top Features (ICIR)</h2>' + renderTabbedSection('ic');
dash.appendChild(c_ic);

// Row 5: FSI (full)
const c_fsi = document.createElement('div');
c_fsi.className = 'card full';
c_fsi.id = 'fsi-tab-wrap';
c_fsi.innerHTML = '<h2>Feature Stability Index (FSI – Warnung wenn > 2)</h2>' + renderTabbedSection('fsi');
dash.appendChild(c_fsi);

// Init charts
requestAnimationFrame(() => {{
  if (PERF_VAL.length) renderPnLChart('pnl-chart');

  for (const asset of ASSETS) {{
    const icCfg = makeChartConfig(asset, IC_DATA, 'rgba(63,185,80,.7)', 'rgba(248,81,73,.7)');
    const icEl  = document.getElementById(`ic-canvas-${{asset}}`);
    if (icCfg && icEl) new Chart(icEl, icCfg);

    const fsiCfg = makeChartConfig(asset, FSI_DATA, 'rgba(88,166,255,.7)', 'rgba(248,81,73,.7)');
    const fsiEl  = document.getElementById(`fsi-canvas-${{asset}}`);
    if (fsiCfg && fsiEl) new Chart(fsiEl, fsiCfg);
  }}
}});
</script>
</body>
</html>"""


# ── Entry Point ────────────────────────────────────────────────────────────────

def generate_dashboard():
    print("\n  [Dashboard] Generiere outputs/dashboard.html...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_dashboard_data()
    html = generate_html(data)

    out_path = os.path.join(OUTPUT_DIR, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✓ Dashboard: {out_path}")
    print(f"    Assets: {len(data['assets'])}  |  "
          f"Perf Rows: {data['n_perf_rows']}  |  "
          f"Trades: {len(data['recent_trades'])}  |  "
          f"Go-Live: {data['golive_score']}%")
    print(f"    file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    generate_dashboard()
