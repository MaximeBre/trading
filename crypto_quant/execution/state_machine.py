"""
execution/state_machine.py – Execution Layer: State Machine
============================================================
Verhindert Churning und verwaltet den tatsächlichen Positionsstatus.
FLAT-Kapital wird aktiv via Aave Yield (+5% p.a.) eingesetzt.

States:
    FLAT     – keine offene Position, Kapital in Aave
    LONG     – Delta-Neutral Position offen
    REDUCING – aktives Abbau einer Position (wird nach min_hold ausgelöst)

Zwei kritische Anti-Churning Regeln:
    1. Min Hold Period: Position wird mindestens N Perioden gehalten
    2. Min Rebalance Delta: Änderung < 10% → kein Rebalancing
       (verhindert teure Mini-Anpassungen)

Aave Integration:
    FLAT-Kapital (= 1 - sum(current_sizes)) bringt Aave Yield.
    Yield wird pro Periode als Return gebucht.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


AAVE_YIELD_PER_PERIOD = 0.05 / 1095  # 5% p.a. auf FLAT-Kapital
MIN_REBALANCE_DELTA   = 0.10          # Mindest-Änderung für Rebalancing

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@dataclass
class StateRecord:
    """Tracking-Objekt für einen einzelnen Zeitschritt."""
    period:       int
    state:        str
    sizes:        Dict[str, float]
    flat_capital: float
    aave_return:  float
    rebalanced:   bool
    crisis_exit:  bool


class FundingArbitrageStateMachine:
    """
    State Machine für Funding Rate Arbitrage Position Management.

    Implementiert:
        - Min Hold Period (Anti-Churning)
        - Min Rebalance Delta (Kosten-Kontrolle)
        - Hard Exit bei Crisis-Regime (p_crisis > 0.10)
        - FLAT-Kapital → Aave Yield
        - Vollständiger Audit-Trail via history

    Usage:
        sm = FundingArbitrageStateMachine(params)
        for each_period:
            sizes, aave_ret = sm.step(target_positions, regime_probs)
            portfolio_return = sum(funding_rates[s] * sizes[s] for s in symbols) + aave_ret
    """

    def __init__(self, params: dict, symbols: list = ASSETS):
        self.params        = params
        self.symbols       = symbols
        self.state         = "FLAT"
        self.periods_held  = 0
        self.current_sizes = {s: 0.0 for s in symbols}
        self.history: list[StateRecord] = []
        self._period       = 0

    def step(self, target_positions: Dict[str, float],
              regime_probs: Optional[Dict[str, Dict[str, float]]] = None,
             ) -> tuple[Dict[str, float], float]:
        """
        Führt einen Zeitschritt durch.

        Args:
            target_positions: {symbol: desired_position_size 0.0–1.0}
                              None → Circuit Breaker: Position einfrieren
            regime_probs:     {symbol: {p_crisis, p_bear, p_neutral, p_bull}}
                              None → kein Crisis Check

        Returns:
            (current_sizes, aave_return_this_period)
        """
        # ── Circuit Breaker: kein Signal → Position einfrieren ─────────────────
        # API down, Datenfehler, oder fehlende Inputs → nichts ändern
        # Kein Panik-Exit, kein blindes Halten einer unbekannten Position
        if target_positions is None:
            self._period      += 1
            self.periods_held += 1
            flat_capital = max(0.0, 1.0 - sum(self.current_sizes.values()))
            self.history.append(StateRecord(
                period       = self._period,
                state        = self.state,
                sizes        = dict(self.current_sizes),
                flat_capital = flat_capital,
                aave_return  = 0.0,   # Konservativ: kein Yield ohne Signal-Bestätigung
                rebalanced   = False,
                crisis_exit  = False,
            ))
            return dict(self.current_sizes), 0.0

        self._period    += 1
        self.periods_held += 1
        rebalanced       = False
        crisis_exit      = False

        # ── Hard Exit: Crisis Regime ───────────────────────────────────────────
        if regime_probs is not None:
            crisis_detected = any(
                regime_probs.get(s, {}).get("p_crisis", 0) >
                self.params.get("crisis_threshold", 0.10)
                for s in self.symbols
            )
            if crisis_detected:
                self.current_sizes = {s: 0.0 for s in self.symbols}
                self.periods_held  = 0
                self.state         = "FLAT"
                crisis_exit        = True

        # ── Minimum Hold Period ────────────────────────────────────────────────
        min_hold = int(self.params.get("min_hold_periods", 3))
        if not crisis_exit and self.periods_held >= min_hold:
            for sym in self.symbols:
                current = self.current_sizes[sym]
                target  = target_positions.get(sym, 0.0)
                delta   = abs(target - current)

                # Nur anpassen wenn Änderung signifikant
                if delta > MIN_REBALANCE_DELTA:
                    self.current_sizes[sym] = target
                    rebalanced = True

            if rebalanced:
                self.periods_held = 0

        # ── State Update ───────────────────────────────────────────────────────
        total_invested = sum(self.current_sizes.values())
        if total_invested < 0.05:
            self.state = "FLAT"
        elif any(
            self.current_sizes[s] < target_positions.get(s, 0) * 0.9
            for s in self.symbols
        ):
            self.state = "REDUCING"
        else:
            self.state = "LONG"

        # ── FLAT-Kapital → Aave Yield ─────────────────────────────────────────
        flat_capital = max(0.0, 1.0 - total_invested)
        aave_return  = flat_capital * AAVE_YIELD_PER_PERIOD

        # ── Audit Trail ───────────────────────────────────────────────────────
        self.history.append(StateRecord(
            period       = self._period,
            state        = self.state,
            sizes        = dict(self.current_sizes),
            flat_capital = flat_capital,
            aave_return  = aave_return,
            rebalanced   = rebalanced,
            crisis_exit  = crisis_exit,
        ))

        return dict(self.current_sizes), aave_return

    def reset(self):
        """Setzt State Machine auf Anfangszustand zurück."""
        self.state         = "FLAT"
        self.periods_held  = 0
        self.current_sizes = {s: 0.0 for s in self.symbols}
        self.history       = []
        self._period       = 0

    def summary(self) -> dict:
        """Gibt Statistiken über den gesamten Simulations-Lauf zurück."""
        if not self.history:
            return {}
        states       = [r.state for r in self.history]
        n            = len(self.history)
        time_flat    = states.count("FLAT") / n
        time_long    = states.count("LONG") / n
        n_rebalance  = sum(r.rebalanced for r in self.history)
        n_crisis_exit = sum(r.crisis_exit for r in self.history)
        avg_aave     = np.mean([r.aave_return for r in self.history])
        total_aave   = sum(r.aave_return for r in self.history)
        avg_invested = np.mean([
            sum(r.sizes.values()) for r in self.history
        ])

        return {
            "n_periods":       n,
            "time_flat_pct":   round(time_flat * 100, 1),
            "time_long_pct":   round(time_long * 100, 1),
            "n_rebalances":    n_rebalance,
            "n_crisis_exits":  n_crisis_exit,
            "avg_invested":    round(avg_invested, 3),
            "total_aave_yield": round(total_aave * 100, 4),
            "avg_aave_per_period": round(avg_aave, 8),
        }
