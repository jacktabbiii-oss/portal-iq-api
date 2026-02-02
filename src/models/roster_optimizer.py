"""Roster Optimizer for Portal IQ.

Integrates NIL valuation, portal prediction, draft projection, and win impact
models to provide comprehensive roster optimization recommendations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pulp import (
        LpMaximize,
        LpMinimize,
        LpProblem,
        LpVariable,
        lpSum,
        LpStatus,
        value,
        PULP_CBC_CMD,
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from .nil_valuator import NILValuator
from .portal_predictor import PortalPredictor
from .draft_projector import DraftProjector
from .win_model import WinImpactModel

logger = logging.getLogger(__name__)


class RosterOptimizer:
    """Comprehensive roster optimization using linear programming and ML models.

    Integrates all Portal IQ models to provide:
    - Optimal NIL budget allocation
    - Portal shopping lists ranked by value/impact
    - Recruiting ROI analysis
    - Full roster reports combining all insights
    """

    # Position groups for optimization
    POSITION_GROUPS = {
        'QB': ['QB'],
        'RB': ['RB', 'FB'],
        'WR': ['WR'],
        'TE': ['TE'],
        'OL': ['OT', 'OG', 'C', 'OL'],
        'DL': ['DE', 'DT', 'NT', 'DL'],
        'EDGE': ['EDGE', 'OLB'],
        'LB': ['LB', 'ILB', 'MLB'],
        'CB': ['CB'],
        'S': ['S', 'FS', 'SS'],
        'K': ['K', 'P'],
    }

    # Minimum starters by position group
    MIN_STARTERS = {
        'QB': 1, 'RB': 1, 'WR': 2, 'TE': 1, 'OL': 5,
        'DL': 2, 'EDGE': 2, 'LB': 2, 'CB': 2, 'S': 2, 'K': 1
    }

    # Recommended depth by position group
    RECOMMENDED_DEPTH = {
        'QB': 3, 'RB': 4, 'WR': 6, 'TE': 3, 'OL': 10,
        'DL': 6, 'EDGE': 4, 'LB': 5, 'CB': 5, 'S': 4, 'K': 2
    }

    # Position value weights for optimization (based on win impact)
    POSITION_VALUE_WEIGHTS = {
        'QB': 1.0, 'EDGE': 0.48, 'OL': 0.48, 'CB': 0.40,
        'WR': 0.36, 'DL': 0.36, 'LB': 0.32, 'S': 0.28,
        'RB': 0.24, 'TE': 0.24, 'K': 0.08
    }

    # Average NIL values by tier and position (for estimation)
    AVG_NIL_BY_TIER_POSITION = {
        'blue_blood': {'QB': 800000, 'RB': 200000, 'WR': 250000, 'TE': 150000,
                       'OL': 100000, 'DL': 150000, 'EDGE': 200000, 'LB': 120000,
                       'CB': 180000, 'S': 120000, 'K': 50000},
        'elite': {'QB': 500000, 'RB': 150000, 'WR': 180000, 'TE': 100000,
                  'OL': 80000, 'DL': 100000, 'EDGE': 150000, 'LB': 80000,
                  'CB': 120000, 'S': 80000, 'K': 30000},
        'power_brand': {'QB': 300000, 'RB': 100000, 'WR': 120000, 'TE': 70000,
                        'OL': 50000, 'DL': 70000, 'EDGE': 100000, 'LB': 50000,
                        'CB': 80000, 'S': 50000, 'K': 20000},
        'p4_mid': {'QB': 150000, 'RB': 60000, 'WR': 80000, 'TE': 40000,
                   'OL': 30000, 'DL': 40000, 'EDGE': 60000, 'LB': 30000,
                   'CB': 50000, 'S': 30000, 'K': 10000},
        'g5_strong': {'QB': 80000, 'RB': 30000, 'WR': 40000, 'TE': 20000,
                      'OL': 15000, 'DL': 20000, 'EDGE': 30000, 'LB': 15000,
                      'CB': 25000, 'S': 15000, 'K': 5000},
        'g5': {'QB': 40000, 'RB': 15000, 'WR': 20000, 'TE': 10000,
               'OL': 8000, 'DL': 10000, 'EDGE': 15000, 'LB': 8000,
               'CB': 12000, 'S': 8000, 'K': 3000},
    }

    # School tier mapping (subset - full list in nil_features.py)
    SCHOOL_TIERS = {
        'Alabama': 'blue_blood', 'Ohio State': 'blue_blood', 'Georgia': 'blue_blood',
        'Texas': 'blue_blood', 'USC': 'blue_blood', 'Michigan': 'blue_blood',
        'Notre Dame': 'blue_blood', 'Oklahoma': 'blue_blood', 'LSU': 'elite',
        'Florida': 'elite', 'Penn State': 'elite', 'Oregon': 'elite',
        'Clemson': 'elite', 'Tennessee': 'elite', 'Texas A&M': 'elite',
        'Miami': 'power_brand', 'Florida State': 'power_brand', 'Auburn': 'power_brand',
        'Wisconsin': 'power_brand', 'Iowa': 'power_brand', 'UCLA': 'power_brand',
    }

    def __init__(
        self,
        nil_valuator: Optional[NILValuator] = None,
        portal_predictor: Optional[PortalPredictor] = None,
        draft_projector: Optional[DraftProjector] = None,
        win_model: Optional[WinImpactModel] = None,
    ):
        """Initialize with ML model instances.

        Args:
            nil_valuator: Trained NILValuator instance
            portal_predictor: Trained PortalPredictor instance
            draft_projector: Trained DraftProjector instance
            win_model: Trained WinImpactModel instance
        """
        self.nil_valuator = nil_valuator
        self.portal_predictor = portal_predictor
        self.draft_projector = draft_projector
        self.win_model = win_model

        if not PULP_AVAILABLE:
            logger.warning("PuLP not available. optimize_nil_budget will use fallback method.")

    def _get_position_group(self, position: str) -> str:
        """Map specific position to position group."""
        position = str(position).upper()
        for group, positions in self.POSITION_GROUPS.items():
            if position in positions:
                return group
        return 'OTHER'

    def _get_school_tier(self, school: str) -> str:
        """Get school tier for budget estimation."""
        return self.SCHOOL_TIERS.get(school, 'p4_mid')

    def _estimate_player_value(
        self,
        player: pd.Series,
        school: str,
    ) -> Dict[str, float]:
        """Estimate player's value contribution.

        Returns dict with nil_value, win_value, draft_value, and total_value.
        """
        position_group = self._get_position_group(player.get('position', 'OTHER'))
        school_tier = self._get_school_tier(school)

        # Base NIL value from tier/position
        base_nil = self.AVG_NIL_BY_TIER_POSITION.get(
            school_tier, self.AVG_NIL_BY_TIER_POSITION['p4_mid']
        ).get(position_group, 50000)

        # Adjust for player rating/production if available
        rating_mult = 1.0
        if 'overall_rating' in player:
            rating = player['overall_rating']
            if rating >= 0.95:
                rating_mult = 2.0
            elif rating >= 0.90:
                rating_mult = 1.5
            elif rating >= 0.85:
                rating_mult = 1.2
            elif rating < 0.75:
                rating_mult = 0.7

        # Use NIL valuator if available
        nil_value = base_nil * rating_mult
        if self.nil_valuator is not None:
            try:
                prediction = self.nil_valuator.predict(player.to_frame().T)
                if prediction and 'predicted_value' in prediction[0]:
                    nil_value = prediction[0]['predicted_value']
            except Exception as e:
                logger.debug(f"NIL prediction failed, using estimate: {e}")

        # Win value from position weight
        position_weight = self.POSITION_VALUE_WEIGHTS.get(position_group, 0.2)
        win_value = position_weight * nil_value / 100000  # Scale to wins

        # Draft value (if projector available)
        draft_value = 0.0
        if self.draft_projector is not None and player.get('eligibility_year', 0) <= 1:
            try:
                projection = self.draft_projector.predict(player.to_frame().T)
                if projection and 'expected_draft_value' in projection[0]:
                    draft_value = projection[0]['expected_draft_value']
            except Exception:
                pass

        # Total value combines all factors
        total_value = nil_value + (win_value * 100000) + (draft_value * 0.1)

        return {
            'nil_value': nil_value,
            'win_value': win_value,
            'draft_value': draft_value,
            'total_value': total_value,
        }

    def optimize_nil_budget(
        self,
        school: str,
        total_budget: float,
        roster_df: pd.DataFrame,
        win_target: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Optimize NIL budget allocation across roster using linear programming.

        Uses PuLP to solve the optimization problem:
        - Maximize total roster value (wins + retention + draft potential)
        - Subject to budget constraint
        - Subject to minimum allocations for key players
        - Subject to positional balance requirements

        Args:
            school: School name for context
            total_budget: Total NIL budget available
            roster_df: DataFrame with player info (name, position, rating, etc.)
            win_target: Optional win target to constrain optimization

        Returns:
            Dict containing:
            - allocations: List of {player, position, recommended_nil, reasoning}
            - total_allocated: Sum of allocations
            - expected_wins: Projected wins with this allocation
            - budget_efficiency: Value per dollar metric
            - position_breakdown: Budget by position group
            - retention_priority: High-risk players flagged for investment
            - optimization_status: 'optimal', 'feasible', or 'fallback'
        """
        if roster_df.empty:
            return {
                'allocations': [],
                'total_allocated': 0,
                'expected_wins': 0,
                'budget_efficiency': 0,
                'position_breakdown': {},
                'retention_priority': [],
                'optimization_status': 'empty_roster',
            }

        # Prepare player data
        players = []
        for idx, row in roster_df.iterrows():
            player_name = row.get('name', row.get('player_name', f'Player_{idx}'))
            position = row.get('position', 'UNKNOWN')
            position_group = self._get_position_group(position)

            # Get player values
            values = self._estimate_player_value(row, school)

            # Check flight risk if predictor available
            flight_risk = 0.3  # Default moderate risk
            if self.portal_predictor is not None:
                try:
                    risk_pred = self.portal_predictor.predict_flight_risk(row.to_frame().T)
                    if risk_pred:
                        flight_risk = risk_pred[0].get('flight_risk_probability', 0.3)
                except Exception:
                    pass

            # Minimum NIL to retain (based on market and risk)
            school_tier = self._get_school_tier(school)
            market_rate = self.AVG_NIL_BY_TIER_POSITION.get(
                school_tier, self.AVG_NIL_BY_TIER_POSITION['p4_mid']
            ).get(position_group, 30000)

            min_nil = market_rate * 0.5  # At least 50% of market to retain
            max_nil = market_rate * 3.0  # Cap at 3x market rate

            players.append({
                'index': idx,
                'name': player_name,
                'position': position,
                'position_group': position_group,
                'value': values['total_value'],
                'win_value': values['win_value'],
                'nil_value': values['nil_value'],
                'flight_risk': flight_risk,
                'min_nil': min_nil,
                'max_nil': max_nil,
                'market_rate': market_rate,
            })

        # Use PuLP if available, otherwise fallback
        if PULP_AVAILABLE:
            result = self._optimize_with_pulp(
                players, total_budget, win_target, school
            )
        else:
            result = self._optimize_fallback(
                players, total_budget, win_target, school
            )

        return result

    def _optimize_with_pulp(
        self,
        players: List[Dict],
        total_budget: float,
        win_target: Optional[float],
        school: str,
    ) -> Dict[str, Any]:
        """Solve NIL allocation using PuLP linear programming."""

        # Create the optimization problem
        prob = LpProblem("NIL_Budget_Allocation", LpMaximize)

        # Decision variables: NIL allocation for each player
        nil_vars = {}
        for p in players:
            nil_vars[p['index']] = LpVariable(
                f"nil_{p['index']}",
                lowBound=0,
                upBound=p['max_nil'],
            )

        # Objective: Maximize total value
        # Value = (allocation / market_rate) * player_value * (1 + flight_risk)
        # Higher flight risk players get more weight (retention value)
        prob += lpSum([
            (nil_vars[p['index']] / max(p['market_rate'], 1)) *
            p['value'] * (1 + p['flight_risk'])
            for p in players
        ]), "Total_Roster_Value"

        # Constraint: Total budget
        prob += lpSum([nil_vars[p['index']] for p in players]) <= total_budget, "Budget_Constraint"

        # Constraint: Minimum allocations for high-value/high-risk players
        for p in players:
            if p['flight_risk'] > 0.5 or p['win_value'] > 0.3:
                prob += nil_vars[p['index']] >= p['min_nil'], f"Min_NIL_{p['index']}"

        # Constraint: Positional balance (don't overspend on one position)
        position_budgets = {}
        for p in players:
            pg = p['position_group']
            if pg not in position_budgets:
                position_budgets[pg] = []
            position_budgets[pg].append(nil_vars[p['index']])

        for pg, vars_list in position_budgets.items():
            max_position_budget = total_budget * 0.35  # No position > 35% of budget
            prob += lpSum(vars_list) <= max_position_budget, f"Max_{pg}_Budget"

        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))

        status = LpStatus[prob.status]

        if status not in ['Optimal', 'Feasible']:
            logger.warning(f"Optimization status: {status}, using fallback")
            return self._optimize_fallback(players, total_budget, win_target, school)

        # Extract results
        allocations = []
        position_breakdown = {}
        retention_priority = []
        total_allocated = 0
        total_win_value = 0

        for p in players:
            allocation = value(nil_vars[p['index']])
            if allocation is None:
                allocation = 0

            total_allocated += allocation
            total_win_value += p['win_value'] * (allocation / max(p['market_rate'], 1))

            # Position breakdown
            pg = p['position_group']
            position_breakdown[pg] = position_breakdown.get(pg, 0) + allocation

            # Determine reasoning
            reasoning = []
            if p['flight_risk'] > 0.6:
                reasoning.append("high flight risk - retention priority")
                retention_priority.append(p['name'])
            if p['win_value'] > 0.4:
                reasoning.append("high win impact")
            if allocation >= p['market_rate'] * 1.5:
                reasoning.append("above market - key player premium")
            elif allocation < p['market_rate'] * 0.7:
                reasoning.append("below market - depth player")

            allocations.append({
                'player': p['name'],
                'position': p['position'],
                'position_group': pg,
                'recommended_nil': round(allocation, 2),
                'market_rate': round(p['market_rate'], 2),
                'vs_market': f"{(allocation / max(p['market_rate'], 1) * 100):.0f}%",
                'flight_risk': round(p['flight_risk'], 2),
                'win_value': round(p['win_value'], 3),
                'reasoning': '; '.join(reasoning) if reasoning else 'standard allocation',
            })

        # Sort by allocation descending
        allocations.sort(key=lambda x: x['recommended_nil'], reverse=True)

        # Calculate efficiency
        budget_efficiency = total_win_value / max(total_allocated, 1) * 1000000

        return {
            'allocations': allocations,
            'total_allocated': round(total_allocated, 2),
            'budget_remaining': round(total_budget - total_allocated, 2),
            'expected_wins': round(total_win_value + 6, 1),  # Base 6 wins + additions
            'budget_efficiency': round(budget_efficiency, 4),
            'position_breakdown': {k: round(v, 2) for k, v in position_breakdown.items()},
            'retention_priority': retention_priority,
            'optimization_status': status.lower(),
            'school': school,
            'total_budget': total_budget,
        }

    def _optimize_fallback(
        self,
        players: List[Dict],
        total_budget: float,
        win_target: Optional[float],
        school: str,
    ) -> Dict[str, Any]:
        """Fallback optimization using heuristic allocation."""

        # Sort players by value-adjusted priority
        # Priority = value * (1 + flight_risk) / market_rate
        for p in players:
            p['priority'] = p['value'] * (1 + p['flight_risk']) / max(p['market_rate'], 1)

        players_sorted = sorted(players, key=lambda x: x['priority'], reverse=True)

        # Allocate budget greedily
        remaining_budget = total_budget
        allocations = []
        position_breakdown = {}
        retention_priority = []
        total_win_value = 0

        # First pass: ensure minimum allocations for high-priority players
        for p in players_sorted:
            if p['flight_risk'] > 0.5 or p['win_value'] > 0.3:
                allocation = min(p['min_nil'], remaining_budget)
                p['allocated'] = allocation
                remaining_budget -= allocation
            else:
                p['allocated'] = 0

        # Second pass: distribute remaining budget by priority
        for p in players_sorted:
            additional = min(
                p['market_rate'] - p['allocated'],
                remaining_budget * p['priority'] / sum(x['priority'] for x in players_sorted),
                p['max_nil'] - p['allocated']
            )
            additional = max(0, additional)
            p['allocated'] += additional
            remaining_budget -= additional

        # Build results
        for p in players_sorted:
            allocation = p['allocated']
            total_win_value += p['win_value'] * (allocation / max(p['market_rate'], 1))

            pg = p['position_group']
            position_breakdown[pg] = position_breakdown.get(pg, 0) + allocation

            if p['flight_risk'] > 0.6:
                retention_priority.append(p['name'])

            reasoning = []
            if p['flight_risk'] > 0.6:
                reasoning.append("high flight risk")
            if p['win_value'] > 0.4:
                reasoning.append("high win impact")

            allocations.append({
                'player': p['name'],
                'position': p['position'],
                'position_group': pg,
                'recommended_nil': round(allocation, 2),
                'market_rate': round(p['market_rate'], 2),
                'vs_market': f"{(allocation / max(p['market_rate'], 1) * 100):.0f}%",
                'flight_risk': round(p['flight_risk'], 2),
                'win_value': round(p['win_value'], 3),
                'reasoning': '; '.join(reasoning) if reasoning else 'standard allocation',
            })

        allocations.sort(key=lambda x: x['recommended_nil'], reverse=True)
        total_allocated = total_budget - remaining_budget

        return {
            'allocations': allocations,
            'total_allocated': round(total_allocated, 2),
            'budget_remaining': round(remaining_budget, 2),
            'expected_wins': round(total_win_value + 6, 1),
            'budget_efficiency': round(total_win_value / max(total_allocated, 1) * 1000000, 4),
            'position_breakdown': {k: round(v, 2) for k, v in position_breakdown.items()},
            'retention_priority': retention_priority,
            'optimization_status': 'fallback_heuristic',
            'school': school,
            'total_budget': total_budget,
        }

    def portal_shopping_list(
        self,
        school: str,
        roster_df: pd.DataFrame,
        budget_remaining: float,
        positions_of_need: Optional[List[str]] = None,
        portal_players_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate ranked portal target list with cost/impact analysis.

        Args:
            school: School seeking players
            roster_df: Current roster for gap analysis
            budget_remaining: Available NIL budget
            positions_of_need: Specific positions to target (optional)
            portal_players_df: Available portal players (optional)

        Returns:
            Dict containing:
            - shopping_list: Ranked list of {player, position, projected_nil,
              fit_score, win_impact, value_rating, reasoning}
            - positions_prioritized: Positions ranked by need
            - budget_strategy: Recommended allocation approach
            - total_targets: Number of realistic targets
            - projected_improvement: Expected wins improvement if targets acquired
        """
        school_tier = self._get_school_tier(school)

        # Analyze current roster gaps
        position_counts = {}
        position_quality = {}

        for _, row in roster_df.iterrows():
            pg = self._get_position_group(row.get('position', 'OTHER'))
            position_counts[pg] = position_counts.get(pg, 0) + 1

            rating = row.get('overall_rating', 0.75)
            if pg not in position_quality:
                position_quality[pg] = []
            position_quality[pg].append(rating)

        # Calculate position needs
        position_needs = {}
        for pg in self.POSITION_GROUPS.keys():
            current = position_counts.get(pg, 0)
            recommended = self.RECOMMENDED_DEPTH.get(pg, 3)

            avg_quality = np.mean(position_quality.get(pg, [0.7]))

            # Need score: quantity gap + quality gap + position value
            quantity_gap = max(0, recommended - current) / recommended
            quality_gap = max(0, 0.85 - avg_quality)  # Target 85% average
            position_value = self.POSITION_VALUE_WEIGHTS.get(pg, 0.2)

            need_score = (quantity_gap * 0.4 + quality_gap * 0.3 + position_value * 0.3)
            position_needs[pg] = {
                'current_count': current,
                'recommended': recommended,
                'avg_quality': round(avg_quality, 3),
                'need_score': round(need_score, 3),
                'quantity_gap': current < recommended,
            }

        # Prioritize positions
        if positions_of_need:
            # User-specified positions get priority
            positions_prioritized = [
                self._get_position_group(p) for p in positions_of_need
            ]
            # Add other positions by need score
            other_positions = sorted(
                [pg for pg in position_needs if pg not in positions_prioritized],
                key=lambda x: position_needs[x]['need_score'],
                reverse=True
            )
            positions_prioritized.extend(other_positions)
        else:
            positions_prioritized = sorted(
                position_needs.keys(),
                key=lambda x: position_needs[x]['need_score'],
                reverse=True
            )

        # Generate shopping list
        shopping_list = []

        # If portal players provided, evaluate them
        if portal_players_df is not None and not portal_players_df.empty:
            for idx, player in portal_players_df.iterrows():
                pg = self._get_position_group(player.get('position', 'OTHER'))

                # Estimate NIL cost
                values = self._estimate_player_value(player, school)
                projected_nil = values['nil_value']

                # Check if affordable
                if projected_nil > budget_remaining:
                    continue

                # Calculate fit score
                fit_score = 0.5  # Base
                if pg in positions_prioritized[:3]:
                    fit_score += 0.3  # High need position
                if player.get('origin_school_tier', 'p4_mid') in ['blue_blood', 'elite']:
                    fit_score += 0.1  # P4 experience
                if player.get('years_remaining', 1) >= 2:
                    fit_score += 0.1  # Multi-year value

                # Portal fit prediction if available
                if self.portal_predictor is not None:
                    try:
                        fit_pred = self.portal_predictor.predict_portal_fit(
                            player.to_frame().T, school
                        )
                        if fit_pred:
                            fit_score = fit_pred[0].get('fit_score', fit_score)
                    except Exception:
                        pass

                # Value rating (fit * win_impact / cost)
                win_impact = values['win_value']
                value_rating = (fit_score * win_impact) / max(projected_nil / 1000000, 0.01)

                # Build reasoning
                reasoning = []
                need_info = position_needs.get(pg, {})
                if need_info.get('quantity_gap'):
                    reasoning.append(f"fills depth need at {pg}")
                if need_info.get('avg_quality', 0.8) < 0.80:
                    reasoning.append(f"upgrades {pg} quality")
                if player.get('overall_rating', 0.75) >= 0.90:
                    reasoning.append("elite talent")
                if projected_nil <= budget_remaining * 0.2:
                    reasoning.append("affordable depth piece")

                shopping_list.append({
                    'player': player.get('name', player.get('player_name', f'Portal_{idx}')),
                    'position': player.get('position', 'UNKNOWN'),
                    'position_group': pg,
                    'origin_school': player.get('origin_school', 'Unknown'),
                    'projected_nil': round(projected_nil, 2),
                    'fit_score': round(fit_score, 3),
                    'win_impact': round(win_impact, 3),
                    'value_rating': round(value_rating, 4),
                    'priority_rank': positions_prioritized.index(pg) + 1 if pg in positions_prioritized else 99,
                    'reasoning': '; '.join(reasoning) if reasoning else 'general roster improvement',
                })
        else:
            # Generate hypothetical targets based on needs
            for rank, pg in enumerate(positions_prioritized[:5], 1):
                market_rate = self.AVG_NIL_BY_TIER_POSITION.get(
                    school_tier, self.AVG_NIL_BY_TIER_POSITION['p4_mid']
                ).get(pg, 50000)

                win_impact = self.POSITION_VALUE_WEIGHTS.get(pg, 0.2)
                need_info = position_needs.get(pg, {})

                # Suggest target profile
                if need_info.get('avg_quality', 0.8) < 0.80:
                    target_type = "quality upgrade"
                    projected_nil = market_rate * 1.5
                elif need_info.get('quantity_gap'):
                    target_type = "depth addition"
                    projected_nil = market_rate * 0.8
                else:
                    target_type = "competition piece"
                    projected_nil = market_rate

                if projected_nil <= budget_remaining:
                    shopping_list.append({
                        'player': f"[Target {pg} - {target_type}]",
                        'position': pg,
                        'position_group': pg,
                        'origin_school': 'TBD',
                        'projected_nil': round(projected_nil, 2),
                        'fit_score': round(0.9 - rank * 0.1, 3),
                        'win_impact': round(win_impact, 3),
                        'value_rating': round(win_impact / (projected_nil / 1000000), 4),
                        'priority_rank': rank,
                        'reasoning': f"{target_type} for position of need",
                    })

        # Sort by value rating
        shopping_list.sort(key=lambda x: x['value_rating'], reverse=True)

        # Calculate projected improvement
        affordable_targets = [t for t in shopping_list if t['projected_nil'] <= budget_remaining]
        projected_improvement = sum(t['win_impact'] for t in affordable_targets[:5])

        # Budget strategy
        if budget_remaining > 2000000:
            strategy = "Pursue 1-2 elite transfers plus quality depth"
        elif budget_remaining > 500000:
            strategy = "Focus on high-value mid-tier transfers at positions of need"
        else:
            strategy = "Target undervalued depth pieces with upside"

        return {
            'shopping_list': shopping_list,
            'positions_prioritized': positions_prioritized,
            'position_needs': position_needs,
            'budget_remaining': budget_remaining,
            'budget_strategy': strategy,
            'total_targets': len(shopping_list),
            'affordable_targets': len(affordable_targets),
            'projected_improvement': round(projected_improvement, 2),
            'school': school,
            'school_tier': school_tier,
        }

    def recruiting_roi_analysis(
        self,
        school: str,
        recruiting_history: pd.DataFrame,
        player_outcomes: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Analyze historical recruiting ROI by star rating and position.

        Args:
            school: School for context
            recruiting_history: Historical recruiting data with star ratings
            player_outcomes: Player career outcomes (production, draft, NIL)

        Returns:
            Dict containing:
            - roi_by_stars: ROI metrics for each star rating (5/4/3/2)
            - roi_by_position: ROI metrics by position group
            - top_value_picks: Best ROI recruits historically
            - underperformers: Recruits that didn't meet projection
            - recommendations: Strategic recruiting recommendations
            - hit_rates: Percentage of recruits meeting expectations by star
        """
        if recruiting_history.empty or player_outcomes.empty:
            return {
                'roi_by_stars': {},
                'roi_by_position': {},
                'top_value_picks': [],
                'underperformers': [],
                'recommendations': ["Insufficient data for analysis"],
                'hit_rates': {},
            }

        # Merge recruiting with outcomes
        merged = recruiting_history.merge(
            player_outcomes,
            on=['player_name', 'school'],
            how='inner',
            suffixes=('_recruit', '_outcome')
        )

        if merged.empty:
            return {
                'roi_by_stars': {},
                'roi_by_position': {},
                'top_value_picks': [],
                'underperformers': [],
                'recommendations': ["No matching player data found"],
                'hit_rates': {},
            }

        # Expected value by star rating (based on industry data)
        expected_value_by_stars = {
            5: {'hit_rate': 0.80, 'expected_nil': 500000, 'expected_draft': 0.50},
            4: {'hit_rate': 0.55, 'expected_nil': 150000, 'expected_draft': 0.25},
            3: {'hit_rate': 0.30, 'expected_nil': 50000, 'expected_draft': 0.08},
            2: {'hit_rate': 0.10, 'expected_nil': 20000, 'expected_draft': 0.02},
        }

        # Calculate ROI by stars
        roi_by_stars = {}
        hit_rates = {}

        for stars in [5, 4, 3, 2]:
            star_players = merged[merged['stars'] == stars]
            if star_players.empty:
                continue

            expected = expected_value_by_stars.get(stars, {})

            # Calculate actual outcomes
            actual_nil = star_players['nil_value'].mean() if 'nil_value' in star_players else 0
            actual_draft_rate = (star_players['was_drafted'].sum() / len(star_players)
                                if 'was_drafted' in star_players else 0)

            # Determine "hit" (met expectations)
            hits = 0
            for _, player in star_players.iterrows():
                player_nil = player.get('nil_value', 0)
                player_drafted = player.get('was_drafted', False)

                # Hit if NIL >= expected OR drafted (for high stars)
                if player_nil >= expected.get('expected_nil', 0) * 0.5:
                    hits += 1
                elif player_drafted and stars >= 4:
                    hits += 1

            actual_hit_rate = hits / len(star_players)

            # ROI calculation
            # ROI = (actual value - expected cost) / expected cost
            # Using NIL as proxy for value
            expected_cost = {5: 300000, 4: 100000, 3: 30000, 2: 10000}.get(stars, 20000)
            roi = (actual_nil - expected_cost) / max(expected_cost, 1)

            roi_by_stars[stars] = {
                'count': len(star_players),
                'avg_nil_value': round(actual_nil, 2),
                'draft_rate': round(actual_draft_rate, 3),
                'hit_rate': round(actual_hit_rate, 3),
                'expected_hit_rate': expected.get('hit_rate', 0.3),
                'roi': round(roi, 3),
                'vs_expected': 'above' if actual_hit_rate > expected.get('hit_rate', 0.3) else 'below',
            }

            hit_rates[f"{stars}_star"] = round(actual_hit_rate, 3)

        # Calculate ROI by position
        roi_by_position = {}

        for _, row in merged.iterrows():
            pg = self._get_position_group(row.get('position', 'OTHER'))
            if pg not in roi_by_position:
                roi_by_position[pg] = {'count': 0, 'total_nil': 0, 'drafted': 0}
            roi_by_position[pg]['count'] += 1
            roi_by_position[pg]['total_nil'] += row.get('nil_value', 0)
            roi_by_position[pg]['drafted'] += 1 if row.get('was_drafted', False) else 0

        for pg in roi_by_position:
            count = roi_by_position[pg]['count']
            roi_by_position[pg] = {
                'count': count,
                'avg_nil': round(roi_by_position[pg]['total_nil'] / max(count, 1), 2),
                'draft_rate': round(roi_by_position[pg]['drafted'] / max(count, 1), 3),
            }

        # Find top value picks (exceeded expectations most)
        merged['value_score'] = (
            merged.get('nil_value', 0) / 100000 +
            merged.get('was_drafted', False).astype(int) * 2 -
            merged.get('stars', 3) * 0.5
        )

        top_value = merged.nlargest(10, 'value_score')
        top_value_picks = [
            {
                'player': row.get('player_name', 'Unknown'),
                'position': row.get('position', 'UNKNOWN'),
                'stars': row.get('stars', 3),
                'nil_value': row.get('nil_value', 0),
                'was_drafted': row.get('was_drafted', False),
                'value_score': round(row['value_score'], 2),
            }
            for _, row in top_value.iterrows()
        ]

        # Find underperformers
        merged['underperform_score'] = (
            merged.get('stars', 3) * 2 -
            merged.get('nil_value', 0) / 100000 -
            merged.get('was_drafted', False).astype(int) * 3
        )

        underperformers_df = merged[merged['stars'] >= 4].nlargest(5, 'underperform_score')
        underperformers = [
            {
                'player': row.get('player_name', 'Unknown'),
                'position': row.get('position', 'UNKNOWN'),
                'stars': row.get('stars', 4),
                'nil_value': row.get('nil_value', 0),
                'was_drafted': row.get('was_drafted', False),
            }
            for _, row in underperformers_df.iterrows()
        ]

        # Generate recommendations
        recommendations = []

        # Best star value
        best_star_roi = max(roi_by_stars.items(), key=lambda x: x[1]['roi'], default=(None, {}))
        if best_star_roi[0]:
            recommendations.append(
                f"{best_star_roi[0]}-star recruits showing best ROI "
                f"({best_star_roi[1]['roi']:.1%}) - consider focusing resources here"
            )

        # Position-specific insights
        best_position = max(roi_by_position.items(),
                          key=lambda x: x[1]['draft_rate'], default=(None, {}))
        if best_position[0]:
            recommendations.append(
                f"{best_position[0]} position showing highest draft rate "
                f"({best_position[1]['draft_rate']:.1%}) - recruiting strength"
            )

        # Hit rate analysis
        for stars, data in roi_by_stars.items():
            if data['hit_rate'] < data['expected_hit_rate'] * 0.7:
                recommendations.append(
                    f"{stars}-star development below expectations - "
                    f"review player development program"
                )

        if not recommendations:
            recommendations.append("Recruiting ROI in line with expectations")

        return {
            'roi_by_stars': roi_by_stars,
            'roi_by_position': roi_by_position,
            'top_value_picks': top_value_picks,
            'underperformers': underperformers,
            'recommendations': recommendations,
            'hit_rates': hit_rates,
            'school': school,
            'total_recruits_analyzed': len(merged),
        }

    def full_roster_report(
        self,
        school: str,
        roster_df: Optional[pd.DataFrame] = None,
        nil_budget: Optional[float] = None,
        recruiting_history: Optional[pd.DataFrame] = None,
        player_outcomes: Optional[pd.DataFrame] = None,
        portal_players_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive roster report combining all model insights.

        Saves report to outputs/reports/{school}_full_roster_report.json and .csv

        Args:
            school: School name
            roster_df: Current roster data
            nil_budget: Total NIL budget (estimated if not provided)
            recruiting_history: Historical recruiting for ROI analysis
            player_outcomes: Player outcomes for ROI analysis
            portal_players_df: Available portal players

        Returns:
            Dict containing all analysis sections plus file paths
        """
        report = {
            'school': school,
            'school_tier': self._get_school_tier(school),
            'generated_at': datetime.now().isoformat(),
            'sections': {},
        }

        # Estimate budget if not provided
        school_tier = self._get_school_tier(school)
        if nil_budget is None:
            tier_budgets = {
                'blue_blood': 15000000,
                'elite': 8000000,
                'power_brand': 5000000,
                'p4_mid': 3000000,
                'g5_strong': 1500000,
                'g5': 500000,
            }
            nil_budget = tier_budgets.get(school_tier, 3000000)

        report['nil_budget'] = nil_budget

        # Create empty roster if not provided
        if roster_df is None or roster_df.empty:
            roster_df = pd.DataFrame()
            report['sections']['roster_summary'] = {
                'total_players': 0,
                'message': 'No roster data provided',
            }
        else:
            # Roster summary
            position_breakdown = {}
            for _, row in roster_df.iterrows():
                pg = self._get_position_group(row.get('position', 'OTHER'))
                position_breakdown[pg] = position_breakdown.get(pg, 0) + 1

            report['sections']['roster_summary'] = {
                'total_players': len(roster_df),
                'position_breakdown': position_breakdown,
                'avg_rating': round(roster_df.get('overall_rating', pd.Series([0.75])).mean(), 3),
            }

        # NIL Budget Optimization
        if not roster_df.empty:
            budget_optimization = self.optimize_nil_budget(
                school=school,
                total_budget=nil_budget,
                roster_df=roster_df,
            )
            report['sections']['nil_optimization'] = budget_optimization
        else:
            report['sections']['nil_optimization'] = {
                'message': 'Requires roster data',
            }

        # Portal Shopping List
        if not roster_df.empty:
            budget_remaining = budget_optimization.get('budget_remaining', nil_budget * 0.3)
            shopping_list = self.portal_shopping_list(
                school=school,
                roster_df=roster_df,
                budget_remaining=budget_remaining,
                portal_players_df=portal_players_df,
            )
            report['sections']['portal_shopping'] = shopping_list
        else:
            report['sections']['portal_shopping'] = {
                'message': 'Requires roster data',
            }

        # Team Flight Risk Analysis
        if not roster_df.empty and self.portal_predictor is not None:
            try:
                flight_report = self.portal_predictor.team_flight_risk_report(
                    roster_df, school
                )
                report['sections']['flight_risk'] = flight_report
            except Exception as e:
                report['sections']['flight_risk'] = {
                    'message': f'Analysis error: {str(e)}',
                }
        else:
            report['sections']['flight_risk'] = {
                'message': 'Requires roster data and trained portal predictor',
            }

        # Win Projection
        if not roster_df.empty and self.win_model is not None:
            try:
                win_projection = self.win_model.predict_team_wins(roster_df, school)
                report['sections']['win_projection'] = win_projection
            except Exception as e:
                report['sections']['win_projection'] = {
                    'message': f'Analysis error: {str(e)}',
                }
        else:
            report['sections']['win_projection'] = {
                'message': 'Requires roster data and trained win model',
            }

        # Roster Gap Analysis
        if not roster_df.empty and self.win_model is not None:
            try:
                gap_analysis = self.win_model.roster_gap_analysis(roster_df, school)
                report['sections']['gap_analysis'] = gap_analysis
            except Exception as e:
                report['sections']['gap_analysis'] = {
                    'message': f'Analysis error: {str(e)}',
                }
        else:
            report['sections']['gap_analysis'] = {
                'message': 'Requires roster data and trained win model',
            }

        # Recruiting ROI Analysis
        if recruiting_history is not None and player_outcomes is not None:
            roi_analysis = self.recruiting_roi_analysis(
                school=school,
                recruiting_history=recruiting_history,
                player_outcomes=player_outcomes,
            )
            report['sections']['recruiting_roi'] = roi_analysis
        else:
            report['sections']['recruiting_roi'] = {
                'message': 'Requires recruiting history and player outcomes data',
            }

        # Executive Summary
        summary_points = []

        if 'nil_optimization' in report['sections']:
            opt = report['sections']['nil_optimization']
            if 'expected_wins' in opt:
                summary_points.append(
                    f"Projected wins with optimized NIL: {opt['expected_wins']}"
                )
            if 'retention_priority' in opt and opt['retention_priority']:
                summary_points.append(
                    f"Critical retention targets: {', '.join(opt['retention_priority'][:3])}"
                )

        if 'portal_shopping' in report['sections']:
            ps = report['sections']['portal_shopping']
            if 'positions_prioritized' in ps:
                summary_points.append(
                    f"Top position needs: {', '.join(ps['positions_prioritized'][:3])}"
                )
            if 'projected_improvement' in ps:
                summary_points.append(
                    f"Portal upside: +{ps['projected_improvement']} wins"
                )

        if 'flight_risk' in report['sections']:
            fr = report['sections']['flight_risk']
            if 'total_at_risk' in fr:
                summary_points.append(
                    f"Players at flight risk: {fr['total_at_risk']}"
                )

        report['executive_summary'] = summary_points if summary_points else [
            "Report generated with limited data - provide roster for full analysis"
        ]

        # Save reports
        output_dir = Path('outputs/reports')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Clean school name for filename
        school_slug = school.lower().replace(' ', '_').replace("'", '')

        # Save JSON
        json_path = output_dir / f'{school_slug}_full_roster_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save CSV summary
        csv_path = output_dir / f'{school_slug}_full_roster_report.csv'

        # Flatten key metrics for CSV
        csv_rows = []

        # Add allocation rows if available
        if 'nil_optimization' in report['sections']:
            allocations = report['sections']['nil_optimization'].get('allocations', [])
            for alloc in allocations:
                csv_rows.append({
                    'section': 'nil_allocation',
                    'player': alloc.get('player', ''),
                    'position': alloc.get('position', ''),
                    'value': alloc.get('recommended_nil', 0),
                    'metric': 'recommended_nil',
                    'notes': alloc.get('reasoning', ''),
                })

        # Add shopping list rows
        if 'portal_shopping' in report['sections']:
            targets = report['sections']['portal_shopping'].get('shopping_list', [])
            for target in targets:
                csv_rows.append({
                    'section': 'portal_target',
                    'player': target.get('player', ''),
                    'position': target.get('position', ''),
                    'value': target.get('projected_nil', 0),
                    'metric': 'projected_nil',
                    'notes': target.get('reasoning', ''),
                })

        if csv_rows:
            csv_df = pd.DataFrame(csv_rows)
            csv_df.to_csv(csv_path, index=False)
        else:
            # Empty CSV with headers
            pd.DataFrame(columns=['section', 'player', 'position', 'value', 'metric', 'notes']).to_csv(
                csv_path, index=False
            )

        report['output_files'] = {
            'json': str(json_path),
            'csv': str(csv_path),
        }

        logger.info(f"Full roster report saved: {json_path}")

        return report
