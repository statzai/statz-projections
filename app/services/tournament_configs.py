"""Per-tournament configuration for the Monte Carlo tournament simulator.

A TournamentConfig describes the structure of a competition (group sizes,
qualifier counts, knockout round sequence, ET / pens rules) so the generic
engine in tournament_simulation_service.py can simulate any tournament
without code changes.

Adding a new tournament = new config below; no engine changes required.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TournamentConfig:
    """All knobs the simulator needs for one tournament edition."""
    name: str
    competition_id: int
    season_id: int

    # Group stage
    num_groups: int                                  # e.g. 12 for WC 2026, 6 for Euros 2024
    teams_per_group: int                             # almost always 4
    advance_per_group: int                           # top N go through (almost always 2)
    best_thirds_advance: int                         # extra 3rd-place qualifiers (8 WC2026, 4 Euros)

    # Knockout — ordered list of round names (used as keys in tournament_projections columns)
    knockout_rounds: List[str]                       # e.g. ['r32','r16','qf','sf','final'] WC2026
                                                     #      ['r16','qf','sf','final'] Euros 2024
    has_third_place_playoff: bool = False            # WC yes, Euros no

    # Extra time + penalties for drawn knockout matches
    et_minutes: int = 30                             # 2 × 15 min halves
    et_lambda_factor: float = 30 / 90                # pro-rate regulation λ to ET duration
    pens_p_favourite: float = 0.51                   # mild edge to favourite in shootout

    # FIFA tiebreaker chain to apply within a group. Step IDs:
    #   1 = group points, 2 = GD, 3 = GF
    #   4 = H2H points, 5 = H2H GD, 6 = H2H GF
    #   7 = fair-play (skipped per 2026-05-15 decision)
    #   8 = drawing of lots (random)
    # NOTE: this default is the PRE-2026 FIFA order (overall GD before H2H).
    # FIFA WC 2026 reversed it (H2H first) — see WC_2026's explicit override.
    # Any new modern-tournament config should set the chain explicitly.
    group_tiebreaker_chain: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8])

    # Cross-group ranking (for best-thirds selection) — H2H doesn't apply
    # across different groups, so we drop steps 4-6.
    cross_group_tiebreaker_chain: List[int] = field(default_factory=lambda: [1, 2, 3, 8])

    # Optional explicit team → group code mapping. When None the engine
    # derives groups from `fixtures.group_id` (sorted ascending → A, B, C…).
    # Useful for tournaments where fixtures aren't loaded but group draws
    # are known.
    group_code_override: Optional[dict] = None


# ---------------------------------------------------------------------------
# FIFA World Cup 2026 — 48 teams, 12 groups of 4, top 2 + 8 best-thirds = 32
# qualify to R32, then standard single-elimination knockout.
# ---------------------------------------------------------------------------
WC_2026 = TournamentConfig(
    name='World Cup 2026',
    competition_id=732,
    season_id=26618,
    num_groups=12,
    teams_per_group=4,
    advance_per_group=2,
    best_thirds_advance=8,
    knockout_rounds=['r32', 'r16', 'qf', 'sf', 'final'],
    has_third_place_playoff=True,
    # FIFA reversed the group tiebreaker order for 2026 (Art. 13): head-to-head
    # (pts→GD→goals) now comes BEFORE overall goal difference, unlike 1998-2022.
    # Step IDs: 1=pts, 4/5/6=H2H pts/GD/goals, 2/3=group GD/goals, 8=lots.
    group_tiebreaker_chain=[1, 4, 5, 6, 2, 3, 8],
    # Cross-group (best-thirds) ranking: H2H meaningless (teams never met) →
    # pts→groupGD→groupGoals→lots, unchanged for 2026.
    cross_group_tiebreaker_chain=[1, 2, 3, 8],
)


# ---------------------------------------------------------------------------
# Euros 2024 reference config (not used now — keeps the pattern visible for
# when we extend to Euros 2028).
# ---------------------------------------------------------------------------
# EURO_2024 = TournamentConfig(
#     name='UEFA Euro 2024',
#     competition_id=1326,
#     season_id=...,
#     num_groups=6,
#     teams_per_group=4,
#     advance_per_group=2,
#     best_thirds_advance=4,
#     knockout_rounds=['r16', 'qf', 'sf', 'final'],
#     has_third_place_playoff=False,
# )


# ---------------------------------------------------------------------------
# Registry: which comps have a bracket worth simulating, and what config.
# Read by InternationalProjectionService at scope-construction time (or by
# any code that needs to ask "does this comp have a bracket?").
# Friendlies / qualifiers / Nations League group stage have no bracket and
# are deliberately absent here — their scope.bracket_config stays None, so
# Step 3 (tournament simulator) skips for them.
# ---------------------------------------------------------------------------
BRACKET_CONFIGS = {
    732: WC_2026,
    # 1326: EURO_2028,             # uncomment when bracket drawn
    # 1114: COPA_2026,
    # 1117: AFCON_2025,
    # 1538: NATIONS_LEAGUE_FINALS, # if/when modelled
}


def get_config_for_comp(competition_id: int) -> Optional[TournamentConfig]:
    """Return the TournamentConfig for the given comp, or None if the comp
    has no bracket to simulate."""
    return BRACKET_CONFIGS.get(competition_id)


def has_bracket(competition_id: int) -> bool:
    return competition_id in BRACKET_CONFIGS
