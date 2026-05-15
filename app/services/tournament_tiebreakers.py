"""FIFA tournament group tiebreaker chain.

Resolves a tied set of teams using a configurable sequence of rules:

  Step 1: Total group points
  Step 2: Goal difference (all group games)
  Step 3: Goals scored (all group games)
  Step 4: H2H points among tied teams only
  Step 5: H2H goal difference among tied teams only
  Step 6: H2H goals scored among tied teams only
  Step 7: Fair-play points (skipped per 2026-05-15 decision)
  Step 8: Drawing of lots (random)

The chain is recursive: after each step the tied teams are re-grouped by
the step's key, and any sub-group still tied falls through to the next
step. H2H steps (4-6) only count fixtures *between the currently-tied
teams*, so a 3-way tie that partially resolves after step 4 has steps 5-6
re-computed for the remaining sub-tie among just those teams.

Used by tournament_simulation_service.py inside the per-sim group stage.
"""
import random
from typing import Callable, List


def resolve_group_order(
    teams: List[int],
    fixtures: list,
    tiebreaker_chain: List[int],
) -> List[int]:
    """Sort teams by the tiebreaker chain, best first.

    teams: list of team_ids in the tied group
    fixtures: list of dicts with keys home_id, away_id, home_goals, away_goals
              (only fixtures involving teams in `teams` are needed; engine
              filters to the group before calling)
    tiebreaker_chain: ordered list of step IDs (1-8) to apply
    """
    return _resolve(teams, fixtures, tiebreaker_chain, step_idx=0)


def _resolve(teams, fixtures, chain, step_idx):
    if len(teams) <= 1:
        return list(teams)
    if step_idx >= len(chain):
        # Exhausted all steps including step 8 (lots) → final random shuffle.
        return random.sample(teams, len(teams))

    step = chain[step_idx]
    if step == 8:
        # Drawing of lots — random ordering, terminates the chain.
        return random.sample(teams, len(teams))

    key_fn = _step_key_fn(step, teams, fixtures)
    pairs = sorted(((t, key_fn(t)) for t in teams), key=lambda x: x[1], reverse=True)

    result = []
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        tied = [t for t, _ in pairs[i:j]]
        if len(tied) == 1:
            result.append(tied[0])
        else:
            result.extend(_resolve(tied, fixtures, chain, step_idx + 1))
        i = j
    return result


def _step_key_fn(step: int, teams_in_scope: List[int], fixtures: list) -> Callable:
    """Return a key function team_id -> sort key for the given tiebreaker step.

    Steps 1-3 use ALL group fixtures (regardless of whether the opponent is
    also tied). Steps 4-6 restrict to fixtures between teams in scope.
    """
    if step in (1, 2, 3):
        scope_fixtures = fixtures
    elif step in (4, 5, 6):
        scope_fixtures = [f for f in fixtures
                          if f['home_id'] in teams_in_scope
                          and f['away_id'] in teams_in_scope]
    else:
        raise ValueError(f"Unsupported tiebreaker step: {step}")

    if step in (1, 4):
        return lambda t: _points(t, scope_fixtures)
    if step in (2, 5):
        return lambda t: _gd(t, scope_fixtures)
    if step in (3, 6):
        return lambda t: _gf(t, scope_fixtures)


def _points(team_id: int, fixtures: list) -> int:
    """3 points for a win, 1 for a draw, 0 for a loss, summed across fixtures."""
    pts = 0
    for f in fixtures:
        if team_id not in (f['home_id'], f['away_id']):
            continue
        is_home = f['home_id'] == team_id
        gf = f['home_goals'] if is_home else f['away_goals']
        ga = f['away_goals'] if is_home else f['home_goals']
        if gf > ga: pts += 3
        elif gf == ga: pts += 1
    return pts


def _gd(team_id: int, fixtures: list) -> int:
    """Goal difference across fixtures."""
    diff = 0
    for f in fixtures:
        if team_id not in (f['home_id'], f['away_id']):
            continue
        is_home = f['home_id'] == team_id
        gf = f['home_goals'] if is_home else f['away_goals']
        ga = f['away_goals'] if is_home else f['home_goals']
        diff += gf - ga
    return diff


def _gf(team_id: int, fixtures: list) -> int:
    """Goals scored across fixtures."""
    scored = 0
    for f in fixtures:
        if team_id not in (f['home_id'], f['away_id']):
            continue
        is_home = f['home_id'] == team_id
        scored += f['home_goals'] if is_home else f['away_goals']
    return scored
