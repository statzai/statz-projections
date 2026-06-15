"""WC Fantasy point projections — Step 6 of InternationalProjectionService.

Reads the per-stat WC player projections that wc_player_stat_service.py
wrote (in long format, into `player_projections` for competition 732),
the matching opponent goal expectations from `fixture_projections`, and
the wc_rounds windows; combines them into a single per-(fixture, player)
fantasy_points expected value using the 2026 FIFA WC Fantasy scoring rules,
and writes to `wc_fantasy_projections`.

Idempotent: DELETE all comp-732 rows from wc_fantasy_projections, then
bulk-insert the new batch. Mirrors how the player + prop projections are
refreshed.

Scoring rules (FIFA 2026 — different from FPL):
  - Appearance:    +1 (any minutes), +1 (60+ min) — assume starter → +2
  - Goals:         GK +9, DEF +7, MID +6, FWD +5
  - Clean sheet:   GK +5, DEF +5, MID +1, FWD 0
  - Assist:        +3 (all positions)
  - Yellow card:   -1 (all)
  - Saves:         GK only — every 3 = +1
  - Tackles:       MID only — every 3 = +1
  - Key passes:    MID only — every 2 = +1
  - Shots on tgt:  FWD only — every 2 = +1

Out of scope for v1 (would need data we don't have or compute):
  - Red card / own goal / won-penalty / conceded-penalty (rare, not projected)
  - Direct-FK-goal +1 bonus (not projected)
  - Scouting Bonus (+2 if scored >4 AND owned <5%) — dynamic, applied post-fact
  - Qualification Booster chip (+2 per starting-XI player whose team advances)
    — applied by the planner when the chip is active, not baked into base xPts

Knockout-round handling: knockout fixtures don't exist in `fixtures` until
the bracket is drawn at end of group stage. While they're missing, this
service writes group-stage rows only. Once knockout fixtures appear we'll
also want to weight by P(team plays this round) — see the tournament
simulator's `tournament_projections` table. Hook left as a TODO.

Threshold maths ("every K = +1"):
  E[floor(X/K)] where X ~ Poisson(λ = expected value of the stat).
  Implemented in `_threshold_points` — same discrete-Poisson approach as
  statz_functions.get_fpl_points uses for `saves_points`, just generalised.
"""
import logging
from datetime import datetime
from typing import Dict, Tuple

from scipy.stats import poisson

from app.repository.wc_fantasy_repo import insert_wc_fantasy_projections_async
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("wc_fantasy_points")

# fixture_player_stats stats_type_ids we read out of player_projections.
# Mirrors player_repo.STATUS_TYPES — only the ones we need to score.
STAT_GOALS = 52
STAT_ASSISTS = 79
STAT_YELLOW_CARDS = 84
STAT_SAVES = 57
STAT_TACKLES = 78
STAT_KEY_PASSES = 117
STAT_SHOTS_ON_TARGET = 86

# Per-position scoring tables. Each value applies as `expected_value × pts`
# for continuous stats (goals, assists, yellows). Threshold stats (saves,
# tackles, key passes, shots-on-target) use the Poisson discretisation in
# `_threshold_points`.
GOAL_PTS = {'GK': 9, 'DEF': 7, 'MID': 6, 'FWD': 5}
CLEAN_SHEET_PTS = {'GK': 5, 'DEF': 5, 'MID': 1, 'FWD': 0}
ASSIST_PTS = 3
YELLOW_CARD_PTS = -1
APPEARANCE_PTS = 2   # +1 starts, +1 60+ min — assume starter both ways
SAVES_PER_POINT = 3
TACKLES_PER_POINT = 3
KEY_PASSES_PER_POINT = 2
SHOTS_PER_POINT = 2


def _threshold_points(lam: float, k: int, max_n: int = 15) -> float:
    """Expected value of floor(X / K) where X ~ Poisson(λ = lam).

    Discrete-Poisson form that matches the FPL `saves_points` block in
    statz_functions.get_fpl_points — slightly more accurate than λ/K for
    the low-λ region where bunching matters. max_n=15 covers >99.99% of
    the mass for any λ we care about (saves ~3, tackles ~2, KP ~1, SoT ~1).
    """
    if lam <= 0:
        return 0.0
    total = 0.0
    for n in range(k, max_n + 1):
        total += poisson.pmf(n, lam) * (n // k)
    return total


def _clean_sheet_prob(opp_expected_goals: float) -> float:
    """P(opponent scores 0) under Poisson(λ = opp_expected_goals)."""
    if opp_expected_goals <= 0:
        return 1.0
    return float(poisson.pmf(0, opp_expected_goals))


def _fantasy_points(stats: Dict[int, float], position: str, opp_goals: float) -> float:
    """Apply the WC scoring rules to one player-fixture's stat bag.

    `stats` is `{stat_type_id: expected_value}`. Missing stats default to 0.
    `position` is one of GK/DEF/MID/FWD. `opp_goals` is the opponent's
    expected goals for the fixture (clean-sheet calc input).
    """
    if position not in GOAL_PTS:
        # Unknown / unsupported position — skip rather than crash. Will
        # show up in the row count delta.
        return 0.0

    g = stats.get(STAT_GOALS, 0.0)
    a = stats.get(STAT_ASSISTS, 0.0)
    y = stats.get(STAT_YELLOW_CARDS, 0.0)

    pts = APPEARANCE_PTS
    pts += g * GOAL_PTS[position]
    pts += a * ASSIST_PTS
    pts += y * YELLOW_CARD_PTS
    pts += _clean_sheet_prob(opp_goals) * CLEAN_SHEET_PTS[position]

    if position == 'GK':
        pts += _threshold_points(stats.get(STAT_SAVES, 0.0), SAVES_PER_POINT)
    if position == 'MID':
        pts += _threshold_points(stats.get(STAT_TACKLES, 0.0), TACKLES_PER_POINT)
        pts += _threshold_points(stats.get(STAT_KEY_PASSES, 0.0), KEY_PASSES_PER_POINT)
    if position == 'FWD':
        pts += _threshold_points(stats.get(STAT_SHOTS_ON_TARGET, 0.0), SHOTS_PER_POINT)

    return round(pts, 2)


async def _load_data(conn, competition_id: int, fixture_ids_filter=None) -> dict:
    """Pull everything in one go.

    - long-format `player_projections` rows for WC (filtered to future fixtures)
    - fixture-level opponent goals for clean sheet (from `fixture_projections`)
    - wc_rounds windows for fixture → round_id mapping (uses kickoff in [start, end))

    fixture_ids_filter: optional list — narrows SELECTs to those fixtures.
    """
    # Pooled conn might carry a stale snapshot from the prior step.
    await conn.rollback()

    # Build the optional fixture-id filter once.
    fp_fid_filter_sql = ""
    fp_fid_filter_params: tuple = ()
    if fixture_ids_filter:
        ph_fp = ",".join(["%s"] * len(fixture_ids_filter))
        fp_fid_filter_sql = f" AND f.id IN ({ph_fp})"
        fp_fid_filter_params = tuple(fixture_ids_filter)

    async with conn.cursor() as cur:
        # Long-format player projection rows (future fixtures only).
        await cur.execute(
            f"""
            SELECT pp.fixture_id, pp.player_id, pp.position, pp.team_id,
                   pp.opponent_id, pp.venue, pp.kickoff_datetime,
                   pp.stats_type_id, pp.stats_value
            FROM player_projections pp
            JOIN fixtures f ON f.id = pp.fixture_id
            WHERE f.competition_id = %s
              AND f.kickoff_datetime > NOW()
              {fp_fid_filter_sql}
            """,
            (competition_id,) + fp_fid_filter_params,
        )
        pp_rows = await cur.fetchall()

        # Fixture-level expected goals — for clean sheet calc, we need the
        # OPPONENT's expected goals (which is home_goals when player is away,
        # away_goals when player is home).
        await cur.execute(
            f"""
            SELECT fp.fixture_id, fp.home_team_id, fp.away_team_id,
                   fp.home_goals, fp.away_goals
            FROM fixture_projections fp
            JOIN fixtures f ON f.id = fp.fixture_id
            WHERE f.competition_id = %s
              AND f.kickoff_datetime > NOW()
              {fp_fid_filter_sql}
            """,
            (competition_id,) + fp_fid_filter_params,
        )
        fp_rows = await cur.fetchall()

        # Fixture → round mapping from `wc_fixtures.round_id` (FIFA's
        # authoritative pre-assignment). Used to be wc_rounds windows by
        # kickoff_datetime, but FIFA's wc_rounds.start_date is the lineup-
        # lock deadline (typically 1h post-first-fixture-kickoff) which
        # left opening fixtures of each round outside their own window →
        # NULL wc_round_id. Direct lookup avoids the window mismatch and
        # leaves wc_rounds.start_date semantics intact for the planner.
        await cur.execute(
            "SELECT fixture_id, round_id FROM wc_fixtures WHERE fixture_id IS NOT NULL"
        )
        round_rows = await cur.fetchall()

    # Pivot player_projections from long → {(fixture_id, player_id): row dict}.
    by_pair: Dict[Tuple[int, int], dict] = {}
    for fid, pid, pos, tid, oid, venue, ko, sid, val in pp_rows:
        key = (fid, pid)
        entry = by_pair.setdefault(key, {
            'fixture_id': fid,
            'player_id': pid,
            'position': pos,
            'team_id': tid,
            'opponent_id': oid,
            'venue': venue,
            'kickoff_datetime': ko,
            'stats': {},
        })
        # Last write wins on duplicate (shouldn't happen — unique key on
        # (fixture, player, stats_type)).
        if val is not None:
            try:
                entry['stats'][int(sid)] = float(val)
            except (TypeError, ValueError):
                pass

    # Fixture lookup → opponent goals per player side.
    fix_meta: Dict[int, dict] = {}
    for fid, ht, at, hg, ag in fp_rows:
        try:
            home_goals = float(hg) if hg is not None else 0.0
            away_goals = float(ag) if ag is not None else 0.0
        except (TypeError, ValueError):
            home_goals = away_goals = 0.0
        fix_meta[fid] = {
            'home_team_id': ht,
            'away_team_id': at,
            'home_goals': home_goals,
            'away_goals': away_goals,
        }

    # {statz_fixture_id: wc_round_id} — direct lookup. Empty rows mean
    # wc_fixtures hasn't been ingested yet; downstream uses .get() so a
    # missing entry just leaves wc_round_id as None.
    round_by_fixture = {fid: rid for fid, rid in round_rows if fid is not None}

    return {
        'players': by_pair,
        'fixtures': fix_meta,
        'round_by_fixture': round_by_fixture,
    }


def _round_for(fixture_id, round_by_fixture) -> int:
    """Look up wc_round_id for a Statz fixture id via the wc_fixtures.round_id
    mapping. Returns None when not found (knockout fixture before bracket
    draw, or wc_fixtures not yet ingested).
    """
    return round_by_fixture.get(fixture_id)


def _build_rows(data: dict) -> list:
    """Pure function: stats bag + opponent goals + round window → INSERT rows."""
    players = data['players']
    fixtures = data['fixtures']
    round_by_fixture = data['round_by_fixture']

    out = []
    for (fid, pid), entry in players.items():
        position = entry['position']
        team_id = entry['team_id']

        meta = fixtures.get(fid)
        if meta is None:
            # No fixture_projections row — can't compute clean sheet. Skip.
            continue

        if team_id == meta['home_team_id']:
            opp_goals = meta['away_goals']
        elif team_id == meta['away_team_id']:
            opp_goals = meta['home_goals']
        else:
            # Player team doesn't match fixture sides (data drift). Fall back
            # to the higher of the two so clean sheet stays conservative.
            opp_goals = max(meta['home_goals'], meta['away_goals'])

        pts = _fantasy_points(entry['stats'], position, opp_goals)
        wc_round_id = _round_for(fid, round_by_fixture)

        out.append((
            fid,
            pid,
            entry['kickoff_datetime'],
            entry['venue'],
            pts,
            wc_round_id,
            team_id,
            entry['opponent_id'],
            position,
        ))

    return out


class WcFantasyPointsService:
    """Compute + write per-(fixture, player) WC Fantasy point projections.

    FIFA-WC-2026-specific scoring rules. Gated by `scope.fantasy_rules ==
    'fifa_wc_2026'` in the orchestrator — non-WC scopes don't reach this
    service. Takes a scope for symmetry with the other sub-services; only
    scope.competition_id is consumed (drives the comp filter in SQL).
    """

    def __init__(self, scope=None):
        if scope is None:
            from app.services.international_projection_service import INTL_SCOPES
            scope = INTL_SCOPES['World Cup']
        self.scope = scope

    async def project(self, commit: bool = True, fixture_ids: list = None) -> dict:
        """fixture_ids: optional — when set, scope projection + DELETE to those fixtures only."""
        logger.info(
            f"{self.scope.competition_name} fantasy points start — "
            f"commit={commit}, fixture_ids={fixture_ids}"
        )

        conn = await get_source_connection()
        try:
            data = await _load_data(
                conn,
                competition_id=self.scope.competition_id,
                fixture_ids_filter=fixture_ids,
            )
            n_players = len(data['players'])
            n_fixtures = len(data['fixtures'])
            n_round_mappings = len(data['round_by_fixture'])
            logger.info(
                f"Loaded {n_players} player-fixture stat bags, {n_fixtures} "
                f"fixture-projection rows, {n_round_mappings} fixture→round mappings"
            )

            if n_players == 0 or n_fixtures == 0:
                logger.warning(
                    "Empty input — run InternationalTeamStatService + WcPlayerStatService first."
                )
                return {
                    'n_rows': 0,
                    'n_player_fixtures': n_players,
                    'n_fixtures': n_fixtures,
                    'committed': False,
                }

            rows = _build_rows(data)
            logger.info(f"WC fantasy points ready: {len(rows)} rows")

            if commit and rows:
                # The INSERT upserts on (fixture_id, player_id), so future
                # fixtures are refreshed in place — no blanket delete needed
                # to keep them current. We delete ONLY to clear projections we
                # no longer want.
                #
                # Rule: a round's projections are kept until the round is
                # FINISHED (all its matches played). The planner shows the
                # in-play round read-only, including projections for fixtures
                # that have already kicked off — so we must NOT prune a fixture
                # just because it's started. Once kickoff passes, the SELECT
                # (kickoff > NOW) stops refreshing that fixture, so its last
                # pre-match projection is simply retained. We only delete rows
                # for rounds whose status is 'complete'.
                #
                # Per-fixture mode still scopes the DELETE to those fixtures.
                async with conn.cursor() as cur:
                    if fixture_ids:
                        del_ph = ",".join(["%s"] * len(fixture_ids))
                        await cur.execute(
                            f"DELETE FROM wc_fantasy_projections WHERE fixture_id IN ({del_ph})",
                            tuple(fixture_ids),
                        )
                    else:
                        await cur.execute(
                            """DELETE wfp FROM wc_fantasy_projections wfp
                               JOIN wc_fixtures wf ON wf.fixture_id = wfp.fixture_id
                               JOIN wc_rounds wr ON wr.id = wf.round_id
                               WHERE wr.status = 'complete'""",
                        )
                await conn.commit()

                await insert_wc_fantasy_projections_async(rows)
                logger.info(f"WC fantasy projections written: {len(rows)} rows")

            return {
                'n_rows': len(rows),
                'n_player_fixtures': n_players,
                'n_fixtures': n_fixtures,
                'committed': commit,
            }
        finally:
            release_source_connection(conn)
