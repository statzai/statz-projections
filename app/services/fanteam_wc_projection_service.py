"""FanTeam WC fantasy point projections.

Sibling service to wc_fantasy_points_service.py — same input data
(wc per-fixture per-player stats from player_projections), different
scoring rules (FanTeam's PL game ruleset, lifted from
statz_functions.get_fanteam_points).

Group fixtures only (future WC fixtures = group stage until the
knockout bracket is drawn). Reads position from
`fanteam_wc_player_mappings.fanteam_position` so the scoring matches
how FanTeam classifies each player (which can differ from FIFA's
classification on the same player).

Idempotent: DELETE rows for the touched competition (or fixtures),
then bulk-insert.
"""
import logging
from typing import Dict, Tuple

from scipy.stats import poisson

from app.repository.fanteam_wc_repo import insert_fanteam_wc_projections_async
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("fanteam_wc_projection")

STAT_GOALS = 52
STAT_ASSISTS = 79
STAT_YELLOW_CARDS = 84
STAT_SAVES = 57
STAT_SHOTS_ON_TARGET = 86

# Scoring tables — lifted verbatim from
# projection_service.py:1865-1881 (PL FanTeam) so the WC version uses
# the exact same ruleset FanTeam pays out under.
FT_PTS = {
    'GK':  {'Goals': 8, 'Assists': 3, 'Shots On Target': 1,   'Saves': 0.5,
            'Penalties Saved': 5, 'Clean Sheet': 4, 'Win': 0.3, 'Lose': -0.3,
            'Goals Conceded': -1, 'Yellow Card': -1},
    'DEF': {'Goals': 6, 'Assists': 3, 'Shots On Target': 0.6, 'Clean Sheet': 4,
            'Win': 0.3, 'Lose': -0.3, 'Goals Conceded': -1, 'Yellow Card': -1},
    'MID': {'Goals': 5, 'Assists': 3, 'Shots On Target': 0.4, 'Clean Sheet': 1,
            'Win': 0.3, 'Lose': -0.3, 'Yellow Card': -1, 'Full Match': 1},
    'FWD': {'Goals': 4, 'Assists': 3, 'Shots On Target': 0.4,
            'Win': 0.3, 'Lose': -0.3, 'Yellow Card': -1, 'Full Match': 1},
}

# Match-outcome probabilities under independent Poisson(λ_home, λ_away).
# Cached for speed — same (h_g, a_g) pair recurs across all players in a
# fixture so we don't want to re-do the double sum 22 times.
def _outcome_probs(lh: float, la: float, n_max: int = 12) -> tuple:
    """Returns (p_home_win, p_draw, p_away_win)."""
    if lh <= 0 and la <= 0:
        return 0.0, 1.0, 0.0
    p_h = [poisson.pmf(h, lh) for h in range(n_max + 1)]
    p_a = [poisson.pmf(a, la) for a in range(n_max + 1)]
    pw = pd = pl = 0.0
    for h in range(n_max + 1):
        for a in range(n_max + 1):
            joint = p_h[h] * p_a[a]
            if h > a:
                pw += joint
            elif h == a:
                pd += joint
            else:
                pl += joint
    return pw, pd, pl


def _gc_points(lam: float) -> float:
    """Tiered GC penalty in Poisson(λ) expectation. Mirror of PL FanTeam
    formula at statz_functions.py:2402-2405:

        E[1{GC∈2-3}] + 2·E[1{GC∈4-5}] + 3·E[1{GC∈6-7}]    (all times -1)
    """
    if lam <= 0:
        return 0.0
    return -1.0 * (
        poisson.pmf(2, lam) + poisson.pmf(3, lam)
        + 2 * (poisson.pmf(4, lam) + poisson.pmf(5, lam))
        + 3 * (poisson.pmf(6, lam) + poisson.pmf(7, lam))
    )


def _clean_sheet_prob(opp_goals: float) -> float:
    if opp_goals <= 0:
        return 1.0
    return float(poisson.pmf(0, opp_goals))


def _fanteam_points(stats: Dict[int, float], position: str,
                    opp_goals: float, own_goals: float,
                    p_win: float, p_loss: float) -> float:
    """Score one player-fixture's stat bag under FanTeam rules."""
    if position not in FT_PTS:
        return 0.0

    table = FT_PTS[position]
    pts = 2.0   # base appearance (+1 + +1 60-min → assume starter)

    g = stats.get(STAT_GOALS, 0.0)
    a = stats.get(STAT_ASSISTS, 0.0)
    y = stats.get(STAT_YELLOW_CARDS, 0.0)
    sot = stats.get(STAT_SHOTS_ON_TARGET, 0.0)

    pts += g * table['Goals']
    pts += a * table['Assists']
    pts += y * table['Yellow Card']
    pts += sot * table['Shots On Target']
    pts += p_win * table['Win']
    pts += p_loss * table['Lose']

    if 'Clean Sheet' in table:
        pts += _clean_sheet_prob(opp_goals) * table['Clean Sheet']
    if 'Goals Conceded' in table:
        # _gc_points already returns the negative penalty (built-in -1
        # multiplier), so add it straight in. table['Goals Conceded']
        # is always -1 today, but kept in the dict for symmetry.
        pts += _gc_points(opp_goals)
    if position == 'GK':
        pts += stats.get(STAT_SAVES, 0.0) * table['Saves']
        # Penalties Saved heuristic from PL FanTeam: ~0.08·opp_goals (10%
        # chance of pen per GC × 16% save rate). Inline rather than
        # extract — tiny term.
        pts += 0.08 * opp_goals * table['Penalties Saved']
    if 'Full Match' in table:
        # Assume starter goes 60+ min: +1.
        pts += 1.0 * table['Full Match']

    return round(pts, 2)


async def _load_data(conn, competition_id: int, fixture_ids_filter=None) -> dict:
    """Pull stat bags + fixture-level expected goals + per-player FanTeam
    position from the mapping table. Future fixtures only (group stage)."""
    await conn.rollback()

    fid_filter_sql = ""
    fid_filter_params: tuple = ()
    if fixture_ids_filter:
        ph = ",".join(["%s"] * len(fixture_ids_filter))
        fid_filter_sql = f" AND f.id IN ({ph})"
        fid_filter_params = tuple(fixture_ids_filter)

    async with conn.cursor() as cur:
        # Stat bags from the per-stat WC projections — same source the FIFA
        # scoring service reads. position column ignored here (we override
        # below from the FanTeam mapping).
        await cur.execute(
            f"""
            SELECT pp.fixture_id, pp.player_id, pp.team_id, pp.opponent_id,
                   pp.venue, pp.kickoff_datetime,
                   pp.stats_type_id, pp.stats_value
            FROM player_projections pp
            JOIN fixtures f ON f.id = pp.fixture_id
            WHERE f.competition_id = %s
              AND f.kickoff_datetime > NOW()
              {fid_filter_sql}
            """,
            (competition_id,) + fid_filter_params,
        )
        pp_rows = await cur.fetchall()

        await cur.execute(
            f"""
            SELECT fp.fixture_id, fp.home_team_id, fp.away_team_id,
                   fp.home_goals, fp.away_goals
            FROM fixture_projections fp
            JOIN fixtures f ON f.id = fp.fixture_id
            WHERE f.competition_id = %s
              AND f.kickoff_datetime > NOW()
              {fid_filter_sql}
            """,
            (competition_id,) + fid_filter_params,
        )
        fp_rows = await cur.fetchall()

        # FanTeam mappings — drives the eligibility filter AND position.
        # Only players present here get a FanTeam projection row.
        await cur.execute(
            """
            SELECT player_id, fanteam_position
            FROM fanteam_wc_player_mappings
            WHERE player_id IS NOT NULL AND unmapped = 0
            """
        )
        ft_map_rows = await cur.fetchall()

        # Direct fixture→round lookup via wc_fixtures.round_id (FIFA's
        # authoritative pre-assignment). Sibling of wc_fantasy_points
        # service — see its comment for why we don't use wc_rounds
        # start/end windows.
        await cur.execute(
            "SELECT fixture_id, round_id FROM wc_fixtures WHERE fixture_id IS NOT NULL"
        )
        round_rows = await cur.fetchall()

    # Build {player_id: 'GK'/'DEF'/'MID'/'FWD'} from FanTeam mapping.
    _pos_map_long = {
        'goalkeeper': 'GK',
        'defender': 'DEF',
        'midfielder': 'MID',
        'forward': 'FWD',
    }
    ft_position: Dict[int, str] = {}
    for pid, fpos in ft_map_rows:
        if pid is None:
            continue
        mapped = _pos_map_long.get((fpos or '').lower())
        if mapped:
            ft_position[pid] = mapped

    by_pair: Dict[Tuple[int, int], dict] = {}
    for fid, pid, tid, oid, venue, ko, sid, val in pp_rows:
        if pid not in ft_position:
            continue   # not a FanTeam-listed player → skip
        key = (fid, pid)
        entry = by_pair.setdefault(key, {
            'fixture_id': fid,
            'player_id': pid,
            'position': ft_position[pid],
            'team_id': tid,
            'opponent_id': oid,
            'venue': venue,
            'kickoff_datetime': ko,
            'stats': {},
        })
        if val is not None:
            try:
                entry['stats'][int(sid)] = float(val)
            except (TypeError, ValueError):
                pass

    fix_meta: Dict[int, dict] = {}
    for fid, ht, at, hg, ag in fp_rows:
        try:
            home_goals = float(hg) if hg is not None else 0.0
            away_goals = float(ag) if ag is not None else 0.0
        except (TypeError, ValueError):
            home_goals = away_goals = 0.0
        pw, pd, pl = _outcome_probs(home_goals, away_goals)
        fix_meta[fid] = {
            'home_team_id': ht,
            'away_team_id': at,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'p_home_win': pw,
            'p_draw': pd,
            'p_away_win': pl,
        }

    round_by_fixture = {fid: rid for fid, rid in round_rows if fid is not None}

    return {
        'players': by_pair,
        'fixtures': fix_meta,
        'round_by_fixture': round_by_fixture,
    }


def _round_for(fixture_id, round_by_fixture) -> int:
    return round_by_fixture.get(fixture_id)


def _build_rows(data: dict) -> list:
    players = data['players']
    fixtures = data['fixtures']
    round_by_fixture = data['round_by_fixture']

    out = []
    for (fid, pid), entry in players.items():
        position = entry['position']
        team_id = entry['team_id']

        meta = fixtures.get(fid)
        if meta is None:
            continue

        if team_id == meta['home_team_id']:
            opp_goals = meta['away_goals']
            own_goals = meta['home_goals']
            p_win = meta['p_home_win']
            p_loss = meta['p_away_win']
        elif team_id == meta['away_team_id']:
            opp_goals = meta['home_goals']
            own_goals = meta['away_goals']
            p_win = meta['p_away_win']
            p_loss = meta['p_home_win']
        else:
            # Team / fixture mismatch — fall back conservatively.
            opp_goals = max(meta['home_goals'], meta['away_goals'])
            own_goals = min(meta['home_goals'], meta['away_goals'])
            p_win = p_loss = 0.0

        pts = _fanteam_points(entry['stats'], position, opp_goals,
                              own_goals, p_win, p_loss)
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


class FanTeamWcProjectionService:
    """Compute + write per-(fixture, player) FanTeam WC fantasy projections.

    Gated by `scope.fanteam_rules == 'fanteam_wc_2026'` in the orchestrator
    (parallel to scope.fantasy_rules for FIFA). Reads the same WC
    per-stat player projections + fixture projections that the FIFA
    scoring service uses; applies FanTeam's PL ruleset and writes to
    `fanteam_wc_projections`.
    """

    def __init__(self, scope=None):
        if scope is None:
            from app.services.international_projection_service import INTL_SCOPES
            scope = INTL_SCOPES['World Cup']
        self.scope = scope

    async def project(self, commit: bool = True, fixture_ids: list = None) -> dict:
        logger.info(
            f"{self.scope.competition_name} FanTeam projection start — "
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
            logger.info(
                f"Loaded {n_players} FanTeam-listed player-fixture stat bags "
                f"across {n_fixtures} fixtures"
            )

            if n_players == 0 or n_fixtures == 0:
                # Same quiet-state gate as wc_fantasy_points_service: no
                # upcoming fixtures (tournament over / not scheduled) makes
                # empty input the expected state, not a fault.
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT COUNT(*) FROM fixtures WHERE competition_id = %s AND kickoff_datetime > NOW()",
                        (self.scope.competition_id,),
                    )
                    n_upcoming = (await cur.fetchone())[0]
                if n_upcoming == 0:
                    logger.info(
                        f"No FanTeam input for {self.scope.competition_name} — expected: "
                        f"no upcoming fixtures (tournament finished or not yet scheduled)."
                    )
                else:
                    logger.warning(
                        "Empty input — ensure InternationalTeamStatService + "
                        "WcPlayerStatService have run AND that fanteam:ingest-wc-csv "
                        "has been run to populate fanteam_wc_player_mappings."
                    )
                return {
                    'n_rows': 0,
                    'n_player_fixtures': n_players,
                    'n_fixtures': n_fixtures,
                    'committed': False,
                }

            rows = _build_rows(data)
            logger.info(f"FanTeam WC projections ready: {len(rows)} rows")

            if commit and rows:
                # Same retention rule as the FIFA table (wc_fantasy_points_
                # service): the INSERT upserts on (fixture_id, player_id) so
                # upcoming fixtures refresh in place, and a round's rows are
                # KEPT until the round completes — a fixture that has kicked
                # off simply stops being refreshed (the SELECT is
                # kickoff > NOW), freezing its last pre-match projection. The
                # old blanket comp-wide DELETE made the FanTeam tab drop
                # played fixtures at kickoff while the FIFA tab kept them.
                async with conn.cursor() as cur:
                    if fixture_ids:
                        del_ph = ",".join(["%s"] * len(fixture_ids))
                        await cur.execute(
                            f"DELETE FROM fanteam_wc_projections WHERE fixture_id IN ({del_ph})",
                            tuple(fixture_ids),
                        )
                    else:
                        await cur.execute(
                            """DELETE fwp FROM fanteam_wc_projections fwp
                               JOIN wc_fixtures wf ON wf.fixture_id = fwp.fixture_id
                               JOIN wc_rounds wr ON wr.id = wf.round_id
                               WHERE wr.status = 'complete'""",
                        )
                await conn.commit()

                await insert_fanteam_wc_projections_async(rows)
                logger.info(f"FanTeam WC projections written: {len(rows)} rows")

            return {
                'n_rows': len(rows),
                'n_player_fixtures': n_players,
                'n_fixtures': n_fixtures,
                'committed': commit,
            }
        finally:
            release_source_connection(conn)
