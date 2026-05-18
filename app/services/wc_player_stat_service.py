"""
World Cup player-stat projections.

Distributes each WC team-stat projection (written by WcTeamStatService into
team_projections) down to the nation's confirmed squad players, by each
player's recency-weighted share of that stat across their international
history.

Runs as step 5 of WcProjectionService.projections(), after the team-stat
step. Mirrors the domestic distribute_team_predictions_to_players step,
with two deliberate differences:

  - Player universe = tournament_squads (the confirmed WC squads) rather
    than players.current_team_id, which points at club teams.
  - The share is computed from a player's INTERNATIONAL fixtures only —
    the pool scoped to INTERNATIONAL_COMP_IDS — with NO different-team
    half-weighting. A player has one national team, so every cap counts
    at full recency weight (domestic halves games for a different team to
    discount loans/internationals — neither applies here).

Pipeline:
  1. Load tournament_squads (competition 732, status='active') — gives the
     player universe + each player's team and position group.
  2. Load international fixtures + fixture_team_stats (share denominators)
     + fixture_player_stats (share numerators), scoped to the squad
     players / squad teams.
  3. Load the WC team_projections rows from step 4.
  4. For each (WC fixture, team) with a confirmed squad, for each player:
       share = Σ(w·player_stat) / Σ(w·team_stat) over the player's intl
               history (w = recency weight); player_value = team_proj × share.
     Goals:   share is blended 50/50 with the player's xG share (xG steadies
              a noisy goal sample); goals-only when there's no xG history.
     Derived: Assists  = assist-share × (team Goals  × 0.82)
              Key Passes = kp-share   × (team Shots  × 0.75)
              Saves      = opponent's (SoT − Goals), assigned to GKs.
              Fouls Drawn = fd-share × the opponent's projected Fouls — a
              team's Fouls Drawn IS the opponent's Fouls, so the share
              denominator is the opponent's fouls across the player's intl
              history (mirrors domestic get_player_stats / get_team_round_predictions).
  5. Idempotent DELETE + upsert into player_projections.

Only nations with a tournament_squads row produce player projections, so
the output grows automatically as more squads are confirmed.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.repository.player_repo import insert_player_async
from app.services.wc_team_stat_service import INTERNATIONAL_COMP_IDS
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("wc_player_stats")

WC_COMP_ID = 732

# Recency weighting — mirrors the domestic player-share curve in
# statz_functions.get_weighted_player_stats: full weight for the most
# recent weeks, then geometric decay. Tunable; international schedules
# are sparse so this leans on a handful of recent caps.
RECENCY_WEIGHT = 0.97
RECENCY_GRACE_WEEKS = 4      # full weight inside the last 4 weeks
RECENCY_EXP_SHIFT = 3        # decay exponent = weeks_since - 3

# Small-sample shrink — a high share off very few caps is unreliable, so
# pull it back toward the mean (same rule as get_player_weighted_average).
SMALL_SAMPLE_N = 10
SMALL_SAMPLE_SHARE_CAP = 0.2
SMALL_SAMPLE_SHRINK = 0.75

# Derived team-level stats — domestic derives these rather than projecting
# them; we mirror the ratios.
ASSISTS_PER_GOAL = 0.82
KEY_PASSES_PER_SHOT = 0.75

# Share-distributed stats.
#   output name -> (player fixture_player_stats stats_type_id,
#                   team   fixture_team_stats   stats_type_id  (denominator),
#                   team_projections column being distributed)
# Note Accurate Passes: the player-level stat (116) is distributed against
# the team-level "Successful Passes" total (81) — same convention as the
# domestic get_player_stats Accurate-Passes branch.
SHARE_STATS: Dict[str, Tuple[int, int, str]] = {
    'Goals':           (52, 52, 'goals'),
    'Shots Total':     (42, 42, 'shots_total'),
    'Shots On Target': (86, 86, 'shots_on_target'),
    'Fouls':           (56, 56, 'fouls'),
    'Yellow Cards':    (84, 84, 'yellowcards'),
    'Tackles':         (78, 78, 'tackles'),
    'Passes':          (80, 80, 'passes'),
    'Accurate Passes': (116, 81, 'successful_passes'),
    'Total Crosses':   (98, 98, 'total_crosses'),
    'Interceptions':   (100, 100, 'interceptions'),
    'Offsides':        (51, 51, 'offsides'),
}

# Derived stats distributed by share, but the team-level total comes from
# a ratio off another projection rather than a team_projections column.
#   output name -> (player fps id, team fts id used as the share denominator)
DERIVED_SHARE_STATS: Dict[str, Tuple[int, int]] = {
    'Assists':    (79, 79),
    'Key Passes': (117, 117),
}

# Fouls Drawn is special: a team's Fouls Drawn = the OPPONENT's Fouls — both
# for the team-level total (the opponent's projected fouls) and the share
# denominator (the opponent's fouls across the player's intl history).
# Mirrors statz_functions.get_player_stats (player numerator = player Fouls
# Drawn id 96; team denominator = opponent Fouls id 56).
FOULS_TEAM_STAT_ID = 56            # fixture_team_stats Fouls
FOULS_DRAWN_PLAYER_STAT_ID = 96    # fixture_player_stats Fouls Drawn

# The Goals share is blended 50/50 with the player's xG share — goals are
# low-frequency and noisy off a thin international sample, xG steadies the
# estimate. Mirrors statz_functions.distribute_team_predictions_to_players;
# falls back to goals-only when the player/team has no xG history.
XG_STAT_ID = 5304   # Expected Goals (xG) — fixture_team_stats + fixture_player_stats
SHOTS_TOTAL_PLAYER_STAT_ID = 42   # used by the xG zero-data guard

_ALL_PLAYER_STAT_IDS = sorted(
    {p for p, _t, _c in SHARE_STATS.values()}
    | {p for p, _t in DERIVED_SHARE_STATS.values()}
    | {FOULS_DRAWN_PLAYER_STAT_ID, XG_STAT_ID}
)
_ALL_TEAM_STAT_IDS = sorted(
    {t for _p, t, _c in SHARE_STATS.values()}
    | {t for _p, t in DERIVED_SHARE_STATS.values()}
    | {XG_STAT_ID}
)

_EMPTY_STAT_DF = pd.DataFrame(columns=['fixture_id', 'value'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(val) -> float:
    """Null-safe float. team_projections.goals is a varchar; the decimal
    columns can be NULL on pre-migration rows."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _weighted_share(
    player_rows: pd.DataFrame,
    team_rows: pd.DataFrame,
    fixtures_lookup: Dict[int, pd.Timestamp],
    target_dt: pd.Timestamp,
) -> Tuple[float, int]:
    """Player's recency-weighted share of a stat = Σ(w·player) / Σ(w·team)
    over the fixtures the player appeared in.

    player_rows / team_rows: DataFrames with columns [fixture_id, value].
    Fixtures with a missing/zero team value are dropped — without that the
    player value still counts toward the numerator while the denominator
    excludes it, inflating the share (the bug the domestic code guards at
    get_player_weighted_average:1563).

    Returns (share, n_fixtures).
    """
    if player_rows.empty or team_rows.empty:
        return 0.0, 0

    merged = player_rows.merge(team_rows, on='fixture_id', suffixes=('_p', '_t'))
    merged = merged[merged['value_t'] > 0]
    if merged.empty:
        return 0.0, 0

    kickoffs = pd.to_datetime(merged['fixture_id'].map(fixtures_lookup))
    weeks = ((target_dt - kickoffs).dt.days // 7).clip(lower=0)
    weight = np.where(
        weeks < RECENCY_GRACE_WEEKS,
        1.0,
        RECENCY_WEIGHT ** (weeks - RECENCY_EXP_SHIFT),
    )

    numerator = float((merged['value_p'] * weight).sum())
    denominator = float((merged['value_t'] * weight).sum())
    if denominator <= 0:
        return 0.0, 0

    share = numerator / denominator
    n = len(merged)
    # Shrink an implausibly high share built off a thin sample.
    if n < SMALL_SAMPLE_N and share > SMALL_SAMPLE_SHARE_CAP:
        share *= SMALL_SAMPLE_SHRINK
    return share, n


def _drop_missing_xg(xg_rows: pd.DataFrame, shot_rows: pd.DataFrame) -> pd.DataFrame:
    """xG zero-data guard — mirrors the xG branch of statz_functions.get_player_stats.

    A shot can't be worth 0.00 xG, so an xG=0 row for a fixture where the
    player DID take shots is missing data, not a genuine zero — drop it so it
    doesn't deflate the xG share. Keep xG>0 rows, and xG=0 rows where the
    player took no shots (a genuine zero)."""
    if xg_rows.empty:
        return xg_rows
    if shot_rows.empty:
        return xg_rows
    shot_fixtures = set(shot_rows[shot_rows['value'] > 0]['fixture_id'])
    drop_mask = (xg_rows['value'] == 0) & xg_rows['fixture_id'].isin(shot_fixtures)
    return xg_rows[~drop_mask]


async def _load_data(conn) -> dict:
    """Pull everything in one connection: confirmed WC squads, the
    international history (team + player stats, scoped to those squads),
    the WC team_projections to distribute, and name lookups."""
    async with conn.cursor() as cur:
        # 1. Confirmed WC squads — the player universe.
        await cur.execute(
            """
            SELECT team_id, player_id, position_group
            FROM tournament_squads
            WHERE competition_id = %s AND status = 'active'
            """,
            (WC_COMP_ID,),
        )
        squad_rows = await cur.fetchall()

        squad_team_ids = sorted({r[0] for r in squad_rows})
        squad_player_ids = sorted({r[1] for r in squad_rows})

        # 2. All finished international fixtures (the 17-comp pool).
        ph_comp = ",".join(["%s"] * len(INTERNATIONAL_COMP_IDS))
        await cur.execute(
            f"""
            SELECT id, kickoff_datetime
            FROM fixtures
            WHERE competition_id IN ({ph_comp})
              AND state_id IN (5, 7, 8)
            """,
            tuple(INTERNATIONAL_COMP_IDS),
        )
        fixture_rows = await cur.fetchall()
        intl_fixture_ids = [r[0] for r in fixture_rows]

        # 3. Team stats for those fixtures, squad teams only (share denominators).
        team_stat_rows = []
        if intl_fixture_ids and squad_team_ids:
            ph_fid = ",".join(["%s"] * len(intl_fixture_ids))
            ph_tid = ",".join(["%s"] * len(squad_team_ids))
            ph_sid = ",".join(["%s"] * len(_ALL_TEAM_STAT_IDS))
            await cur.execute(
                f"""
                SELECT fixture_id, team_id, stats_type_id, value
                FROM fixture_team_stats
                WHERE fixture_id IN ({ph_fid})
                  AND team_id IN ({ph_tid})
                  AND stats_type_id IN ({ph_sid})
                """,
                tuple(intl_fixture_ids) + tuple(squad_team_ids) + tuple(_ALL_TEAM_STAT_IDS),
            )
            team_stat_rows = await cur.fetchall()

        # 4. Player stats for those fixtures, squad players only (numerators).
        player_stat_rows = []
        if intl_fixture_ids and squad_player_ids:
            ph_fid = ",".join(["%s"] * len(intl_fixture_ids))
            ph_pid = ",".join(["%s"] * len(squad_player_ids))
            ph_sid = ",".join(["%s"] * len(_ALL_PLAYER_STAT_IDS))
            await cur.execute(
                f"""
                SELECT fixture_id, player_id, team_id, stats_type_id, value
                FROM fixture_player_stats
                WHERE fixture_id IN ({ph_fid})
                  AND player_id IN ({ph_pid})
                  AND stats_type_id IN ({ph_sid})
                """,
                tuple(intl_fixture_ids) + tuple(squad_player_ids) + tuple(_ALL_PLAYER_STAT_IDS),
            )
            player_stat_rows = await cur.fetchall()

        # 4b. Fouls (56) for ALL teams in those fixtures — needed to derive
        # each squad team's "Fouls Drawn" (= the opponent's fouls in that
        # fixture). Loaded unscoped by team because the opponent is often
        # not itself a squad team.
        all_fouls_rows = []
        if intl_fixture_ids:
            ph_fid = ",".join(["%s"] * len(intl_fixture_ids))
            await cur.execute(
                f"""
                SELECT fixture_id, team_id, value
                FROM fixture_team_stats
                WHERE fixture_id IN ({ph_fid})
                  AND stats_type_id = %s
                """,
                tuple(intl_fixture_ids) + (FOULS_TEAM_STAT_ID,),
            )
            all_fouls_rows = await cur.fetchall()

        # 5. WC team_projections written by the team-stat step.
        await cur.execute(
            """
            SELECT tp.fixture_id, tp.team_id, tp.opponent_id, tp.venue,
                   tp.kickoff_datetime, tp.goals, tp.shots_total,
                   tp.shots_on_target, tp.fouls, tp.yellowcards, tp.tackles,
                   tp.passes, tp.successful_passes, tp.total_crosses,
                   tp.interceptions, tp.offsides
            FROM team_projections tp
            JOIN fixtures f ON f.id = tp.fixture_id
            WHERE f.competition_id = %s
            """,
            (WC_COMP_ID,),
        )
        team_proj_rows = await cur.fetchall()

        # 6. Name lookups.
        await cur.execute("SELECT id, name FROM teams")
        teams_rows = await cur.fetchall()

        players_rows = []
        if squad_player_ids:
            ph_pid = ",".join(["%s"] * len(squad_player_ids))
            await cur.execute(
                f"SELECT id, display_name FROM players WHERE id IN ({ph_pid})",
                tuple(squad_player_ids),
            )
            players_rows = await cur.fetchall()

    # --- assemble ---
    fixtures_lookup = {
        r[0]: pd.to_datetime(r[1]) for r in fixture_rows
    }

    squads: Dict[int, List[Tuple[int, str]]] = {}
    for team_id, player_id, position_group in squad_rows:
        squads.setdefault(team_id, []).append((player_id, position_group))

    team_stats_df = pd.DataFrame(
        team_stat_rows, columns=['fixture_id', 'team_id', 'stats_type_id', 'value']
    )
    if not team_stats_df.empty:
        team_stats_df['value'] = pd.to_numeric(team_stats_df['value'], errors='coerce')
        team_stats_df = team_stats_df.dropna(subset=['value'])

    player_stats_df = pd.DataFrame(
        player_stat_rows,
        columns=['fixture_id', 'player_id', 'team_id', 'stats_type_id', 'value'],
    )
    if not player_stats_df.empty:
        player_stats_df['value'] = pd.to_numeric(player_stats_df['value'], errors='coerce')
        player_stats_df = player_stats_df.dropna(subset=['value'])

    # team "Fouls Drawn" per (fixture, team) = the OPPONENT's Fouls. Self-merge
    # the all-team fouls rows on fixture_id and keep opposite-team pairs.
    all_fouls_df = pd.DataFrame(all_fouls_rows, columns=['fixture_id', 'team_id', 'value'])
    team_fouls_drawn_df = pd.DataFrame(columns=['fixture_id', 'team_id', 'value'])
    if not all_fouls_df.empty:
        all_fouls_df['value'] = pd.to_numeric(all_fouls_df['value'], errors='coerce')
        all_fouls_df = all_fouls_df.dropna(subset=['value'])
        cross = all_fouls_df.merge(all_fouls_df, on='fixture_id', suffixes=('_self', '_opp'))
        cross = cross[cross['team_id_self'] != cross['team_id_opp']]
        team_fouls_drawn_df = (
            cross[['fixture_id', 'team_id_self', 'value_opp']]
            .rename(columns={'team_id_self': 'team_id', 'value_opp': 'value'})
            .drop_duplicates(subset=['fixture_id', 'team_id'])
            .reset_index(drop=True)
        )

    tp_cols = [
        'fixture_id', 'team_id', 'opponent_id', 'venue', 'kickoff_datetime',
        'goals', 'shots_total', 'shots_on_target', 'fouls', 'yellowcards',
        'tackles', 'passes', 'successful_passes', 'total_crosses',
        'interceptions', 'offsides',
    ]
    team_projections = [dict(zip(tp_cols, r)) for r in team_proj_rows]

    return {
        'squads': squads,
        'fixtures_lookup': fixtures_lookup,
        'team_stats_df': team_stats_df,
        'player_stats_df': player_stats_df,
        'team_fouls_drawn_df': team_fouls_drawn_df,
        'team_projections': team_projections,
        'teams': {r[0]: r[1] for r in teams_rows},
        'players': {r[0]: r[1] for r in players_rows},
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class WcPlayerStatService:
    """Compute + write per-player stat projections for upcoming WC fixtures."""

    async def project(self, commit: bool = True) -> dict:
        logger.info(f"WC player-stat projection start — commit={commit}")

        conn = await get_source_connection()
        try:
            data = await _load_data(conn)
            squads = data['squads']
            team_projections = data['team_projections']
            teams = data['teams']
            players = data['players']
            fixtures_lookup = data['fixtures_lookup']

            if not squads:
                logger.warning("No confirmed WC squads in tournament_squads — nothing to project.")
                return {'n_player_rows': 0, 'n_squads': 0, 'committed': False}
            if not team_projections:
                logger.warning("No WC team_projections found — run WcTeamStatService first.")
                return {'n_player_rows': 0, 'n_squads': len(squads), 'committed': False}

            logger.info(
                f"Loaded {len(squads)} confirmed squads, {len(team_projections)} "
                f"team-projection rows, {len(data['player_stats_df'])} intl player-stat "
                f"rows, {len(data['team_stats_df'])} intl team-stat rows"
            )

            # Pre-group history for O(1) lookup inside the loop.
            #   (player_id, team_id, stats_type_id) -> [fixture_id, value]
            pstats_groups = {
                key: grp[['fixture_id', 'value']]
                for key, grp in data['player_stats_df'].groupby(
                    ['player_id', 'team_id', 'stats_type_id'], sort=False
                )
            } if not data['player_stats_df'].empty else {}
            #   (team_id, stats_type_id) -> [fixture_id, value]
            tstats_groups = {
                key: grp[['fixture_id', 'value']]
                for key, grp in data['team_stats_df'].groupby(
                    ['team_id', 'stats_type_id'], sort=False
                )
            } if not data['team_stats_df'].empty else {}
            #   team_id -> [fixture_id, value] where value = opponent's fouls
            #   (the team's "Fouls Drawn") — share denominator for Fouls Drawn.
            team_fouls_drawn_groups = {
                tid: grp[['fixture_id', 'value']]
                for tid, grp in data['team_fouls_drawn_df'].groupby('team_id', sort=False)
            } if not data['team_fouls_drawn_df'].empty else {}

            # team Saves = opponent's (SoT − Goals) — look up the other row
            # in the same fixture.
            tp_by_fixture: Dict[int, Dict[int, dict]] = {}
            for tp in team_projections:
                tp_by_fixture.setdefault(tp['fixture_id'], {})[tp['team_id']] = tp

            output_rows = []
            n_skipped_no_squad = 0
            for tp in team_projections:
                team_id = tp['team_id']
                squad = squads.get(team_id)
                if not squad:
                    n_skipped_no_squad += 1
                    continue

                target_dt = pd.to_datetime(tp['kickoff_datetime'])
                team_name = teams.get(team_id, str(team_id))
                opp_name = teams.get(tp['opponent_id'], str(tp['opponent_id']))

                # team Saves from the opponent's projection row.
                opp_tp = tp_by_fixture.get(tp['fixture_id'], {}).get(tp['opponent_id'])
                if opp_tp is not None:
                    team_saves = max(
                        _f(opp_tp['shots_on_target']) - _f(opp_tp['goals']), 0.0
                    )
                else:
                    team_saves = 0.0

                team_goals = _f(tp['goals'])
                team_shots = _f(tp['shots_total'])
                derived_totals = {
                    'Assists': team_goals * ASSISTS_PER_GOAL,
                    'Key Passes': team_shots * KEY_PASSES_PER_SHOT,
                }

                for player_id, position_group in squad:
                    row = {
                        'fixture_id': tp['fixture_id'],
                        'kickoff_datetime': target_dt,
                        'player_id': player_id,
                        'Player': players.get(player_id, str(player_id)),
                        'Position': position_group,
                        'Team': team_name,
                        'Opponent': opp_name,
                        'Venue': tp['venue'],
                        # No pre-tournament lineup data — Start? stays 'No'
                        # for v1 (a lineup predictor is a later enhancement).
                        'Start?': 'No',
                    }

                    # Share-distributed stats.
                    for out_stat, (p_sid, t_sid, tp_col) in SHARE_STATS.items():
                        p_rows = pstats_groups.get((player_id, team_id, p_sid), _EMPTY_STAT_DF)
                        t_rows = tstats_groups.get((team_id, t_sid), _EMPTY_STAT_DF)
                        share, _n = _weighted_share(p_rows, t_rows, fixtures_lookup, target_dt)

                        # Goals: blend the goals share 50/50 with the xG share
                        # (mirrors domestic distribute_team_predictions_to_players).
                        # Goals-only fallback when there's no xG history.
                        if out_stat == 'Goals':
                            xg_p = pstats_groups.get((player_id, team_id, XG_STAT_ID), _EMPTY_STAT_DF)
                            xg_t = tstats_groups.get((team_id, XG_STAT_ID), _EMPTY_STAT_DF)
                            # xG zero-data guard: drop xG=0 fixtures where the
                            # player took shots (missing data, not a real 0).
                            shot_rows = pstats_groups.get(
                                (player_id, team_id, SHOTS_TOTAL_PLAYER_STAT_ID), _EMPTY_STAT_DF
                            )
                            xg_p = _drop_missing_xg(xg_p, shot_rows)
                            xg_share, _xn = _weighted_share(xg_p, xg_t, fixtures_lookup, target_dt)
                            if xg_share > 0:
                                share = (share + xg_share) / 2.0

                        row[out_stat] = round(_f(tp[tp_col]) * share, 2)

                    # Derived stats (Assists, Key Passes).
                    for out_stat, (p_sid, t_sid) in DERIVED_SHARE_STATS.items():
                        p_rows = pstats_groups.get((player_id, team_id, p_sid), _EMPTY_STAT_DF)
                        t_rows = tstats_groups.get((team_id, t_sid), _EMPTY_STAT_DF)
                        share, _n = _weighted_share(p_rows, t_rows, fixtures_lookup, target_dt)
                        row[out_stat] = round(derived_totals[out_stat] * share, 2)

                    # Fouls Drawn — team total = the opponent's projected
                    # Fouls; share denominator = the opponent's fouls across
                    # the player's intl history (team_fouls_drawn_groups).
                    fd_player_rows = pstats_groups.get(
                        (player_id, team_id, FOULS_DRAWN_PLAYER_STAT_ID), _EMPTY_STAT_DF
                    )
                    fd_team_rows = team_fouls_drawn_groups.get(team_id, _EMPTY_STAT_DF)
                    fd_share, _n = _weighted_share(
                        fd_player_rows, fd_team_rows, fixtures_lookup, target_dt
                    )
                    opp_fouls_total = _f(opp_tp['fouls']) if opp_tp is not None else 0.0
                    row['Fouls Drawn'] = round(opp_fouls_total * fd_share, 2)

                    # Saves — keepers get the team total; outfielders 0
                    # (mirrors the domestic GK-only Saves assignment).
                    row['Saves'] = round(team_saves, 2) if position_group == 'GK' else 0.0

                    output_rows.append(row)

            logger.info(
                f"WC player-stat projection ready: {len(output_rows)} player-fixture "
                f"rows across {len({r['Team'] for r in output_rows})} nations, "
                f"skipped {n_skipped_no_squad} team-fixtures with no confirmed squad"
            )

            if commit and output_rows:
                df = pd.DataFrame(output_rows)
                df['kickoff_datetime'] = pd.to_datetime(df['kickoff_datetime'])

                # Idempotent: clear existing WC player rows before re-insert.
                async with conn.cursor() as cur:
                    await cur.execute(
                        """DELETE pp FROM player_projections pp
                           JOIN fixtures f ON f.id = pp.fixture_id
                           WHERE f.competition_id = %s""",
                        (WC_COMP_ID,),
                    )
                await conn.commit()

                teams_df = pd.DataFrame(
                    list(data['teams'].items()), columns=['id', 'name']
                )
                await insert_player_async(df, teams=teams_df, competition_id=WC_COMP_ID)
                logger.info(f"WC player-stat projections written: {len(df)} rows")

            return {
                'n_player_rows': len(output_rows),
                'n_squads': len(squads),
                'n_team_projection_rows': len(team_projections),
                'committed': commit,
            }
        finally:
            release_source_connection(conn)
