"""
World Cup player-stat projections.

Distributes each WC team-stat projection (written by InternationalTeamStatService into
team_projections) down to the nation's confirmed squad players, by each
player's recency-weighted share of that stat across a 30-game history.

Still a share model (player_value = team_projection × share). The window:

  - Main window = the player's last 30 appearances. International caps are
    taken first; if fewer than 30, the remainder is FILLED with the player's
    most recent CLUB games. International football is sparse, so club games
    populate the recency-weighted window and stabilise the estimate.
  - Per game the "team" for the share denominator is whoever the player
    played for (national team for caps, club for club games).
  - Club games are weighted lower than caps (CLUB_GAME_WEIGHT).
  - Players with NO international AND NO club history are skipped + logged.

  - The Goals/xG blend uses a SEPARATE xG window — the player's recent games
    that actually carry an xG row, international-first then club-filled. A
    high-cap player from an xG-barren confederation (CAF/AFC) has a main
    window of 30 xG-less caps, yet plenty of club xG; the separate window
    reaches it. Goals-only when the xG window is below XG_MIN_GAMES.

Known v1 limitation kept deliberately: a share is teammate-dependent and
doesn't fully travel club↔country, so club-filled games import a little of
the player's club role. Accepted for now; a rate-based rework is the v2.

Pipeline:
  1. Load the FIFA-game player pool (wc_players JOIN wc_squads, linked
     subset via WcAutoLinker). Position comes from wc_players.position
     (GK/DEF/MID/FWD), which is what the fantasy game scores by.
  2. Build each player's 30-game main window + xG window, then load the
     player + team stats for those fixtures.
  3. Load the WC team_projections rows from step 4.
  4. For each (WC fixture, team) in the FIFA pool, for each player:
       share = Σ(w·player_stat) / Σ(w·team_stat) over the main window.
     Goals:   share blended 50/50 with the player's xG share — the xG share
              uses the SEPARATE xG window so a player with no xG in their
              cap history still uses club xG.
     Derived: Assists = assist-share × (team Goals × 0.82)
              Key Passes = kp-share × (team Shots × 0.75)
              Saves = opponent's (SoT − Goals), assigned to GKs.
              Fouls Drawn = fd-share × the opponent's projected Fouls.
  5. Idempotent DELETE + upsert into player_projections.
  6. Poisson-distribute those expected-value lines across the 1+/2+/3+ prop
     markets and write player_prop_projections — the table the
     /projections/player-props page reads. Mirrors projection_service.py.
"""
import logging
from typing import Dict, List, Tuple

import pandas as pd

from app.repository.player_repo import insert_player_async
from app.repository.player_stat_repo import insert_players_stats_async
from app.services.statz_functions import get_poisson_probs
from app.services.international_team_stat_service import INTERNATIONAL_COMP_IDS
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("wc_player_stats")

# --- Game window -----------------------------------------------------------
GAME_WINDOW = 30             # last N VALID appearances per player (intl-first, club-filled)
# Raw window is intentionally wider than GAME_WINDOW so the share calc can
# walk past fixtures with missing team_stats (sparse-coverage stats like
# xG / Successful Passes on older intl fixtures) and still find 30 valid
# fixtures with a real team denominator. _weighted_share early-breaks once
# GAME_WINDOW valid fixtures are accumulated, so dense-coverage stats see
# identical behaviour to the previous GAME_WINDOW=30 cap. Memory: 1330
# players × 90 entries ≈ 6MB additional, negligible. Mirrors the domestic
# 2026-06-05 filter-before-window fix in statz_functions.get_player_stats.
GAME_WINDOW_RAW = GAME_WINDOW * 3  # walk up to 3x GAME_WINDOW seeking valid fixtures
# No club lookback limit — matches the domestic projection path which loads
# all of a player's stats and lets the GAME_WINDOW cap + recency weights
# decide what's relevant. The previous 18-month cap was a load-bound that
# created inconsistent behaviour for veterans with thin recent xG samples
# (Lukaku: 9 xG-eligible games inside 18mo, but plenty more at Roma/Inter
# pre-2024 that the window-builder couldn't reach).

# --- Recency weighting -----------------------------------------------------
# 0.9975/week — VERY gentle decay. Over a 30-game window this keeps games
# spanning multiple intl tournament cycles (2022 WC, Euro 2024, current
# qualifying) weighted near-evenly. A 3-year-old cap carries 68% weight,
# 5-year-old 53%. Loosened from 0.995 to 0.9975 on 2026-06-04 to pull older
# tournament-era caps further toward full influence — players don't get 30
# caps in tight succession, so the share calc needs the older games to
# stabilise.
RECENCY_WEIGHT = 0.9975
RECENCY_GRACE_WEEKS = 4      # full weight inside the last 4 weeks
RECENCY_EXP_SHIFT = 3        # decay exponent = weeks_since - 3

# Club games count for less than international caps — the share we ultimately
# want is the player's NATIONAL-team role; club games stabilise it but the
# discount keeps caps dominant. Applied on top of the recency weight.
# Lowered 2026-06-03 from 0.5 → 0.3 to pull share more toward intl pedigree.
CLUB_GAME_WEIGHT = 0.3

# Minutes-played filter — mirror domestic statz_functions.py:1372 which
# drops fixtures where the player played 45 mins or less. Late-sub cameos
# inflate per-90 share calcs.
MINUTES_PLAYED_STAT_ID = 119
MIN_MINUTES_THRESHOLD = 45

# Goals share blend: how much weight on raw goals vs xG. Goals are noisy
# and high-variance; xG is steadier. Shifted 2026-06-03 from 50/50 to 30/70
# favoring xG. The 0.7 weight is the FULL weight applied when xG window
# is at or above XG_FULL_WEIGHT_N; thinner samples scale the weight down
# linearly so a player with 5-10 xG games doesn't have their projection
# dominated by a noisy partial sample. Audit on 2026-06-04 showed top-30
# scorers including Lukaku (9 xG games) and Havertz (15) were getting
# 70% blend weight from too few xG fixtures — tiered scaling fixes this.
GOALS_BLEND_GOALS_WEIGHT = 0.3
GOALS_BLEND_XG_WEIGHT = 0.7
XG_FULL_WEIGHT_N = 20

# International-cap-count shrink — players with few caps in the last 24
# months have an unreliable intl share; dampen their projections so a
# 1-cap player who scored doesn't get treated as a goal-per-game. Counted
# AFTER the >45 min filter, so only "meaningful" caps count.
INTL_CAP_SHRINK_LOOKBACK_MONTHS = 24
def _intl_cap_shrink(caps_24mo: int) -> float:
    if caps_24mo < 5:
        return 0.6
    if caps_24mo < 10:
        return 0.8
    return 1.0

# Small-sample shrink — a high share off very few games is unreliable, so
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
# denominator (the opponent's fouls across the player's history).
FOULS_TEAM_STAT_ID = 56            # fixture_team_stats Fouls
FOULS_DRAWN_PLAYER_STAT_ID = 96    # fixture_player_stats Fouls Drawn

# The Goals share is blended 50/50 with the player's xG share — goals are
# low-frequency and noisy, xG steadies the estimate. The xG share runs over
# its OWN window (player_xg_windows) — the player's recent games that carry
# an xG row, international-first — separate from the main window so a
# high-cap player from an xG-barren confederation still reaches their club
# xG. Skipped when the xG window has fewer than XG_MIN_GAMES games.
XG_STAT_ID = 5304     # Expected Goals (xG)
XG_MIN_GAMES = 5      # min xG-window games before the xG blend is used at all

# --- Player-prop markets ---------------------------------------------------
# Once the expected-value lines are built, each is Poisson-distributed across
# the 1+/2+/3+ thresholds to get prop probabilities (P(X >= line)) — the rows
# player_prop_projections stores. Mirrors the domestic perc_stats/lines in
# projection_service.py. Yellow Cards is 1+ only (2+ = a red card, ~0
# probability, not a useful market). The 'Fouls' expected-value column is
# renamed to the 'Fouls Committed' market before distribution.
PROP_STATS = ['Shots On Target', 'Fouls Committed', 'Fouls Drawn',
              'Goals', 'Tackles', 'Shots Total', 'Offsides']
PROP_LINES = [1, 2, 3]

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

# Share-denominator reconciliation map: {player_fps_id: team_fts_id}.
# Every share/derived stat is a "one event = one player" counting stat, so
# the team total SHOULD equal the sum of the players' values. In practice
# the two feeds disagree in a meaningful fraction of (esp. international)
# fixtures — sometimes the team row is missing/low (Assists ~33% of intl
# games, Key Passes ~24%), sometimes a thin player import makes the
# player-sum low. BOTH failure modes are undercounts, so we take
# max(team_row, player_sum) as the denominator: it's a no-op where the
# feeds agree, picks player-sum where the team row is short, and picks the
# team row where the player import is thin. xG (modeled, not a count) and
# Fouls Drawn (opponent-derived) are deliberately excluded.
_COUNTING_DENOM_MAP: Dict[int, int] = {
    **{p: t for p, t, _c in SHARE_STATS.values()},
    **{p: t for p, t in DERIVED_SHARE_STATS.values()},
}
_COUNTING_PLAYER_IDS = sorted(_COUNTING_DENOM_MAP.keys())
_INTL_COMP_SET = set(INTERNATIONAL_COMP_IDS)


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


def _build_window(
    apps: List[Tuple[int, pd.Timestamp, int, int]],
) -> List[Tuple[int, pd.Timestamp, bool, int]]:
    """Pick a player's RAW game window from their appearances — international
    games first (most recent), then club games fill the remainder, up to
    GAME_WINDOW_RAW. _weighted_share then walks the raw window and early-
    breaks at GAME_WINDOW VALID fixtures (where team_val > 0). Wider raw
    window means sparse-coverage stats can still find a full 30-fixture
    sample without changing dense-stat behaviour.

    Input appearances are (fixture_id, kickoff, competition_id, team_id).
    Returns (fixture_id, kickoff, is_club, team_id) entries.
    """
    intl = sorted((a for a in apps if a[2] in _INTL_COMP_SET),
                  key=lambda a: a[1], reverse=True)
    club = sorted((a for a in apps if a[2] not in _INTL_COMP_SET),
                  key=lambda a: a[1], reverse=True)
    chosen = intl[:GAME_WINDOW_RAW]
    if len(chosen) < GAME_WINDOW_RAW:
        chosen = chosen + club[:GAME_WINDOW_RAW - len(chosen)]
    return [(fid, ko, comp not in _INTL_COMP_SET, tid)
            for (fid, ko, comp, tid) in chosen]


def _weighted_share(
    window: List[Tuple[int, pd.Timestamp, bool, int]],
    player_vals: Dict[int, float],
    team_vals: Dict[int, float],
    target_dt: pd.Timestamp,
) -> Tuple[float, int]:
    """Recency- and club-weighted share = Σ(w·player) / Σ(w·team) over the
    given window.

    window      : list of (fixture_id, kickoff, is_club, team_id).
    player_vals : {fixture_id: player's stat value}.
    team_vals   : {fixture_id: the relevant team's stat value} — fixtures
                  missing here (or ≤ 0) are dropped.

    w = recency × (CLUB_GAME_WEIGHT if club else 1.0).
    Returns (share, n_fixtures_used).
    """
    num = 0.0
    den = 0.0
    n = 0
    for fixture_id, kickoff, is_club, _team_id in window:
        team_val = team_vals.get(fixture_id)
        if team_val is None or team_val <= 0:
            continue
        weeks = (target_dt - kickoff).days // 7
        if weeks < 0:
            weeks = 0
        if weeks < RECENCY_GRACE_WEEKS:
            recency = 1.0
        else:
            recency = RECENCY_WEIGHT ** (weeks - RECENCY_EXP_SHIFT)
        weight = recency * (CLUB_GAME_WEIGHT if is_club else 1.0)
        num += weight * player_vals.get(fixture_id, 0.0)
        den += weight * team_val
        n += 1
        # Early-break: once we have GAME_WINDOW valid fixtures the share
        # is stable. Walking further into the raw window just buries the
        # recent signal under deeper-time, weight-suppressed noise.
        if n >= GAME_WINDOW:
            break

    if den <= 0:
        return 0.0, 0
    share = num / den
    # Shrink an implausibly high share built off a thin sample.
    if n < SMALL_SAMPLE_N and share > SMALL_SAMPLE_SHARE_CAP:
        share *= SMALL_SAMPLE_SHRINK
    return share, n


async def _load_data(conn, competition_id: int, squad_provider, fixture_ids_filter=None) -> dict:
    """Pull everything: per-nation squad pool (via squad_provider strategy),
    each player's main 30-game window + xG window, the player/team stats
    for those fixtures, the team_projections rows to distribute (filtered
    to competition_id), and name lookups.

    The squad_provider returns {team_id: [(player_id, position), ...]}.
    For WC scope it reads wc_squads; for non-WC scopes (friendlies / quals)
    it derives from recent intl appearances — see intl_squad_provider.py.
    """
    # End any transaction inherited on this pooled connection so the reads
    # below get a FRESH snapshot — a pooled connection can carry a stale
    # InnoDB REPEATABLE READ snapshot taken before InternationalTeamStatService (the
    # immediately-prior step) committed its team_projections rows.
    await conn.rollback()

    # 1. Squad pool — delegated to a SquadProvider. WC scope uses
    # WcSquadProvider (wc_players JOIN wc_squads); non-WC scopes inject
    # RecentCapsSquadProvider via WcPlayerStatService.__init__.
    # Returns {team_id: [(player_id, position_group), ...]}.
    #
    # The WC-only docstring on the previous inline SQL covered:
    #   - ~1,352 player rows across 48 nations from FIFA WC fantasy
    #   - position comes straight from FIFA (no Sportmonks drift)
    #   - 102 unlinked FIFA players (player_id NULL) excluded
    # All of that is preserved by WcSquadProvider; the provider call
    # is the same query.
    squads = await squad_provider.load(conn)
    squad_player_ids = sorted({pid for members in squads.values() for pid, _pos in members})

    async with conn.cursor() as cur:

        # 2. Appearances — every finished game each squad player featured in:
        #    all international caps + club games within the lookback window.
        # >45 min minutes-played filter — mirrors the domestic
        # statz_functions.get_player_weighted_average path which excludes
        # fixtures where the player played 45 mins or fewer (line 1372 of
        # statz_functions.py). Cameos / late subs inflate per-90 share
        # calcs — a 6-min appearance with a goal looks like a 9-goal-
        # per-90 player. Filter at the appearance-loading step so every
        # downstream share calc inherits it.
        # state_id filter dropped — historical Sportmonks fixtures have
        # state_id=NULL (~67% of pre-current-season PL/Serie A/La Liga
        # rows). Filtering IN (5,7,8) was excluding ~10-15 years of
        # career data for veterans (Lukaku's Roma/Inter/Chelsea games,
        # Ronaldo's Madrid era, etc.). A row in fixture_player_stats
        # is itself sufficient evidence the fixture was played — no
        # need to gate on state.
        appearance_rows = []
        if squad_player_ids:
            ph_pid = ",".join(["%s"] * len(squad_player_ids))
            await cur.execute(
                f"""
                SELECT DISTINCT fps.player_id, fps.fixture_id, fps.team_id,
                       f.kickoff_datetime, f.competition_id
                FROM fixture_player_stats fps
                JOIN fixtures f ON f.id = fps.fixture_id
                JOIN fixture_player_stats mp ON mp.fixture_id = fps.fixture_id
                                            AND mp.player_id  = fps.player_id
                                            AND mp.stats_type_id = {MINUTES_PLAYED_STAT_ID}
                                            AND mp.value > {MIN_MINUTES_THRESHOLD}
                WHERE fps.player_id IN ({ph_pid})
                  AND f.kickoff_datetime < NOW()
                """,
                tuple(squad_player_ids),
            )
            appearance_rows = await cur.fetchall()

        # 2b. xG appearances — every finished game a squad player has an xG
        #     row in AND played > 45 mins. Drives the SEPARATE xG window;
        #     not date-bounded (the intl-first 30-cap slice bounds it).
        xg_row_data = []
        if squad_player_ids:
            ph_pid = ",".join(["%s"] * len(squad_player_ids))
            await cur.execute(
                f"""
                SELECT DISTINCT fps.player_id, fps.fixture_id, fps.team_id,
                       f.kickoff_datetime, f.competition_id
                FROM fixture_player_stats fps
                JOIN fixtures f ON f.id = fps.fixture_id
                JOIN fixture_player_stats mp ON mp.fixture_id = fps.fixture_id
                                            AND mp.player_id  = fps.player_id
                                            AND mp.stats_type_id = {MINUTES_PLAYED_STAT_ID}
                                            AND mp.value > {MIN_MINUTES_THRESHOLD}
                WHERE fps.player_id IN ({ph_pid})
                  AND fps.stats_type_id = %s
                  AND f.kickoff_datetime < NOW()
                """,
                tuple(squad_player_ids) + (XG_STAT_ID,),
            )
            xg_row_data = await cur.fetchall()

        # --- build the per-player windows ---
        def _group_appearances(rows) -> Dict[int, list]:
            grouped: Dict[int, list] = {}
            seen = set()
            for player_id, fixture_id, team_id, kickoff, competition_id in rows:
                key = (player_id, fixture_id)
                if key in seen:
                    continue
                seen.add(key)
                grouped.setdefault(player_id, []).append(
                    (fixture_id, pd.to_datetime(kickoff), int(competition_id), team_id)
                )
            return grouped

        appearances = _group_appearances(appearance_rows)
        xg_appearances = _group_appearances(xg_row_data)

        # Per-player intl cap count in last 24mo — drives _intl_cap_shrink
        # in the per-fixture loop. Counted against the >45-min-filtered
        # appearances list (so 1 short cameo doesn't count as a cap).
        # Anchored on "now" rather than per-target-fixture because cap
        # counts are stable across the upcoming-fixture set.
        _cap_anchor = pd.Timestamp.now() - pd.Timedelta(days=INTL_CAP_SHRINK_LOOKBACK_MONTHS * 30)
        intl_cap_counts: Dict[int, int] = {}
        for player_id, apps_list in appearances.items():
            intl_cap_counts[player_id] = sum(
                1 for (_fid, ko, comp, _tid) in apps_list
                if comp in _INTL_COMP_SET and ko >= _cap_anchor
            )

        player_windows: Dict[int, List[Tuple[int, pd.Timestamp, bool, int]]] = {}
        skipped_no_data: List[int] = []
        for player_id in squad_player_ids:
            apps = appearances.get(player_id, [])
            if not apps:
                skipped_no_data.append(player_id)
                continue
            player_windows[player_id] = _build_window(apps)

        player_xg_windows: Dict[int, List[Tuple[int, pd.Timestamp, bool, int]]] = {}
        for player_id in squad_player_ids:
            xg_apps = xg_appearances.get(player_id, [])
            if xg_apps:
                player_xg_windows[player_id] = _build_window(xg_apps)

        window_fixture_ids = sorted(
            {fid for w in player_windows.values() for (fid, _ko, _ic, _tid) in w}
            | {fid for w in player_xg_windows.values() for (fid, _ko, _ic, _tid) in w}
        )

        # 3. Player stats for the window fixtures (share numerators).
        player_stat_rows = []
        if window_fixture_ids and squad_player_ids:
            ph_fid = ",".join(["%s"] * len(window_fixture_ids))
            ph_pid = ",".join(["%s"] * len(squad_player_ids))
            ph_sid = ",".join(["%s"] * len(_ALL_PLAYER_STAT_IDS))
            await cur.execute(
                f"""
                SELECT fixture_id, player_id, stats_type_id, value
                FROM fixture_player_stats
                WHERE fixture_id IN ({ph_fid})
                  AND player_id IN ({ph_pid})
                  AND stats_type_id IN ({ph_sid})
                """,
                tuple(window_fixture_ids) + tuple(squad_player_ids) + tuple(_ALL_PLAYER_STAT_IDS),
            )
            player_stat_rows = await cur.fetchall()

        # 4. Team stats for the window fixtures, ALL teams — covers both the
        #    national teams and the clubs the players turned out for, plus
        #    (via stat 56) the opponent fouls for the Fouls Drawn denominator.
        #    Mirror domestic: filter `fixtures.stats_imported = 1` so empty-
        #    import fixtures don't pollute team denominators. Same as
        #    statz_functions.py:333 / :375 in the domestic path.
        team_stat_rows = []
        team_player_sum_rows = []
        if window_fixture_ids:
            ph_fid = ",".join(["%s"] * len(window_fixture_ids))
            ph_sid = ",".join(["%s"] * len(_ALL_TEAM_STAT_IDS))
            await cur.execute(
                f"""
                SELECT fts.fixture_id, fts.team_id, fts.stats_type_id, fts.value
                FROM fixture_team_stats fts
                JOIN fixtures f ON f.id = fts.fixture_id
                WHERE fts.fixture_id IN ({ph_fid})
                  AND fts.stats_type_id IN ({ph_sid})
                  AND f.stats_imported = 1
                """,
                tuple(window_fixture_ids) + tuple(_ALL_TEAM_STAT_IDS),
            )
            team_stat_rows = await cur.fetchall()

            # Per-(fixture, team, stat) SUM of player values for every
            # counting stat. Used below to set each share denominator to
            # max(team_row, player_sum) — see _COUNTING_DENOM_MAP. The team
            # and player feeds should agree (one event = one player) but
            # often don't; both disagreement modes are undercounts, so the
            # larger is the better estimate of the true team total. Same
            # stats_imported=1 gate as the team-stats query so thin imports
            # don't feed a too-low player-sum into the reconciliation.
            ph_csid = ",".join(["%s"] * len(_COUNTING_PLAYER_IDS))
            await cur.execute(
                f"""
                SELECT fps.fixture_id, fps.team_id, fps.stats_type_id, SUM(fps.value) AS total
                FROM fixture_player_stats fps
                JOIN fixtures f ON f.id = fps.fixture_id
                WHERE fps.fixture_id IN ({ph_fid})
                  AND fps.stats_type_id IN ({ph_csid})
                  AND f.stats_imported = 1
                GROUP BY fps.fixture_id, fps.team_id, fps.stats_type_id
                """,
                tuple(window_fixture_ids) + tuple(_COUNTING_PLAYER_IDS),
            )
            team_player_sum_rows = await cur.fetchall()

        # 5. WC team_projections written by the team-stat step.
        # Per-fixture mode narrows to just the requested fixtures.
        tp_fid_filter_sql = ""
        tp_fid_filter_params: tuple = ()
        if fixture_ids_filter:
            ph_tp = ",".join(["%s"] * len(fixture_ids_filter))
            tp_fid_filter_sql = f" AND tp.fixture_id IN ({ph_tp})"
            tp_fid_filter_params = tuple(fixture_ids_filter)
        await cur.execute(
            f"""
            SELECT tp.fixture_id, tp.team_id, tp.opponent_id, tp.venue,
                   tp.kickoff_datetime, tp.goals, tp.shots_total,
                   tp.shots_on_target, tp.fouls, tp.yellowcards, tp.tackles,
                   tp.passes, tp.successful_passes, tp.total_crosses,
                   tp.interceptions, tp.offsides
            FROM team_projections tp
            JOIN fixtures f ON f.id = tp.fixture_id
            WHERE f.competition_id = %s
              {tp_fid_filter_sql}
            """,
            (competition_id,) + tp_fid_filter_params,
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
    # squads already comes shaped {team_id: [(player_id, position), ...]}
    # from squad_provider.load() above — no further reshape required.

    # pstats[(player_id, fixture_id, stat_id)] = value
    pstats: Dict[Tuple[int, int, int], float] = {}
    for fixture_id, player_id, stats_type_id, value in player_stat_rows:
        pstats[(player_id, fixture_id, stats_type_id)] = _f(value)

    # tstats[(team_id, fixture_id, stat_id)] = value
    tstats: Dict[Tuple[int, int, int], float] = {}
    # fixture_fouls[fixture_id] = {team_id: fouls} — to derive opponent fouls
    fixture_fouls: Dict[int, Dict[int, float]] = {}
    for fixture_id, team_id, stats_type_id, value in team_stat_rows:
        v = _f(value)
        tstats[(team_id, fixture_id, stats_type_id)] = v
        if stats_type_id == FOULS_TEAM_STAT_ID:
            fixture_fouls.setdefault(fixture_id, {})[team_id] = v

    # Reconcile every counting-stat denominator to max(team_row, player_sum).
    # No-op where the feeds agree; recovers the team total where the team
    # row is missing/low (Assists, Key Passes worst) without corrupting it
    # where a thin player import would undercount (team_row wins then).
    for fixture_id, team_id, player_sid, total in team_player_sum_rows:
        team_sid = _COUNTING_DENOM_MAP.get(int(player_sid))
        if team_sid is None:
            continue
        key = (int(team_id), int(fixture_id), team_sid)
        tstats[key] = max(tstats.get(key, 0.0), _f(total))

    # opp_fouls[(team_id, fixture_id)] = the OTHER team's fouls in that fixture
    opp_fouls: Dict[Tuple[int, int], float] = {}
    for fixture_id, team_fouls in fixture_fouls.items():
        for team_id in team_fouls:
            other = [v for tid, v in team_fouls.items() if tid != team_id]
            if other:
                opp_fouls[(team_id, fixture_id)] = sum(other) / len(other)

    tp_cols = [
        'fixture_id', 'team_id', 'opponent_id', 'venue', 'kickoff_datetime',
        'goals', 'shots_total', 'shots_on_target', 'fouls', 'yellowcards',
        'tackles', 'passes', 'successful_passes', 'total_crosses',
        'interceptions', 'offsides',
    ]
    team_projections = [dict(zip(tp_cols, r)) for r in team_proj_rows]

    return {
        'squads': squads,
        'player_windows': player_windows,
        'player_xg_windows': player_xg_windows,
        'skipped_no_data': skipped_no_data,
        'pstats': pstats,
        'tstats': tstats,
        'opp_fouls': opp_fouls,
        'team_projections': team_projections,
        'teams': {r[0]: r[1] for r in teams_rows},
        'players': {r[0]: r[1] for r in players_rows},
        'intl_cap_counts': intl_cap_counts,
    }


# ---------------------------------------------------------------------------
# Row building (pure function of the loaded data — extracted so it is
# directly testable / dry-runnable without a DB write).
# ---------------------------------------------------------------------------

def _build_player_rows(data: dict) -> Tuple[list, int]:
    """Run the share distribution over the loaded data. Returns
    (output_rows, n_skipped_team_fixtures_with_no_squad)."""
    squads = data['squads']
    player_windows = data['player_windows']
    player_xg_windows = data['player_xg_windows']
    pstats = data['pstats']
    tstats = data['tstats']
    opp_fouls = data['opp_fouls']
    team_projections = data['team_projections']
    teams = data['teams']
    players = data['players']
    intl_cap_counts = data.get('intl_cap_counts', {})
    player_odds = data.get('player_odds', {}) or {}
    odds_blend_weight = float(data.get('odds_blend_weight', 0.3))

    # Player-prop blend helpers hoisted out of the per-row hot loop.
    # PLAYER_BLEND_STAT_NAMES is the single source of truth for the
    # stat-name → stats_type_id map (shared with statz_functions.py).
    from app.services.odds_blend import blend_player_stat, PLAYER_BLEND_STAT_NAMES

    # team Saves = opponent's (SoT − Goals) — look up the other row in fixture.
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

        opp_tp = tp_by_fixture.get(tp['fixture_id'], {}).get(tp['opponent_id'])
        if opp_tp is not None:
            team_saves = max(_f(opp_tp['shots_on_target']) - _f(opp_tp['goals']), 0.0)
        else:
            team_saves = 0.0

        team_goals = _f(tp['goals'])
        team_shots = _f(tp['shots_total'])
        derived_totals = {
            'Assists': team_goals * ASSISTS_PER_GOAL,
            'Key Passes': team_shots * KEY_PASSES_PER_SHOT,
        }
        opp_fouls_total = _f(opp_tp['fouls']) if opp_tp is not None else 0.0

        for player_id, position_group in squad:
            window = player_windows.get(player_id)
            if not window:
                continue  # no history — skipped + logged in project()

            # Cap-shrink: dampen the share by player's recent intl
            # presence. Counted off the >45-min-filtered appearances so a
            # cameo doesn't count as a cap. Applied uniformly to every
            # share-distributed stat below (Goals, Shots, Assists, ...).
            cap_shrink = _intl_cap_shrink(intl_cap_counts.get(player_id, 0))

            row = {
                'fixture_id': tp['fixture_id'],
                'kickoff_datetime': target_dt,
                'player_id': player_id,
                'Player': players.get(player_id, str(player_id)),
                'Position': position_group,
                'Team': team_name,
                'Opponent': opp_name,
                'Venue': tp['venue'],
                # No pre-tournament lineup data — Start? stays 'No' for v1.
                'Start?': 'No',
            }

            # Share-distributed stats.
            for out_stat, (p_sid, t_sid, tp_col) in SHARE_STATS.items():
                player_vals = {
                    fid: pstats.get((player_id, fid, p_sid), 0.0)
                    for (fid, _ko, _ic, _tid) in window
                }
                team_vals = {
                    fid: tstats.get((tid, fid, t_sid))
                    for (fid, _ko, _ic, tid) in window
                }
                share, _n = _weighted_share(window, player_vals, team_vals, target_dt)

                # Goals: blend the goals share 50/50 with the xG share. The
                # xG share runs over the player's SEPARATE xG window (recent
                # games that actually carry xG), so a player whose caps lack
                # xG still gets it from club games. Skipped when the xG
                # window is too thin to be reliable.
                if out_stat == 'Goals':
                    xg_window = player_xg_windows.get(player_id, [])
                    if len(xg_window) >= XG_MIN_GAMES:
                        xg_player_vals = {
                            fid: pstats.get((player_id, fid, XG_STAT_ID), 0.0)
                            for (fid, _ko, _ic, _tid) in xg_window
                        }
                        xg_team_vals = {
                            fid: tstats.get((tid, fid, XG_STAT_ID))
                            for (fid, _ko, _ic, tid) in xg_window
                        }
                        xg_share, _xn = _weighted_share(
                            xg_window, xg_player_vals, xg_team_vals, target_dt
                        )
                        if xg_share > 0:
                            # Tiered xG weight: full GOALS_BLEND_XG_WEIGHT
                            # when xg window has >=XG_FULL_WEIGHT_N games,
                            # linearly less when thinner. Keeps the 30/70
                            # G/xG blend honest for players with rich xG
                            # history while preventing thin samples (5-15
                            # games) from dominating a player's projection.
                            xg_w = min(
                                GOALS_BLEND_XG_WEIGHT,
                                len(xg_window) / XG_FULL_WEIGHT_N * GOALS_BLEND_XG_WEIGHT,
                            )
                            goals_w = 1.0 - xg_w
                            share = share * goals_w + xg_share * xg_w

                row[out_stat] = round(_f(tp[tp_col]) * share * cap_shrink, 2)

            # Derived stats (Assists, Key Passes).
            for out_stat, (p_sid, t_sid) in DERIVED_SHARE_STATS.items():
                player_vals = {
                    fid: pstats.get((player_id, fid, p_sid), 0.0)
                    for (fid, _ko, _ic, _tid) in window
                }
                team_vals = {
                    fid: tstats.get((tid, fid, t_sid))
                    for (fid, _ko, _ic, tid) in window
                }
                share, _n = _weighted_share(window, player_vals, team_vals, target_dt)
                row[out_stat] = round(derived_totals[out_stat] * share * cap_shrink, 2)

            # Fouls Drawn — team total = the opponent's projected Fouls;
            # share denominator = the opponent's fouls across the window.
            fd_player_vals = {
                fid: pstats.get((player_id, fid, FOULS_DRAWN_PLAYER_STAT_ID), 0.0)
                for (fid, _ko, _ic, _tid) in window
            }
            fd_team_vals = {
                fid: opp_fouls.get((tid, fid))
                for (fid, _ko, _ic, tid) in window
            }
            fd_share, _n = _weighted_share(window, fd_player_vals, fd_team_vals, target_dt)
            row['Fouls Drawn'] = round(opp_fouls_total * fd_share * cap_shrink, 2)

            # Saves — keepers get the team total; outfielders 0.
            row['Saves'] = round(team_saves, 2) if position_group == 'GK' else 0.0

            # Player-prop blend (Goals / Shots Total / Shots On Target /
            # Assists / Fouls / Tackles / Fouls Drawn / Yellow Cards / Passes).
            # Mutates row[stat] in place; missing-ladder rows fall through
            # untouched. Skipped for GKs on Shots / SoT / Tackles / Goals / Assists —
            # the keeper model is ~0 there, so a bookie λ blended over it
            # produces absurd values. Goals + Assists are skipped because a
            # keeper with anytime scorer/assist odds is a name-collision mismap (outfield
            # "Mladen Jurkas" / midfielder "Éderson" priced as scorers and
            # resolved onto same-named GKs) — never legitimate signal. Fouls/
            # Passes/Cards stay blendable for GKs (real GK markets).
            if player_odds:
                for _stat_name, _stat_type_id in PLAYER_BLEND_STAT_NAMES.items():
                    if _stat_name not in row:
                        continue
                    if position_group == 'GK' and _stat_name in ('Shots Total', 'Shots On Target', 'Tackles', 'Goals', 'Assists'):
                        continue
                    _ladders = (player_odds
                                .get(int(tp['fixture_id']), {})
                                .get(int(player_id), {})
                                .get(_stat_type_id, {}))
                    row[_stat_name] = round(
                        blend_player_stat(
                            float(row[_stat_name]), _ladders,
                            _stat_type_id, odds_blend_weight,
                        ),
                        2,
                    )

            output_rows.append(row)

    return output_rows, n_skipped_no_squad


def _build_prop_rows(player_df: pd.DataFrame) -> pd.DataFrame:
    """Poisson-distribute the per-player expected-value lines across the
    1+/2+/3+ prop markets — the rows player_prop_projections stores.

    Pure function of the player-stat DataFrame (extracted so it is directly
    dry-runnable without a DB write). Mirrors the domestic block in
    projection_service.py: P(X >= line) under Poisson(λ = expected value).
    """
    if player_df.empty:
        return player_df
    # The expected-value 'Fouls' column feeds the 'Fouls Committed' market.
    prop_input = player_df.rename(columns={'Fouls': 'Fouls Committed'})
    probs = get_poisson_probs(prop_input, PROP_STATS, PROP_LINES)
    if 'Yellow Cards' in prop_input.columns:
        yellow = get_poisson_probs(prop_input, ['Yellow Cards'], [1])
        probs = pd.concat([probs, yellow], ignore_index=True)
    return probs.round(2)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class WcPlayerStatService:
    """Compute + write per-player stat projections for upcoming international
    fixtures. Defaults to WC scope when called without an IntlProjectionScope.

    File still named wc_player_stat_service.py because the *historic*
    squad source (wc_squads JOIN wc_players) is FIFA-WC-specific. Non-WC
    scopes inject a different SquadProvider — see intl_squad_provider.py
    for the WcSquadProvider / RecentCapsSquadProvider strategies.
    """

    def __init__(self, scope=None, squad_provider=None):
        # Lazy import — international_projection_service imports this class
        # at module-load.
        if scope is None:
            from app.services.international_projection_service import INTL_SCOPES
            scope = INTL_SCOPES['World Cup']
        self.scope = scope
        if squad_provider is None:
            from app.services.intl_squad_provider import WcSquadProvider
            squad_provider = WcSquadProvider()
        self.squad_provider = squad_provider

    async def project(self, commit: bool = True, fixture_ids: list = None) -> dict:
        """fixture_ids: optional — when set, scope projection + DELETE to those fixtures only."""
        logger.info(
            f"{self.scope.competition_name} player-stat projection start — "
            f"commit={commit}, fixture_ids={fixture_ids}"
        )

        conn = await get_source_connection()
        try:
            data = await _load_data(
                conn,
                competition_id=self.scope.competition_id,
                squad_provider=self.squad_provider,
                fixture_ids_filter=fixture_ids,
            )
            squads = data['squads']
            player_windows = data['player_windows']
            player_xg_windows = data['player_xg_windows']
            skipped_no_data = data['skipped_no_data']
            team_projections = data['team_projections']
            teams = data['teams']
            players = data['players']

            if not squads:
                if self.scope.has_squad_source:
                    logger.warning(
                        f"No linked FIFA-game players in wc_players for {self.scope.competition_name} "
                        f"— nothing to project. Run `php artisan wc:link --players-only` on statz."
                    )
                else:
                    # INFO not WARNING: for recent-caps comps (friendlies/quals)
                    # an empty pool is the expected "nothing to project right
                    # now" state (no upcoming fixtures, or only minnow nations
                    # with no caps). Not a fault — keep it out of the digest.
                    logger.info(
                        f"No squads loaded for {self.scope.competition_name} — "
                        f"RecentCapsSquadProvider returned an empty pool. Likely no "
                        f"upcoming fixtures, or every selected nation has zero "
                        f"recent caps in international comps."
                    )
                return {'n_player_rows': 0, 'n_squads': 0, 'committed': False}
            if not team_projections:
                logger.warning(
                    f"No team_projections found for {self.scope.competition_name} "
                    f"— run InternationalTeamStatService first."
                )
                return {'n_player_rows': 0, 'n_squads': len(squads), 'committed': False}

            # Pre-load player-prop odds (Goals / Shots Total / SoT / Assists)
            # and stash on data dict so _build_player_rows stays pure. Same
            # cascade as domestic; blend weight flows from the competition's
            # odds_beta (0.5 for WC, 0.7 friendlies) — matching the team-goal
            # blend rather than a fixed 0.3. WC has no pre-tournament confirmed
            # lineups, so no load_confirmed_lineups call here. Run AFTER the
            # empty-squad / empty-team_projections early returns so we don't
            # fire per-book SELECTs on a degenerate run.
            from app.services.odds_blend import (
                load_player_odds, PLAYER_BLEND_BOOKS, PLAYER_BLEND_STAT_IDS,
            )
            _wc_fix_ids = sorted({tp['fixture_id'] for tp in team_projections})
            data['player_odds'] = await load_player_odds(
                conn, _wc_fix_ids, PLAYER_BLEND_STAT_IDS, PLAYER_BLEND_BOOKS,
            )
            # Ignored downstream — blend_player_stat uses the fixed
            # PLAYER_ODDS_BLEND_WEIGHT (0.3, all comps). Kept for the data
            # contract; set to 0.3 so it isn't misleading.
            data['odds_blend_weight'] = 0.3

            # Squad players with no usable history at all — skipped from
            # projection. INFO not WARNING: expected for obscure / deep-squad
            # names we simply have no match data for (can't project a player
            # with zero history). Kept queryable at INFO; out of the digest.
            if skipped_no_data:
                pid_to_team = {
                    pid: tid for tid, members in squads.items() for (pid, _pg) in members
                }
                detail = ", ".join(
                    f"{players.get(pid, pid)} [{teams.get(pid_to_team.get(pid), '?')}]"
                    for pid in skipped_no_data
                )
                logger.info(
                    f"{self.scope.competition_name} player-stat: skipped "
                    f"{len(skipped_no_data)} squad players with no international "
                    f"or club history (no data to project): {detail}"
                )

            logger.info(
                f"Loaded {len(squads)} confirmed squads, {len(team_projections)} "
                f"team-projection rows, {len(player_windows)} players with a game "
                f"window, {len(player_xg_windows)} with an xG window, "
                f"{len(data['pstats'])} player-stat / {len(data['tstats'])} team-stat cells"
            )

            output_rows, n_skipped_no_squad = _build_player_rows(data)

            logger.info(
                f"{self.scope.competition_name} player-stat projection ready: "
                f"{len(output_rows)} player-fixture rows across "
                f"{len({r['Team'] for r in output_rows})} nations, "
                f"skipped {n_skipped_no_squad} team-fixtures with no squad, "
                f"{len(skipped_no_data)} players skipped (no history)"
            )

            n_prop_rows = 0
            if commit and output_rows:
                df = pd.DataFrame(output_rows)
                df['kickoff_datetime'] = pd.to_datetime(df['kickoff_datetime'])

                # Step 6: prop probabilities derived from the same df.
                prop_df = _build_prop_rows(df)
                n_prop_rows = len(prop_df)

                # Idempotent: clear existing WC player + prop rows first.
                # Per-fixture mode scopes the DELETE to those fixtures
                # only — don't wipe other WC fixtures' player rows.
                async with conn.cursor() as cur:
                    if fixture_ids:
                        del_ph = ",".join(["%s"] * len(fixture_ids))
                        await cur.execute(
                            f"DELETE FROM player_projections WHERE fixture_id IN ({del_ph})",
                            tuple(fixture_ids),
                        )
                        await cur.execute(
                            f"DELETE FROM player_prop_projections WHERE fixture_id IN ({del_ph})",
                            tuple(fixture_ids),
                        )
                    else:
                        await cur.execute(
                            """DELETE pp FROM player_projections pp
                               JOIN fixtures f ON f.id = pp.fixture_id
                               WHERE f.competition_id = %s""",
                            (self.scope.competition_id,),
                        )
                        await cur.execute(
                            """DELETE ppp FROM player_prop_projections ppp
                               JOIN fixtures f ON f.id = ppp.fixture_id
                               WHERE f.competition_id = %s""",
                            (self.scope.competition_id,),
                        )
                await conn.commit()

                teams_df = pd.DataFrame(
                    list(data['teams'].items()), columns=['id', 'name']
                )
                await insert_player_async(df, teams=teams_df, competition_id=self.scope.competition_id)
                logger.info(
                    f"{self.scope.competition_name} player-stat projections written: {len(df)} rows"
                )

                await insert_players_stats_async(
                    prop_df, teams=teams_df, competition_id=self.scope.competition_id
                )
                logger.info(
                    f"{self.scope.competition_name} player-prop projections written: {n_prop_rows} rows"
                )

            return {
                'n_player_rows': len(output_rows),
                'n_prop_rows': n_prop_rows,
                'n_squads': len(squads),
                'n_team_projection_rows': len(team_projections),
                'n_skipped_no_data': len(skipped_no_data),
                'committed': commit,
            }
        finally:
            release_source_connection(conn)
