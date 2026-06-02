"""
World Cup 2026 fixture projections.

Self-contained — ratings AND projections are both refreshed every run,
mirroring the domestic projection pattern.

Score prediction is the **same code path as domestic** (calls the shared
get_result_probs + find_inputs_for_probs helpers in statz_functions).
The only differences are the ratings source (international_ratings rather
than the domestic get_ratings pipeline) and the per-fixture lambda
construction (single AVG_GOALS + host bonus, no league-specific weightings).

Pipeline:
1. Compute fresh Statz international ratings (TEB v5 FIFA opp + MV v8
   nudge + symmetric caps) for today's date, committing to team_ratings.
2. Cross-Poisson per fixture from those ratings: λ_h = (h.atk/100) ×
   (a.def/100) × AVG_GOALS, λ_a similarly. 1.10× host bonus for
   USA/Canada/Mexico at home.
3. Model 1X2 from get_result_probs(home_goals, away_goals, boost).
4. If bet365 1X2 odds exist: de-vig (divide by overround) + linear
   blend at ODDS_BETA → adjusted 1X2 → find_inputs_for_probs to re-solve
   λs that produce the adjusted 1X2.
5. Downstream markets (CS, O1.5, O2.5, BTTS) from the final Poisson grid.
6. Write to fixture_projections (idempotent — DELETE comp=732 + INSERT).

Skipped: fixtures with unknown team rating (knockout bracket placeholders
like "1st Group A vs 3rd Group C/E/F/H/I").

After the fixture markets: a Monte Carlo tournament simulation, then
per-team stat projections (InternationalTeamStatService → team_projections), then
per-player stat projections (WcPlayerStatService → player_projections,
scoped to nations with a confirmed tournament_squads entry).
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from scipy.stats import poisson

from app.services.international_ratings import compute_international_ratings
from app.services.statz_functions import get_result_probs, find_inputs_for_probs
from app.services.tournament_configs import WC_2026
from app.services.tournament_simulation_service import TournamentSimulator
from app.services.international_team_stat_service import InternationalTeamStatService
from app.services.wc_player_stat_service import WcPlayerStatService
from app.services.wc_fantasy_points_service import WcFantasyPointsService
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("international_projection")

# team_ratings storage bucket for international football. All 211 national
# teams' ratings live under this single competition_id regardless of which
# intl comp is being projected — friendlies, qualifiers, WC all read from
# the same pool. Distinct from `scope.competition_id` which is the comp
# whose FIXTURES are being projected.
INTL_RATINGS_BUCKET_COMP_ID = 732
AVG_GOALS = 1.3

# odds_beta = bet365 goal-line blend weight (0 = pure model, 1 = pure bookie).
# Bumped to 0.5 alongside the goals odds-blend cascade rewrite 2026-05-29 —
# matches euro_comp_projection_service. boost=1.0 = no draw inflation
# (international games don't need the draw nudge that league football
# gets per-comp via projection_config).
ODDS_BETA = 0.5
BOOST = 1.0


@dataclass(frozen=True)
class IntlProjectionScope:
    """Per-comp configuration for the international projection pipeline.

    competition_id    — the comp whose fixtures are being projected
    competition_name  — human-readable name (matches competitions.name)
    hosts             — host nation names (only WC has these); empty for
                        non-tournament intl comps
    host_bonus/penalty — λ multipliers when a host plays at home / opp
                        plays in a host country
    bracket_config    — TournamentConfig for tournaments with a knockout
                        bracket (WC, Euros, Copa, AFCON). None for
                        friendlies / qualifiers / Nations League group
                        stage etc. — those skip Step 3.
    has_squad_source  — True when wc_squads/wc_players (or future Euros
                        equivalent) carries the named squad. False routes
                        the player-stat service through the recent-caps
                        provider in intl_squad_provider.py.
    fantasy_rules     — 'fifa_wc_2026' enables Step 6 (WC fantasy points).
                        None for non-WC comps; future Euros fantasy keys
                        would be added here.
    """
    competition_id: int
    competition_name: str
    hosts: frozenset = field(default_factory=frozenset)
    host_bonus: float = 1.0
    host_penalty: float = 1.0
    bracket_config: object = None
    has_squad_source: bool = False
    fantasy_rules: object = None


# Registry of every comp the international projection pipeline knows about.
# Adding a comp = adding an entry here + flipping competitions.is_projected
# = 1 in the DB. Renamed comps trip the name lookup; keys must match
# competitions.name exactly.
INTL_SCOPES = {
    'World Cup': IntlProjectionScope(
        competition_id=732,
        competition_name='World Cup',
        hosts=frozenset({'United States', 'Mexico', 'Canada'}),
        host_bonus=1.10,
        host_penalty=0.90,
        bracket_config=WC_2026,
        has_squad_source=True,
        fantasy_rules='fifa_wc_2026',
    ),
    'Friendly International': IntlProjectionScope(
        competition_id=1082,
        competition_name='Friendly International',
        hosts=frozenset(),       # no host nation
        host_bonus=1.0,          # → no λ adjustment
        host_penalty=1.0,
        bracket_config=None,     # no bracket → Step 3 skipped
        has_squad_source=False,  # no tournament_squads entry → RecentCapsSquadProvider
        fantasy_rules=None,      # no FIFA fantasy → Step 6 skipped
    ),
}


class InternationalProjectionService:
    """Routes any international fixture (national-team comp) through the
    same pipeline. WC is one comp among several — friendlies, qualifiers,
    Euros etc. are added by registering scopes in INTL_SCOPES.

    Stateless — instance method only for parity with EuroCompProjectionService.
    """

    @staticmethod
    def is_international_comp(league: str) -> bool:
        return league in INTL_SCOPES

    async def _fetch_upcoming_team_ids(
        self,
        competition_id: int,
        fixture_ids_filter,
    ) -> list:
        """Return the distinct team_ids playing in upcoming fixtures of
        competition_id (optionally narrowed by fixture_ids_filter).
        Used to scope RecentCapsSquadProvider for non-WC intl comps."""
        fid_sql = ""
        fid_params: tuple = ()
        if fixture_ids_filter:
            ph = ",".join(["%s"] * len(fixture_ids_filter))
            fid_sql = f" AND id IN ({ph})"
            fid_params = tuple(fixture_ids_filter)
        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT DISTINCT team_id FROM (
                        SELECT home_team_id AS team_id FROM fixtures
                        WHERE competition_id = %s
                          AND kickoff_datetime > NOW()
                          AND state_id = 1
                          {fid_sql}
                        UNION
                        SELECT away_team_id AS team_id FROM fixtures
                        WHERE competition_id = %s
                          AND kickoff_datetime > NOW()
                          AND state_id = 1
                          {fid_sql}
                    ) t
                    WHERE team_id IS NOT NULL
                    """,
                    (competition_id,) + fid_params + (competition_id,) + fid_params,
                )
                rows = await cur.fetchall()
            return sorted({int(r[0]) for r in rows})
        finally:
            release_source_connection(conn)

    async def projections(self, league_request=None, commit: bool = True) -> dict:
        """Compute + (optionally) write international fixture projections.

        Self-contained: refreshes Statz international ratings inline
        (committing to team_ratings) before computing fixture projections.
        Score prediction uses the same shared helpers as domestic
        (get_result_probs + find_inputs_for_probs).

        Per-comp behaviour is driven by the IntlProjectionScope looked up
        from INTL_SCOPES by request.league. Hosts/host_bonus, bracket
        config, squad source, fantasy rules all flow from the scope.

        Returns a stats dict:
          {n_total, n_projected, n_blended, n_skipped_unknown_team,
           n_ratings, ratings_snapshot_date, committed, competition_id, ...}
        """
        league_name = league_request.league if league_request else 'World Cup'
        if league_name not in INTL_SCOPES:
            raise ValueError(
                f"InternationalProjectionService called for unknown league {league_name!r}. "
                f"Known: {sorted(INTL_SCOPES.keys())}"
            )
        scope = INTL_SCOPES[league_name]

        # Per-fixture mode (set via LeagueRequest.fixture_ids) skips
        # the slow fixture-independent / bracket-wide steps:
        #   - Step 1 (rating refresh): uses last nightly snapshot from
        #     team_ratings instead of recomputing
        #   - Step 3 (tournament simulator): bracket-wide, not per-fixture
        # Steps 2/4/5/6 each filter SQL by the supplied fixture_ids.
        fixture_ids_filter = None
        if league_request is not None and getattr(league_request, 'fixture_ids', None):
            fixture_ids_filter = [int(x) for x in league_request.fixture_ids]

        logger.info(
            f"International projection start — comp={scope.competition_name} "
            f"(id={scope.competition_id}), commit={commit}, odds_beta={ODDS_BETA}, "
            f"boost={BOOST}, fixture_ids={fixture_ids_filter}"
        )

        # Step 1: refresh Statz ratings inline (skipped in per-fixture mode).
        ratings: dict
        if fixture_ids_filter:
            # Read latest cached team_ratings rows from team_ratings table
            # without recomputing. Fast — single SQL query.
            _r_conn = await get_source_connection()
            try:
                async with _r_conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT t.name, tr.attack, tr.defense
                        FROM team_ratings tr
                        JOIN teams t ON t.id = tr.team_id
                        WHERE tr.competition_id = %s
                          AND tr.date = (
                              SELECT MAX(date) FROM team_ratings
                              WHERE competition_id = %s AND team_id = tr.team_id
                          )
                        """,
                        (INTL_RATINGS_BUCKET_COMP_ID, INTL_RATINGS_BUCKET_COMP_ID),
                    )
                    rows = await cur.fetchall()
                ratings = {r[0]: (float(r[1]), float(r[2])) for r in rows}
            finally:
                release_source_connection(_r_conn)
            logger.info(f"Per-fixture mode: loaded {len(ratings)} ratings from cache (skipped recompute)")
        else:
            ratings_date = date.today()
            statz_df, _ = await compute_international_ratings(ratings_date, commit=commit)
            ratings = {
                row['team_name']: (float(row['attack']), float(row['defense']))
                for _, row in statz_df.iterrows()
            } if not statz_df.empty else {}
            logger.info(f"Refreshed {len(ratings)} Statz ratings for {ratings_date}")

        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                # Upcoming WC fixtures + bet365 1X2 odds (LEFT JOIN, null-safe).
                # When fixture_ids_filter is set, scope the SELECT to that
                # list — typically exactly one fixture for the per-fixture
                # re-projection trigger.
                fid_filter_sql = ""
                fid_filter_params: tuple = ()
                if fixture_ids_filter:
                    placeholders = ",".join(["%s"] * len(fixture_ids_filter))
                    fid_filter_sql = f" AND f.id IN ({placeholders})"
                    fid_filter_params = tuple(fixture_ids_filter)
                await cur.execute(
                    f"""
                    SELECT f.id, f.home_team_id, f.away_team_id, f.kickoff_datetime,
                           th.name AS h, ta.name AS a,
                           bo.home_win_odd, bo.draw_odd, bo.away_win_odd
                    FROM fixtures f
                    JOIN teams th ON th.id = f.home_team_id
                    JOIN teams ta ON ta.id = f.away_team_id
                    LEFT JOIN bet365_fixture_odds bo ON bo.fixture_id = f.id
                    WHERE f.competition_id = %s
                      AND f.kickoff_datetime > NOW()
                      AND f.state_id = 1
                      {fid_filter_sql}
                    ORDER BY f.kickoff_datetime
                    """,
                    (scope.competition_id,) + fid_filter_params,
                )
                fixtures = await cur.fetchall()
                logger.info(
                    f"Loaded {len(fixtures)} upcoming {scope.competition_name} fixtures"
                )

            # Pre-load bet365 goals over/under for these fixtures. The
            # blend cascade (paths 1-3) uses per-team and match-total
            # ladders directly; path 4 (legacy 1X2) is the fallback.
            from app.services.odds_blend import (
                load_goals_odds_for_fixtures,
                compute_final_goals_and_probs,
            )
            wc_fixture_ids = [row[0] for row in fixtures]
            goals_odds_map = await load_goals_odds_for_fixtures(conn, wc_fixture_ids)

            n_blended = 0
            n_skipped_unknown_team = 0
            inserts = []
            for fid, h_tid, a_tid, ko, home, away, oh, od, oa in fixtures:
                if home not in ratings or away not in ratings:
                    n_skipped_unknown_team += 1
                    continue

                # λ from cross-Poisson rating product + host bonus
                h_atk, h_def = ratings[home]
                a_atk, a_def = ratings[away]
                home_goals = (h_atk / 100) * (a_def / 100) * AVG_GOALS
                away_goals = (a_atk / 100) * (h_def / 100) * AVG_GOALS
                if home in scope.hosts:
                    home_goals *= scope.host_bonus
                    away_goals *= scope.host_penalty
                elif away in scope.hosts:
                    away_goals *= scope.host_bonus
                    home_goals *= scope.host_penalty

                # Score prediction via the shared odds-blend cascade.
                # Paths 1-3 consume bet365 goals over/under directly;
                # path 4 (legacy 1X2 blend + reverse-solve) is the
                # fallback; path 5 (no odds at all) leaves model output
                # unchanged.
                bookie_1x2_pct = None
                if oh and od and oa:
                    home_odds_pct = (1.0 / float(oh)) * 100
                    draw_odds_pct = (1.0 / float(od)) * 100
                    away_odds_pct = (1.0 / float(oa)) * 100
                    bookie_margin = 1 + (home_odds_pct + draw_odds_pct + away_odds_pct - 100) / 100
                    bookie_1x2_pct = (
                        home_odds_pct / bookie_margin / 100.0,
                        draw_odds_pct / bookie_margin / 100.0,
                        away_odds_pct / bookie_margin / 100.0,
                    )
                    n_blended += 1

                new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                    compute_final_goals_and_probs(
                        fid,
                        float(home_goals), float(away_goals),
                        bookie_1x2_pct,
                        goals_odds_map.get(fid, {}),
                        ODDS_BETA,
                        BOOST,
                    )
                )

                home_clean_sheet = poisson.pmf(0, new_away_goals)
                away_clean_sheet = poisson.pmf(0, new_home_goals)
                x = np.arange(0, 9)
                y = np.arange(0, 9)
                X, Y = np.meshgrid(x, y)
                Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
                over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
                over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
                both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
                # === End shared score-prediction block ===

                inserts.append((
                    fid, h_tid, a_tid,
                    str(round(new_home_goals, 2)), str(round(new_away_goals, 2)),
                    round(adjusted_home_win_prob, 2),
                    round(adjusted_away_win_prob, 2),
                    round(adjusted_draw_prob, 2),
                    round(home_clean_sheet * 100, 2),
                    round(away_clean_sheet * 100, 2),
                    round(over_1_goals, 2),
                    round(over_2_goals, 2),
                    round(both_teams_score_prob, 2),
                    ko,
                ))

            logger.info(
                f"{scope.competition_name} projection ready: total={len(fixtures)} "
                f"projected={len(inserts)} blended={n_blended} "
                f"skipped_unknown_team={n_skipped_unknown_team}"
            )

            if commit and inserts:
                async with conn.cursor() as cur:
                    # Per-fixture mode scopes the DELETE to the requested
                    # fixture_ids so we don't wipe the rest of the WC
                    # projection rows. Full-comp mode keeps the original
                    # "delete-all-WC-then-reinsert" idempotent pattern.
                    if fixture_ids_filter:
                        del_ph = ",".join(["%s"] * len(fixture_ids_filter))
                        await cur.execute(
                            f"DELETE FROM fixture_projections WHERE fixture_id IN ({del_ph})",
                            tuple(fixture_ids_filter),
                        )
                    else:
                        await cur.execute(
                            """
                            DELETE fp FROM fixture_projections fp
                            JOIN fixtures f ON f.id = fp.fixture_id
                            WHERE f.competition_id = %s
                            """,
                            (scope.competition_id,),
                        )
                    deleted = cur.rowcount
                    now = datetime.now()
                    rows = [(
                        fid, h_tid, a_tid, hg, ag,
                        hw, aw, dw, hcs, acs, o15, o25, btts,
                        ko, now, now
                    ) for fid, h_tid, a_tid, hg, ag, hw, aw, dw, hcs, acs, o15, o25, btts, ko in inserts]
                    for i in range(0, len(rows), 100):
                        await cur.executemany(
                            """INSERT INTO fixture_projections
                               (fixture_id, home_team_id, away_team_id,
                                home_goals, away_goals,
                                home_win_percent, away_win_percent, draw_percent,
                                home_clean_sheet_percent, away_clean_sheet_percent,
                                over_15_goals_percent, over_25_goals_percent,
                                both_teams_shore_percent,
                                kickoff_datetime, created_at, updated_at)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            rows[i:i+100],
                        )
                await conn.commit()
                logger.info(f"WROTE: deleted={deleted}, inserted={len(rows)}")
        finally:
            release_source_connection(conn)

        # Step 3: Monte Carlo tournament simulator (bracket-wide).
        # Skipped in per-fixture mode — the sim reads the entire bracket
        # fixture projection set and walks the bracket; a single-fixture
        # change doesn't shift bracket outcomes enough to be worth a full
        # 10k-sim rerun. The next nightly full pass will refresh it.
        # Also skipped for non-bracket comps (friendlies, quals, Nations
        # League group stage) — their scope.bracket_config is None.
        sim_result = None
        if commit and not fixture_ids_filter and scope.bracket_config is not None:
            sim_result = await TournamentSimulator().run(scope.bracket_config, num_sims=10_000)
            logger.info(f"{scope.competition_name} tournament simulation: {sim_result}")
        elif scope.bracket_config is None:
            logger.info(
                f"Step 3: skipped (no bracket config for {scope.competition_name})"
            )

        # Step 4: Per-team stat projections.
        team_stat_result = None
        if commit:
            try:
                team_stat_result = await InternationalTeamStatService(scope=scope).project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"{scope.competition_name} team-stat projection: {team_stat_result}")
            except Exception as e:
                logger.exception(f"{scope.competition_name} team-stat projection failed: {e}")
                team_stat_result = {'error': str(e)}

        # Step 5: Per-player stat projections.
        #
        # Squad-source strategy depends on the scope:
        #   - WC scope (has_squad_source=True) → WcSquadProvider, reads
        #     wc_squads (the FIFA fantasy roster).
        #   - Non-WC scopes (friendlies / quals / Nations League — no
        #     formal roster) → RecentCapsSquadProvider, derives from
        #     fixture_player_stats minutes played in the last 24 months
        #     for each nation in the upcoming fixture batch.
        player_stat_result = None
        if commit:
            try:
                from app.services.intl_squad_provider import (
                    WcSquadProvider, RecentCapsSquadProvider,
                )
                if scope.has_squad_source:
                    squad_provider = WcSquadProvider()
                else:
                    # Pre-fetch the team_ids playing in the upcoming
                    # fixture batch for this scope. RecentCapsSquadProvider
                    # uses them to know which nations to source squads for.
                    team_ids_in_scope = await self._fetch_upcoming_team_ids(
                        scope.competition_id, fixture_ids_filter,
                    )
                    squad_provider = RecentCapsSquadProvider(team_ids=team_ids_in_scope)
                    logger.info(
                        f"{scope.competition_name} player-stat squad source: "
                        f"recent-caps for {len(team_ids_in_scope)} nations"
                    )
                player_stat_result = await WcPlayerStatService(
                    scope=scope, squad_provider=squad_provider,
                ).project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"{scope.competition_name} player-stat projection: {player_stat_result}")
            except Exception as e:
                logger.exception(f"{scope.competition_name} player-stat projection failed: {e}")
                player_stat_result = {'error': str(e)}

        # Step 6: Per-(fixture, player) Fantasy point projections.
        # FIFA WC fantasy scoring is the only ruleset wired today. Other
        # comps (Euros / Copa / AFCON) will need their own scoring-rule
        # configs before opting in via scope.fantasy_rules.
        fantasy_points_result = None
        if commit and scope.fantasy_rules == 'fifa_wc_2026':
            try:
                fantasy_points_result = await WcFantasyPointsService(scope=scope).project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"{scope.competition_name} fantasy points projection: {fantasy_points_result}")
            except Exception as e:
                logger.exception(f"{scope.competition_name} fantasy points projection failed: {e}")
                fantasy_points_result = {'error': str(e)}

        return {
            'n_total': len(fixtures),
            'n_projected': len(inserts),
            'n_blended': n_blended,
            'n_skipped_unknown_team': n_skipped_unknown_team,
            'n_ratings': len(ratings),
            'ratings_snapshot_date': 'cached' if fixture_ids_filter else str(ratings_date),
            'fixture_ids': fixture_ids_filter,
            'competition_id': scope.competition_id,
            'competition_name': scope.competition_name,
            'committed': commit,
            'tournament_simulation': sim_result,
            'team_stat_projection': team_stat_result,
            'player_stat_projection': player_stat_result,
            'fantasy_points_projection': fantasy_points_result,
        }
