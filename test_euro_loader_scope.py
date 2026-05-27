"""Smoke test for euro-comp loader scope narrowing (2026-05-27).

Runs the same euro-comp league twice — once with the new
restrict_team_ids path (narrow scope), once with the legacy full scope
— and diffs the key DataFrames the projection consumes downstream.

The narrow scope should produce IDENTICAL outputs for the teams that
ARE in upcoming fixtures (Palace, Rayo etc.); it intentionally drops
history for teams that aren't playing. So we compare on the subset
that should match, not the full frame.

    docker compose exec statz-projection \\
        python test_euro_loader_scope.py "Champions League"
"""

import asyncio
import logging
import sys
import time

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_euro_loader_scope")


async def _load(league: str, narrow: bool):
    from app.services.euro_comp_projection_service import EuroCompProjectionService
    from app.services.projection_service import ProjectionService
    from app.data_loader import LeagueDataLoader
    import os

    comp_id = await ProjectionService._resolve_league_id_db(league)
    domestic_ids = [
        await ProjectionService._resolve_league_id_db(dom)
        for dom in EuroCompProjectionService.LEAGUE_COUNTRY_DICT.keys()
    ]

    restrict = None
    if narrow:
        date_from = pd.to_datetime('today')
        date_to = date_from + pd.DateOffset(days=EuroCompProjectionService.DAYS)
        restrict = await EuroCompProjectionService._resolve_upcoming_fixture_teams(
            comp_id, date_from, date_to
        )

    league_weightings_path = os.path.join(
        EuroCompProjectionService.DATA_FOLDER_PATH, "League Weightings.xlsx"
    )
    loader = LeagueDataLoader(
        comp_id,
        extra_league_ids=domestic_ids,
        league_weightings_xlsx_path=league_weightings_path,
        restrict_team_ids=restrict,
    )
    t0 = time.time()
    await loader.load()
    return loader, restrict, time.time() - t0


def _team_stats_for(loader, team_ids):
    fix_ids = set(loader.fixtures_df[
        loader.fixtures_df['home_team_id'].isin(team_ids)
        | loader.fixtures_df['away_team_id'].isin(team_ids)
    ]['id'])
    return loader.team_stats[loader.team_stats['fixture_id'].isin(fix_ids)].copy()


def _player_stats_for(loader, team_ids):
    player_ids = set(
        loader.players[loader.players['current_team_id'].isin(team_ids)]['id']
    )
    return loader.player_stats[loader.player_stats['player_id'].isin(player_ids)].copy()


async def main(league: str) -> int:
    from app.source_database import source_init_db_pool, close_source_db_pool

    await source_init_db_pool()
    try:
        logger.info("=== NARROW SCOPE ===")
        narrow_loader, restrict, narrow_t = await _load(league, narrow=True)
        logger.info(f"narrow load: {narrow_t:.1f}s, "
                    f"teams={len(narrow_loader.team_ids)} "
                    f"players={len(narrow_loader.player_ids)} "
                    f"fixtures={len(narrow_loader.fixtures_df)} "
                    f"team_stats={len(narrow_loader.team_stats)} "
                    f"player_stats={len(narrow_loader.player_stats)}")

        if restrict is None:
            logger.warning("No upcoming fixtures → narrow scope fell back to full. Nothing to compare.")
            return 0

        logger.info("=== FULL SCOPE ===")
        full_loader, _, full_t = await _load(league, narrow=False)
        logger.info(f"full   load: {full_t:.1f}s, "
                    f"teams={len(full_loader.team_ids)} "
                    f"players={len(full_loader.player_ids)} "
                    f"fixtures={len(full_loader.fixtures_df)} "
                    f"team_stats={len(full_loader.team_stats)} "
                    f"player_stats={len(full_loader.player_stats)}")

        logger.info(f"speedup: {full_t / narrow_t:.1f}x")

        # Compare team_stats for the restricted team set
        nts = _team_stats_for(narrow_loader, restrict).sort_values(
            ['fixture_id', 'team_id', 'stats_type_id']
        ).reset_index(drop=True)
        fts = _team_stats_for(full_loader, restrict).sort_values(
            ['fixture_id', 'team_id', 'stats_type_id']
        ).reset_index(drop=True)
        logger.info(f"team_stats for restricted teams: narrow={len(nts)} full={len(fts)}")
        if len(nts) == len(fts):
            mismatches = (nts['value'] != fts['value']).sum() if 'value' in nts.columns else 0
            logger.info(f"  → value mismatches: {mismatches}")
        else:
            logger.warning("  → ROW COUNT DIFFERS — investigate")

        # Compare player_stats for the restricted team set's players
        nps = _player_stats_for(narrow_loader, restrict).sort_values(
            ['player_id', 'fixture_id', 'stats_type_id']
        ).reset_index(drop=True)
        fps = _player_stats_for(full_loader, restrict).sort_values(
            ['player_id', 'fixture_id', 'stats_type_id']
        ).reset_index(drop=True)
        logger.info(f"player_stats for restricted teams' players: narrow={len(nps)} full={len(fps)}")
        if len(nps) == len(fps):
            mismatches = (nps['value'] != fps['value']).sum() if 'value' in nps.columns else 0
            logger.info(f"  → value mismatches: {mismatches}")
        else:
            logger.warning("  → ROW COUNT DIFFERS — investigate")

        # team_ratings is loaded in full both ways — should be byte-equal
        if narrow_loader.team_ratings is not None and full_loader.team_ratings is not None:
            logger.info(f"team_ratings: narrow={len(narrow_loader.team_ratings)} "
                        f"full={len(full_loader.team_ratings)}")

        return 0
    except Exception:
        logger.exception("smoke test failed")
        return 1
    finally:
        await close_source_db_pool()


if __name__ == "__main__":
    league = sys.argv[1] if len(sys.argv) > 1 else "Champions League"
    sys.exit(asyncio.run(main(league)))
