"""Squad providers for international fixture projections.

The squad source differs per tournament. For the WC we have a named
roster: `wc_squads` joined to `wc_players` carries the 26-player FIFA
fantasy pool with explicit position labels. For friendlies / qualifiers /
non-tournament intl comps no such roster exists — every game pools from
"whoever the manager calls up" which we can only approximate via
fixture_player_stats.

Two providers behind a single Protocol so the player-stat service can
swap implementations per scope without branching internally:

  - WcSquadProvider          — wc_players JOIN wc_squads (existing)
  - RecentCapsSquadProvider  — derived from fixture_player_stats minutes
                               played in the last N months for each
                               national team in scope.

Returned shape is identical for both: a dict mapping team_id to a list
of (player_id, position_group) tuples. position_group is one of GK /
DEF / MID / FWD.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, Protocol, Sequence
import logging

logger = logging.getLogger('intl_squad_provider')

# Public type alias — what _load_data consumes downstream.
SquadPool = dict  # {team_id: [(player_id, position_group), ...]}


class SquadProvider(Protocol):
    """Strategy interface: load the player pool for a set of national teams.

    team_ids: when provided, restrict to these teams. WcSquadProvider
    ignores this (loads all wc_squads). RecentCapsSquadProvider requires
    it (the source query is per-team).
    as_of: anchor date for recency calculations. Defaults to now().
    """
    async def load(
        self,
        conn,
        team_ids: Optional[Sequence[int]] = None,
        *,
        as_of: Optional[datetime] = None,
    ) -> SquadPool:
        ...


class WcSquadProvider:
    """Existing WC behaviour, lifted out of wc_player_stat_service._load_data.

    Reads the FIFA WC fantasy roster: 26 players per nation, each tagged
    GK/DEF/MID/FWD from FIFA's source. The `players.position_id` /
    Sportmonks position is intentionally NOT used here — there's a known
    drift where a club LW gets tagged DEF for his country, which throws
    the fantasy scoring.

    team_ids filter is supported but optional. WC is a single-tournament
    universe so the full pool is usually wanted.
    """

    async def load(
        self,
        conn,
        team_ids: Optional[Sequence[int]] = None,
        *,
        as_of: Optional[datetime] = None,
    ) -> SquadPool:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT s.team_id, p.player_id, p.position
                FROM wc_players p
                JOIN wc_squads s ON s.id = p.squad_id
                WHERE p.player_id IS NOT NULL
                  AND s.team_id IS NOT NULL
                  AND p.status = 'playing'
                """,
            )
            rows = await cur.fetchall()
        pool: SquadPool = {}
        team_ids_set = set(team_ids) if team_ids else None
        for team_id, pid, pos in rows:
            if team_ids_set is not None and team_id not in team_ids_set:
                continue
            pool.setdefault(int(team_id), []).append((int(pid), pos))
        return pool


class RecentCapsSquadProvider:
    """Derives a per-nation squad from recent international appearances.

    For each requested team, finds players who:
      - appeared in an international fixture in the last `lookback_months`
      - played at least `min_minutes` minutes in that appearance
      - made at least `min_caps` such appearances

    Returns up to `limit_per_team` players per nation, ordered by most
    recent appearance then by total caps. Position resolved via the
    4-step fallback in `resolve_player_position` below.

    Used for friendlies, WC quals, Euro quals, Nations League — anywhere
    a formal named roster doesn't exist.
    """

    def __init__(
        self,
        *,
        team_ids: Optional[Sequence[int]] = None,
        lookback_months: int = 24,
        min_minutes: int = 45,
        min_caps: int = 1,
        limit_per_team: int = 30,
        intl_comp_ids: Optional[list] = None,
    ):
        # team_ids stashed at construction so the orchestrator can
        # pre-compute the upcoming-fixture roster once and inject it.
        # load() also accepts team_ids and overrides this default if
        # passed.
        self.team_ids = list(team_ids) if team_ids else []
        self.lookback_months = lookback_months
        self.min_minutes = min_minutes
        self.min_caps = min_caps
        self.limit_per_team = limit_per_team
        # Lazy default to avoid a circular import at module load.
        self._intl_comp_ids = intl_comp_ids

    def _comp_ids(self) -> list:
        if self._intl_comp_ids is None:
            from app.services.international_team_stat_service import INTERNATIONAL_COMP_IDS
            self._intl_comp_ids = list(INTERNATIONAL_COMP_IDS)
        return self._intl_comp_ids

    async def load(
        self,
        conn,
        team_ids: Optional[Sequence[int]] = None,
        *,
        as_of: Optional[datetime] = None,
    ) -> SquadPool:
        # Fall back to the team_ids stashed at construction time.
        if not team_ids:
            team_ids = self.team_ids
        if not team_ids:
            logger.debug("RecentCapsSquadProvider called with empty team_ids — returning empty pool")
            return {}
        # Local import — datetime is module-imported.
        anchor = as_of or datetime.utcnow()
        intl_comp_ids = self._comp_ids()
        comp_ph = ",".join(["%s"] * len(intl_comp_ids))

        # Phase 1: per-team candidate player_id pool from fixture_player_stats.
        candidates_by_team: dict = {}
        async with conn.cursor() as cur:
            for tid in team_ids:
                await cur.execute(
                    f"""
                    SELECT fps.player_id,
                           COUNT(DISTINCT fps.fixture_id) AS n_caps,
                           MAX(f.kickoff_datetime) AS last_cap
                    FROM fixture_player_stats fps
                    JOIN fixtures f ON f.id = fps.fixture_id
                    WHERE fps.team_id = %s
                      AND fps.stats_type_id = 119
                      AND fps.value >= %s
                      AND f.state_id IN (5, 7, 8)
                      AND f.kickoff_datetime > DATE_SUB(%s, INTERVAL %s MONTH)
                      AND f.competition_id IN ({comp_ph})
                    GROUP BY fps.player_id
                    HAVING n_caps >= %s
                    ORDER BY last_cap DESC, n_caps DESC
                    LIMIT %s
                    """,
                    (int(tid), self.min_minutes, anchor, self.lookback_months,
                     *intl_comp_ids, self.min_caps, self.limit_per_team),
                )
                rows = await cur.fetchall()
                candidates_by_team[int(tid)] = [int(r[0]) for r in rows]

        # Phase 2: resolve positions in one batch query.
        all_pids = sorted({pid for pids in candidates_by_team.values() for pid in pids})
        positions = await resolve_player_position(conn, all_pids)

        pool: SquadPool = {}
        for tid, pids in candidates_by_team.items():
            pool[tid] = [(pid, positions.get(pid, 'MID')) for pid in pids]
            if not pool[tid]:
                # INFO not WARNING: expected for minnow nations with no
                # recent caps in our DB — benign data coverage, not a fault.
                # The digest only surfaces WARNING/ERROR.
                logger.info(
                    "RecentCapsSquadProvider: no recent caps for team_id=%s "
                    "(min_minutes=%d, min_caps=%d, lookback_months=%d) — "
                    "fixture's player projections will be empty",
                    tid, self.min_minutes, self.min_caps, self.lookback_months,
                )
        return pool


async def resolve_player_position(conn, player_ids: list) -> dict:
    """4-step fallback: wc_players.position → players.position string →
    fixture_player_lineup.position → 'MID' default.

    Returns {player_id: 'GK'/'DEF'/'MID'/'FWD'}.
    """
    if not player_ids:
        return {}
    out: dict = {}
    ph = ",".join(["%s"] * len(player_ids))
    async with conn.cursor() as cur:
        # Step 1: wc_players (authoritative for any player on a FIFA WC squad).
        await cur.execute(
            f"""
            SELECT player_id, position
            FROM wc_players
            WHERE player_id IN ({ph}) AND player_id IS NOT NULL
            """,
            tuple(player_ids),
        )
        for pid, pos in await cur.fetchall():
            if pos:
                out[int(pid)] = pos

        # Step 2: players.position string (Sportmonks profile).
        # Column is varchar but Sportmonks data sometimes returns NULL or
        # surprise values — defensive isinstance guard.
        missing = [p for p in player_ids if p not in out]
        if missing:
            ph2 = ",".join(["%s"] * len(missing))
            await cur.execute(
                f"SELECT id, position FROM players WHERE id IN ({ph2})",
                tuple(missing),
            )
            STR_MAP = {
                'goalkeeper': 'GK',
                'defender': 'DEF',
                'midfielder': 'MID',
                'attacker': 'FWD',
                'forward': 'FWD',
            }
            for pid, pos_str in await cur.fetchall():
                if isinstance(pos_str, str) and pos_str.lower() in STR_MAP:
                    out[int(pid)] = STR_MAP[pos_str.lower()]

        # Step 3: fixture_player_lineup — most recent finished fixture.
        # Reads detailed_position_code (varchar) NOT position (int — formation
        # slot number, useless for position-group resolution).
        missing = [p for p in player_ids if p not in out]
        if missing:
            ph3 = ",".join(["%s"] * len(missing))
            await cur.execute(
                f"""
                SELECT fpl.player_id, fpl.detailed_position_code
                FROM fixture_player_lineup fpl
                JOIN fixtures f ON f.id = fpl.fixture_id
                WHERE fpl.player_id IN ({ph3})
                  AND f.state_id IN (5, 7, 8)
                ORDER BY f.kickoff_datetime DESC
                """,
                tuple(missing),
            )
            # detailed_position_code is a Sportmonks slug like 'right-back',
            # 'central-midfielder', 'striker', 'goalkeeper'. Map by infix
            # match — covers all variants without an exhaustive enum.
            for pid, code in await cur.fetchall():
                if int(pid) in out:
                    continue
                if not isinstance(code, str):
                    continue
                low = code.lower()
                if 'keeper' in low:
                    out[int(pid)] = 'GK'
                elif 'back' in low or 'defender' in low or 'centre-back' in low or 'center-back' in low:
                    out[int(pid)] = 'DEF'
                elif 'midfield' in low:
                    out[int(pid)] = 'MID'
                elif (
                    'forward' in low
                    or 'striker' in low
                    or 'attacker' in low
                    or 'winger' in low
                ):
                    out[int(pid)] = 'FWD'

    # Step 4: default to MID for anyone still unresolved.
    for pid in player_ids:
        if int(pid) not in out:
            out[int(pid)] = 'MID'
            # DEBUG not WARNING: a benign per-player data fallback (no position
            # on record → MID). Noisy + expected; kept reachable at DEBUG.
            logger.debug(
                "resolve_player_position: no position for player_id=%s — defaulting to MID",
                pid,
            )
    return out
