"""
League projection outcomes — rule-driven market generation.

Replaces the hardcoded per-league `lines` in
statz_functions.get_avg_table_with_probs with sections derived from
`standings.standing_rule_type_id`: positions sharing a Sportmonks rule form
a section, and each section becomes a market. No hardcoded lines, no fixed
columns, rename-proof. See docs/league-projections-redesign.md.

Dual-write companion to predicted_table_repo.insert_predicted_table_async:
that owns the projected table (points / position / goals + the legacy
`*_percent` columns); this owns the rule-driven `league_projection_outcomes`
rows. Both run for one cycle so the new numbers can be diffed for parity
before the read side cuts over.
"""
import logging

from app.repository.db_utils import execute, execute_chunked, fetch_all

logger = logging.getLogger("league_outcome_repo")

# Brazil Serie A is keyed by a fixed competition_id — mirrors the special
# case in predicted_table_repo.insert_predicted_table_async.
_BRAZIL_COMPETITION_ID = 648

_INSERT_SQL = """
INSERT INTO league_projection_outcomes
    (competition_id, season_id, team_id, market_key, rule_type_id,
     position_from, position_to, band_probability, cumulative_probability,
     created_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
ON DUPLICATE KEY UPDATE
    rule_type_id           = VALUES(rule_type_id),
    position_from          = VALUES(position_from),
    position_to            = VALUES(position_to),
    band_probability       = VALUES(band_probability),
    cumulative_probability = VALUES(cumulative_probability),
    updated_at             = NOW()
"""


def resolve_competition_id(comps, league):
    """League name -> competition_id, mirroring insert_predicted_table_async."""
    if league == 'Brazil Serie A':
        return _BRAZIL_COMPETITION_ID
    match = comps.loc[comps['name'] == league, 'id']
    if match.empty:
        raise Exception(f"League {league} not found in comps")
    return int(match.iloc[0])


async def resolve_current_season_id(competition_id):
    """The current season id for a competition, or None if none is flagged."""
    rows = await fetch_all(
        "SELECT id FROM seasons WHERE competition_id = %s AND is_current = 1 "
        "ORDER BY id DESC LIMIT 1",
        (competition_id,),
    )
    return int(rows[0][0]) if rows else None


async def is_split_format_competition(competition_id, season_id):
    """A competition is split-format if its current season carries league
    standings across more than one stage — the Scottish Premiership
    (top-6 / bottom-6) and the Austrian / Belgian / Danish / Greek
    "Championship Round" + "Relegation Round" formats.

    Counts distinct stage_id among STANDINGS rows only, so knockout
    promotion play-offs (no standings -> one stage) and CL / EL (many
    stages, standings only on the league stage) correctly read as one
    stage and are NOT skipped. See docs/league-projections-redesign.md.
    """
    rows = await fetch_all(
        "SELECT COUNT(DISTINCT stage_id) FROM standings "
        "WHERE competition_id = %s AND season_id = %s",
        (competition_id, season_id),
    )
    return bool(rows) and rows[0][0] is not None and int(rows[0][0]) > 1


def build_markets(standings_rows, rule_meta):
    """Group standings positions into rule-driven markets.

    standings_rows : list of (team_id, position, standing_rule_type_id).
    rule_meta      : {rule_type_id: (market_key, direction)} from standing_rule_types.

    Returns an ordered list of market dicts:
      {market_key, rule_type_id, direction, positions:set, position_from, position_to}.
    'win' (band {1}) is always present. Positions are grouped BY rule_type_id
    (not by contiguous run) so a non-contiguous rule — e.g. La Liga's Europa
    rule at positions 6 and 9 — yields one market with positions {6,9}.
    """
    by_rule = {}
    for _team_id, position, rule_type_id in standings_rows:
        if rule_type_id is None or position is None:
            continue
        by_rule.setdefault(int(rule_type_id), set()).add(int(position))

    # 'win' is unconditional — every league has a "win the league" market.
    markets = {
        'win': {
            'market_key': 'win', 'rule_type_id': None,
            'direction': 'top', 'positions': {1},
        }
    }
    for rule_type_id, positions in by_rule.items():
        meta = rule_meta.get(rule_type_id)
        if meta is None:
            # Rule type Sportmonks uses that isn't seeded in
            # standing_rule_types — skip gracefully (matches the read side).
            logger.warning(
                f"[league_outcomes] standing rule_type_id {rule_type_id} not in "
                f"standing_rule_types — section skipped (positions {sorted(positions)})"
            )
            continue
        market_key, direction = meta
        if market_key in markets:
            # Two rule_type_ids mapping to the same market_key — e.g. 289 and
            # 111385 both 'conference_league'. Same market: merge positions.
            markets[market_key]['positions'] |= positions
        else:
            markets[market_key] = {
                'market_key': market_key, 'rule_type_id': rule_type_id,
                'direction': direction, 'positions': set(positions),
            }

    out = []
    for m in markets.values():
        m['position_from'] = min(m['positions'])
        m['position_to'] = max(m['positions'])
        out.append(m)
    # Stable, readable order: by where the band starts then ends.
    out.sort(key=lambda m: (m['position_from'], m['position_to']))
    return out


def build_outcome_rows(all_tables, name_to_id, standings_rows, rule_meta,
                       competition_id, season_id):
    """Pure: compute league_projection_outcomes value tuples from the sim.

    name_to_id : {team name: team_id} scoped to the competition's current
        roster. The sim keys teams by name; globally-ambiguous club names
        (two clubs both "Liverpool" / "Everton") MUST be resolved against the
        roster, not the global teams table, or the row lands under the wrong
        team_id.

    For each (team, market):
      band_probability       = P(finish inside the market's position set)
      cumulative_probability = P(finish <= position_to)   for top-anchored markets
                               P(finish >= position_from) for bottom (relegation)

    Returns (value_tuples, markets) — markets is returned too so the caller
    can log / dry-run the derived market list.
    """
    markets = build_markets(standings_rows, rule_meta)
    if all_tables is None or len(all_tables) == 0:
        return [], markets

    num_sims = int(all_tables['Simulation'].nunique())
    if num_sims == 0:
        return [], markets

    # Position distribution per team — index=Team, columns=Position, value=count.
    pos_pivot = (all_tables.groupby(['Team', 'Position']).size()
                 .unstack(fill_value=0))
    position_cols = list(pos_pivot.columns)

    values = []
    for team in pos_pivot.index:
        team_id = name_to_id.get(team)
        if team_id is None:
            logger.warning(f"[league_outcomes] sim team '{team}' not in competition "
                           f"roster — outcome row skipped")
            continue
        dist = pos_pivot.loc[team]
        for m in markets:
            band_cols = [p for p in position_cols if p in m['positions']]
            band = dist[band_cols].sum() / num_sims * 100.0 if band_cols else 0.0
            if m['direction'] == 'bottom':
                cum_cols = [p for p in position_cols if p >= m['position_from']]
            else:
                cum_cols = [p for p in position_cols if p <= m['position_to']]
            cumulative = dist[cum_cols].sum() / num_sims * 100.0 if cum_cols else 0.0
            values.append((
                competition_id, season_id, int(team_id), m['market_key'],
                m['rule_type_id'], m['position_from'], m['position_to'],
                round(float(band), 2), round(float(cumulative), 2),
            ))
    return values, markets


async def write_league_outcomes_async(all_tables, teams, comps, league):
    """Compute + write league_projection_outcomes for one competition.

    Skips split-format competitions (a single continuous projected table
    can't represent them). Idempotent per (competition, season): DELETE then
    re-insert, so a market that disappears from the standings rules doesn't
    leave a stale row behind.
    """
    competition_id = resolve_competition_id(comps, league)
    season_id = await resolve_current_season_id(competition_id)
    if season_id is None:
        logger.warning(f"[league_outcomes:{league}] no current season for competition "
                        f"{competition_id} — skipping outcomes")
        return 0

    if await is_split_format_competition(competition_id, season_id):
        logger.info(f"[league_outcomes:{league}] split-format competition — "
                    f"skipping outcome generation")
        return 0

    standings_rows = await fetch_all(
        "SELECT team_id, position, standing_rule_type_id FROM standings "
        "WHERE competition_id = %s AND season_id = %s "
        "AND standing_rule_type_id IS NOT NULL",
        (competition_id, season_id),
    )
    rule_rows = await fetch_all("SELECT id, market_key, direction FROM standing_rule_types")
    rule_meta = {int(r[0]): (r[1], r[2]) for r in rule_rows}

    # Competition-scoped team name -> id map. The sim keys teams by name, and
    # resolving those names against the full teams table is ambiguous (two
    # clubs both "Liverpool"). The current-season standings roster is the
    # authoritative, unambiguous team set for this competition.
    roster_rows = await fetch_all(
        "SELECT DISTINCT team_id FROM standings "
        "WHERE competition_id = %s AND season_id = %s",
        (competition_id, season_id),
    )
    id_to_name = dict(zip(teams['id'], teams['name']))
    name_to_id = {}
    for (tid,) in roster_rows:
        nm = id_to_name.get(tid)
        if nm is not None:
            name_to_id[nm] = int(tid)

    values, markets = build_outcome_rows(
        all_tables, name_to_id, standings_rows, rule_meta, competition_id, season_id
    )
    if not values:
        logger.warning(f"[league_outcomes:{league}] no outcome rows built — skipping write")
        return 0

    # Idempotent per (competition, season).
    await execute(
        "DELETE FROM league_projection_outcomes WHERE competition_id = %s AND season_id = %s",
        (competition_id, season_id),
    )
    written = await execute_chunked(
        _INSERT_SQL, values, label=f"[league_projection_outcomes:{league}]"
    )
    logger.info(
        f"[league_outcomes:{league}] wrote {written} rows for season {season_id} — "
        f"{len(markets)} markets: {', '.join(m['market_key'] for m in markets)}"
    )
    return written
