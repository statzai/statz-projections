"""
Monte Carlo tournament simulator.

Generic engine — drives FIFA WC, Euros, Copa America, AFCON, CL knockout
phase. Per-tournament structure comes from a TournamentConfig
(tournament_configs.py); FIFA tiebreaker chain comes from
tournament_tiebreakers.py.

Pipeline per run(config, num_sims=10000):
  1. Load qualifying teams + group assignments + Statz ratings
  2. Load group-fixture lambdas from fixture_projections (already bet365-
     blended in the daily WC projection step)
  3. For each of `num_sims` iterations:
       a. Sample each group fixture's score from Poisson(λ_h, λ_a)
       b. Resolve group standings with the FIFA tiebreaker chain
       c. Pick advance_per_group + best_thirds_advance qualifiers
       d. Seed qualifiers into a standard single-elimination bracket
       e. Walk the bracket, simulating each match (90' + ET if drawn,
          then coin flip with p_favourite for pens)
  4. Aggregate per-sim outcomes into per-team probabilities
  5. Upsert into tournament_projections

Knockout fixture lambdas are computed per-sim from team ratings — bet365
can't price emergent matchups, so we use the pure cross-Poisson model
with the standard 1.3 AVG_GOALS.
"""
import logging
import math
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from app.services.tournament_configs import TournamentConfig
from app.services.tournament_tiebreakers import resolve_group_order
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("tournament_simulation")

AVG_GOALS = 1.3      # cross-Poisson scaling for knockout fixtures
PENS_NOISE = 0.49    # coin-flip baseline; favourite tweak applied on top


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

async def _load_data(conn, config: TournamentConfig) -> dict:
    """Fetch everything the simulator needs in one DB round-trip set."""
    async with conn.cursor() as cur:
        # Ratings (Atk/Def per team)
        await cur.execute(
            """
            SELECT t.id, t.name, tr.attack, tr.defense
            FROM team_ratings tr JOIN teams t ON t.id = tr.team_id
            WHERE tr.competition_id = %s AND tr.inverse = 'No'
              AND tr.date = (
                SELECT MAX(date) FROM team_ratings
                WHERE competition_id = %s AND inverse = 'No' AND date <= CURDATE()
              )
            """,
            (config.competition_id, config.competition_id),
        )
        ratings = {}
        team_name_by_id = {}
        for tid, tname, atk, defn in await cur.fetchall():
            ratings[tid] = (float(atk), float(defn))
            team_name_by_id[tid] = tname

        # Group-stage fixtures + bet365-blended lambdas (from fixture_projections)
        # We only load fixtures with both teams known (placeholders skipped).
        await cur.execute(
            """
            SELECT f.id, f.group_id, f.home_team_id, f.away_team_id,
                   fp.home_goals, fp.away_goals,
                   COALESCE(fhs.home_score, 0), COALESCE(fhs.away_score, 0),
                   f.state_id
            FROM fixtures f
            LEFT JOIN fixture_projections fp ON fp.fixture_id = f.id
            LEFT JOIN fixtures fhs ON fhs.id = f.id
            WHERE f.competition_id = %s
              AND f.group_id IS NOT NULL
              AND f.home_team_id IS NOT NULL
              AND f.away_team_id IS NOT NULL
              AND f.kickoff_datetime > '2026-05-01'
            ORDER BY f.group_id, f.kickoff_datetime
            """,
            (config.competition_id,),
        )
        group_fixtures = []
        groups = defaultdict(set)
        for fid, group_id, h_id, a_id, h_lam, a_lam, h_actual, a_actual, state_id in await cur.fetchall():
            if h_lam is None or a_lam is None:
                # No projection row yet — derive cross-Poisson on the fly
                if h_id in ratings and a_id in ratings:
                    h_atk, h_def = ratings[h_id]
                    a_atk, a_def = ratings[a_id]
                    h_lam = (h_atk / 100) * (a_def / 100) * AVG_GOALS
                    a_lam = (a_atk / 100) * (h_def / 100) * AVG_GOALS
                else:
                    continue
            else:
                h_lam, a_lam = float(h_lam), float(a_lam)
            group_fixtures.append({
                'fixture_id': fid,
                'group_id': group_id,
                'home_id': h_id,
                'away_id': a_id,
                'home_lambda': h_lam,
                'away_lambda': a_lam,
                'played': state_id == 5,
                'actual_home': h_actual,
                'actual_away': a_actual,
            })
            groups[group_id].add(h_id)
            groups[group_id].add(a_id)

    # Sort group_ids ascending → assign labels A, B, C, ... (consistent
    # with how Sportmonks orders them per draw).
    group_codes = {}
    for letter_idx, gid in enumerate(sorted(groups.keys())):
        group_codes[gid] = chr(ord('A') + letter_idx)

    teams_by_group = {gid: list(team_set) for gid, team_set in groups.items()}

    return {
        'ratings': ratings,
        'team_name_by_id': team_name_by_id,
        'group_fixtures': group_fixtures,
        'teams_by_group': teams_by_group,
        'group_codes': group_codes,
    }


# ----------------------------------------------------------------------------
# Single-sim simulation
# ----------------------------------------------------------------------------

def _sample_score(lam_h: float, lam_a: float) -> Tuple[int, int]:
    """Draw a fixture score from independent Poissons."""
    return int(np.random.poisson(lam_h)), int(np.random.poisson(lam_a))


def _simulate_group_stage(
    teams_by_group: Dict[int, List[int]],
    group_fixtures: list,
    config: TournamentConfig,
) -> Tuple[dict, dict]:
    """Returns ({group_id: ordered_team_ids_best_first}, {team_id: {points, gd, gf, group_id}})."""
    # Bucket fixtures per group
    fixtures_by_group = defaultdict(list)
    for f in group_fixtures:
        fixtures_by_group[f['group_id']].append(f)

    group_orderings = {}
    team_stats = {}

    for group_id, teams in teams_by_group.items():
        # Sample each fixture's score (or use actual if already played)
        sim_fixtures = []
        for f in fixtures_by_group[group_id]:
            if f['played']:
                hg, ag = f['actual_home'], f['actual_away']
            else:
                hg, ag = _sample_score(f['home_lambda'], f['away_lambda'])
            sim_fixtures.append({
                'home_id': f['home_id'], 'away_id': f['away_id'],
                'home_goals': hg, 'away_goals': ag,
            })

        # Resolve standings using the configured tiebreaker chain
        ordering = resolve_group_order(teams, sim_fixtures, config.group_tiebreaker_chain)
        group_orderings[group_id] = ordering

        # Record points/GD/GF per team for cross-group ranking + storage
        for t_id in teams:
            pts = gd = gf = 0
            for f in sim_fixtures:
                if t_id not in (f['home_id'], f['away_id']):
                    continue
                is_home = f['home_id'] == t_id
                tgf = f['home_goals'] if is_home else f['away_goals']
                tga = f['away_goals'] if is_home else f['home_goals']
                if tgf > tga: pts += 3
                elif tgf == tga: pts += 1
                gd += tgf - tga
                gf += tgf
            team_stats[t_id] = {'points': pts, 'gd': gd, 'gf': gf, 'group_id': group_id}

    return group_orderings, team_stats


def _select_qualifiers(
    group_orderings: dict,
    team_stats: dict,
    config: TournamentConfig,
) -> Tuple[List[int], List[int], List[int]]:
    """Return (winners, runners_up, best_thirds) lists of team_ids."""
    winners, runners_up, third_place = [], [], []
    for group_id, order in group_orderings.items():
        if len(order) >= 1: winners.append(order[0])
        if len(order) >= 2: runners_up.append(order[1])
        if len(order) >= 3: third_place.append(order[2])

    # Rank third-place teams by [points desc, gd desc, gf desc, random]
    third_ranked = sorted(
        third_place,
        key=lambda t: (
            -team_stats[t]['points'],
            -team_stats[t]['gd'],
            -team_stats[t]['gf'],
            random.random(),
        ),
    )
    best_thirds = third_ranked[: config.best_thirds_advance]
    return winners, runners_up, best_thirds


def _seed_bracket(
    winners: List[int], runners_up: List[int], best_thirds: List[int],
    ratings: Dict[int, Tuple[float, float]],
) -> List[int]:
    """Seed qualifiers into a bracket order. Winners get top seeds, then
    runners-up, then best-thirds, sorted within each tier by overall rating.

    Returns ordered list of qualifier team_ids — index 0 is top seed, last
    index is bottom seed. R32 matchups are then (seed[0] vs seed[-1],
    seed[1] vs seed[-2], …) in the standard knockout pattern.

    NOTE: this is a balanced approximation, not FIFA's exact bracket
    pairings (which depend on group_letter slot rules + which 3rds qualify
    from which groups). The published WC 2026 bracket structure can be
    coded as an explicit pairing table in a follow-up if calibration shows
    it matters.
    """
    def _ovr(t):
        atk, defn = ratings.get(t, (100.0, 100.0))
        return atk - defn

    return (
        sorted(winners, key=_ovr, reverse=True)
        + sorted(runners_up, key=_ovr, reverse=True)
        + sorted(best_thirds, key=_ovr, reverse=True)
    )


def _simulate_knockout_match(
    home_id: int, away_id: int,
    ratings: Dict[int, Tuple[float, float]],
    config: TournamentConfig,
) -> int:
    """Returns the winning team_id. Models 90' + ET + pens if drawn."""
    h_atk, h_def = ratings.get(home_id, (100.0, 100.0))
    a_atk, a_def = ratings.get(away_id, (100.0, 100.0))
    lam_h = (h_atk / 100) * (a_def / 100) * AVG_GOALS
    lam_a = (a_atk / 100) * (h_def / 100) * AVG_GOALS

    hg, ag = _sample_score(lam_h, lam_a)
    if hg > ag: return home_id
    if ag > hg: return away_id

    # Drawn → ET (30 min, pro-rated λ)
    et_h = lam_h * config.et_lambda_factor
    et_a = lam_a * config.et_lambda_factor
    et_hg, et_ag = _sample_score(et_h, et_a)
    if et_hg > et_ag: return home_id
    if et_ag > et_hg: return away_id

    # Still drawn → penalties. Mild edge to favourite (higher Overall).
    home_ovr = h_atk - h_def
    away_ovr = a_atk - a_def
    if home_ovr > away_ovr:
        p_home = config.pens_p_favourite
    elif away_ovr > home_ovr:
        p_home = 1 - config.pens_p_favourite
    else:
        p_home = 0.5
    return home_id if random.random() < p_home else away_id


def _simulate_knockout(
    seeded: List[int],
    ratings: Dict[int, Tuple[float, float]],
    config: TournamentConfig,
) -> Dict[int, str]:
    """Walk the bracket from R32 (or R16 etc.) down to the final.

    Returns {team_id: stage_reached} where stage_reached is the LAST round
    the team played in (e.g. 'r16' = won R32 then lost R16, 'final' = won
    SF then lost final, 'winner' = won the final).
    """
    rounds = config.knockout_rounds  # e.g. ['r32', 'r16', 'qf', 'sf', 'final']
    stage_reached = {}  # team_id -> stage_label

    current_round = seeded[:]
    for round_idx, round_name in enumerate(rounds):
        next_round = []
        n = len(current_round)
        # Standard bracket pairing: seed 0 vs seed -1, seed 1 vs seed -2 …
        for i in range(n // 2):
            home = current_round[i]
            away = current_round[n - 1 - i]
            winner = _simulate_knockout_match(home, away, ratings, config)
            loser = away if winner == home else home
            stage_reached[loser] = round_name  # eliminated at this round
            next_round.append(winner)
        current_round = next_round
        if round_idx == len(rounds) - 1:
            # Final round — winner takes 'winner' label
            if len(current_round) == 1:
                stage_reached[current_round[0]] = 'winner'

    return stage_reached


def _simulate_one(
    data: dict, config: TournamentConfig,
) -> Dict[int, dict]:
    """Run one Monte Carlo iteration. Returns per-team outcome dict."""
    group_orderings, team_stats = _simulate_group_stage(
        data['teams_by_group'], data['group_fixtures'], config,
    )
    winners, runners_up, best_thirds = _select_qualifiers(group_orderings, team_stats, config)
    qualifiers = set(winners + runners_up + best_thirds)

    seeded = _seed_bracket(winners, runners_up, best_thirds, data['ratings'])
    knockout_stage = _simulate_knockout(seeded, data['ratings'], config)

    outcomes = {}
    for t_id, stats in team_stats.items():
        position_in_group = group_orderings[stats['group_id']].index(t_id) + 1
        outcomes[t_id] = {
            'group_id': stats['group_id'],
            'group_position': position_in_group,
            'group_points': stats['points'],
            'group_gd': stats['gd'],
            'group_gf': stats['gf'],
            'qualified': t_id in qualifiers,
            'is_group_winner': position_in_group == 1,
            'is_group_bottom': position_in_group == config.teams_per_group,
            'is_best_third': t_id in best_thirds,
            'knockout_stage': knockout_stage.get(t_id, None),  # None = eliminated at groups
        }
    return outcomes


# ----------------------------------------------------------------------------
# Aggregation + DB write
# ----------------------------------------------------------------------------

def _aggregate(all_sim_outcomes: List[dict], config: TournamentConfig,
               group_codes: Dict[int, str]) -> Dict[int, dict]:
    """Convert num_sims worth of per-team outcomes into per-team probabilities."""
    num_sims = len(all_sim_outcomes)
    teams = set()
    for s in all_sim_outcomes:
        teams.update(s.keys())

    # Build round-reach counts. A team reaches round X if their
    # knockout_stage label is X *or later*. We define stage_order so
    # 'sf' > 'qf', 'final' > 'sf', etc.
    stage_order = config.knockout_rounds + ['winner']
    stage_rank = {s: i for i, s in enumerate(stage_order)}

    agg = {}
    for t_id in teams:
        ctr = defaultdict(int)
        group_positions = []
        group_points = []
        group_id = None

        for sim in all_sim_outcomes:
            if t_id not in sim:
                continue
            o = sim[t_id]
            group_id = o['group_id']
            group_positions.append(o['group_position'])
            group_points.append(o['group_points'])
            if o['is_group_winner']: ctr['win_group'] += 1
            if o['qualified']: ctr['qualify'] += 1
            if o['is_group_bottom']: ctr['bottom'] += 1

            if o['knockout_stage'] == 'winner':
                ctr['winner'] += 1
            stage = o['knockout_stage']
            if stage is not None:
                # Team played in this round (eliminated here OR won it)
                played_idx = stage_rank[stage]
                # Tally "reached round X" for every X up to played_idx
                for s_idx, s_name in enumerate(config.knockout_rounds):
                    if played_idx >= s_idx:
                        ctr[f"reach_{s_name}"] += 1

        agg[t_id] = {
            'group_id': group_id,
            'group_code': group_codes.get(group_id),
            'expected_group_position': sum(group_positions) / len(group_positions),
            'expected_group_points': sum(group_points) / len(group_points),
            'win_group_percent': 100.0 * ctr['win_group'] / num_sims,
            'qualify_percent': 100.0 * ctr['qualify'] / num_sims,
            'finish_bottom_group_percent': 100.0 * ctr['bottom'] / num_sims,
            'win_tournament_percent': 100.0 * ctr['winner'] / num_sims,
        }
        for round_name in config.knockout_rounds:
            agg[t_id][f"reach_{round_name}_percent"] = 100.0 * ctr[f"reach_{round_name}"] / num_sims

    return agg


async def _write_to_db(
    conn, agg: Dict[int, dict], config: TournamentConfig, num_sims: int,
) -> None:
    """Upsert into tournament_projections — one row per team."""
    now = datetime.now()
    rows = []
    for t_id, p in agg.items():
        # Map dynamic round columns. WC has r32→r16→qf→sf→final; Euros
        # has r16→qf→sf→final (no r32). Columns absent from this config
        # stay NULL.
        rounds_pct = {
            'reach_r32_percent': p.get('reach_r32_percent'),
            'reach_r16_percent': p.get('reach_r16_percent'),
            'reach_qf_percent': p.get('reach_qf_percent'),
            'reach_sf_percent': p.get('reach_sf_percent'),
            'reach_final_percent': p.get('reach_final_percent'),
        }
        rows.append((
            config.competition_id, config.season_id, t_id,
            p['group_code'],
            round(p['expected_group_position'], 2),
            round(p['expected_group_points'], 2),
            round(p['win_group_percent'], 2),
            round(p['qualify_percent'], 2),
            round(p['finish_bottom_group_percent'], 2),
            None if rounds_pct['reach_r32_percent'] is None else round(rounds_pct['reach_r32_percent'], 2),
            None if rounds_pct['reach_r16_percent'] is None else round(rounds_pct['reach_r16_percent'], 2),
            None if rounds_pct['reach_qf_percent'] is None else round(rounds_pct['reach_qf_percent'], 2),
            None if rounds_pct['reach_sf_percent'] is None else round(rounds_pct['reach_sf_percent'], 2),
            None if rounds_pct['reach_final_percent'] is None else round(rounds_pct['reach_final_percent'], 2),
            round(p['win_tournament_percent'], 2),
            num_sims, now, now,
        ))

    async with conn.cursor() as cur:
        # Idempotent: delete-by-competition then bulk insert. Mirrors the
        # league_projections pattern.
        await cur.execute(
            "DELETE FROM tournament_projections WHERE competition_id = %s AND season_id = %s",
            (config.competition_id, config.season_id),
        )
        for i in range(0, len(rows), 100):
            await cur.executemany(
                """INSERT INTO tournament_projections
                   (competition_id, season_id, team_id,
                    group_code, expected_group_position, expected_group_points,
                    win_group_percent, qualify_percent, finish_bottom_group_percent,
                    reach_r32_percent, reach_r16_percent, reach_qf_percent,
                    reach_sf_percent, reach_final_percent, win_tournament_percent,
                    num_sims, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows[i:i+100],
            )
    await conn.commit()


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

class TournamentSimulator:
    """Generic tournament Monte Carlo engine. Stateless instance."""

    async def run(self, config: TournamentConfig, num_sims: int = 10_000) -> dict:
        logger.info(f"Tournament sim start — {config.name}, sims={num_sims}")

        conn = await get_source_connection()
        try:
            data = await _load_data(conn, config)
            n_teams = sum(len(ts) for ts in data['teams_by_group'].values())
            logger.info(
                f"Loaded {n_teams} teams across {len(data['teams_by_group'])} groups, "
                f"{len(data['group_fixtures'])} group fixtures, "
                f"{len(data['ratings'])} rating rows"
            )

            all_outcomes = []
            log_every = max(1, num_sims // 10)
            for sim_idx in range(num_sims):
                all_outcomes.append(_simulate_one(data, config))
                if (sim_idx + 1) % log_every == 0:
                    logger.info(f"  sim {sim_idx + 1}/{num_sims}")

            agg = _aggregate(all_outcomes, config, data['group_codes'])
            await _write_to_db(conn, agg, config, num_sims)
            logger.info(f"Tournament sim done — {len(agg)} teams written")

            return {
                'name': config.name,
                'num_sims': num_sims,
                'teams': len(agg),
                'groups': len(data['teams_by_group']),
            }
        finally:
            release_source_connection(conn)
