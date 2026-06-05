"""
Monte Carlo tournament simulator.

Cached FIFA WC 2026 best-thirds allocation table at
  /app/app/data/wc_2026_best_thirds_table.json
(495 rows from FIFA Annex C / Wikipedia knockout-stage article, scraped
2026-05-15). When present the simulator uses FIFA's deterministic slot
assignment for any qualifying-third combination; when absent it falls
back to greedy constraint matching.



Generic engine — drives FIFA WC, Euros, Copa America, AFCON, CL knockout
phase. Per-tournament structure comes from a TournamentConfig
(tournament_configs.py); FIFA tiebreaker chain comes from
tournament_tiebreakers.py; the deterministic FIFA knockout bracket
structure is parsed directly from fixture placeholder names in the DB.

Pipeline per run(config, num_sims=10000):
  1. Load qualifying teams + group assignments + Statz ratings
  2. Load group-fixture lambdas from fixture_projections (already bet365-
     blended in the daily WC projection step)
  3. Load the knockout bracket structure: parse placeholder names like
     "1st Group A" / "3rd Group C/E/F/H/I" / "Winner Match 73" /
     "Winner Quarter-final 1" to build a slot→fixture graph that
     mirrors FIFA's published bracket exactly.
  4. For each of `num_sims` iterations:
       a. Sample each group fixture's score from Poisson(λ_h, λ_a)
       b. Resolve group standings with the FIFA tiebreaker chain
       c. Pick advance_per_group + best_thirds_advance qualifiers
       d. Fill knockout slot graph with actual team IDs (handling
          "3rd from X/Y/Z/A/B" best-third allocation)
       e. Walk the bracket, simulating each match (90' + ET if drawn,
          then coin flip with p_favourite for pens)
  5. Aggregate per-sim outcomes into per-team probabilities
  6. Upsert into tournament_projections

Knockout fixture lambdas are computed per-sim from team ratings — bet365
can't price emergent matchups, so we use the pure cross-Poisson model
with the standard 1.3 AVG_GOALS.
"""
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.tournament_configs import TournamentConfig
from app.services.tournament_tiebreakers import resolve_group_order
from app.source_database import get_source_connection, release_source_connection

WC_BEST_THIRDS_TABLE_PATH = '/app/app/data/wc_2026_best_thirds_table.json'

_BEST_THIRDS_TABLE_CACHE = None


def _load_best_thirds_table() -> Optional[dict]:
    """Cached load of FIFA's WC 2026 Annex C best-thirds allocation table.

    Returns None if the file is missing (caller falls back to greedy
    constraint matching). The table maps each qualifying-set of 8 group
    letters (as a sorted concatenated string like 'ABCDEFGH') to a dict
    of {qualifying_group_letter: slot_label} where slot_label is one of
    '1A', '1B', '1D', '1E', '1G', '1I', '1K', '1L' — the 8 R32 slots
    reserved for best-thirds.
    """
    global _BEST_THIRDS_TABLE_CACHE
    if _BEST_THIRDS_TABLE_CACHE is not None:
        return _BEST_THIRDS_TABLE_CACHE
    if not os.path.exists(WC_BEST_THIRDS_TABLE_PATH):
        return None
    with open(WC_BEST_THIRDS_TABLE_PATH) as f:
        raw = json.load(f)
    _BEST_THIRDS_TABLE_CACHE = raw.get('combinations', {})
    return _BEST_THIRDS_TABLE_CACHE

logger = logging.getLogger("tournament_simulation")

AVG_GOALS = 1.3      # cross-Poisson scaling for knockout fixtures
PENS_NOISE = 0.49    # coin-flip baseline; favourite tweak applied on top


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

async def _load_data(conn, config: TournamentConfig) -> dict:
    """Fetch everything the simulator needs in one DB round-trip set."""
    async with conn.cursor() as cur:
        # Ratings (Atk/Def per team) + FIFA confederation (for the
        # Continental Betting market — best finisher per confederation).
        await cur.execute(
            """
            SELECT t.id, t.name, t.confederation, tr.attack, tr.defense
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
        confederation_by_team = {}
        for tid, tname, conf, atk, defn in await cur.fetchall():
            ratings[tid] = (float(atk), float(defn))
            team_name_by_id[tid] = tname
            confederation_by_team[tid] = conf

        # Group-stage fixtures + bet365-blended lambdas (from fixture_projections)
        # We only load fixtures with both teams known (placeholders skipped).
        await cur.execute(
            """
            SELECT f.id, f.group_id, f.home_team_id, f.away_team_id,
                   fp.home_goals, fp.away_goals,
                   COALESCE(f.home_team_goals, 0), COALESCE(f.away_team_goals, 0),
                   f.state_id
            FROM fixtures f
            LEFT JOIN fixture_projections fp ON fp.fixture_id = f.id
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

    # Bracket structure from placeholder names (sportmonks convention)
    bracket = await _load_bracket_structure(conn, config, group_codes)

    return {
        'ratings': ratings,
        'team_name_by_id': team_name_by_id,
        'confederation_by_team': confederation_by_team,
        'group_fixtures': group_fixtures,
        'teams_by_group': teams_by_group,
        'group_codes': group_codes,
        'bracket': bracket,
    }


# ----------------------------------------------------------------------------
# Bracket structure parser — replaces the old balanced-seeded approximation
# with FIFA's actual deterministic bracket as encoded in fixture placeholder
# names. Mirrors the logic in
# c:\laragon\www\statz\app\Http\Controllers\CompetitionKnockout.php
# (`attachParentsPositional`, lines ~653-729).
# ----------------------------------------------------------------------------

# Slot definitions parsed from fixture placeholder team names.
#   ('winner', 'A')                        → 1st of Group A
#   ('runner_up', 'B')                     → 2nd of Group B
#   ('best_third', ['A','B','C','D','F'])  → one of the qualifying 3rds from these groups
#   ('parent', round_name, match_idx)      → winner of an earlier knockout match
SLOT_WINNER_RE = re.compile(r'^1st\s+Group\s+([A-Z])\s*$', re.IGNORECASE)
SLOT_RUNNER_UP_RE = re.compile(r'^2(?:nd|nd position)\s+Group\s+([A-Z])\s*$', re.IGNORECASE)
SLOT_BEST_THIRD_RE = re.compile(r'^3rd\s+Group\s+([A-Z](?:[/\s]+[A-Z])+)\s*$', re.IGNORECASE)
SLOT_MATCH_RE = re.compile(r'^(Winner|Loser)\s+Match\s+(\d+)\s*$', re.IGNORECASE)
SLOT_QF_RE = re.compile(r'^(Winner|Loser)\s+Quarter-finals?\s+(\d+)\s*$', re.IGNORECASE)
SLOT_SF_RE = re.compile(r'^(Winner|Loser)\s+Semi-finals?\s+(\d+)\s*$', re.IGNORECASE)


def _parse_slot(name: str):
    """Parse a placeholder team name into a slot tuple. Returns None if unrecognised."""
    name = (name or '').strip()
    m = SLOT_WINNER_RE.match(name)
    if m:
        return ('winner', m.group(1).upper())
    m = SLOT_RUNNER_UP_RE.match(name)
    if m:
        return ('runner_up', m.group(1).upper())
    m = SLOT_BEST_THIRD_RE.match(name)
    if m:
        letters = sorted(set(re.findall(r'[A-Z]', m.group(1).upper())))
        return ('best_third', letters)
    m = SLOT_MATCH_RE.match(name)
    if m:
        return ('match_ref', m.group(1).lower(), int(m.group(2)))
    m = SLOT_QF_RE.match(name)
    if m:
        return ('qf_ref', m.group(1).lower(), int(m.group(2)))
    m = SLOT_SF_RE.match(name)
    if m:
        return ('sf_ref', m.group(1).lower(), int(m.group(2)))
    return None


# Maps round_name (config.knockout_rounds entries) to the DB stage_name patterns
# Sportmonks uses. Adjust here when new tournaments don't follow these labels.
_STAGE_NAME_PATTERNS = {
    'r32': ['Round of 32'],
    'r16': ['Round of 16'],
    'qf': ['Quarter-finals', 'Quarter-final'],
    'sf': ['Semi-finals', 'Semi-final'],
    'final': ['Final'],
}


async def _load_bracket_structure(conn, config: TournamentConfig, group_codes: Dict[int, str]) -> dict:
    """Parse the knockout-round fixtures into a bracket graph.

    Returns:
      {
        round_name: [
          {'fixture_id': int, 'home_slot': slot, 'away_slot': slot}
          ...
        ],
        '_match_base': int   # FIFA match number of the first R32 fixture (e.g. 73)
      }
    """
    # Find DB stage_ids matching each round in this tournament. Filter to the
    # active tournament edition by stage start_date.
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT DISTINCT s.id, s.name, s.starting_at
            FROM stages s
            JOIN fixtures f ON f.stage_id = s.id
            WHERE f.competition_id = %s AND f.kickoff_datetime > %s
            ORDER BY s.starting_at, s.id
            """,
            (config.competition_id, '2026-05-01'),
        )
        stage_rows = await cur.fetchall()

    stage_id_by_round = {}
    for round_name in config.knockout_rounds:
        for pattern in _STAGE_NAME_PATTERNS.get(round_name, []):
            for sid, sname, _ in stage_rows:
                if sname == pattern:
                    stage_id_by_round[round_name] = sid
                    break
            if round_name in stage_id_by_round:
                break

    bracket = {}
    match_base = None
    for round_name in config.knockout_rounds:
        sid = stage_id_by_round.get(round_name)
        if sid is None:
            continue

        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT f.id, th.name AS home_name, ta.name AS away_name, f.kickoff_datetime
                FROM fixtures f
                JOIN teams th ON th.id = f.home_team_id
                JOIN teams ta ON ta.id = f.away_team_id
                WHERE f.stage_id = %s
                ORDER BY f.kickoff_datetime, f.id
                """,
                (sid,),
            )
            fixture_rows = await cur.fetchall()

        round_matches = []
        for fid, h_name, a_name, _ in fixture_rows:
            home_slot = _parse_slot(h_name)
            away_slot = _parse_slot(a_name)
            round_matches.append({
                'fixture_id': fid,
                'home_slot': home_slot,
                'away_slot': away_slot,
            })

            # Derive match_base from the first "Winner Match N" reference in R16:
            # the smallest N across all R16 home/away slots is the first R32 match.
            if round_name == 'r16' and match_base is None:
                for slot in (home_slot, away_slot):
                    if slot and slot[0] == 'match_ref':
                        n = slot[2]
                        match_base = n if match_base is None else min(match_base, n)
        bracket[round_name] = round_matches

    bracket['_match_base'] = match_base or 1
    return bracket


# ----------------------------------------------------------------------------


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


def _fill_r32_slots(
    r32_matches: list,
    winners_by_group: Dict[str, int],
    runners_up_by_group: Dict[str, int],
    best_thirds_by_group: Dict[str, int],
    qualifying_third_groups: set,
) -> List[Tuple[int, int]]:
    """Resolve placeholder slots in R32 fixtures to actual team IDs.

    Returns ordered list of (home_team_id, away_team_id) pairs, one per
    R32 fixture in the order they appear in r32_matches.

    Best-third slot allocation: if FIFA's Annex C lookup table is
    available, use it — for each qualifying-set of 8 group letters the
    table gives a deterministic slot assignment. Otherwise fall back to
    greedy constraint matching (process most-constrained slots first).
    """
    third_slots = []   # list of (match_idx, side, allowed_groups, winner_group_letter)
    resolved = [[None, None] for _ in r32_matches]

    for i, m in enumerate(r32_matches):
        # Track the "winner" partner letter on the same fixture so we can
        # look up FIFA's slot labels like '1A' / '1G' etc.
        winner_partner_letter = None
        for side, slot in (('home', m['home_slot']), ('away', m['away_slot'])):
            if slot and slot[0] == 'winner':
                winner_partner_letter = slot[1]

        for side, slot in (('home', m['home_slot']), ('away', m['away_slot'])):
            if slot is None:
                continue
            side_idx = 0 if side == 'home' else 1
            if slot[0] == 'winner':
                resolved[i][side_idx] = winners_by_group.get(slot[1])
            elif slot[0] == 'runner_up':
                resolved[i][side_idx] = runners_up_by_group.get(slot[1])
            elif slot[0] == 'best_third':
                third_slots.append((i, side_idx, list(slot[1]), winner_partner_letter))

    # === Attempt FIFA lookup first ===
    fifa_table = _load_best_thirds_table()
    fifa_assigned = False
    if fifa_table:
        key = ''.join(sorted(qualifying_third_groups))
        allocation = fifa_table.get(key)
        if allocation:
            # FIFA's table: {qualifying_group_letter: slot_label like '1A'}
            # Invert to {slot_label: qualifying_group_letter}
            slot_to_group = {v: k for k, v in allocation.items()}
            unfilled = []
            for match_idx, side_idx, allowed, winner_letter in third_slots:
                if winner_letter is None:
                    unfilled.append((match_idx, side_idx, allowed, winner_letter))
                    continue
                slot_label = '1' + winner_letter
                qual_letter = slot_to_group.get(slot_label)
                if qual_letter and qual_letter in qualifying_third_groups:
                    resolved[match_idx][side_idx] = best_thirds_by_group.get(qual_letter)
                else:
                    unfilled.append((match_idx, side_idx, allowed, winner_letter))
            if not unfilled:
                fifa_assigned = True

    # === Fall back to greedy if FIFA lookup unavailable or incomplete ===
    if not fifa_assigned:
        available = set(qualifying_third_groups)
        # Re-derive any already-assigned (FIFA partial) — track which thirds
        # are still unplaced
        for match_idx, side_idx, _, _ in third_slots:
            if resolved[match_idx][side_idx] is not None:
                # Find which group letter this team belongs to and remove from available
                for g, tid in best_thirds_by_group.items():
                    if tid == resolved[match_idx][side_idx] and g in available:
                        available.remove(g)
                        break

        # Greedy: process tightest slots first
        unfilled = [s for s in third_slots if resolved[s[0]][s[1]] is None]
        unfilled.sort(key=lambda s: len([g for g in s[2] if g in available]))
        for match_idx, side_idx, allowed, _ in unfilled:
            candidates = [g for g in allowed if g in available]
            if not candidates:
                candidates = list(available) or list(allowed)
            chosen = random.choice(candidates)
            if chosen in available:
                available.remove(chosen)
            resolved[match_idx][side_idx] = best_thirds_by_group.get(chosen)

    return [tuple(p) for p in resolved]


def _simulate_knockout_match(
    home_id: int, away_id: int,
    ratings: Dict[int, Tuple[float, float]],
    config: TournamentConfig,
) -> Tuple[int, int, int]:
    """Simulate a knockout match.

    Returns (winner_id, home_goals, away_goals). The goal counts are
    90' + ET only — penalty-shootout goals are deliberately NOT counted
    so the per-team goal aggregates match how official competition stats
    treat shootouts. Models 90' + ET + pens if drawn.
    """
    h_atk, h_def = ratings.get(home_id, (100.0, 100.0))
    a_atk, a_def = ratings.get(away_id, (100.0, 100.0))
    lam_h = (h_atk / 100) * (a_def / 100) * AVG_GOALS
    lam_a = (a_atk / 100) * (h_def / 100) * AVG_GOALS

    hg, ag = _sample_score(lam_h, lam_a)
    if hg > ag: return home_id, hg, ag
    if ag > hg: return away_id, hg, ag

    # Drawn → ET (30 min, pro-rated λ)
    et_h = lam_h * config.et_lambda_factor
    et_a = lam_a * config.et_lambda_factor
    et_hg, et_ag = _sample_score(et_h, et_a)
    tot_h, tot_a = hg + et_hg, ag + et_ag
    if et_hg > et_ag: return home_id, tot_h, tot_a
    if et_ag > et_hg: return away_id, tot_h, tot_a

    # Still drawn → penalties. Mild edge to favourite (higher Overall).
    # Shootout goals are not added to tot_h / tot_a.
    home_ovr = h_atk - h_def
    away_ovr = a_atk - a_def
    if home_ovr > away_ovr:
        p_home = config.pens_p_favourite
    elif away_ovr > home_ovr:
        p_home = 1 - config.pens_p_favourite
    else:
        p_home = 0.5
    winner = home_id if random.random() < p_home else away_id
    return winner, tot_h, tot_a


def _simulate_fifa_knockout(
    bracket: dict,
    r32_team_pairs: List[Tuple[int, int]],
    ratings: Dict[int, Tuple[float, float]],
    config: TournamentConfig,
) -> Tuple[Dict[int, str], Dict[int, dict]]:
    """Walk the FIFA bracket from R32 (or R16) down to the final, using
    the parsed slot graph from `bracket`.

    Returns (stage_reached, knockout_goals):
      stage_reached  — {team_id: last round played} (e.g. 'r16' = won R32
                       then lost R16, 'winner' = won the final).
      knockout_goals — {team_id: {'gf': int, 'ga': int}} goals scored /
                       conceded across all knockout matches the team
                       played (90' + ET, shootout goals excluded).
    """
    stage_reached = {}
    knockout_goals = defaultdict(lambda: {'gf': 0, 'ga': 0})
    match_base = bracket.get('_match_base', 1)

    # Round-by-round: per-fixture winners stored so subsequent rounds can
    # look them up via match-number / quarter-final / semi-final refs.
    winners_by_round = {}  # round_name -> [winner_id per fixture in order]
    losers_by_round = {}   # round_name -> [loser_id per fixture in order]
    fifa_number_offsets = {}  # round_name -> starting FIFA match number

    for round_idx, round_name in enumerate(config.knockout_rounds):
        round_matches = bracket.get(round_name, [])
        winners = []
        losers = []

        for fixture_idx, m in enumerate(round_matches):
            # Determine home / away team for this fixture
            if round_name == 'r32':
                home_id, away_id = r32_team_pairs[fixture_idx]
            else:
                home_id = _resolve_ref(m['home_slot'], winners_by_round, losers_by_round,
                                       fifa_number_offsets, match_base)
                away_id = _resolve_ref(m['away_slot'], winners_by_round, losers_by_round,
                                       fifa_number_offsets, match_base)

            if home_id is None or away_id is None:
                # Couldn't resolve a slot (placeholder doesn't match known patterns
                # or upstream match failed). Skip this fixture in this sim.
                winners.append(None)
                losers.append(None)
                continue

            winner_id, hg, ag = _simulate_knockout_match(home_id, away_id, ratings, config)
            knockout_goals[home_id]['gf'] += hg
            knockout_goals[home_id]['ga'] += ag
            knockout_goals[away_id]['gf'] += ag
            knockout_goals[away_id]['ga'] += hg
            loser_id = away_id if winner_id == home_id else home_id
            stage_reached[loser_id] = round_name
            winners.append(winner_id)
            losers.append(loser_id)

        winners_by_round[round_name] = winners
        losers_by_round[round_name] = losers
        # Track FIFA match numbering offsets for "Winner Match N" lookups
        if round_name == 'r32':
            fifa_number_offsets['r32'] = match_base
        elif round_name == 'r16':
            fifa_number_offsets['r16'] = match_base + len(bracket.get('r32', []))

        # Final round — winner gets 'winner' label
        if round_idx == len(config.knockout_rounds) - 1 and len(winners) == 1 and winners[0] is not None:
            stage_reached[winners[0]] = 'winner'

    return stage_reached, dict(knockout_goals)


def _resolve_ref(slot, winners_by_round, losers_by_round, fifa_offsets, match_base):
    """Resolve a parent-match reference slot to a team_id.
       Slot tuples handled here:
         ('match_ref', 'winner'|'loser', fifa_match_num)
         ('qf_ref', 'winner'|'loser', qf_idx)        — 1-indexed
         ('sf_ref', 'winner'|'loser', sf_idx)        — 1-indexed
    """
    if slot is None:
        return None
    kind = slot[0]
    if kind == 'match_ref':
        outcome, fifa_num = slot[1], slot[2]
        source = winners_by_round if outcome == 'winner' else losers_by_round
        # FIFA "Match N" with N in 73..88 = R32 indices 0..15; 89..96 = R16 indices 0..7.
        r32_len = len(source.get('r32') or [])
        r32_base = fifa_offsets.get('r32', match_base)
        r16_base = fifa_offsets.get('r16', r32_base + r32_len)
        if 'r32' in source and r32_base <= fifa_num < r32_base + r32_len:
            return source['r32'][fifa_num - r32_base]
        if 'r16' in source:
            r16_len = len(source.get('r16') or [])
            if r16_base <= fifa_num < r16_base + r16_len:
                return source['r16'][fifa_num - r16_base]
    elif kind == 'qf_ref':
        outcome, idx = slot[1], slot[2] - 1
        source = winners_by_round if outcome == 'winner' else losers_by_round
        return (source.get('qf') or [None])[idx] if idx < len(source.get('qf') or []) else None
    elif kind == 'sf_ref':
        outcome, idx = slot[1], slot[2] - 1
        source = winners_by_round if outcome == 'winner' else losers_by_round
        return (source.get('sf') or [None])[idx] if idx < len(source.get('sf') or []) else None
    return None


def _simulate_one(
    data: dict, config: TournamentConfig,
) -> Dict[int, dict]:
    """Run one Monte Carlo iteration. Returns per-team outcome dict."""
    group_orderings, team_stats = _simulate_group_stage(
        data['teams_by_group'], data['group_fixtures'], config,
    )
    winners, runners_up, best_thirds = _select_qualifiers(group_orderings, team_stats, config)
    qualifiers = set(winners + runners_up + best_thirds)

    # Build per-group-letter lookups for the bracket slot resolver
    group_codes = data['group_codes']
    winners_by_group = {}
    runners_up_by_group = {}
    thirds_by_group = {}   # ALL 3rd-place teams (only 8 of 12 qualify)
    for gid, ordering in group_orderings.items():
        code = group_codes.get(gid)
        if code is None:
            continue
        if len(ordering) >= 1: winners_by_group[code] = ordering[0]
        if len(ordering) >= 2: runners_up_by_group[code] = ordering[1]
        if len(ordering) >= 3: thirds_by_group[code] = ordering[2]

    qualifying_third_groups = {
        code for code, tid in thirds_by_group.items() if tid in best_thirds
    }

    bracket = data['bracket']
    r32_team_pairs = _fill_r32_slots(
        bracket.get('r32', []),
        winners_by_group, runners_up_by_group, thirds_by_group,
        qualifying_third_groups,
    )

    knockout_stage, knockout_goals = _simulate_fifa_knockout(
        bracket, r32_team_pairs, data['ratings'], config,
    )

    outcomes = {}
    for t_id, stats in team_stats.items():
        group_order = group_orderings[stats['group_id']]
        position_in_group = group_order.index(t_id) + 1
        kg = knockout_goals.get(t_id, {'gf': 0, 'ga': 0})
        group_ga = stats['gf'] - stats['gd']
        outcomes[t_id] = {
            'group_id': stats['group_id'],
            'group_position': position_in_group,
            'group_points': stats['points'],
            'group_gd': stats['gd'],
            'group_gf': stats['gf'],
            'group_ga': group_ga,
            'qualified': t_id in qualifiers,
            'is_group_winner': position_in_group == 1,
            'is_best_third': t_id in best_thirds,
            # Bottom of group = last place (position 4 in a 4-team group).
            'is_group_bottom': position_in_group == len(group_order),
            'knockout_stage': knockout_stage.get(t_id, None),  # None = eliminated at groups
            # Whole-tournament goals: group stage + knockout (90'+ET).
            'total_gf': stats['gf'] + kg['gf'],
            'total_ga': group_ga + kg['ga'],
        }
    return outcomes


# ----------------------------------------------------------------------------
# Aggregation + DB write
# ----------------------------------------------------------------------------

def _aggregate(all_sim_outcomes: List[dict], config: TournamentConfig,
               group_codes: Dict[int, str],
               confederation_by_team: Dict[int, Optional[str]],
               ) -> Tuple[Dict[int, dict], Dict[str, dict]]:
    """Convert num_sims worth of per-team outcomes into projection rows.

    Returns (team_agg, group_agg):
      team_agg  — {team_id: {...}}    → tournament_projections
      group_agg — {group_code: {...}} → tournament_group_projections

    Per-sim cross-team metrics (highest/lowest scoring team, best finisher
    per confederation, highest-scoring group, the champion's group) are
    resolved inside the sim loop; ties split the credit fractionally so
    every probability column still sums correctly across teams/groups.
    """
    num_sims = len(all_sim_outcomes)
    if num_sims == 0:
        return {}, {}

    teams = set()
    for s in all_sim_outcomes:
        teams.update(s.keys())

    # stage_order so 'sf' > 'qf', 'final' > 'sf', 'winner' tops the chain.
    stage_order = config.knockout_rounds + ['winner']
    stage_rank = {s: i for i, s in enumerate(stage_order)}

    # Per-team running tallies.
    ctr = {t: defaultdict(float) for t in teams}
    sum_group_pos = defaultdict(float)
    sum_group_pts = defaultdict(float)
    sum_group_gf = defaultdict(float)
    sum_group_ga = defaultdict(float)
    # total_gf kept as a per-sim sample list (not just a sum) so we can
    # derive the mean AND the min/max/p10/p90 range. total_ga stays a sum —
    # we only expose a range for goals-for. ~48 teams × num_sims floats ≈ 4MB.
    total_gf_samples = defaultdict(list)
    sum_total_ga = defaultdict(float)
    team_group_id = {}

    # Whole-tournament total goals per sim (sum of every team's total_gf in
    # that sim = all goals scored in the tournament, each counted once). The
    # mean equals Σ per-team expected_goals_for; the spread is computed here
    # rather than by summing per-team ranges (which would over-state it).
    tournament_total_samples: List[int] = []

    # Per-group running tallies (keyed by group_code).
    grp_win_tournament = defaultdict(float)   # sims whose champion is from this group
    grp_highest_scoring = defaultdict(float)  # fractional credit on ties
    grp_goals_samples = defaultdict(list)     # per-sim group-stage total goals

    for sim in all_sim_outcomes:
        # ---- per-team tallies ----
        for t_id, o in sim.items():
            team_group_id[t_id] = o['group_id']
            sum_group_pos[t_id] += o['group_position']
            sum_group_pts[t_id] += o['group_points']
            sum_group_gf[t_id] += o['group_gf']
            sum_group_ga[t_id] += o['group_ga']
            total_gf_samples[t_id].append(o['total_gf'])
            sum_total_ga[t_id] += o['total_ga']
            c = ctr[t_id]
            if o['is_group_winner']: c['win_group'] += 1
            if o['qualified']: c['qualify'] += 1
            if o['is_group_bottom']: c['group_bottom'] += 1
            stage = o['knockout_stage']
            if stage == 'winner':
                c['winner'] += 1
            if stage is not None:
                played_idx = stage_rank[stage]
                for s_idx, s_name in enumerate(config.knockout_rounds):
                    if played_idx >= s_idx:
                        c[f"reach_{s_name}"] += 1

        # ---- highest / lowest scoring team this sim (whole-tournament gf) ----
        gf_by_team = {t: o['total_gf'] for t, o in sim.items()}
        if gf_by_team:
            max_gf = max(gf_by_team.values())
            min_gf = min(gf_by_team.values())
            top = [t for t, g in gf_by_team.items() if g == max_gf]
            bot = [t for t, g in gf_by_team.items() if g == min_gf]
            for t in top: ctr[t]['highest_scorer'] += 1.0 / len(top)
            for t in bot: ctr[t]['lowest_scorer'] += 1.0 / len(bot)
            # Whole-tournament total goals this sim — sum of every team's
            # total_gf (each goal counted once, by the scoring team).
            tournament_total_samples.append(sum(gf_by_team.values()))

        # ---- best finisher per confederation this sim ----
        # finish_key ranks by knockout stage first, then group form so
        # group-stage exits still order sensibly within a confederation.
        by_conf = defaultdict(list)
        for t_id, o in sim.items():
            conf = confederation_by_team.get(t_id)
            if not conf:
                continue
            stage = o['knockout_stage']
            srank = stage_rank[stage] if stage is not None else -1
            finish_key = (srank, o['group_points'], o['group_gd'], o['group_gf'])
            by_conf[conf].append((t_id, finish_key))
        for entries in by_conf.values():
            best_key = max(k for _, k in entries)
            leaders = [t for t, k in entries if k == best_key]
            for t in leaders:
                ctr[t]['best_in_continent'] += 1.0 / len(leaders)

        # ---- group-level: total goals, highest-scoring group, champion's group ----
        sim_group_goals = defaultdict(int)
        champion_group = None
        for o in sim.values():
            code = group_codes.get(o['group_id'])
            if code is None:
                continue
            sim_group_goals[code] += o['group_gf']
            if o['knockout_stage'] == 'winner':
                champion_group = code
        for code, g in sim_group_goals.items():
            grp_goals_samples[code].append(g)
        if champion_group is not None:
            grp_win_tournament[champion_group] += 1
        if sim_group_goals:
            max_g = max(sim_group_goals.values())
            top_groups = [c for c, g in sim_group_goals.items() if g == max_g]
            for c in top_groups:
                grp_highest_scoring[c] += 1.0 / len(top_groups)

    # ---- assemble per-team rows ----
    agg = {}
    for t_id in teams:
        c = ctr[t_id]
        gid = team_group_id.get(t_id)
        gf_arr = np.asarray(total_gf_samples[t_id], dtype=float)
        gf_mean = float(gf_arr.mean()) if gf_arr.size else 0.0
        agg[t_id] = {
            'group_id': gid,
            'group_code': group_codes.get(gid),
            'expected_group_position': sum_group_pos[t_id] / num_sims,
            'expected_group_points': sum_group_pts[t_id] / num_sims,
            'win_group_percent': 100.0 * c['win_group'] / num_sims,
            'qualify_percent': 100.0 * c['qualify'] / num_sims,
            'win_tournament_percent': 100.0 * c['winner'] / num_sims,
            'finish_bottom_group_percent': 100.0 * c['group_bottom'] / num_sims,
            'best_in_continent_percent': 100.0 * c['best_in_continent'] / num_sims,
            'highest_scoring_team_percent': 100.0 * c['highest_scorer'] / num_sims,
            'lowest_scoring_team_percent': 100.0 * c['lowest_scorer'] / num_sims,
            'expected_group_goals_for': sum_group_gf[t_id] / num_sims,
            'expected_group_goals_against': sum_group_ga[t_id] / num_sims,
            'expected_goals_for': gf_mean,
            'expected_goals_against': sum_total_ga[t_id] / num_sims,
            'expected_goal_difference': gf_mean - (sum_total_ga[t_id] / num_sims),
            # Whole-tournament goals-for range across sims. min/max = single
            # most extreme sim; p10/p90 = stable floor/ceiling band.
            'goals_for_min': float(gf_arr.min()) if gf_arr.size else 0.0,
            'goals_for_max': float(gf_arr.max()) if gf_arr.size else 0.0,
            'goals_for_p10': float(np.percentile(gf_arr, 10)) if gf_arr.size else 0.0,
            'goals_for_p90': float(np.percentile(gf_arr, 90)) if gf_arr.size else 0.0,
        }
        for round_name in config.knockout_rounds:
            agg[t_id][f"reach_{round_name}_percent"] = 100.0 * c[f"reach_{round_name}"] / num_sims

    # ---- assemble per-group rows ----
    group_agg = {}
    all_codes = set(grp_goals_samples) | set(grp_win_tournament) | set(grp_highest_scoring)
    for code in all_codes:
        samples = grp_goals_samples.get(code, [])
        group_agg[code] = {
            'win_tournament_percent': 100.0 * grp_win_tournament.get(code, 0.0) / num_sims,
            'highest_scoring_percent': 100.0 * grp_highest_scoring.get(code, 0.0) / num_sims,
            'expected_goals': (sum(samples) / len(samples)) if samples else 0.0,
        }

    # ---- whole-tournament total-goals distribution ----
    tt_arr = np.asarray(tournament_total_samples, dtype=float)
    tournament_agg = {
        'total_goals_mean': float(tt_arr.mean()) if tt_arr.size else 0.0,
        'total_goals_min': float(tt_arr.min()) if tt_arr.size else 0.0,
        'total_goals_max': float(tt_arr.max()) if tt_arr.size else 0.0,
        'total_goals_p10': float(np.percentile(tt_arr, 10)) if tt_arr.size else 0.0,
        'total_goals_p90': float(np.percentile(tt_arr, 90)) if tt_arr.size else 0.0,
    }

    return agg, group_agg, tournament_agg


async def _write_to_db(
    conn, agg: Dict[int, dict], group_agg: Dict[str, dict],
    tournament_agg: Dict[str, float],
    config: TournamentConfig, num_sims: int,
) -> None:
    """Upsert simulator output — one row per team into
    tournament_projections, one row per group into
    tournament_group_projections, one row per (competition, season) into
    tournament_goal_summary. Idempotent per (competition, season)."""
    now = datetime.now()

    team_rows = []
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
        team_rows.append((
            config.competition_id, config.season_id, t_id,
            p['group_code'],
            round(p['expected_group_position'], 2),
            round(p['expected_group_points'], 2),
            round(p['win_group_percent'], 2),
            round(p['qualify_percent'], 2),
            None if rounds_pct['reach_r32_percent'] is None else round(rounds_pct['reach_r32_percent'], 2),
            None if rounds_pct['reach_r16_percent'] is None else round(rounds_pct['reach_r16_percent'], 2),
            None if rounds_pct['reach_qf_percent'] is None else round(rounds_pct['reach_qf_percent'], 2),
            None if rounds_pct['reach_sf_percent'] is None else round(rounds_pct['reach_sf_percent'], 2),
            None if rounds_pct['reach_final_percent'] is None else round(rounds_pct['reach_final_percent'], 2),
            round(p['win_tournament_percent'], 2),
            round(p['expected_group_goals_for'], 2),
            round(p['expected_group_goals_against'], 2),
            round(p['expected_goals_for'], 2),
            round(p['goals_for_min'], 2),
            round(p['goals_for_max'], 2),
            round(p['goals_for_p10'], 2),
            round(p['goals_for_p90'], 2),
            round(p['expected_goals_against'], 2),
            round(p['expected_goal_difference'], 2),
            round(p['finish_bottom_group_percent'], 2),
            round(p['best_in_continent_percent'], 2),
            round(p['highest_scoring_team_percent'], 2),
            round(p['lowest_scoring_team_percent'], 2),
            num_sims, now, now,
        ))

    group_rows = []
    for code, g in group_agg.items():
        group_rows.append((
            config.competition_id, config.season_id, code,
            round(g['win_tournament_percent'], 2),
            round(g['highest_scoring_percent'], 2),
            round(g['expected_goals'], 2),
            num_sims, now, now,
        ))

    async with conn.cursor() as cur:
        # Idempotent: delete-by-(competition, season) then bulk insert.
        # Mirrors the league_projections pattern.
        await cur.execute(
            "DELETE FROM tournament_projections WHERE competition_id = %s AND season_id = %s",
            (config.competition_id, config.season_id),
        )
        for i in range(0, len(team_rows), 100):
            await cur.executemany(
                """INSERT INTO tournament_projections
                   (competition_id, season_id, team_id,
                    group_code, expected_group_position, expected_group_points,
                    win_group_percent, qualify_percent,
                    reach_r32_percent, reach_r16_percent, reach_qf_percent,
                    reach_sf_percent, reach_final_percent, win_tournament_percent,
                    expected_group_goals_for, expected_group_goals_against,
                    expected_goals_for, goals_for_min, goals_for_max,
                    goals_for_p10, goals_for_p90,
                    expected_goals_against, expected_goal_difference,
                    finish_bottom_group_percent, best_in_continent_percent,
                    highest_scoring_team_percent, lowest_scoring_team_percent,
                    num_sims, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                team_rows[i:i+100],
            )

        await cur.execute(
            "DELETE FROM tournament_group_projections WHERE competition_id = %s AND season_id = %s",
            (config.competition_id, config.season_id),
        )
        if group_rows:
            await cur.executemany(
                """INSERT INTO tournament_group_projections
                   (competition_id, season_id, group_code,
                    win_tournament_percent, highest_scoring_percent, expected_goals,
                    num_sims, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                group_rows,
            )

        # Whole-tournament total-goals distribution — one row per
        # (competition, season). Delete-then-insert keeps it idempotent.
        await cur.execute(
            "DELETE FROM tournament_goal_summary WHERE competition_id = %s AND season_id = %s",
            (config.competition_id, config.season_id),
        )
        await cur.execute(
            """INSERT INTO tournament_goal_summary
               (competition_id, season_id, total_goals_mean, total_goals_min,
                total_goals_max, total_goals_p10, total_goals_p90,
                num_sims, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                config.competition_id, config.season_id,
                round(tournament_agg['total_goals_mean'], 2),
                round(tournament_agg['total_goals_min'], 2),
                round(tournament_agg['total_goals_max'], 2),
                round(tournament_agg['total_goals_p10'], 2),
                round(tournament_agg['total_goals_p90'], 2),
                num_sims, now, now,
            ),
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

            agg, group_agg, tournament_agg = _aggregate(
                all_outcomes, config, data['group_codes'], data['confederation_by_team'],
            )
            await _write_to_db(conn, agg, group_agg, tournament_agg, config, num_sims)
            logger.info(
                f"Tournament sim done — {len(agg)} teams, {len(group_agg)} groups written; "
                f"total goals mean={tournament_agg['total_goals_mean']:.1f} "
                f"[{tournament_agg['total_goals_min']:.0f}, {tournament_agg['total_goals_max']:.0f}] "
                f"p10={tournament_agg['total_goals_p10']:.1f} p90={tournament_agg['total_goals_p90']:.1f}"
            )

            return {
                'name': config.name,
                'num_sims': num_sims,
                'teams': len(agg),
                'groups': len(data['teams_by_group']),
            }
        finally:
            release_source_connection(conn)
