"""Bookmaker odds blending for goal projections.

Reels the model's expected goals (λ_h, λ_a) toward bookie expected goals
when bookie totals markets are priced. Replaces the older blend-in-
probability-space + 2D reverse-solve approach with a cleaner cascade
that exploits the goals over/under markets directly.

Cascade per fixture, per bookmaker (bet365 first, others as fallback):

  Path 1  per-team goals O/U ladders → fit each side's λ independently
  Path 1.5 single-team ladder + match total → derive missing side by subtraction
  Path 2  match total + 1X2 → fit match λ, reverse-solve share to match 1X2
  Path 3  match total only → fit match λ, split via model's per-team ratio
  Path 4  1X2 only → caller's existing reverse-solve (not handled here)
  Path 5  nothing → return None; caller leaves model unchanged

The caller blends our returned (λ_h_bookie, λ_a_bookie) with their model
output at the service-specific blend weight:

    λ_h_final = (1-w) * λ_h_model + w * λ_h_bookie

If we return None, the caller falls back to its existing 1X2-only path.
"""

import math
from typing import Optional, Tuple

import logging
logger = logging.getLogger("odds_blend")

# Bookmakers tried in order. Each entry maps to the column names / source
# in bet365_totals_odds and the equivalents for the other books. v1 is
# bet365-only; fallback books wired in once their totals tables are
# confirmed to cover the goals market.
BOOKIE_PRIORITY = ['bet365']

# Per-stat bookmaker priority for team-stat blending. bet365 always
# tried first per user rule (2026-05-29); rest ordered by per-team
# coverage observed on UCL final. Maps the bookie market name (column
# `market` in *_totals_odds) to the priority list.
TEAM_STAT_BOOKIE_PRIORITY = {
    'corners': ['bet365', 'midnite', 'boylesports', 'coral', 'ladbrokes'],
    'cards':   ['bet365', 'midnite', 'boylesports', 'coral'],
    'shots':   ['midnite', 'boylesports', 'coral'],     # no bet365 coverage
    'sot':     ['midnite', 'boylesports', 'coral'],     # no bet365 coverage
    'fouls':   ['midnite'],                              # midnite only
    'tackles': ['boylesports'],                          # boyle only
}

# Mapping from the team_projections column (statz internal stat name)
# to the bookie market key in TEAM_STAT_BOOKIE_PRIORITY.
STAT_COLUMN_TO_MARKET = {
    'Corners':         'corners',
    'Yellowcards':     'cards',
    'Shots Total':     'shots',
    'Shots On Target': 'sot',
    'Fouls':           'fouls',
    'Tackles':         'tackles',
}


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    # Log-space to avoid overflow on factorial
    logp = -lam + k * math.log(lam)
    for i in range(2, k + 1):
        logp -= math.log(i)
    return math.exp(logp)


def _poisson_p_geq(k: int, lam: float) -> float:
    s = 0.0
    for i in range(k):
        s += _poisson_pmf(i, lam)
    return 1.0 - s


def _margin_stripped_over_prob(over_price: Optional[float], under_price: Optional[float]) -> Optional[float]:
    """Return margin-stripped P(over) if BOTH sides priced; raw 1/over if
    only over is priced; 1 - raw 1/under if only under is priced; None
    if neither. Single-side fallback inflates λ slightly (~5%) but is
    still much better than nothing for fixtures where bet365 prices only
    one side of a line.
    """
    if over_price and under_price:
        po = 1.0 / over_price
        pu = 1.0 / under_price
        return po / (po + pu)
    if over_price:
        return 1.0 / over_price
    if under_price:
        return 1.0 - 1.0 / under_price
    return None


def fit_lambda_from_ladder(ladder: list) -> Optional[float]:
    """Fit a single Poisson λ to a list of over/under lines via
    squared-error MLE in probability space.

    ladder = [(line: float, over_price: float|None, under_price: float|None), ...]

    For each usable line, compute margin-stripped P(X > floor(line)) and
    minimise the sum of squared errors over all lines:

        λ* = argmin Σ_i (P_poisson(X ≥ k_i, λ) - p_i_bookie)^2

    This automatically down-weights extreme lines: their margin-stripped
    probabilities are tiny and barely change with λ, so they contribute
    little to the fit. Near-50/50 lines, by contrast, are highly
    λ-sensitive and dominate. No special tier logic needed.

    Filters out lines with degenerate prices (≤ 1.05 or ≥ 30): those
    are bet365 no-take stubs, not real prices — they carry no info.

    Returns None if no usable line.
    """
    if not ladder:
        return None

    # Build the (k, p_over) pairs to fit. Drop lines where the
    # margin-stripped over-probability is outside [0.10, 0.90] —
    # outside that band three things go wrong simultaneously:
    #   1. The Poisson tail is insensitive to λ → line carries no
    #      real info anyway (squared error stays near 0 for any λ).
    #   2. Bookmaker margins skew asymmetrically at the extremes
    #      (long-shots carry wider margin than the lay side), so the
    #      "margin-stripped" probability isn't actually stripped clean.
    #   3. Most extreme-price lines are bookmaker stubs ("won't take
    #      this bet" placeholder prices) rather than real markets.
    # Filtering on the derived probability subsumes both "over @ 20.0"
    # (P_over ~0.05) and "over @ 1.05" (P_over ~0.95) without
    # hardcoding magic price thresholds. Works the same for over-only
    # ladders (single-side raw 1/price still goes through the filter).
    rows = []
    max_line_kept = 0.0
    for line, over_price, under_price in ladder:
        p_over = _margin_stripped_over_prob(over_price, under_price)
        if p_over is None:
            continue
        if p_over < 0.10 or p_over > 0.90:
            continue
        k = int(math.floor(line)) + 1
        rows.append((k, p_over))
        if line > max_line_kept:
            max_line_kept = line

    if not rows:
        return None

    # Data-driven λ search cap: the deepest kept line + 5 buffer.
    # True λ is essentially always below the deepest priced line —
    # otherwise the bookmaker would offer further lines. +5 gives the
    # grid headroom without slowing the fit. Self-scales to any stat:
    # goals (line up to ~6.5 → cap ~12), corners (~16 → ~21), tackles
    # (~22 → ~27), passes (~30 → ~35).
    lambda_upper_hundredths = int((max_line_kept + 5.0) * 100)

    best_lam = None
    best_err = float('inf')
    lam_hundredths = 5
    while lam_hundredths <= lambda_upper_hundredths:
        lam = lam_hundredths / 100.0
        err = 0.0
        for k, p_bookie in rows:
            err += (_poisson_p_geq(k, lam) - p_bookie) ** 2
        if err < best_err:
            best_err = err
            best_lam = lam
        lam_hundredths += 1

    return best_lam


def reverse_solve_share(lambda_total: float, p_h: float, p_d: float, p_a: float) -> float:
    """Find share_home ∈ [0.05, 0.95] such that Poisson joint
    distribution of (lambda_total * share, lambda_total * (1-share))
    best matches the target 1X2 probabilities.

    1D bounded grid search (step 0.01) + local refinement. Result fits
    the 1X2 spread while honouring the goals-total constraint exactly.
    Used by Path 2 only.
    """
    best_s = 0.5
    best_err = float('inf')

    # Coarse search
    si = 5
    while si <= 95:
        s = si / 100.0
        lh = lambda_total * s
        la = lambda_total * (1.0 - s)
        ph, pd, pa = _hda_from_lambdas(lh, la)
        err = (ph - p_h) ** 2 + (pd - p_d) ** 2 + (pa - p_a) ** 2
        if err < best_err:
            best_err = err
            best_s = s
        si += 1

    # Refine around best_s with step 0.005
    for delta_hundredths in range(-10, 11):
        s = best_s + delta_hundredths * 0.005
        if s < 0.05 or s > 0.95:
            continue
        lh = lambda_total * s
        la = lambda_total * (1.0 - s)
        ph, pd, pa = _hda_from_lambdas(lh, la)
        err = (ph - p_h) ** 2 + (pd - p_d) ** 2 + (pa - p_a) ** 2
        if err < best_err:
            best_err = err
            best_s = s

    return best_s


def _hda_from_lambdas(lh: float, la: float) -> Tuple[float, float, float]:
    """Joint Poisson grid 0..9 → (P_home, P_draw, P_away)."""
    ph = pd_ = pa = 0.0
    for x in range(10):
        px = _poisson_pmf(x, lh)
        for y in range(10):
            py = _poisson_pmf(y, la)
            p = px * py
            if x > y:
                ph += p
            elif x == y:
                pd_ += p
            else:
                pa += p
    return ph, pd_, pa


def derive_bookie_lambdas(
    fixture_id: int,
    lambda_h_model: float,
    lambda_a_model: float,
    bookie_1x2: Optional[Tuple[float, float, float]],
    goals_odds: dict,
) -> Optional[Tuple[float, float]]:
    """Try paths 1 → 3 to derive bookie-implied (λ_h, λ_a) for the
    fixture. Returns None if no path succeeded; caller should then
    fall through to its existing 1X2-only logic.

    bookie_1x2: margin-stripped (p_home, p_draw, p_away) or None.
    goals_odds: dict keyed by bookmaker name → dict with:
        'match':  list[(line, over, under)]    # team_id IS NULL rows
        'home':   list[(line, over, under)]    # team_id = home_team_id
        'away':   list[(line, over, under)]    # team_id = away_team_id

    Order of fallback through bookmakers controlled by BOOKIE_PRIORITY.
    """
    for bookie in BOOKIE_PRIORITY:
        result = _try_paths_for_bookie(
            goals_odds.get(bookie, {}),
            lambda_h_model, lambda_a_model,
            bookie_1x2,
        )
        if result is not None:
            return result
    return None


def _try_paths_for_bookie(
    bookie_data: dict,
    lambda_h_model: float,
    lambda_a_model: float,
    bookie_1x2: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float]]:
    home_ladder = bookie_data.get('home', [])
    away_ladder = bookie_data.get('away', [])
    match_ladder = bookie_data.get('match', [])

    lambda_h_bookie = fit_lambda_from_ladder(home_ladder)
    lambda_a_bookie = fit_lambda_from_ladder(away_ladder)
    lambda_total_bookie = fit_lambda_from_ladder(match_ladder)

    # PATH 1 — both per-team ladders
    if lambda_h_bookie is not None and lambda_a_bookie is not None:
        return lambda_h_bookie, lambda_a_bookie

    # PATH 1.5 — single per-team + match total → derive missing side
    if lambda_h_bookie is not None and lambda_total_bookie is not None:
        return lambda_h_bookie, max(0.05, lambda_total_bookie - lambda_h_bookie)
    if lambda_a_bookie is not None and lambda_total_bookie is not None:
        return max(0.05, lambda_total_bookie - lambda_a_bookie), lambda_a_bookie

    # PATH 2 — match total + 1X2 → reverse-solve share
    if lambda_total_bookie is not None and bookie_1x2 is not None:
        p_h, p_d, p_a = bookie_1x2
        share = reverse_solve_share(lambda_total_bookie, p_h, p_d, p_a)
        return lambda_total_bookie * share, lambda_total_bookie * (1.0 - share)

    # PATH 3 — match total only → split via model ratio
    if lambda_total_bookie is not None:
        denom = lambda_h_model + lambda_a_model
        share = (lambda_h_model / denom) if denom > 0 else 0.5
        return lambda_total_bookie * share, lambda_total_bookie * (1.0 - share)

    # No goals-ladder data → caller falls back to its own 1X2-only path
    return None


def blend_lambdas(
    lambda_h_model: float,
    lambda_a_model: float,
    lambda_h_bookie: float,
    lambda_a_bookie: float,
    w: float,
) -> Tuple[float, float]:
    """Linear blend in goal space.
        λ_final = (1-w) * λ_model + w * λ_bookie
    """
    lh = (1.0 - w) * lambda_h_model + w * lambda_h_bookie
    la = (1.0 - w) * lambda_a_model + w * lambda_a_bookie
    return lh, la


async def load_goals_odds_for_fixtures(conn, fixture_ids: list) -> dict:
    """Pre-load bet365 goals over/under for a batch of fixtures.

    Returns dict keyed by fixture_id:
        {fid: {'bet365': {'match': [...], 'home': [...], 'away': [...]}}}

    Each list element is (line: float, over_price: float|None, under_price: float|None).
    Rows deduped via MAX(price) per (fixture, team_id, line, side) — the
    underlying table can carry multiples from repeated fetch cycles.

    Multi-bookie fallback (Ladbrokes/Coral/Midnite/BoyleSports) will
    layer in alongside the bet365 query once their totals tables are
    confirmed to cover the goals market with comparable depth.
    """
    if not fixture_ids:
        return {}

    # Map fixture → (home_team_id, away_team_id) so we can tag per-team rows.
    fix_ph = ",".join(["%s"] * len(fixture_ids))
    async with conn.cursor() as cur:
        await cur.execute(
            f"SELECT id, home_team_id, away_team_id FROM fixtures WHERE id IN ({fix_ph})",
            tuple(fixture_ids),
        )
        fixture_teams = {row[0]: (row[1], row[2]) for row in await cur.fetchall()}

    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT fixture_id, team_id, line, side, MAX(price) AS price
            FROM bet365_totals_odds
            WHERE market = 'goals' AND fixture_id IN ({fix_ph})
            GROUP BY fixture_id, team_id, line, side
            """,
            tuple(fixture_ids),
        )
        rows = await cur.fetchall()

    # Build per-fixture, per-role line→(over,under) dicts then flatten.
    # Roles: 'match' (team_id IS NULL), 'home' / 'away' by mapping team_id.
    buckets = {}  # buckets[(fid, role)][line] = {'over': p, 'under': p}
    for fid, team_id, line, side, price in rows:
        teams = fixture_teams.get(fid)
        if not teams:
            continue
        home_tid, away_tid = teams
        if team_id is None:
            role = 'match'
        elif team_id == home_tid:
            role = 'home'
        elif team_id == away_tid:
            role = 'away'
        else:
            continue  # team_id we don't recognise for this fixture
        key = (fid, role)
        buckets.setdefault(key, {}).setdefault(float(line), {})[side] = float(price)

    result = {}
    for (fid, role), by_line in buckets.items():
        ladder = []
        for line, sides in sorted(by_line.items()):
            ladder.append((line, sides.get('over'), sides.get('under')))
        result.setdefault(fid, {}).setdefault('bet365', {})[role] = ladder

    logger.info(
        "Loaded bet365 goals O/U for %d/%d fixtures",
        len(result), len(fixture_ids),
    )
    return result


async def load_team_stat_odds(conn, fixture_ids: list, market: str, books: list) -> dict:
    """Generalised totals-odds loader for any market across multiple
    books. Same shape as load_goals_odds_for_fixtures but parameterised:

        market ∈ {'goals', 'corners', 'cards', 'shots', 'sot', 'fouls', 'tackles', ...}
        books  ⊆ {'bet365', 'ladbrokes', 'coral', 'midnite', 'boylesports'}

    Returns nested dict keyed by fixture_id:
        {fid: {book: {'match': [...], 'home': [...], 'away': [...]}}}

    Each list element is (line, over_price, under_price). Books not
    carrying the market for a fixture get an empty dict; downstream
    cascade will fall through them.

    Single SELECT per book — all schemas identical (fixture_id, team_id,
    market, line, side, price). Rows deduped via MAX(price) per
    (fixture, team, line, side) to handle the multi-fetch repeats.
    """
    if not fixture_ids or not books:
        return {}

    # Fixture → (home_team_id, away_team_id) map for tagging per-team rows.
    fix_ph = ",".join(["%s"] * len(fixture_ids))
    async with conn.cursor() as cur:
        await cur.execute(
            f"SELECT id, home_team_id, away_team_id FROM fixtures WHERE id IN ({fix_ph})",
            tuple(fixture_ids),
        )
        fixture_teams = {row[0]: (row[1], row[2]) for row in await cur.fetchall()}

    result = {}
    for book in books:
        table = f"{book}_totals_odds"
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT fixture_id, team_id, line, side, MAX(price) AS price
                FROM {table}
                WHERE market = %s AND fixture_id IN ({fix_ph})
                GROUP BY fixture_id, team_id, line, side
                """,
                (market,) + tuple(fixture_ids),
            )
            rows = await cur.fetchall()

        buckets = {}  # buckets[(fid, role)][line] = {'over': p, 'under': p}
        for fid, team_id, line, side, price in rows:
            teams = fixture_teams.get(fid)
            if not teams:
                continue
            home_tid, away_tid = teams
            if team_id is None:
                role = 'match'
            elif team_id == home_tid:
                role = 'home'
            elif team_id == away_tid:
                role = 'away'
            else:
                continue
            buckets.setdefault((fid, role), {}).setdefault(float(line), {})[side] = float(price)

        for (fid, role), by_line in buckets.items():
            ladder = [(line, sides.get('over'), sides.get('under'))
                      for line, sides in sorted(by_line.items())]
            result.setdefault(fid, {}).setdefault(book, {})[role] = ladder

    n_with_data = sum(1 for fid in fixture_ids if result.get(fid))
    logger.info(
        "Loaded %s odds for %d/%d fixtures across %s",
        market, n_with_data, len(fixture_ids), ",".join(books),
    )
    return result


def derive_team_stat_lambdas(
    odds_for_fixture: dict,
    model_home: float,
    model_away: float,
    books_priority: list,
) -> Optional[Tuple[float, float]]:
    """Cascade for team-stat (corners, cards, shots etc.) lambdas.

    Per priority book: try per-team ladders first (full or partial),
    then match-total split via model ratio. First book that returns a
    usable result wins.

    No 1X2 analog for team stats — cascade is shorter than goals:
      Path 1   per-team ladders (both teams)     → fit each side
      Path 1.5 one per-team + match total        → derive missing side
      Path 2   match total only                  → split via model ratio
      Path 3   nothing for this book             → fall through to next

    Returns (lambda_home_bookie, lambda_away_bookie) or None.
    """
    for book in books_priority:
        book_data = odds_for_fixture.get(book, {})
        if not book_data:
            continue

        home_ladder = book_data.get('home', [])
        away_ladder = book_data.get('away', [])
        match_ladder = book_data.get('match', [])

        lam_h = fit_lambda_from_ladder(home_ladder)
        lam_a = fit_lambda_from_ladder(away_ladder)
        lam_t = fit_lambda_from_ladder(match_ladder)

        # Path 1 — both per-team ladders
        if lam_h is not None and lam_a is not None:
            return lam_h, lam_a

        # Path 1.5 — one per-team + match total
        if lam_h is not None and lam_t is not None:
            return lam_h, max(0.01, lam_t - lam_h)
        if lam_a is not None and lam_t is not None:
            return max(0.01, lam_t - lam_a), lam_a

        # Path 2 — match total only, split by model ratio
        if lam_t is not None:
            denom = model_home + model_away
            if denom > 0:
                share = model_home / denom
                return lam_t * share, lam_t * (1.0 - share)

        # This book has nothing usable — fall through.

    return None


def blend_team_stat(
    model_home: float,
    model_away: float,
    odds_for_fixture: dict,
    market: str,
    blend_weight: float,
) -> Tuple[float, float]:
    """Blend a single team-stat (e.g. corners) per fixture in goal space.

    Returns (final_home, final_away). Falls back to model unchanged
    if no book in the priority list has usable data.
    """
    books = TEAM_STAT_BOOKIE_PRIORITY.get(market, ['bet365'])
    bookie_lambdas = derive_team_stat_lambdas(
        odds_for_fixture, model_home, model_away, books,
    )
    if bookie_lambdas is None:
        return model_home, model_away

    lh_b, la_b = bookie_lambdas
    fh = (1.0 - blend_weight) * model_home + blend_weight * lh_b
    fa = (1.0 - blend_weight) * model_away + blend_weight * la_b
    return fh, fa


def compute_final_goals_and_probs(
    fixture_id: int,
    lambda_h_model: float,
    lambda_a_model: float,
    bookie_1x2_pct: Optional[Tuple[float, float, float]],
    goals_odds: dict,
    odds_weight: float,
    boost: float,
) -> Tuple[float, float, float, float, float]:
    """Single entry point for the goal-blend logic across all 3 services.

    Returns: (final_home_goals, final_away_goals,
              final_home_win_pct, final_draw_pct, final_away_win_pct)

    Path priority:
      1-3 (this module): if goals O/U markets present, derive bookie λs
                         and blend in goal space.
      4   (legacy):      if only 1X2 present, fall back to the original
                         probability-space blend + 2D reverse-solve.
      5   (no odds):     model unchanged.

    bookie_1x2_pct: margin-stripped (p_h, p_d, p_a) as fractions, or None.
    goals_odds: nested dict keyed by bookmaker → {match, home, away}.
    odds_weight: per-service blend weight (0.3 domestic/WC, 0.5 euro comp).
    boost: draw-bias multiplier (1.1 across all services today).
    """
    # Try the new cascade first.
    bookie_lambdas = derive_bookie_lambdas(
        fixture_id, lambda_h_model, lambda_a_model,
        bookie_1x2_pct, goals_odds,
    )

    if bookie_lambdas is not None:
        # Paths 1 / 1.5 / 2 / 3 — blend in goal space.
        lh_b, la_b = bookie_lambdas
        new_h, new_a = blend_lambdas(
            lambda_h_model, lambda_a_model, lh_b, la_b, odds_weight,
        )
        # Compute final H/D/A from the blended λs so they're internally
        # consistent with the output goals.
        ph, pd_, pa = _hda_from_lambdas(new_h, new_a)
        # Apply the same draw-boost shape used by get_result_probs so
        # downstream percentages match what every other path emits.
        pd_boosted = pd_ * boost
        remaining = 1.0 - pd_boosted
        if (ph + pa) > 0:
            ph_norm = (ph / (ph + pa)) * remaining
            pa_norm = (pa / (ph + pa)) * remaining
        else:
            ph_norm = pa_norm = remaining / 2.0
        return new_h, new_a, ph_norm * 100, pd_boosted * 100, pa_norm * 100

    # PATH 4 — legacy 1X2-only blend + 2D reverse-solve. Same maths as
    # before the cascade was added; preserved for fixtures without
    # goals O/U markets.
    if bookie_1x2_pct is None:
        # PATH 5 — no odds at all. Use model unchanged; emit model H/D/A.
        from app.services.statz_functions import get_result_probs
        h, d, a = get_result_probs(lambda_h_model, lambda_a_model, boost)
        return lambda_h_model, lambda_a_model, h, d, a

    from app.services.statz_functions import get_result_probs, find_inputs_for_probs
    p_h, p_d, p_a = bookie_1x2_pct
    bookie_h_pct, bookie_d_pct, bookie_a_pct = p_h * 100, p_d * 100, p_a * 100

    model_h_pct, model_d_pct, model_a_pct = get_result_probs(
        lambda_h_model, lambda_a_model, boost,
    )

    adj_h = model_h_pct + (bookie_h_pct - model_h_pct) * odds_weight
    adj_d = model_d_pct + (bookie_d_pct - model_d_pct) * odds_weight
    adj_a = model_a_pct + (bookie_a_pct - model_a_pct) * odds_weight

    new_h, new_a = find_inputs_for_probs(
        lambda_h_model, lambda_a_model,
        adj_h, adj_d, adj_a, boost,
    )
    return float(new_h), float(new_a), adj_h, adj_d, adj_a
