"""
FPL squad-optimisation ILPs — the pure solver core for the on-demand endpoint.

Two entry points, both pure (no DB, no file I/O — the caller passes the fully
built input and gets the result dict back):

  solve_build(players, horizon, from_gw, season_id, budget, scope)
      Best legal squad from scratch by projected points over the horizon, with
      the recommended XI + captain each week. Backs gw1_draft / wildcard /
      freehit (freehit = horizon 1; wildcard = a manager's budget).

  solve_transfer(payload)
      Best set of transfers FROM an existing squad — owned players valued at
      their SELL price, others at BUY; budget = bank + squad sell value;
      objective = Σ(best XI + captain over horizon) − 4 × max(0, transfers − FT).
      Unbounded transfers/hits; taken only when they pay.

Mirrors scripts/fpl/solve_squad.py + scripts/fpl/transfer_plan.py in the statz
repo (the dev/cron runners). Kept in sync by hand — same math, no CLI/JSON I/O.
scipy.optimize.milp (HiGHS).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

BUDGET = 100.0
SQUAD = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
XI_SIZE = 11
CLUB_CAP = 3
HIT = 4.0
POSNAME = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}


def solve_build(players, horizon, from_gw, season_id, budget=BUDGET, scope='preseason'):
    """players: [{id,name,pos,club,price,xpts[H],eligible}]. Returns the draft dict."""
    P = len(players)
    H = horizon
    pos = [pl['pos'] for pl in players]
    price = [pl['price'] for pl in players]
    club = [pl['club'] for pl in players]
    xpts = [pl['xpts'] for pl in players]

    def xi(p): return p
    def yi(p, g): return P + g * P + p
    def ci(p, g): return P + H * P + g * P + p
    N = P + 2 * H * P

    c = np.zeros(N)
    for p in range(P):
        for g in range(H):
            c[yi(p, g)] = -xpts[p][g]
            c[ci(p, g)] = -xpts[p][g]

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = [0]
    def add(coefs, l, u):
        for col, val in coefs:
            rows.append(r[0]); cols.append(col); vals.append(val)
        lb.append(l); ub.append(u); r[0] += 1

    for ps, cnt in SQUAD.items():
        add([(xi(p), 1) for p in range(P) if pos[p] == ps], cnt, cnt)
    add([(xi(p), price[p]) for p in range(P)], 0, budget)
    for cl in set(club):
        add([(xi(p), 1) for p in range(P) if club[p] == cl], 0, CLUB_CAP)
    for p in range(P):
        for g in range(H):
            add([(yi(p, g), 1), (xi(p), -1)], -np.inf, 0)
            add([(ci(p, g), 1), (yi(p, g), -1)], -np.inf, 0)
    for g in range(H):
        add([(yi(p, g), 1) for p in range(P)], XI_SIZE, XI_SIZE)
        add([(ci(p, g), 1) for p in range(P)], 1, 1)
        for ps in (1, 2, 3, 4):
            add([(yi(p, g), 1) for p in range(P) if pos[p] == ps], XI_MIN[ps], XI_MAX[ps])

    ub_v = np.ones(N)
    for p in range(P):
        if not players[p].get('eligible', True):
            for g in range(H):
                ub_v[yi(p, g)] = 0
                ub_v[ci(p, g)] = 0

    A = coo_matrix((vals, (rows, cols)), shape=(r[0], N))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(N), bounds=Bounds(np.zeros(N), ub_v),
               options={'time_limit': 300, 'mip_rel_gap': 0.0})
    if res.x is None:
        raise RuntimeError(f"no solution (status {res.status})")
    x = res.x

    squad_idx = [p for p in range(P) if x[xi(p)] > 0.5]
    squad = []
    for p in sorted(squad_idx, key=lambda p: (pos[p], -sum(xpts[p]))):
        squad.append({
            'player_id': players[p]['id'], 'name': players[p]['name'],
            'position': POSNAME[pos[p]], 'club': club[p], 'price': price[p],
            'xpts': [round(v, 2) for v in xpts[p]], 'six_gw': round(sum(xpts[p]), 1),
            'starts': [bool(x[yi(p, g)] > 0.5) for g in range(H)],
        })
    per_gw = []
    for g in range(H):
        cap_p = next(p for p in squad_idx if x[ci(p, g)] > 0.5)
        counts, pts = {2: 0, 3: 0, 4: 0}, 0.0
        for p in squad_idx:
            if x[yi(p, g)] > 0.5:
                counts[pos[p]] = counts.get(pos[p], 0) + 1
                pts += xpts[p][g]
        pts += xpts[cap_p][g]
        per_gw.append({
            'gw': from_gw + g,
            'xi': [players[p]['id'] for p in squad_idx if x[yi(p, g)] > 0.5],
            'captain': players[cap_p]['id'],
            'formation': f"{counts[2]}-{counts[3]}-{counts[4]}",
            'points': round(pts, 1),
        })
    return {
        'season_id': season_id, 'scope': scope, 'from_gameweek': from_gw,
        'horizon': H, 'budget': budget, 'objective_pts': round(-res.fun, 1),
        'squad': squad, 'per_gw': per_gw,
    }


def solve_transfer(data):
    """
    data: {players:[{id,name,pos,club,price,xpts[H],eligible,owned,sell}],
           season_id, from_gameweek, horizon, bank, free_transfers}
    Returns the plan dict (out/in legs, transfers, hits, bank_after, ...).
    """
    players = data['players']
    season_id = int(data.get('season_id', 0))
    from_gw = int(data.get('from_gameweek', 1))
    bank = float(data.get('bank', 0.0))
    ft = int(data.get('free_transfers', 1))
    P = len(players)
    H = int(data.get('horizon') or len(players[0]['xpts']))

    pos = [pl['pos'] for pl in players]
    club = [pl['club'] for pl in players]
    xpts = [pl['xpts'] for pl in players]
    owned = [bool(pl.get('owned')) for pl in players]
    # keep/sell an owned player at his sell price, buy anyone else at buy price.
    cost = [float(pl['sell']) if owned[p] else float(pl['price']) for p, pl in enumerate(players)]
    budget = bank + sum(cost[p] for p in range(P) if owned[p])

    def xi(p): return p
    def yi(p, g): return P + g * P + p
    def ci(p, g): return P + H * P + g * P + p
    H_VAR = P + 2 * H * P            # the single hit-count integer var
    N = H_VAR + 1

    c = np.zeros(N)
    for p in range(P):
        for g in range(H):
            c[yi(p, g)] = -xpts[p][g]
            c[ci(p, g)] = -xpts[p][g]
    c[H_VAR] = HIT                    # minimise → +4 per hit

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = [0]
    def add(coefs, l, u):
        for col, val in coefs:
            rows.append(r[0]); cols.append(col); vals.append(val)
        lb.append(l); ub.append(u); r[0] += 1

    for ps, cnt in SQUAD.items():
        add([(xi(p), 1) for p in range(P) if pos[p] == ps], cnt, cnt)
    add([(xi(p), cost[p]) for p in range(P)], 0, budget)
    for cl in set(club):
        add([(xi(p), 1) for p in range(P) if club[p] == cl], 0, CLUB_CAP)
    for p in range(P):
        for g in range(H):
            add([(yi(p, g), 1), (xi(p), -1)], -np.inf, 0)
            add([(ci(p, g), 1), (yi(p, g), -1)], -np.inf, 0)
    for g in range(H):
        add([(yi(p, g), 1) for p in range(P)], XI_SIZE, XI_SIZE)
        add([(ci(p, g), 1) for p in range(P)], 1, 1)
        for ps in (1, 2, 3, 4):
            add([(yi(p, g), 1) for p in range(P) if pos[p] == ps], XI_MIN[ps], XI_MAX[ps])
    # transfers k = incoming (= outgoing, squad size fixed). hit var h >= k - FT.
    add([(xi(p), 1) for p in range(P) if not owned[p]] + [(H_VAR, -1)], -np.inf, ft)

    ub_v = np.ones(N)
    ub_v[H_VAR] = max(0, 15 - ft)
    for p in range(P):
        if not players[p].get('eligible', True):
            for g in range(H):
                ub_v[yi(p, g)] = 0
                ub_v[ci(p, g)] = 0

    A = coo_matrix((vals, (rows, cols)), shape=(r[0], N))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(N), bounds=Bounds(np.zeros(N), ub_v),
               options={'time_limit': 300, 'mip_rel_gap': 0.0})
    if res.x is None:
        raise RuntimeError(f"no solution (status {res.status})")
    x = res.x

    new_squad = {p for p in range(P) if x[xi(p)] > 0.5}
    old_squad = {p for p in range(P) if owned[p]}
    transfers_out = old_squad - new_squad
    transfers_in = new_squad - old_squad
    k = len(transfers_in)
    hits = max(0, k - ft)
    gross = -sum(c[yi(p, g)] * x[yi(p, g)] + c[ci(p, g)] * x[ci(p, g)]
                 for p in range(P) for g in range(H))
    net = gross - HIT * hits

    captain_gws = {}
    for g in range(H):
        cap_p = next((p for p in new_squad if x[ci(p, g)] > 0.5), None)
        if cap_p is not None:
            captain_gws.setdefault(players[cap_p]['id'], []).append(from_gw + g)

    def leg(p, is_in):
        d = {'id': players[p]['id'], 'name': players[p]['name'],
             'position': players[p]['pos'], 'club': players[p]['club'],
             'per_gw_pts': [round(v, 2) for v in xpts[p]], 'six_gw': round(sum(xpts[p]), 1)}
        if is_in:
            d['price'] = players[p]['price']
            d['captain_gws'] = captain_gws.get(players[p]['id'], [])
        else:
            d['sell'] = cost[p]
        return d

    in_price = sum(players[p]['price'] for p in transfers_in)
    out_sell = sum(cost[p] for p in transfers_out)
    return {
        'season_id': season_id, 'from_gameweek': from_gw, 'horizon': H,
        'free_transfers': ft, 'bank': bank,
        'transfers': k, 'hits': hits, 'hit_cost': round(HIT * hits, 1),
        'squad_pts_gross': round(gross, 1), 'net_gain_vs_hits': round(net, 1),
        'bank_after': round(bank + out_sell - in_price, 1),
        'ft_after': max(0, ft - k),
        'out': [leg(p, False) for p in transfers_out],
        'in': [leg(p, True) for p in transfers_in],
    }
