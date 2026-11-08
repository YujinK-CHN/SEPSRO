"""
Meta-solvers: empirical payoff tensor -> one distribution per seat.

Design follows OpenSpiel's psro_v2/meta_strategies.py: a meta-solver is a pure
function of the payoff tensor, so the PSRO loop never knows which one it runs
and a new solver is one registered function.

Not ported yet from psro_v2: alpharank (the PSROSolver default), rm (regret
matching) and uniform_biased.

Input  payoffs   shape [n_0, ..., n_{N-1}, N]; payoffs[..., k] is seat k's payoff
                 (the layout of meta_game.PayoffTable.tensor()).
Output list of N probability vectors, entry k of length n_k.

    uniform   every member equally -> PSRO reduces to fictitious play
    nash      exact Nash by linear programming; two-player zero-sum only
    prd       projected replicator dynamics (Lanctot et al., 2017); any N, any sum
    auto      nash if the table is two-player zero-sum, else prd
"""

from __future__ import annotations

import numpy as np

from .registry import Registry

META_SOLVERS = Registry('meta-solver')


def is_two_player_zero_sum(payoffs, tol=1e-6):
    if payoffs.shape[-1] != 2:
        return False
    scale = max(1.0, float(np.abs(payoffs).max()))
    return float(np.abs(payoffs.sum(axis=-1)).max()) <= tol * scale


@META_SOLVERS.register('uniform')
def uniform(payoffs, **_):
    return [np.full(n, 1.0 / n) for n in payoffs.shape[:-1]]


@META_SOLVERS.register('nash')
def nash(payoffs, **_):
    """Maximin LP for each player of a two-player zero-sum game.

    Estimated tables are only approximately zero-sum, so the game solved is
    A = (P_0 - P_1) / 2, the zero-sum part of the table.
    """
    from scipy.optimize import linprog

    if not is_two_player_zero_sum(payoffs, tol=1e-2):
        raise ValueError("Meta-solver 'nash' needs a two-player zero-sum table; use 'prd'.")
    A = (payoffs[..., 0] - payoffs[..., 1]) / 2.0
    return [_maximin(A, linprog), _maximin(-A.T, linprog)]


def _maximin(A, linprog):
    """argmax_x min_j (x^T A)_j over the simplex. Variables: x (m), v."""
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0                                     # maximise v
    A_ub = np.hstack([-A.T, np.ones((n, 1))])        # v - (x^T A)_j <= 0
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f'Nash LP failed: {res.message}')
    x = np.clip(res.x[:m], 0.0, None)
    return x / x.sum()


@META_SOLVERS.register('prd')
def prd(payoffs, iterations=50_000, dt=1e-3, gamma=1e-10, average_last=None, **_):
    """Projected replicator dynamics, as OpenSpiel's projected_replicator_dynamics.py.

    Each step moves every seat's distribution along the replicator flow and
    projects it (exact L2 projection) back onto the simplex with every entry
    >= gamma, so no member's mass reaches zero. Returns the average of the
    trajectory -- all of it, including the uniform start, unless `average_last`
    limits it to the last that many iterates.

    Defaults are the values psro_v2_example.py passes (prd_iterations=50000,
    prd_gamma=1e-10); the library defaults are 1e5 and 1e-6.
    """
    num_seats = payoffs.shape[-1]
    strategies = [np.full(n, 1.0 / n) for n in payoffs.shape[:-1]]
    every = average_last is None
    totals = [s.copy() if every else np.zeros_like(s) for s in strategies]

    for t in range(iterations):
        new = []
        for k in range(num_seats):
            values = _values_against(payoffs[..., k], strategies, k)
            delta = strategies[k] * (values - values @ strategies[k])
            new.append(_simplex_projection(strategies[k] + dt * delta, gamma))
        strategies = new
        if every or t >= iterations - average_last:
            totals = [tot + s for tot, s in zip(totals, strategies)]
    return [tot / tot.sum() for tot in totals]


def _values_against(tensor, strategies, seat):
    """Expected payoff of each of seat's pure members against the others' mixtures."""
    out = tensor
    # Contract the highest axes first so the remaining axis indices stay valid.
    for axis in reversed(range(len(strategies))):
        if axis != seat:
            out = np.tensordot(out, strategies[axis], axes=([axis], [0]))
    return out


def _simplex_projection(x, gamma=0.0):
    """Closest point in L2 to x on {p : sum p = 1, p_i >= gamma}.

    Wang & Carreira-Perpinan's sort-based projection, exactly as OpenSpiel's
    _simplex_projection.
    """
    n = len(x)
    idx = np.arange(1, n + 1)
    u = np.sort(x)[::-1]
    u_tmp = (1 - np.cumsum(u) - (n - idx) * gamma) / idx
    rho = np.searchsorted(u + u_tmp <= gamma, True)
    return np.maximum(x + u_tmp[rho - 1], gamma)


@META_SOLVERS.register('auto')
def auto(payoffs, **kwargs):
    if is_two_player_zero_sum(payoffs):
        return nash(payoffs, **kwargs)
    return prd(payoffs, **kwargs)
