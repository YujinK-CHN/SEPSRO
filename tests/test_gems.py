"""
GEMS pieces, checked against known answers, plus end-to-end runs.

    omwu_step         on RPS with exact payoffs, OMWU's iterates converge to the
                      uniform Nash (optimistic MWU converges in the last iterate
                      on zero-sum games, plain MWU only on average)
    eb_ucb            the formula of Eq. 7
    generator         policies start near uniform, respect action masks, and a
                      code's policy is a deterministic function of it
    GEMS              runs on native, PettingZoo (AEC + Parallel) and OpenSpiel
                      envs; anchors grow and are capped
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import parse_env  # noqa: E402
from sepsro.envs import make_env  # noqa: E402

torch = pytest.importorskip('torch')

from sepsro.learners.generator import GeneratorPolicy, make_generator  # noqa: E402
from sepsro.methods.gems import eb_ucb, omwu_step, return_range  # noqa: E402

HAS_OS = True
try:
    import pyspiel  # noqa: F401
except ImportError:
    HAS_OS = False


def test_omwu_last_iterate_converges_on_rps():
    payoffs = (make_env(parse_env('rps')).payoffs + 1) / 2    # rewards in [0, 1]
    A, B = payoffs[..., 0], payoffs[..., 1]
    x, y = np.array([0.6, 0.3, 0.1]), np.array([0.2, 0.2, 0.6])
    vx_prev, vy_prev = np.zeros(3), np.zeros(3)
    for _ in range(20000):
        vx, vy = A @ y, B.T @ x
        x, y = omwu_step(x, vx, vx_prev, 0.1), omwu_step(y, vy, vy_prev, 0.1)
        vx_prev, vy_prev = vx, vy
    np.testing.assert_allclose(x, [1 / 3] * 3, atol=0.02)
    np.testing.assert_allclose(y, [1 / 3] * 3, atol=0.02)


def test_omwu_ignores_the_mixture_value_constant():
    sigma, v = np.array([0.5, 0.5]), np.array([0.8, 0.2])
    a = omwu_step(sigma, v, np.zeros(2), 0.03)
    assert a[0] > 0.5 and abs(a.sum() - 1) < 1e-12


def test_eb_ucb_formula():
    y = [0.0, 1.0, 1.0, 0.0, 1.0]
    delta = 0.5
    n, mean, var = 5, 0.6, np.var(y, ddof=1)
    expected = mean + math.sqrt(2 * var * math.log(3 / delta) / n) + 3 * math.log(3 / delta) / (n - 1)
    assert abs(eb_ucb(y, delta) - expected) < 1e-12
    # Same mean, lower variance -> smaller bonus.
    assert eb_ucb([0.6] * 5, delta) < eb_ucb(y, delta)


@pytest.mark.parametrize('kind', ['hyper', 'conditional'])
def test_generator_policies(kind):
    torch.manual_seed(0)
    gen = make_generator(kind, obs_shape=(5,), num_actions=4, latent_dim=8,
                         hyper_hidden=(32, 32), policy_hidden=16)
    rng = np.random.default_rng(0)
    obs, mask = rng.standard_normal(5).astype(np.float32), np.array([True, False, True, True])
    for _ in range(5):
        z = rng.standard_normal(8)
        probs = GeneratorPolicy(gen, z).action_probs(obs, mask)
        assert probs[1] == 0 and abs(probs.sum() - 1) < 1e-9
        if kind == 'hyper':
            assert probs[mask].min() > 0.15     # starts near uniform over legal actions
        np.testing.assert_allclose(probs, GeneratorPolicy(gen, z).action_probs(obs, mask))


def test_return_range_native():
    env = make_env(parse_env('rps'))
    assert return_range(env, np.random.default_rng(0)) == {'player_0': (-1.0, 1.0),
                                                           'player_1': (-1.0, 1.0)}


@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_return_range_openspiel():
    env = make_env(parse_env('kuhn_poker'))
    assert return_range(env, np.random.default_rng(0))['player_0'] == (-2.0, 2.0)


# Tiny settings: these check the pipeline, not learning.
_SMALL = ['--gems-iterations', '3', '--gems-mc-opponents', '2', '--gems-mc-rollouts', '1',
          '--gems-oracle-opponents', '2', '--gems-oracle-rollouts', '1',
          '--gems-mutation-pool', '3', '--gems-random-pool', '2',
          '--gems-abr-steps', '2', '--gems-abr-batch-anchors', '3',
          '--gems-hyper-hidden', '16', '--gems-policy-hidden', '8', '--eval-episodes', '5']


def _run(tmp_path, env_name, *extra):
    import main
    main.main(['--method', 'gems', '--env', env_name, '--budget', '10000000',
               '--results-dir', str(tmp_path), *_SMALL, *extra])
    run_dir = tmp_path / 'check' / parse_env(env_name)['name'] / 'gems' / 'seed0'
    train = [json.loads(line) for line in (run_dir / 'train.jsonl').read_text().splitlines()]
    evals = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    return train, evals


def test_gems_end_to_end_native(tmp_path):
    train, evals = _run(tmp_path, 'rps', '--eval-metrics', 'nash_conv', 'vs_random')
    iterations = [r for r in train if r['kind'] == 'gems_iteration']
    assert [r['iteration'] for r in iterations] == [1, 2, 3]
    assert iterations[-1]['anchors'] == {'player_0': 4, 'player_1': 4}   # k0 + one per iteration
    for r in iterations:
        for sigma in r['meta_strategy'].values():
            assert abs(sum(sigma) - 1) < 1e-9
    assert evals[-1]['final'] and 'nash_conv' in evals[-1]


def test_gems_anchor_cap(tmp_path):
    train, _ = _run(tmp_path, 'rps', '--gems-max-anchors', '2', '--gems-replacement', 'worst_ev')
    last = [r for r in train if r['kind'] == 'gems_iteration'][-1]
    assert last['anchors'] == {'player_0': 2, 'player_1': 2}


@pytest.mark.parametrize('generator', ['hyper', 'conditional'])
def test_gems_runs_on_pettingzoo(tmp_path, generator):
    pytest.importorskip('pettingzoo')
    _run(tmp_path, 'tic_tac_toe', '--gems-generator', generator)            # AEC, masks
    _run(tmp_path / 'p', 'pettingzoo:pettingzoo.classic.rps_v2@parallel')  # Parallel API


@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_gems_runs_on_openspiel(tmp_path):
    _, evals = _run(tmp_path, 'kuhn_poker', '--eval-metrics', 'nash_conv')
    assert all(0 <= e['nash_conv'] < 5 for e in evals)
