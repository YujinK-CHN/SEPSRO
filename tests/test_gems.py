"""
GEMS pieces, checked against known answers, plus end-to-end runs.

    omwu_step         on RPS with exact payoffs, OMWU's iterates converge to the
                      uniform Nash (optimistic MWU converges in the last iterate
                      on zero-sum games, plain MWU only on average)
    eb_ucb            the formula of Eq. 7
    generator         policies start near uniform, respect action masks, and a
                      code's policy is a deterministic function of it; the
                      hypernetwork's once-per-code batch path equals the
                      per-row one, in values and in gradients
    ABR-TR            the PPO update finds the best response to a fixed player
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


@pytest.mark.parametrize('obs_shape', [(7,), (4, 84, 84)])
def test_hyper_generator_groups_codes_exactly(obs_shape):
    """Batches repeating a few codes (as in ABR-TR) give the same logits and
    gradients as generating weights for every row."""
    torch.manual_seed(0)
    gen = make_generator('hyper', obs_shape, 5, 3, (8,), 6)
    codes = torch.randn(3, 3)
    z = codes[torch.tensor([0, 0, 1, 2, 1, 0, 2, 2])]
    obs = (torch.randint(0, 256, (8, *obs_shape), dtype=torch.uint8) if len(obs_shape) == 3
           else torch.randn(8, *obs_shape))
    weights = torch.randn(8, 5)

    grouped = gen(obs, z)
    (grouped * weights).sum().backward()
    grads = [p.grad.clone() for p in gen.parameters()]
    gen.zero_grad()
    per_row = gen.logits_from_params(gen.params(z), obs)
    (per_row * weights).sum().backward()

    assert torch.allclose(grouped, per_row, atol=1e-5)
    for a, p in zip(grads, gen.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-5)


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
    run_dir = tmp_path / 'check' / 'stats' / 'seed0'
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



def _abr_on_rps(extra):
    """ABR-TR alone, one anchor, against a player that always plays rock."""
    from config import parse_args
    from sepsro.budget import BudgetTracker
    from sepsro.methods.gems import GEMS
    from sepsro.policies import FixedActionPolicy
    args = parse_args(['--method', 'gems', '--env', 'rps', '--gems-abr-steps', '10',
                       '--gems-abr-batch-anchors', '16', '--gems-abr-lr', '1e-2', *extra])
    env = make_env(args.env_preset)
    torch.manual_seed(0)
    gems = GEMS(args=args, env_fn=lambda: make_env(args.env_preset), spec=env.spec,
                budget=BudgetTracker(10 ** 9), logger=None, evaluator=None,
                rng=np.random.default_rng(0))
    gems.env = env
    gems.range = return_range(env, np.random.default_rng(1))
    seat, other = env.spec.agents
    gems._opponents = lambda p: {other: FixedActionPolicy(0)}
    z = gems.anchors[seat][0]
    gems.anchors[seat] = z[None]
    stats = gems._abr_tr(seat, z)
    step = env.reset(3)
    probs = GeneratorPolicy(gems.gen[seat], z, args.gems_temperature).action_probs(
        step.observations[seat], step.legal_masks[seat])
    return stats, float(probs[1])


@pytest.mark.parametrize('extra, updates', [
    ([], 16),                                                    # PPO: 4 epochs x 4 minibatches
    (['--gems-abr-epochs', '1', '--gems-abr-minibatches', '1', '--gems-abr-clip', '0'], 1),
])
def test_abr_tr_ppo_update(extra, updates):
    stats, paper = _abr_on_rps(extra)
    assert stats['updates'] == updates and 0 <= stats['clip_frac'] <= 1
    if updates > 1:   # 10 PPO batches find the best response to rock (paper)
        assert paper > 0.6, paper
