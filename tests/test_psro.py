"""
PSRO pieces, each checked against something known in closed form.

    meta-solvers   Nash of RPS / matching pennies is uniform; PRD finds it too
    oracle         against 'always rock', the best response is 'paper'
    PSRO           on RPS the meta-strategy approaches uniform
    NashConv       Kuhn poker: the uniform policy's NashConv is a known constant,
                   and a mixture is aggregated by reach, not by naive averaging
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, parse_env, resolve  # noqa: E402
from sepsro.budget import BudgetTracker  # noqa: E402
from sepsro.envs import make_env  # noqa: E402
from sepsro.meta_solvers import META_SOLVERS, is_two_player_zero_sum  # noqa: E402
from sepsro.policies import FixedActionPolicy, MixturePolicy, RandomPolicy  # noqa: E402

torch = pytest.importorskip('torch')

HAS_OS = True
try:
    import pyspiel  # noqa: F401
except ImportError:
    HAS_OS = False


def _matrix(name):
    return make_env(parse_env(name)).payoffs


# ------------------------------------------------------------ meta-solvers
@pytest.mark.parametrize('game', ['rps', 'matching_pennies'])
@pytest.mark.parametrize('solver', ['nash', 'prd', 'auto'])
def test_meta_solvers_find_uniform_nash(game, solver):
    payoffs = _matrix(game)
    for dist in META_SOLVERS.get(solver)(payoffs):
        np.testing.assert_allclose(dist, np.full(len(dist), 1 / len(dist)), atol=0.02)


def test_nash_on_biased_rps():
    # Rock beating scissors pays 2. Checked by the indifference condition (every
    # strategy in the support earns the same) rather than a hard-coded vector.
    payoffs = _matrix('rps').copy()
    payoffs[0, 2] = [2, -2]
    payoffs[2, 0] = [-2, 2]
    x, y = META_SOLVERS.get('nash')(payoffs)
    A = payoffs[..., 0]
    row_values = A @ y            # every row in the support earns the same
    col_values = x @ A
    support = x > 1e-6
    assert np.ptp(row_values[support]) < 1e-6
    assert np.ptp(col_values[y > 1e-6]) < 1e-6


def test_nash_refuses_general_sum():
    with pytest.raises(ValueError):
        META_SOLVERS.get('nash')(_matrix('prisoners_dilemma'))
    assert not is_two_player_zero_sum(_matrix('prisoners_dilemma'))
    # auto falls back to PRD, which finds the dominant strategy (defect = 1).
    x, y = META_SOLVERS.get('auto')(_matrix('prisoners_dilemma'), iterations=20_000, dt=1e-2)
    assert x[1] > 0.95 and y[1] > 0.95


def test_prd_three_players():
    # Three-player game where action 1 strictly dominates for every seat.
    payoffs = np.zeros((2, 2, 2, 3))
    for idx in np.ndindex(2, 2, 2):
        payoffs[idx] = [float(a) for a in idx]
    dists = META_SOLVERS.get('prd')(payoffs, iterations=20_000, dt=1e-2)
    assert all(d[1] > 0.95 for d in dists)


# ----------------------------------------------------------------- oracles
def _args(*extra):
    return resolve(get_config().parse_args(['--env', 'rps', *extra]))


def _oracle_vs_rock(kind, *extra):
    """Train player_1 against 'always rock' with the RL oracle; return its policy."""
    from sepsro.oracles import RLOracle, make_learner_factory

    args = _args(*extra)
    env = make_env(args.env_preset)
    oracle = RLOracle(env.spec, make_learner_factory(kind, env.spec, args),
                      number_training_episodes=2000, seed=0)
    budget = BudgetTracker(10**7)
    mixtures = {a: MixturePolicy([FixedActionPolicy(0)], [1.0]) for a in env.spec.agents}
    out = oracle(env, mixtures, np.random.default_rng(0), budget)
    assert all(n > 2000 for _, n in out.values())   # every seat trained past the threshold
    return out['player_1'][0]


@pytest.mark.parametrize('loss', ['qpg', 'rpg', 'rm', 'a2c'])
def test_pg_oracle_best_responds(loss):
    policy = _oracle_vs_rock('pg', '--pg-loss', loss, '--pg-hidden', '32', '32',
                             '--pg-pi-lr', '1e-2')
    probs = policy.action_probs(np.ones(1, np.float32), np.ones(3, bool))
    assert probs[1] > 0.9, (loss, probs)     # paper beats rock


def test_dqn_oracle_best_responds():
    policy = _oracle_vs_rock('dqn', '--dqn-hidden', '32', '32', '--dqn-lr', '1e-3',
                             '--dqn-min-buffer-size-to-learn', '100')
    probs = policy.action_probs(np.ones(1, np.float32), np.ones(3, bool))
    np.testing.assert_allclose(probs, [0, 1, 0])   # greedy, and paper


def test_ppo_oracle_best_responds():
    policy = _oracle_vs_rock('ppo', '--psro-ppo-lr', '1e-2', '--psro-ppo-rollout-steps', '64')
    probs = policy.action_probs(np.ones(1, np.float32), np.ones(3, bool))
    assert probs[1] > 0.9, probs     # paper beats rock


def test_ppo_oracle_settings_and_final_update():
    """--psro-ppo-* replace the ppo group's values only for the oracle; the last
    partial rollout is spent by finish()."""
    from sepsro.learners.ppo import PPOConfig, PPOOracleLearner, psro_ppo_config
    from sepsro.rollout import play_episode
    args = _args('--ppo-lr', '1e-4', '--ppo-clip', '0.1', '--psro-ppo-lr', '3e-3')
    config = psro_ppo_config(args)
    assert (config.lr, config.clip) == (3e-3, 0.1)
    assert PPOConfig.from_args(args).lr == 1e-4          # the scorer's learner is untouched

    env = make_env(parse_env('rps'))
    config.rollout_steps = 10
    learner = PPOOracleLearner(env.spec.obs_shape['player_0'], 3, config, seed=0)
    rng = np.random.default_rng(0)
    joint = {'player_0': learner.acting_policy(), 'player_1': FixedActionPolicy(0)}
    for _ in range(13):
        learner.end_episode(play_episode(env, joint, rng, record=True).trajectories['player_0'])
    assert learner.updates == 1                          # one full rollout of 10
    learner.finish()
    assert learner.updates == 2 and not learner._buffer  # plus the last 3 transitions


def test_ppo_reward_scale_scales_targets_only():
    """GAE targets scale with --ppo-reward-scale; the logged return stays raw."""
    from sepsro.learners.ppo import PPOConfig, PPOLearner
    from sepsro.rollout import Transition
    episode = [Transition(np.ones(1, np.float32), np.ones(3, bool), 1, reward=r,
                          terminated=(i == 2), info={'logp': -1.1, 'value': 0.0})
               for i, r in enumerate([100.0, 0.0, 200.0])]
    out = {}
    for scale in (1.0, 0.01):
        learner = PPOLearner((1,), 3, PPOConfig(reward_scale=scale), seed=0)
        out[scale] = learner._batch([episode])[-1].numpy()
        assert learner.update([episode])['mean_return'] == 300.0
    np.testing.assert_allclose(out[0.01], 0.01 * out[1.0], rtol=1e-6)


@pytest.mark.parametrize('kind', ['pg', 'dqn', 'ppo'])
def test_learners_respect_action_mask(kind):
    from sepsro.oracles import make_learner_factory
    env = make_env(parse_env('rps'))
    learner = make_learner_factory(kind, env.spec, _args())('player_0', seed=0)
    mask = np.array([False, True, False])
    rng = np.random.default_rng(0)
    for policy in (learner.policy(), learner.acting_policy()):
        assert all(policy.act(np.ones(1, np.float32), mask, rng) == 1 for _ in range(50))


def test_count_weighted_choice_balances_seats():
    from sepsro.oracles import random_count_weighted_choice
    rng = np.random.default_rng(0)
    picks = [random_count_weighted_choice([0, 99], rng) for _ in range(2000)]
    assert np.mean(picks) < 0.05   # the seat with fewer episodes is picked ~100x as often


# -------------------------------------------------------------------- PSRO
def test_psro_rps_end_to_end(tmp_path):
    import main
    main.main(['--method', 'psro', '--env', 'rps', '--budget', '10000000', '--seed', '0',
               '--psro-iterations', '6', '--psro-number-training-episodes', '1500',
               '--psro-sims-per-entry', '100', '--pg-hidden', '32', '32', '--pg-pi-lr', '1e-2',
               '--psro-prd-iterations', '2000', '--eval-metrics', 'nash_conv',
               '--eval-interval', '500', '--results-dir', str(tmp_path)])
    run_dir = tmp_path / 'check' / 'rps' / 'psro' / 'seed0'
    train = [json.loads(line) for line in (run_dir / 'train.jsonl').read_text().splitlines()]
    iterations = [r for r in train if r['kind'] == 'psro_iteration']
    assert [r['iteration'] for r in iterations] == list(range(7))   # setup + 6 iterations
    # Population grew by one response per seat per iteration.
    assert iterations[-1]['population_sizes'] == {'player_0': 7, 'player_1': 7}
    assert any(r['kind'] == 'oracle_episode' for r in train)
    evals = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    assert evals[-1]['final']
    # The random seed policy is already Nash in RPS (NashConv 0). The first best
    # response moves away from it; PSRO must then converge back towards it.
    nash_convs = [e['nash_conv'] for e in evals]
    assert nash_convs[-1] < 0.6 * max(nash_convs), nash_convs


def test_matrix_nash_conv():
    from sepsro.exploitability import matrix_nash_conv
    env = make_env(parse_env('rps'))
    uniform = {a: RandomPolicy() for a in env.spec.agents}
    rock = {a: FixedActionPolicy(0) for a in env.spec.agents}
    assert abs(matrix_nash_conv(env, uniform)) < 1e-12
    assert abs(matrix_nash_conv(env, rock) - 2.0) < 1e-12   # each seat gains 1 by paper


def test_psro_budget_is_charged(tmp_path):
    import main
    main.main(['--method', 'psro', '--env', 'rps', '--budget', '3000',
               '--psro-number-training-episodes', '100000', '--psro-sims-per-entry', '10',
               '--pg-hidden', '16', '--psro-prd-iterations', '100',
               '--eval-episodes', '10', '--results-dir', str(tmp_path)])
    run_dir = tmp_path / 'check' / 'rps' / 'psro' / 'seed0'
    final = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()][-1]
    # The oracle stopped on the budget, long before its episode threshold, and
    # the hard stop leaves no overshoot.
    assert final['budget_used'] == 3000


def test_prd_matches_openspiel_projection():
    from sepsro.meta_solvers import _simplex_projection
    # The worked example in OpenSpiel's docstring.
    np.testing.assert_allclose(_simplex_projection(np.array([0.4, 0.7, 0.6]), 0.2),
                               [0.2, 0.45, 0.35])


# ---------------------------------------------------------------- NashConv
@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_nash_conv_uniform_kuhn_matches_openspiel():
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import exploitability
    from sepsro.exploitability import nash_conv

    env = make_env(parse_env('kuhn_poker'))
    ours = nash_conv(env, {a: RandomPolicy() for a in env.spec.agents}, None, 0)['nash_conv']
    reference = exploitability.nash_conv(env.game, os_policy.UniformRandomPolicy(env.game))
    assert abs(ours - reference) < 1e-9
    assert abs(ours - 0.9166666666666666) < 1e-6   # = 11/12, the known value


@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_mixture_is_aggregated_by_reach():
    """'Always bet' + 'always check', mixed 50/50 per episode, is not 'bet with p=0.5' everywhere.

    After player 0 has bet, only the always-bet member can be playing, so at
    player 0's later decisions the aggregate must follow always-bet exactly.
    """
    from sepsro.exploitability import aggregate

    env = make_env(parse_env('kuhn_poker'))
    mix = MixturePolicy([FixedActionPolicy(1), FixedActionPolicy(0)], [0.5, 0.5])
    tabular = aggregate(env, {'player_0': mix, 'player_1': RandomPolicy()})
    # Kuhn actions: 0 = pass, 1 = bet. Deal, player 0 decides, then after
    # pass / bet player 0 decides again -- a state only always-check reaches.
    for card in range(3):
        state = env.game.new_initial_state()
        state.apply_action(card)            # chance: player 0's card
        state.apply_action((card + 1) % 3)  # chance: player 1's card
        first = tabular.policy_for_key(state.information_state_string(0))
        np.testing.assert_allclose(first, [0.5, 0.5])
        state.apply_action(0)   # player 0 passes
        state.apply_action(1)   # player 1 bets
        second = tabular.policy_for_key(state.information_state_string(0))
        np.testing.assert_allclose(second, [1.0, 0.0])   # always-check passes (folds)


@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_psro_kuhn_exact_br(tmp_path):
    """PSRO with exact best responses (psro_v2_example's default oracle) on Kuhn poker."""
    import main
    main.main(['--method', 'psro', '--env', 'kuhn_poker', '--budget', '10000000',
               '--psro-oracle', 'br', '--psro-iterations', '8', '--psro-sims-per-entry', '300',
               '--eval-metrics', 'nash_conv', '--eval-interval', '2000',
               '--results-dir', str(tmp_path)])
    run_dir = tmp_path / 'check' / 'kuhn_poker' / 'psro' / 'seed0'
    nash_convs = [json.loads(line)['nash_conv']
                  for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    assert nash_convs[0] > 0.9 and nash_convs[-1] < 0.15, nash_convs


@pytest.mark.skipif(not HAS_OS, reason='open_spiel not installed')
def test_br_oracle_is_exact():
    """Against uniform play, the oracle's response must earn exactly OpenSpiel's BR value."""
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import exploitability as os_expl
    from open_spiel.python.algorithms import expected_game_score
    from sepsro.exploitability import aggregate
    from sepsro.oracles import BestResponseOracle

    env = make_env(parse_env('kuhn_poker'))
    mixtures = {a: MixturePolicy([RandomPolicy()], [1.0]) for a in env.spec.agents}
    responses = BestResponseOracle(env.spec)(env, mixtures, None, None)
    uniform = os_policy.UniformRandomPolicy(env.game)
    for p, seat in enumerate(env.spec.agents):
        expected = os_expl.best_response(env.game, uniform, p)['best_response_value']
        joint = {a: (responses[seat][0] if a == seat else RandomPolicy()) for a in env.spec.agents}
        tab = aggregate(env, joint)
        value = expected_game_score.policy_value(env.game.new_initial_state(), [tab, tab])[p]
        assert abs(value - expected) < 1e-9, (seat, value, expected)


# ------------------------------------------------------------- PettingZoo
HAS_PZ = True
try:
    import pettingzoo  # noqa: F401
except ImportError:
    HAS_PZ = False


@pytest.mark.skipif(not HAS_PZ, reason='pettingzoo not installed')
@pytest.mark.parametrize('env_name, oracle', [
    ('tic_tac_toe', 'pg'),                                    # AEC, action masks
    ('connect_four', 'dqn'),                                  # AEC, action masks
    ('tic_tac_toe', 'ppo'),                                   # AEC, action masks
    ('pettingzoo:pettingzoo.classic.rps_v2@parallel', 'pg'),  # Parallel API, truncation
])
def test_psro_runs_on_pettingzoo(tmp_path, env_name, oracle):
    import main
    main.main(['--method', 'psro', '--env', env_name, '--psro-oracle', oracle,
               '--budget', '10000000', '--psro-iterations', '2',
               '--psro-number-training-episodes', '30', '--psro-sims-per-entry', '5',
               '--pg-hidden', '32', '--dqn-hidden', '32', '--dqn-min-buffer-size-to-learn', '20',
               '--psro-prd-iterations', '200', '--eval-episodes', '5',
               '--results-dir', str(tmp_path)])
    name = parse_env(env_name)['name']
    run_dir = tmp_path / 'check' / name / 'psro' / 'seed0'
    train = [json.loads(line) for line in (run_dir / 'train.jsonl').read_text().splitlines()]
    last = [r for r in train if r['kind'] == 'psro_iteration'][-1]
    assert last['iteration'] == 2
    assert set(last['population_sizes'].values()) == {3}   # random + 2 responses per seat
    evals = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    assert evals[-1]['final'] and 'vs_random/mean' in evals[-1]
