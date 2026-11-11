"""
The experiment protocol: hard budget stop, evaluation at fixed env-step counts,
per-env settings, saved final policies, and the post-hoc metrics.

    budget        a hard tracker stops at exactly `total`; watermarks fire at
                  every multiple of the interval below it
    methods       PSRO and GEMS are scored at the same env-step counts, end at
                  exactly the budget, and save a loadable final joint policy
    overrides     ENV_OVERRIDES apply unless a flag is given explicitly
    exploit       approximate NashConv recovers a known exact NashConv
    crossplay     the matrix and summary on runs with known payoffs
"""

import json
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENV_OVERRIDES, parse_args  # noqa: E402
from sepsro.budget import BudgetExhausted, BudgetTracker  # noqa: E402
from sepsro.policies import FixedActionPolicy  # noqa: E402


# ---------------------------------------------------------------- budget
def test_hard_budget_stops_exactly():
    budget = BudgetTracker(10, hard=True)
    for _ in range(9):
        budget.consume(1)
    with pytest.raises(BudgetExhausted):
        budget.consume(1)
    assert budget.used == 10


def test_soft_budget_never_raises():
    budget = BudgetTracker(3)
    for _ in range(5):
        budget.consume(1)
    assert budget.exhausted and budget.used == 5


def test_watermarks_fire_below_total():
    budget = BudgetTracker(10, hard=True)
    marks = []
    budget.watch(3, marks.append)
    with pytest.raises(BudgetExhausted):
        for _ in range(20):
            budget.consume(1)
    assert marks == [3, 6, 9]


def test_several_watchers_fire_independently():
    budget = BudgetTracker(10, hard=False)
    evals, beats = [], []
    budget.watch(5, evals.append)
    budget.watch(2, beats.append)
    budget.consume(7)
    budget.consume(3)
    assert evals == [5] and beats == [2, 4, 6, 8]


# --------------------------------------------------------------- methods
_TINY = {
    'psro': ['--psro-number-training-episodes', '200', '--psro-sims-per-entry', '20',
             '--pg-hidden', '8', '--psro-prd-iterations', '100'],
    'gems': ['--gems-mc-opponents', '1', '--gems-mc-rollouts', '1',
             '--gems-oracle-opponents', '1', '--gems-oracle-rollouts', '2',
             '--gems-mutation-pool', '2', '--gems-random-pool', '2', '--gems-abr-steps', '2',
             '--gems-abr-batch-anchors', '2', '--gems-hyper-hidden', '8',
             '--gems-policy-hidden', '8'],
}


@pytest.mark.parametrize('method', ['psro', 'gems'])
def test_methods_scored_at_fixed_points_and_stop_exactly(tmp_path, method):
    pytest.importorskip('torch')
    import main
    main.main(['--method', method, '--env', 'rps', '--budget', '2000', '--eval-interval', '250',
               '--eval-episodes', '5', '--results-dir', str(tmp_path), *_TINY[method]])
    run_dir = tmp_path / 'check' / 'rps' / method / 'seed0'
    records = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    # Same x-positions for every method, whatever its iteration length.
    assert [r['budget_used'] for r in records] == list(range(0, 2000, 250)) + [2000]
    assert records[-1]['final'] and sum(r['final'] for r in records) == 1
    # Wall-clock stamps on every record, of both streams, non-decreasing.
    train = [json.loads(line) for line in (run_dir / 'train.jsonl').read_text().splitlines()]
    for stream in (records, train):
        assert all('wall_time' in r and 'elapsed_s' in r for r in stream)
        assert [r['elapsed_s'] for r in stream] == sorted(r['elapsed_s'] for r in stream)
    policies = main.load_policies(run_dir / 'final_policies.pkl')
    assert set(policies) == {'player_0', 'player_1'}
    rng = np.random.default_rng(0)
    for policy in policies.values():
        policy.reset(rng)
        assert policy.act(np.ones(1, np.float32), np.ones(3, bool), rng) in range(3)


def test_gems_output_is_frozen_while_training(tmp_path):
    """policies() between iterations must not change when ABR-TR trains the live generator."""
    torch = pytest.importorskip('torch')
    import main
    from config import parse_args as parse
    from sepsro.budget import BudgetTracker as Budget
    from sepsro.envs import make_env
    from sepsro.methods.gems import GEMS

    args = parse(['--method', 'gems', '--env', 'rps', *_TINY['gems']])
    spec = make_env(args.env_preset).spec
    gems = GEMS(args=args, env_fn=lambda: make_env(args.env_preset), spec=spec,
                budget=Budget(10 ** 9), logger=None, evaluator=None,
                rng=np.random.default_rng(0))
    before = gems.policies()['player_0'].policies[0]
    obs, mask = np.ones(1, np.float32), np.ones(3, bool)
    probs = before.action_probs(obs, mask)
    with torch.no_grad():
        for param in gems.gen['player_0'].parameters():
            param.add_(1.0)
    assert np.allclose(before.action_probs(obs, mask), probs)
    assert main   # imported for the registry side effects


# -------------------------------------------------------------- overrides
def test_env_overrides_apply_unless_explicit():
    args = parse_args(['--env', 'space_invaders'])
    for key, value in ENV_OVERRIDES['space_invaders'].items():
        assert getattr(args, key) == value, key
    assert args.env_overrides == ENV_OVERRIDES['space_invaders']
    # An explicit flag wins, even when it equals the generic default.
    args = parse_args(['--env', 'space_invaders', '--psro-sims-per-entry', '1000',
                       '--eval-episodes=200'])
    assert args.psro_sims_per_entry == 1000 and args.eval_episodes == 200
    assert 'psro_sims_per_entry' not in args.env_overrides
    # Other envs keep the published defaults.
    args = parse_args(['--env', 'kuhn_poker'])
    assert args.psro_sims_per_entry == 1000 and args.gems_eta == 0.03
    assert args.env_overrides == ENV_OVERRIDES['kuhn_poker']


def test_every_override_names_a_real_setting():
    for env in ENV_OVERRIDES:
        parse_args(['--env', env])   # raises on an unknown key


# ---------------------------------------------------------------- posthoc
def _fake_run(path, actions, method='fixed', seed=0):
    """A run directory on RPS whose final policies always play `actions` (one per seat)."""
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps({
        'env': 'rps', 'seed': seed, 'method': method, 'exp_name': 'test', 'budget': 1,
        'max_episode_steps': None, 'obs_type': 'info_state'}))
    policies = {f'player_{i}': FixedActionPolicy(a) for i, a in enumerate(actions)}
    with open(path / 'final_policies.pkl', 'wb') as handle:
        pickle.dump(policies, handle)
    return path


def test_exploit_recovers_exact_nash_conv(tmp_path):
    """Both seats always play rock: exact NashConv 2 (each gains 1 by playing paper)."""
    pytest.importorskip('torch')
    import posthoc
    run = _fake_run(tmp_path / 'rock', [0, 0])
    result = posthoc.exploit(str(run), ['--exploit-steps', '4000',
                                        '--exploit-eval-episodes', '200',
                                        '--ppo-rollout-steps', '256', '--ppo-lr', '1e-2'])
    assert result['exact_nash_conv'] == pytest.approx(2.0)
    assert result['approx_nash_conv'] > 1.6, result['approx_nash_conv']
    saved = json.loads((run / 'exploit.json').read_text())
    assert set(saved['seats']) == {'player_0', 'player_1'}
    with pytest.raises(SystemExit):              # no silent overwrite
        posthoc.exploit(str(run), ['--exploit-steps', '10'])


def test_crossplay_known_payoffs(tmp_path):
    import posthoc
    _fake_run(tmp_path / 'a' / 'seed0', [0, 0], method='rock')
    _fake_run(tmp_path / 'b' / 'seed0', [1, 1], method='paper')
    per_run, per_method = posthoc.crossplay([str(tmp_path)], episodes=5)
    # Rock's seats only ever meet paper: -1 each; paper's +1.
    assert per_method['rock']['mean'] == -1.0 and per_method['paper']['mean'] == 1.0
    assert (tmp_path / 'crossplay.json').exists()
    assert all(r['self_play'] == [0.0, 0.0] for r in per_run.values())


def test_gems_prints_progress(tmp_path, capsys):
    """Phase lines and step heartbeats in stdout; they do not change the run."""
    pytest.importorskip('torch')
    import main
    common = ['--method', 'gems', '--env', 'rps', '--budget', '600', '--eval-interval', '300',
              '--eval-episodes', '5', *_TINY['gems']]
    main.main([*common, '--results-dir', str(tmp_path / 'on'), '--gems-progress-interval', '100'])
    out = capsys.readouterr().out
    for phase in ('estimate/player_0', 'oracle/player_1', 'abr/player_0'):
        assert f'] [gems] it 1 {phase}: ' in out
    assert '] [gems] 100 steps: it ' in out and 'iteration 1 done at ' in out
    main.main([*common, '--results-dir', str(tmp_path / 'off'), '--gems-progress-interval', '0'])
    assert '[gems]' not in capsys.readouterr().out

    def strip(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return [{k: v for k, v in r.items() if k not in ('wall_time', 'elapsed_s', 'phase_seconds')}
                for r in rows]
    for name in ('eval.jsonl', 'train.jsonl'):
        on, off = (strip(tmp_path / x / 'check' / 'rps' / 'gems' / 'seed0' / name) for x in ('on', 'off'))
        assert on == off
