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
    run_dir = tmp_path / 'check' / 'stats' / 'seed0'
    records = [json.loads(line) for line in (run_dir / 'eval.jsonl').read_text().splitlines()]
    # Same x-positions for every method, whatever its iteration length.
    assert [r['budget_used'] for r in records] == list(range(0, 2000, 250)) + [2000]
    assert records[-1]['final'] and sum(r['final'] for r in records) == 1
    # Wall-clock stamps on every record, of both streams, non-decreasing.
    train = [json.loads(line) for line in (run_dir / 'train.jsonl').read_text().splitlines()]
    for stream in (records, train):
        assert all('wall_time' in r and 'elapsed_s' in r for r in stream)
        assert [r['elapsed_s'] for r in stream] == sorted(r['elapsed_s'] for r in stream)
    policies = main.load_policies(tmp_path / 'check' / 'checkpoints' / 'seed0' / 'final.pkl')
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


def test_poker_logs_exact_nash_conv_by_default():
    for env in ('kuhn_poker', 'leduc_poker'):
        assert parse_args(['--env', env]).eval_metrics == ['nash_conv', 'vs_random', 'self_play']
    assert parse_args(['--env', 'kuhn_poker', '--eval-metrics', 'vs_random']).eval_metrics == ['vs_random']
    assert parse_args(['--env', 'space_invaders']).eval_metrics == ['vs_random', 'self_play']


def test_every_override_names_a_real_setting():
    for env in ENV_OVERRIDES:
        parse_args(['--env', env])   # raises on an unknown key


# ---------------------------------------------------------------- posthoc
def _fake_run(round_dir, actions, method='fixed', seed=0, steps=()):
    """Seed `seed` of a round on RPS (budget 100) whose policies always play `actions`.

    Laid out as main.py writes it: <round>/stats/seed<N>/config.json and
    <round>/checkpoints/seed<N>/ with final.pkl and step_<N>.pkl for N in `steps`.
    Returns the run's stats directory.
    """
    path = round_dir / 'stats' / f'seed{seed}'
    path.mkdir(parents=True)
    ckpt_dir = round_dir / 'checkpoints' / f'seed{seed}'
    ckpt_dir.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps({
        'env': 'rps', 'seed': seed, 'method': method, 'exp_name': round_dir.name, 'budget': 100,
        'max_episode_steps': None, 'obs_type': 'info_state'}))
    policies = {f'player_{i}': FixedActionPolicy(a) for i, a in enumerate(actions)}
    for name in ['final'] + [f'step_{n}' for n in steps]:
        with open(ckpt_dir / f'{name}.pkl', 'wb') as handle:
            pickle.dump(policies, handle)
    return path


def test_exploit_recovers_exact_nash_conv(tmp_path):
    """Both seats always play rock: exact NashConv 2 (each gains 1 by playing paper)."""
    pytest.importorskip('torch')
    import posthoc
    run = _fake_run(tmp_path / 'rock', [0, 0])
    [result] = posthoc.exploit(str(run), ['--exploit-steps', '4000',
                                          '--exploit-eval-episodes', '200',
                                          '--ppo-rollout-steps', '256', '--ppo-lr', '1e-2'])
    assert result['exact_nash_conv'] == pytest.approx(2.0)
    assert result['approx_nash_conv'] > 1.6, result['approx_nash_conv']
    saved = json.loads((run / 'exploit' / 'final.json').read_text())
    assert set(saved['seats']) == {'player_0', 'player_1'}
    assert saved['checkpoint'] == 'final' and saved['budget_used'] == 100
    with pytest.raises(SystemExit):              # no silent overwrite
        posthoc.exploit(str(run), ['--exploit-steps', '10'])


def test_checkpoint_selection():
    import posthoc
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        run = _fake_run(Path(tmp) / 'run', [0, 0], steps=(0, 26, 50, 77))
        cfg = posthoc._read_config(str(run))
        pick = lambda which: [name for _, name in posthoc.select_checkpoints(cfg, which)]
        assert pick('final') == ['final'] and pick('step_26') == ['step_26']
        assert pick('all') == ['step_0', 'step_26', 'step_50', 'step_77', 'final']
        # Fractions of the budget (100): the nearest checkpoint; 1 is the final one.
        assert pick('0,0.25,0.5,0.75,1') == ['step_0', 'step_26', 'step_50', 'step_77', 'final']
        assert pick('0.5,step_50') == ['step_50']          # no duplicates
        for bad in ('step_3', '1.5', 'latest'):
            with pytest.raises(SystemExit):
                pick(bad)


def test_exploit_scores_several_checkpoints_and_plot_draws_them(tmp_path):
    pytest.importorskip('torch')
    import plot_results
    import posthoc
    run = _fake_run(tmp_path / 'test', [0, 0], steps=(0, 50))
    (run / 'eval.jsonl').write_text(json.dumps({'method': 'fixed', 'env': 'rps', 'seed': 0,
                                                'budget': 100, 'budget_used': 0,
                                                'vs_random/mean': 0.0}) + '\n')
    argv = ['--exploit-steps', '300', '--exploit-eval-episodes', '20', '--ppo-rollout-steps', '64']
    results = posthoc.exploit(str(run), argv, '0,0.5,1')
    assert [r['budget_used'] for r in results] == [0, 50, 100]
    assert sorted(os.listdir(run / 'exploit')) == ['final.json', 'step_0.json', 'step_50.json']
    with pytest.raises(SystemExit):               # any already-scored one stops it
        posthoc.exploit(str(run), argv, 'step_50')
    loaded = plot_results.load_run(str(run))
    assert 'approx_nash_conv' in loaded['metrics']
    assert plot_results.default_metric([loaded]) == 'approx_nash_conv'
    xs, centre, *_ = plot_results.aggregate([loaded], 'approx_nash_conv')
    assert list(xs) == [0, 50, 100]
    assert list(centre) == pytest.approx([r['approx_nash_conv'] for r in results])


def test_checkpoint_every_and_overwrite(tmp_path):
    import main
    argv = ['--method', 'random', '--env', 'rps', '--budget', '100', '--eval-interval', '10',
            '--eval-episodes', '2', '--results-dir', str(tmp_path)]
    main.main(argv + ['--checkpoint-every', '3'])
    ckpt_dir = tmp_path / 'check' / 'checkpoints' / 'seed0'
    assert sorted(p.stem for p in ckpt_dir.glob('*.pkl')) == sorted(
        ['step_0', 'step_30', 'step_60', 'step_90', 'final'])
    # A re-run refuses while old checkpoints are there, and --overwrite clears them.
    run_dir = tmp_path / 'check' / 'stats' / 'seed0'
    (run_dir / 'eval.jsonl').unlink()
    (run_dir / 'train.jsonl').unlink()
    with pytest.raises(SystemExit):
        main.main(argv + ['--checkpoint-every', '3'])
    main.main(argv + ['--checkpoint-every', '0', '--overwrite'])   # alone in its round: allowed
    assert [p.stem for p in ckpt_dir.glob('*.pkl')] == ['final']


def test_crossplay_within_and_against_rounds(tmp_path):
    import posthoc
    # Round 'mixed': seed 0 always rock, seed 1 always paper. Round 'rock': two rock seeds.
    _fake_run(tmp_path / 'mixed', [0, 0], seed=0)
    _fake_run(tmp_path / 'mixed', [1, 1], seed=1)
    _fake_run(tmp_path / 'rock', [0, 0], method='rock', seed=0)
    _fake_run(tmp_path / 'rock', [0, 0], method='rock', seed=1)
    results = posthoc.crossplay(str(tmp_path / 'mixed'), against=['rock'], episodes=5)
    within, vs = results['within'], results['vs_rock']
    # Within: the rock seed only meets paper (-1 in both seats), the paper seed rock (+1).
    assert within['per_seed'][0]['mean_vs_others'] == -1.0
    assert within['per_seed'][1]['mean_vs_others'] == 1.0
    assert within['summary']['mean'] == 0.0 and within['per_seed'][0]['self_play'] == [0.0, 0.0]
    # Against the rock round: rock ties (0), paper wins (+1); in both seat assignments.
    assert vs['per_seed'][0]['mean_vs_them'] == 0.0 and vs['per_seed'][1]['mean_vs_them'] == 1.0
    assert vs['per_seed'][1]['their_return'] == -1.0 and vs['against_method'] == 'rock'
    out = tmp_path / 'mixed' / 'stats' / 'crossplay'
    assert sorted(os.listdir(out)) == ['vs_rock.json', 'within.json']
    with pytest.raises(SystemExit):              # no silent overwrite
        posthoc.crossplay(str(tmp_path / 'mixed'), episodes=5)
    with pytest.raises(SystemExit):              # one seed and no other round: nothing to do
        _fake_run(tmp_path / 'single', [0, 0])
        posthoc.crossplay(str(tmp_path / 'single'), episodes=5)


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
        on, off = (strip(tmp_path / x / 'check' / 'stats' / 'seed0' / name) for x in ('on', 'off'))
        assert on == off


class _NetPolicy(FixedActionPolicy):
    """A policy holding a network, as PSRO's and GEMS's do (module level: picklable)."""

    def __init__(self, seed):
        import torch
        super().__init__(0)
        torch.manual_seed(seed)
        self.net = torch.nn.Linear(4, 3)


def test_checkpoints_store_each_network_once(tmp_path):
    torch = pytest.importorskip('torch')
    from sepsro.checkpoints import OBJECTS_DIR, load_policies, save_policies
    from sepsro.policies import MixturePolicy
    Net = _NetPolicy

    a, b = Net(0), Net(1)
    save_policies({'p0': MixturePolicy([a, a], np.ones(2) / 2)}, str(tmp_path / 'step_0.pkl'))
    save_policies({'p0': MixturePolicy([a, b], np.ones(2) / 2)}, str(tmp_path / 'step_1.pkl'))
    objects = tmp_path / OBJECTS_DIR
    assert len(list(objects.iterdir())) == 2          # a once (twice used, two files), b once
    with torch.no_grad():
        a.net.weight.add_(1.0)                        # a changes: a new object, old one kept
    save_policies({'p0': MixturePolicy([a, b], np.ones(2) / 2)}, str(tmp_path / 'final.pkl'))
    assert len(list(objects.iterdir())) == 3
    first = load_policies(str(tmp_path / 'step_0.pkl'))['p0']
    assert first.policies[0].net is first.policies[1].net   # shared stays shared
    old = load_policies(str(tmp_path / 'step_1.pkl'))['p0'].policies[0].net
    new = load_policies(str(tmp_path / 'final.pkl'))['p0'].policies[0].net
    assert torch.allclose(new.weight, old.weight + 1.0)
