"""
plot_results.py on synthetic logs with known answers: arm grouping, names,
seed aggregation (exact and interpolated), truncated logs, training
diagnostics, and the CLI end to end.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_results as pr  # noqa: E402


def _write_run(root, exp, env, method, seed, points, variant=None, train=None, truncated=False):
    run_dir = os.path.join(root, exp, 'stats', f'seed{seed}')
    os.makedirs(run_dir)
    meta = {'method': method, 'env': env, 'seed': seed, 'budget': 1000, **(variant or {})}
    lines = [json.dumps({**meta, 'budget_used': x, 'budget_fraction': x / 1000,
                         'final': i == len(points) - 1, 'vs_random/mean': y,
                         'self_play/p0': y, 'self_play/p1': -y})
             for i, (x, y) in enumerate(points)]
    text = '\n'.join(lines) + '\n'
    if truncated:
        text += '{"budget_used": 99'           # a killed run's half-written line
    with open(os.path.join(run_dir, 'eval.jsonl'), 'w') as f:
        f.write(text)
    with open(os.path.join(run_dir, 'train.jsonl'), 'w') as f:
        f.write(''.join(json.dumps(r) + '\n' for r in train or []))
    return run_dir


@pytest.fixture
def results(tmp_path):
    root = str(tmp_path)
    for seed, offset in ((0, 0.0), (1, 2.0)):
        _write_run(root, 'kuhn_psro', 'kuhn', 'psro', seed, [(0, 1 + offset), (100, 3 + offset)],
                   variant={'oracle': 'pg', 'meta_solver': 'auto'},
                   train=[{'kind': 'oracle_episode', 'iteration': 1, 'seat': 'p0', 'return': r}
                          for r in (1.0, 3.0)])
    # Another configuration of the same method is another round, as main.py requires.
    _write_run(root, 'kuhn_psro_br', 'kuhn', 'psro', 0, [(0, 10.0), (100, 10.0)],
               variant={'oracle': 'br', 'meta_solver': 'auto'})
    # Two GEMS seeds whose eval points fall at different env steps.
    _write_run(root, 'kuhn_gems', 'kuhn', 'gems', 0, [(0, 0.0), (50, 5.0), (100, 10.0)],
               variant={'generator': 'hyper'}, truncated=True,
               train=[{'kind': 'gems_iteration', 'iteration': 1, 'mixture_value': {'p0': 0.5},
                       'oracle': {'p0': {'mean': 0.7}}}])
    _write_run(root, 'kuhn_gems', 'kuhn', 'gems', 1, [(10, 1.0), (110, 11.0)],
               variant={'generator': 'hyper'})
    return root


def test_arms_are_split_by_variant(results):
    groups = pr.group_runs(pr.load_runs(results))
    assert sorted(len(m) for m in groups.values()) == [1, 2, 2]   # br, pg, gems
    names = sorted(pr.series_names(groups))
    # One series per round, named after it.
    assert names == ['kuhn_gems', 'kuhn_psro', 'kuhn_psro_br']
    # Without rounds (or two series in one), the differing fields name them.
    arms = [dict(a, experiment='x') for a in groups]
    assert sorted(pr.series_names(arms)) == ['gems', 'psro | oracle=br', 'psro | oracle=pg']


def test_rounds_and_default_output(results):
    two = [os.path.join(results, 'kuhn_psro'), os.path.join(results, 'kuhn_gems')]
    runs = pr.load_runs(two)
    assert {r['arm']['experiment'] for r in runs} == {'kuhn_psro', 'kuhn_gems'}
    assert pr.default_out(runs, ['vs_random/mean']) == os.path.join(
        results, 'plots', 'kuhn_gems_vs_kuhn_psro__vs_random-mean.png')


def test_aggregate_exact_and_interpolated(results):
    groups = pr.group_runs(pr.load_runs(results))
    by_name = dict(zip(pr.series_names(groups), groups.values()))
    x, centre, lo, hi, n = pr.aggregate(by_name['kuhn_psro'], 'vs_random/mean')
    np.testing.assert_allclose(x, [0, 100])
    np.testing.assert_allclose(centre, [2, 4])                       # mean of 1,3 and 3,5
    np.testing.assert_allclose(hi - centre, [1, 1])                  # std
    x, centre, _, _, n = pr.aggregate(by_name['kuhn_gems'], 'vs_random/mean', band='none')
    assert n == 2 and x[0] == 10 and x[-1] == 100                    # overlap of both seeds
    np.testing.assert_allclose(centre, x / 10)                       # both seeds lie on y = x/10


def test_truncated_log_and_self_play_mean(results):
    runs = pr.load_runs(results, filters=[('method', 'gems')])
    seed0 = [r for r in runs if r['seed'] == 0][0]
    assert len(seed0['evals']) == 3                                  # half line skipped
    assert all(r['self_play/mean'] == 0 for r in seed0['evals'])
    assert 'self_play/mean' in seed0['metrics']


def test_train_curves(results):
    runs = pr.load_runs(results)
    psro = pr.train_curves([r for r in runs if r['arm'].get('oracle') == 'pg'])
    it, val = psro['PSRO best-response training return']['p0']
    assert list(it) == [1] and val[0] == 2.0                         # mean of 1 and 3, both seeds
    gems = pr.train_curves([r for r in runs if r['arm']['method'] == 'gems'])
    assert gems['GEMS mixture value  (normalised)']['p0'][1][0] == 0.5


def test_cli_table_and_figures(results, tmp_path, capsys):
    pr.main([results, '--filter', 'oracle=pg', '--table'])
    out = capsys.readouterr().out
    assert 'vs_random/mean' in out and '[2 seed(s)]' in out
    pytest.importorskip('matplotlib')
    fig = str(tmp_path / 'fig.png')
    pr.main([results, '--metric', 'vs_random/mean', '--metric', 'self_play/mean',
             '--train', '--out', fig])
    assert os.path.getsize(fig) > 0 and os.path.getsize(str(tmp_path / 'fig_train.png')) > 0


def test_unknown_metric_is_reported(results):
    with pytest.raises(SystemExit):
        pr.main([results, '--metric', 'nash_conv', '--table'])
