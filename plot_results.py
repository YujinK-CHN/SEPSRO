"""
Plot runs on the shared budget axis.

Reads every eval.jsonl under a results directory, groups runs into series,
aggregates across seeds, and draws them against env steps spent -- the common
currency that makes methods comparable.

    python plot_results.py results/main
    python plot_results.py results/main --metric vs_random/mean --metric self_play/first_0
    python plot_results.py results --filter env=space_invaders --band minmax --out fig.pdf
    python plot_results.py results/main --table          # the numbers, no figure
    python plot_results.py results/main --list-metrics   # what the logs contain
    python plot_results.py results/main --train          # + training diagnostics figure

A series is one experimental arm: experiment name (the <exp-name> directory),
env, method, and the method's variant fields that main.py stamps on every eval
record (e.g. PSRO's oracle, GEMS's generator). Every seed of an arm is one run.
Legend names show only the fields that differ between the plotted series.

Seeds rarely share exact x-values (an evaluation fires at the first episode end
past each watermark), so each seed is interpolated onto a common grid over the
range all of them cover.

--train draws a second figure from train.jsonl, per iteration:
    PSRO   best-response training return (oracle_episode records), per seat
    GEMS   mixture value sigma . v_hat (normalised to [0, 1]) and the value of
           the candidate the EB-UCB oracle picked, per seat
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

# --------------------------------------------------------------------------- #
# Palette: SESiL-II's. Categorical slots assigned in series order.
# --------------------------------------------------------------------------- #
THEME = {
    'light': {
        'surface': '#fcfcfb', 'ink': '#0b0b0b', 'ink_2': '#52514e', 'muted': '#8a8985',
        'grid': '#e1e0d9',
        'series': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                   '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
    },
    'dark': {
        'surface': '#1a1a19', 'ink': '#ffffff', 'ink_2': '#c3c2b7', 'muted': '#8a8985',
        'grid': '#2c2c2a',
        'series': ['#3987e5', '#d95926', '#199e70', '#c98500',
                   '#d55181', '#008300', '#9085e9', '#e66767'],
    },
}

METRIC_LABELS = {
    'nash_conv': 'NashConv  (lower is better)',
    'exploitability': 'Exploitability  (lower is better)',
    'vs_random/mean': 'Return vs random opponents  (mean over seats)',
    'self_play/mean': 'Return in self-play  (mean over seats)',
}

# Numeric fields of an eval record that are not metrics.
NOT_METRICS = {'seed', 'budget', 'budget_used', 'budget_fraction', 'elapsed_s'}
# Fields that never distinguish one arm from another.
NOT_SERIES = {'seed', 'final', 'budget_used', 'budget_fraction', 'wall_time', 'elapsed_s'}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def read_jsonl(path):
    """Records of a JSONL file; a truncated last line (killed run) is skipped."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def find_runs(root):
    """Every directory under `root` holding an eval.jsonl."""
    return sorted(dirpath for dirpath, _, files in os.walk(root) if 'eval.jsonl' in files)


def _experiment_of(run_dir):
    """<exp-name> from .../<exp-name>/<env>/<method>/seed<N>, else the parent path."""
    parts = os.path.normpath(run_dir).split(os.sep)
    return parts[-4] if len(parts) >= 4 else os.path.dirname(run_dir)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_run(run_dir):
    evals = read_jsonl(os.path.join(run_dir, 'eval.jsonl'))
    if not evals:
        return None
    first = evals[0]
    arm = {k: v for k, v in first.items()
           if k not in NOT_SERIES and (k == 'budget' or not _is_number(v))}
    arm['experiment'] = _experiment_of(run_dir)
    metrics = sorted({k for r in evals for k, v in r.items()
                      if _is_number(v) and k not in NOT_METRICS})
    for r in evals:   # mean of vs_random already exists; add one for self_play
        seats = [v for k, v in r.items() if k.startswith('self_play/') and _is_number(v)]
        if seats:
            r['self_play/mean'] = float(np.mean(seats))
    if any('self_play/mean' in r for r in evals):
        metrics = sorted(set(metrics) | {'self_play/mean'})
    return {'dir': run_dir, 'arm': arm, 'seed': first.get('seed'), 'evals': evals,
            'train': read_jsonl(os.path.join(run_dir, 'train.jsonl')), 'metrics': metrics}


def load_runs(root, filters=()):
    runs = [r for r in (load_run(d) for d in find_runs(root)) if r]
    for key, value in filters:
        runs = [r for r in runs if str(r['arm'].get(key)) == value]
    return runs


def group_runs(runs):
    """arm (as a sorted tuple of items) -> runs (one per seed)."""
    groups = defaultdict(list)
    for run in runs:
        groups[tuple(sorted(run['arm'].items(), key=lambda kv: kv[0]))].append(run)
    return dict(groups)


def series_names(arms):
    """Readable names showing only the fields that differ between arms."""
    arms = [dict(a) for a in arms]
    keys = sorted({k for a in arms for k in a})
    core = ('experiment', 'env', 'method')

    def differs(k):
        # A method's own field (oracle, generator, ...) counts only among the
        # arms that have it: PSRO's meta_solver is not "different" just
        # because GEMS runs lack it -- the method name already says that.
        values = [a.get(k) for a in arms] if k in core else \
            [a[k] for a in arms if a.get(k) is not None]
        return len({str(v) for v in values}) > 1

    varying = [k for k in keys if differs(k)]
    if not varying:
        return [arms[0].get('method', 'run') for _ in arms]
    order = ['experiment', 'env', 'method'] + [k for k in varying
                                               if k not in ('experiment', 'env', 'method')]
    names = []
    for a in arms:
        parts = []
        for k in order:
            # A field another method has but this one does not (e.g. PSRO's
            # oracle on a GEMS run) says nothing about this arm: leave it out.
            if k not in varying or a.get(k) is None:
                continue
            parts.append(str(a[k]) if k in ('experiment', 'env', 'method') else f'{k}={a[k]}')
        names.append(' | '.join(parts))
    return names


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(runs, metric, band='std', x_key='budget_used'):
    """Collapse seeds into (x, centre, lo, hi, n_seeds), or None if no run has the metric."""
    curves = []
    for run in runs:
        points = [(r[x_key], r[metric]) for r in run['evals'] if metric in r]
        if points:
            xs, ys = (np.asarray(v, dtype=float) for v in zip(*sorted(points)))
            curves.append((xs, ys))
    if not curves:
        return None

    if len({tuple(np.round(xs, 6)) for xs, _ in curves}) == 1:
        grid = curves[0][0]
        stack = np.vstack([ys for _, ys in curves])
    else:
        lo_x = max(xs[0] for xs, _ in curves)
        hi_x = min(xs[-1] for xs, _ in curves)
        if hi_x <= lo_x:          # no common range: show the longest run alone
            xs, ys = max(curves, key=lambda c: c[0][-1])
            curves, grid, stack = [(xs, ys)], xs, ys[None]
        else:
            grid = np.linspace(lo_x, hi_x, max(len(xs) for xs, _ in curves))
            stack = np.vstack([np.interp(grid, xs, ys) for xs, ys in curves])

    centre = stack.mean(axis=0)
    if stack.shape[0] == 1 or band == 'none':
        lo = hi = centre
    elif band == 'minmax':
        lo, hi = stack.min(axis=0), stack.max(axis=0)
    elif band == 'iqr':
        lo, hi = np.percentile(stack, 25, axis=0), np.percentile(stack, 75, axis=0)
    else:
        sd = stack.std(axis=0)
        lo, hi = centre - sd, centre + sd
    return grid, centre, lo, hi, stack.shape[0]


def train_curves(runs):
    """panel -> {seat: (iterations, values averaged over seeds)} from train.jsonl."""
    per_panel = defaultdict(lambda: defaultdict(list))   # panel -> seat -> [ {it: value} per run ]
    for run in runs:
        psro = defaultdict(lambda: defaultdict(list))
        gems_mix, gems_pick = defaultdict(dict), defaultdict(dict)
        for r in run['train']:
            if r.get('kind') == 'oracle_episode' and 'return' in r:
                psro[r['seat']][r['iteration']].append(r['return'])
            elif r.get('kind') == 'gems_iteration':
                for seat, v in (r.get('mixture_value') or {}).items():
                    gems_mix[seat][r['iteration']] = v
                for seat, info in (r.get('oracle') or {}).items():
                    gems_pick[seat][r['iteration']] = info.get('mean')
        for seat, by_it in psro.items():
            per_panel['PSRO best-response training return'][seat].append(
                {it: float(np.mean(v)) for it, v in by_it.items()})
        for seat, by_it in gems_mix.items():
            per_panel['GEMS mixture value  (normalised)'][seat].append(by_it)
        for seat, by_it in gems_pick.items():
            per_panel['GEMS chosen candidate value  (normalised)'][seat].append(by_it)

    out = {}
    for panel, seats in per_panel.items():
        out[panel] = {}
        for seat, per_run in seats.items():
            common = sorted(set.intersection(*(set(d) for d in per_run)))
            if common:
                out[panel][seat] = (np.array(common),
                                    np.array([np.mean([d[i] for d in per_run]) for i in common]))
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def print_table(series, metric, band):
    print(f'\n{metric}   (centre = mean across seeds, band = {band})\n')
    for name, (x, centre, lo, hi, n) in series.items():
        print(f'{name}   [{n} seed(s)]')
        print(f'  {"env steps":>12}  {"value":>10}  {"lo":>10}  {"hi":>10}')
        for xi, ci, li, hi_ in zip(x, centre, lo, hi):
            print(f'  {xi:>12.0f}  {ci:>10.4f}  {li:>10.4f}  {hi_:>10.4f}')
        print()


def _style(ax, t):
    ax.set_facecolor(t['surface'])
    ax.grid(True, color=t['grid'], lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['muted'])
    ax.tick_params(colors=t['ink_2'], labelsize=9)


def render(per_metric, out_path, mode='light', title=None):
    """per_metric: metric -> {series name: aggregate tuple}. One panel per metric."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType in PDFs (venues reject Type 3)
    t = THEME[mode]

    names = list(dict.fromkeys(n for s in per_metric.values() for n in s))
    colour = {n: t['series'][i % len(t['series'])] for i, n in enumerate(names)}
    k = len(per_metric)
    fig, axes = plt.subplots(1, k, figsize=(6.4 * k, 4.4), dpi=200, squeeze=False)
    fig.patch.set_facecolor(t['surface'])
    for ax, (metric, series) in zip(axes[0], per_metric.items()):
        _style(ax, t)
        for name, (x, centre, lo, hi, n) in series.items():
            if not np.allclose(lo, hi):
                ax.fill_between(x, lo, hi, color=colour[name], alpha=0.16, lw=0)
            ax.plot(x, centre, color=colour[name], lw=2, marker='o', ms=3.5,
                    mec=t['surface'], mew=1.0, label=f'{name}  (n={n})')
        ax.set_xlabel('Env steps (budget used)', color=t['ink_2'])
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), color=t['ink_2'])
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 3), frameon=False,
               fontsize=9, labelcolor=t['ink'], bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, color=t['ink'])
    fig.tight_layout(rect=(0, 0.08 + 0.04 * ((len(labels) - 1) // 3), 1, 1))
    fig.savefig(out_path, facecolor=t['surface'], bbox_inches='tight')
    plt.close(fig)


def render_train(per_series, out_path, mode='light'):
    """per_series: series name -> train_curves() output. One panel per diagnostic."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    t = THEME[mode]
    panels = sorted({p for curves in per_series.values() for p in curves})
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.0), dpi=200,
                             squeeze=False)
    fig.patch.set_facecolor(t['surface'])
    names = list(per_series)
    # Colour = series (same slots as the main figure), line style = seat.
    styles = ['-', '--', ':', '-.']
    seats = sorted({s for c in per_series.values() for p in c.values() for s in p})
    for ax, panel in zip(axes[0], panels):
        _style(ax, t)
        for i, name in enumerate(names):
            for seat, (x, y) in sorted(per_series[name].get(panel, {}).items()):
                ax.plot(x, y, color=t['series'][i % len(t['series'])], lw=1.8,
                        ls=styles[seats.index(seat) % len(styles)], marker='o', ms=3)
        ax.set_title(panel, color=t['ink'], fontsize=10)
        ax.set_xlabel('Iteration', color=t['ink_2'])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=t['series'][i % len(t['series'])], lw=2.2, label=n)
               for i, n in enumerate(names)]
    handles += [Line2D([], [], color=t['ink_2'], lw=1.6, ls=styles[j % len(styles)], label=s)
                for j, s in enumerate(seats)]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 4), frameon=False,
               fontsize=9, labelcolor=t['ink'], bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1 + 0.04 * ((len(handles) - 1) // 4), 1, 1))
    fig.savefig(out_path, facecolor=t['surface'], bbox_inches='tight')
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def default_metric(runs):
    available = {m for r in runs for m in r['metrics']}
    for metric in ('nash_conv', 'vs_random/mean', 'self_play/mean'):
        if metric in available:
            return metric
    return sorted(available)[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', help='results directory to search for eval.jsonl files')
    p.add_argument('--metric', action='append', default=None,
                   help='metric to plot (repeat for several panels). Default: nash_conv, '
                        'else vs_random/mean.')
    p.add_argument('--filter', action='append', default=[], metavar='KEY=VALUE',
                   help='keep only runs whose arm has this field value, e.g. env=space_invaders, '
                        'method=gems, oracle=pg, experiment=main (repeatable)')
    p.add_argument('--band', default='std', choices=['std', 'minmax', 'iqr', 'none'],
                   help='spread across seeds')
    p.add_argument('--out', default=None, help='image path (.png / .pdf / .svg); '
                                               'default <root>/plot.png')
    p.add_argument('--mode', default='light', choices=['light', 'dark'])
    p.add_argument('--title', default=None)
    p.add_argument('--table', action='store_true', help='print the numbers instead of plotting')
    p.add_argument('--list-metrics', action='store_true', help='list runs, arms and metrics')
    p.add_argument('--train', action='store_true',
                   help='also draw training diagnostics from train.jsonl (<out>_train.png)')
    args = p.parse_args(argv)

    filters = [tuple(f.split('=', 1)) for f in args.filter]
    runs = load_runs(args.root, filters)
    if not runs:
        raise SystemExit(f'No eval.jsonl under {args.root}' + (f' matching {args.filter}' if filters else ''))
    groups = group_runs(runs)
    names = dict(zip(groups, series_names(groups)))

    if args.list_metrics:
        for arm, members in groups.items():
            seeds = sorted(r['seed'] for r in members)
            print(f'{names[arm]}   seeds {seeds}')
            print(f'    arm: {dict(arm)}')
            print(f'    metrics: {", ".join(sorted({m for r in members for m in r["metrics"]}))}')
        return

    metrics = args.metric or [default_metric(runs)]
    per_metric = {}
    for metric in metrics:
        series = {}
        for arm, members in groups.items():
            agg = aggregate(members, metric, args.band)
            if agg is not None:
                series[names[arm]] = agg
        if not series:
            raise SystemExit(f'No run has metric {metric!r}; see --list-metrics.')
        per_metric[metric] = series

    if args.table:
        for metric, series in per_metric.items():
            print_table(series, metric, args.band)
        return

    out = args.out or os.path.join(args.root, 'plot.png')
    render(per_metric, out, args.mode, args.title)
    print(f'wrote {out}')
    if args.train:
        stem, ext = os.path.splitext(out)
        train_out = f'{stem}_train{ext}'
        curves = {names[arm]: train_curves(members) for arm, members in groups.items()}
        if render_train(curves, train_out, args.mode):
            print(f'wrote {train_out}')
        else:
            print('no training diagnostics in train.jsonl')


if __name__ == '__main__':
    main()
