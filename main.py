"""
Single entry point for every method.

    python main.py --method random --env kuhn_poker   --budget 100000 --seed 0
    python main.py --method psro   --env leduc_poker  --budget 1000000 --seed 0

Normally invoked through run.sh.

main.py builds the pieces every method shares -- environment factory, budget,
logger, evaluator, RNG -- and hands them to the method named by --method. At the
same --budget every method gets the same number of environment steps, the same
evaluation watermarks and the same metrics, so their eval.jsonl files can be
plotted on one axis without any post-hoc alignment.

Output: one folder per round (--exp-name): one method with one configuration
in one env, every seed of it.

results/<exp-name>/
    round.json               the round's settings (all but the seed); a seed
                             whose settings differ is refused
    stats/seed<N>/           the statistics (small; copy stats/ back to plot)
        config.json          every setting, after per-env overrides (see config.ENV_OVERRIDES)
        eval.jsonl           one record per evaluation: before training, every
                             --eval-interval env steps, and at the end; each
                             names the checkpoint of the policy it scored
        train.jsonl          method internals
    checkpoints/seed<N>/     the policies (large on Atari)
        step_<N>.pkl         the joint policy evaluated at env step N (every
                             --checkpoint-every-th evaluation point)
        final.pkl            the joint policy scored by the final evaluation
        objects/             their networks, each stored once (sepsro/checkpoints.py)

posthoc.py reads the checkpoints (approximate NashConv, cross-play) and writes
its results into stats/.

The budget is a hard stop: training ends at exactly --budget env steps, and
the final evaluation scores what the method had committed by then.
"""

import os
import json
import time

import numpy as np

from config import parse_args, save_config
from sepsro.budget import BudgetExhausted, BudgetTracker
from sepsro.checkpoints import OBJECTS_DIR, load_policies, save_policies  # noqa: F401
from sepsro.envs import make_env
from sepsro.evaluation import Evaluator
from sepsro.logging import RunLogger
from sepsro.registry import METHODS
from sepsro.utils import set_seed


LOG_FILES = ('eval.jsonl', 'train.jsonl')
FINAL_CHECKPOINT = 'final'
ROUND_FILE = 'round.json'
# Settings that may differ between the seeds of one round: the seed itself,
# where files go, and logging that does not change the run.
NOT_ROUND_SETTINGS = {'seed', 'run_dir', 'checkpoint_dir', 'round_dir', 'results_dir',
                      'overwrite', 'device', 'gems_progress_interval',
                      'psro_log_every_episodes'}


def checkpoint_path(checkpoint_dir, name):
    return os.path.join(checkpoint_dir, f'{name}.pkl')


def _earlier_files(run_dir, checkpoint_dir):
    logs = [os.path.join(run_dir, f) for f in LOG_FILES]
    exploit_dir = os.path.join(run_dir, 'exploit')
    if os.path.isdir(exploit_dir):
        logs += [os.path.join(exploit_dir, f) for f in os.listdir(exploit_dir)]
    if os.path.isdir(checkpoint_dir):
        logs += [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
                 if f.endswith('.pkl')]
        objects = os.path.join(checkpoint_dir, OBJECTS_DIR)
        if os.path.isdir(objects):
            logs += [os.path.join(objects, f) for f in os.listdir(objects)]
    return [f for f in logs if os.path.isfile(f)]


def round_settings(args):
    """What defines the round: every setting but the seed, paths and logging."""
    settings = json.loads(json.dumps(vars(args), default=str, sort_keys=True))
    return {k: v for k, v in settings.items() if k not in NOT_ROUND_SETTINGS}


def _other_seeds(args):
    stats = os.path.dirname(os.path.normpath(args.run_dir))
    if not os.path.isdir(stats):
        return []
    return sorted(d for d in os.listdir(stats)
                  if d != os.path.basename(os.path.normpath(args.run_dir))
                  and os.path.exists(os.path.join(stats, d, 'eval.jsonl')))


def _read_round(path):
    for _ in range(50):   # another seed may be writing it right now
        try:
            with open(path, encoding='utf-8') as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None
        except ValueError:
            time.sleep(0.1)
    raise SystemExit(f'{path} is not valid JSON.')


def round_conflict(args):
    """Why this seed cannot join its round, or None. Writes nothing."""
    recorded = _read_round(os.path.join(args.round_dir, ROUND_FILE))
    if recorded is None:
        return None
    mine = round_settings(args)
    diffs = sorted(k for k in set(mine) | set(recorded) if mine.get(k) != recorded.get(k))
    others = _other_seeds(args)
    if not diffs or not others:   # same round, or nothing left of the old one
        return None
    shown = ', '.join(f'{k}: {recorded.get(k)!r} -> {mine.get(k)!r}' for k in diffs[:8])
    more = f' (+{len(diffs) - 8} more)' if len(diffs) > 8 else ''
    return (f'round {args.round_dir} was run with other settings ({", ".join(others)} '
            f'exist): {shown}{more}. A round is one configuration: use another --exp-name.')


def run_conflict(args):
    """Why main.py would refuse this run, or None (what submit.sh checks up front)."""
    existing = _earlier_files(args.run_dir, args.checkpoint_dir)
    if existing and not args.overwrite:
        return (f'{args.run_dir} already holds an earlier run ({len(existing)} files, e.g. '
                f'{existing[0]}). Use a different --exp-name, or --overwrite to replace it.')
    return round_conflict(args)


def _write_round(args):
    """Record the round's settings; the first seed to get there wins, the rest compare."""
    path = os.path.join(args.round_dir, ROUND_FILE)
    data = json.dumps(round_settings(args), indent=2, sort_keys=True)
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as handle:
        handle.write(data)
    if _read_round(path) is not None:   # an outdated round (its seeds are gone) is replaced
        os.replace(tmp, path)
        return
    try:
        os.link(tmp, path)              # atomic: fails if another seed created it first
    except FileExistsError:
        pass
    finally:
        os.remove(tmp)


def prepare_run_dir(args):
    """Create the run's directories, refusing to mix a new run into an old one.

    The JSONL writers append, so a re-run -- or a second configuration under
    the same --exp-name -- would otherwise land in the same eval.jsonl as the
    first and be read back as one run; old checkpoints and exploit results
    would no longer match the logs. A round holds one configuration only
    (round.json), so its seeds can be averaged.
    """
    problem = run_conflict(args)
    if problem:
        raise SystemExit(problem)
    os.makedirs(args.round_dir, exist_ok=True)
    _write_round(args)
    problem = round_conflict(args)   # a seed started at the same time may have won
    if problem:
        raise SystemExit(problem)
    for f in _earlier_files(args.run_dir, args.checkpoint_dir):
        os.remove(f)
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)


class Checkpointer:
    """Saves the policy each evaluation scores: every `every`-th point, and the final one."""

    def __init__(self, checkpoint_dir, budget, every):
        self.dir = checkpoint_dir
        self.budget = budget
        self.every = int(every)
        self.points = 0

    def __call__(self, policies, final):
        if final:
            name = FINAL_CHECKPOINT
        else:
            index, self.points = self.points, self.points + 1
            # A point at the very end is the final one, saved under that name.
            if self.every <= 0 or index % self.every or self.budget.exhausted:
                return None
            name = f'step_{self.budget.used}'
        save_policies(policies, checkpoint_path(self.dir, name))
        return name


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    def env_fn():
        return make_env(args.env_preset, max_episode_steps=args.max_episode_steps,
                        obs_type=args.obs_type)

    probe = env_fn()
    spec = probe.spec
    probe.close()

    prepare_run_dir(args)
    save_config(args, os.path.join(args.run_dir, 'config.json'))
    # Everything that defines an experimental arm goes into meta, so it is on
    # every eval record and the plotter can tell arms apart. The method adds
    # its own variant fields below, once it exists.
    logger = RunLogger(args.run_dir, meta={'method': args.method, 'env': args.env_name,
                                           'seed': args.seed, 'budget': args.budget})

    budget = BudgetTracker(args.budget, hard=True)
    evaluator = Evaluator(env_fn, budget, logger, metrics=args.eval_metrics,
                          interval=args.eval_interval, episodes=args.eval_episodes,
                          seed=args.seed,
                          checkpoint=Checkpointer(args.checkpoint_dir, budget,
                                                  args.checkpoint_every))

    method_cls = METHODS.get(args.method)
    method = method_cls(args=args, env_fn=env_fn, spec=spec, budget=budget, logger=logger,
                        evaluator=evaluator, rng=np.random.default_rng([args.seed, 0]))
    logger.meta.update(method.variant())
    print(f'[sepsro] {args.method} on {args.env_name} ({spec.backend}, '
          f'{spec.num_agents} agents, {"turn-based" if spec.turn_based else "simultaneous"}) '
          f'budget={args.budget} seed={args.seed} -> {args.run_dir}', flush=True)

    try:
        evaluator.start(method)
        try:
            method.train()
        except BudgetExhausted:
            pass
        evaluator.evaluate(method, final=True)   # also saves final.pkl
    finally:
        evaluator.close()
        logger.close()


if __name__ == '__main__':
    main()
