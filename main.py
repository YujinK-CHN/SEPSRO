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

Output, in two trees with the same <exp-name>/<env>/<method>/seed<seed>/ layout:

results/...  the statistics (small; what you copy back to plot)
    config.json          every setting, after per-env overrides (see config.ENV_OVERRIDES)
    eval.jsonl           one record per evaluation: before training, every
                         --eval-interval env steps, and at the end; each names
                         the checkpoint of the policy it scored
    train.jsonl          method internals

results/checkpoints/...  the policies (--checkpoint-root; large on Atari)
    step_<N>.pkl         the joint policy evaluated at env step N (every
                         --checkpoint-every-th evaluation point)
    final.pkl            the joint policy scored by the final evaluation
    objects/             their networks, each stored once (sepsro/checkpoints.py)

posthoc.py reads the checkpoints (approximate NashConv, cross-play) and writes
its results back into the statistics tree.

The budget is a hard stop: training ends at exactly --budget env steps, and
the final evaluation scores what the method had committed by then.
"""

import os
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


def prepare_run_dir(run_dir, checkpoint_dir, overwrite):
    """Create run_dir and checkpoint_dir, refusing to mix a new run into an old one.

    The JSONL writers append, so a re-run -- or a second arm under the same
    --exp-name -- would otherwise land in the same eval.jsonl as the first and
    be read back as one run; old checkpoints and exploit results would no
    longer match the logs.
    """
    existing = _earlier_files(run_dir, checkpoint_dir)
    if existing and not overwrite:
        raise SystemExit(f'{run_dir} already holds an earlier run ({len(existing)} files, e.g. '
                         f'{existing[0]}). Use a different --exp-name, or --overwrite to replace it.')
    for f in existing:
        os.remove(f)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


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

    prepare_run_dir(args.run_dir, args.checkpoint_dir, args.overwrite)
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
