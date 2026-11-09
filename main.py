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

Output: results/<exp-name>/<env>/<method>/seed<seed>/
    config.json          every setting, after per-env overrides (see config.ENV_OVERRIDES)
    eval.jsonl           one record per evaluation: before training, every
                         --eval-interval env steps, and at the end
    train.jsonl          method internals
    final_policies.pkl   the joint policy scored by the final evaluation, for
                         the post-hoc metrics in posthoc.py (approximate
                         NashConv, cross-play)

The budget is a hard stop: training ends at exactly --budget env steps, and
the final evaluation scores what the method had committed by then.
"""

import os
import pickle

import numpy as np

from config import parse_args, save_config
from sepsro.budget import BudgetExhausted, BudgetTracker
from sepsro.envs import make_env
from sepsro.evaluation import Evaluator
from sepsro.logging import RunLogger
from sepsro.registry import METHODS
from sepsro.utils import set_seed


LOG_FILES = ('eval.jsonl', 'train.jsonl', 'final_policies.pkl')


def prepare_run_dir(run_dir, overwrite):
    """Create run_dir, refusing to mix a new run into an old run's logs.

    The JSONL writers append, so a re-run -- or a second arm under the same
    --exp-name -- would otherwise land in the same eval.jsonl as the first and
    be read back as one run.
    """
    existing = [f for f in LOG_FILES if os.path.exists(os.path.join(run_dir, f))]
    if existing and not overwrite:
        raise SystemExit(f'{run_dir} already holds {", ".join(existing)} from an earlier run. '
                         'Use a different --exp-name, or --overwrite to replace it.')
    for f in existing:
        os.remove(os.path.join(run_dir, f))
    os.makedirs(run_dir, exist_ok=True)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    def env_fn():
        return make_env(args.env_preset, max_episode_steps=args.max_episode_steps,
                        obs_type=args.obs_type)

    probe = env_fn()
    spec = probe.spec
    probe.close()

    prepare_run_dir(args.run_dir, args.overwrite)
    save_config(args, os.path.join(args.run_dir, 'config.json'))
    # Everything that defines an experimental arm goes into meta, so it is on
    # every eval record and the plotter can tell arms apart. The method adds
    # its own variant fields below, once it exists.
    logger = RunLogger(args.run_dir, meta={'method': args.method, 'env': args.env_name,
                                           'seed': args.seed, 'budget': args.budget})

    budget = BudgetTracker(args.budget, hard=True)
    evaluator = Evaluator(env_fn, budget, logger, metrics=args.eval_metrics,
                          interval=args.eval_interval, episodes=args.eval_episodes,
                          seed=args.seed)

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
        evaluator.evaluate(method, final=True)
        save_policies(method.policies(), os.path.join(args.run_dir, 'final_policies.pkl'))
    finally:
        evaluator.close()
        logger.close()


def save_policies(policies, path):
    """Pickle the final joint policy (agent -> Policy). Load with load_policies()."""
    with open(path, 'wb') as handle:
        pickle.dump(policies, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_policies(path):
    """The joint policy a run saved. Only load files your own runs wrote (pickle)."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    main()
