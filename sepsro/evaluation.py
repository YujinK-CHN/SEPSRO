"""
Method-agnostic evaluation at budget watermarks.

Every method exposes its current joint policy through Method.policies(); the
Evaluator scores it with the metrics named in --eval-metrics at fixed points of
the budget (every --eval-interval env steps, plus once at the end). Because the
watermarks and metrics are the same for every method, eval.jsonl from any two
runs is directly comparable.

Evaluation rollouts are NOT charged to the training budget.

Adding a metric: write fn(env, policies, rng, episodes) -> dict of scalars and
decorate it with @METRICS.register('<name>'). Exploitability / NashConv for
OpenSpiel games belongs here once the policy -> pyspiel bridge exists.
"""

from __future__ import annotations

import numpy as np

from .policies import RandomPolicy
from .registry import METRICS
from .rollout import average_returns


@METRICS.register('self_play')
def self_play(env, policies, rng, episodes):
    """Mean return of each seat when the joint policy plays itself."""
    means = average_returns(env, policies, rng, episodes)
    return {f'self_play/{a}': v for a, v in means.items()}


@METRICS.register('vs_random')
def vs_random(env, policies, rng, episodes):
    """Mean return of each seat's policy when every OTHER seat plays uniformly at random."""
    out = {}
    for seat in env.spec.agents:
        joint = {a: (policies[a] if a == seat else RandomPolicy()) for a in env.spec.agents}
        out[f'vs_random/{seat}'] = average_returns(env, joint, rng, episodes)[seat]
    out['vs_random/mean'] = float(np.mean(list(out.values())))
    return out


class Evaluator:

    def __init__(self, env_fn, budget, logger, metrics, interval, episodes, seed):
        for name in metrics:
            METRICS.get(name)   # fail at start-up, not at the first watermark
        self.env = env_fn()
        self.budget = budget
        self.logger = logger
        self.metrics = list(metrics)
        self.interval = int(interval)
        self.episodes = int(episodes)
        # Separate stream from training so evaluating more or less often never
        # changes what a method trains on.
        self.rng = np.random.default_rng([seed, 1])
        self._next = 0

    def maybe_evaluate(self, method):
        """Call from a method's training loop; evaluates once per crossed watermark."""
        if self.budget.used >= self._next:
            self.evaluate(method)
            while self._next <= self.budget.used:
                self._next += self.interval

    def evaluate(self, method, final=False):
        policies = method.policies()
        record = {'budget_used': self.budget.used, 'budget_fraction': self.budget.fraction,
                  'final': final}
        for name in self.metrics:
            record.update(METRICS.get(name)(self.env, policies, self.rng, self.episodes))
        self.logger.eval(record)
        return record

    def close(self):
        self.env.close()
