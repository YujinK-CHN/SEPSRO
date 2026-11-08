"""
Uniform-random joint policy. Not a research method: it is the pipeline check.

It exercises every shared component -- env construction, rollouts, budget
charging, evaluation watermarks, logging -- without learning anything, so
`python main.py --method random --env <env>` verifies a new environment or
machine end-to-end, and gives the floor every real method should beat.
"""

from __future__ import annotations

from ..policies import RandomPolicy
from ..registry import METHODS
from ..rollout import play_episode
from .base import Method


@METHODS.register('random')
class RandomBaseline(Method):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policies = {a: RandomPolicy() for a in self.spec.agents}
        self.env = self.env_fn()

    def train(self):
        while not self.budget.exhausted:
            self.evaluator.maybe_evaluate(self)
            result = play_episode(self.env, self._policies, self.rng, budget=self.budget)
            self.logger.train('episode', {'budget_used': self.budget.used,
                                          'length': result.length,
                                          'returns': result.returns})
        self.env.close()

    def policies(self):
        return self._policies
