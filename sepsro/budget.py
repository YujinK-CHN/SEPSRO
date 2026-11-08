"""
Training budget, in ENVIRONMENT STEPS.

The budget is the common currency that makes methods comparable: PSRO spends it
on best-response training plus empirical-game simulation, PPO spends it on
self-play rollouts, GEMS on whatever it does -- and at the same --budget every
method has consumed the same amount of experience. That is what lets their
curves share one x-axis without post-hoc alignment.

One unit is one call to env.step(): one decision in a turn-based game, one
joint action in a simultaneous one.

Episodes are not cut at the budget boundary, so a run overshoots `total` by at
most one episode (or one batch, for a method that collects in batches). The
final eval record logs the true `budget_used`.

What is charged: every step a method takes while training, INCLUDING the
simulations used to estimate payoff tables (they are experience the method
needed). What is not: evaluation rollouts done by the Evaluator, which measure
the method rather than train it.
"""

from __future__ import annotations


class BudgetExhausted(Exception):
    """Raised by consume(strict=True) when a charge would overrun the budget."""


class BudgetTracker:

    def __init__(self, total: int):
        if total <= 0:
            raise ValueError(f'Budget must be positive, got {total}.')
        self.total = int(total)
        self.used = 0

    def consume(self, steps: int, strict=False):
        if strict and self.used + steps > self.total:
            raise BudgetExhausted(f'{self.used} + {steps} > {self.total}')
        self.used += int(steps)

    @property
    def remaining(self):
        return max(0, self.total - self.used)

    @property
    def exhausted(self):
        return self.used >= self.total

    @property
    def fraction(self):
        return min(1.0, self.used / self.total)

    def state_dict(self):
        return {'total': self.total, 'used': self.used}
