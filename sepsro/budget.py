"""
Training budget, in ENVIRONMENT STEPS.

The budget is the common currency that makes methods comparable: PSRO spends it
on best-response training plus empirical-game simulation, PPO spends it on
self-play rollouts, GEMS on whatever it does -- and at the same --budget every
method has consumed the same amount of experience. That is what lets their
curves share one x-axis without post-hoc alignment.

One unit is one call to env.step(): one decision in a turn-based game, one
joint action in a simultaneous one.

Hard stop (hard=True, what main.py uses): the step that brings `used` to
`total` raises BudgetExhausted, wherever the method is -- mid-episode,
mid-iteration. Every method therefore spends exactly `total` steps and is
scored on what it had committed by then, so no method gets extra experience
by finishing an iteration it started. main.py catches the exception and runs
the final evaluation. Library code that builds its own tracker (tests, the
post-hoc best-response trainer) keeps the soft default and checks `exhausted`.

Watermarks: watch(every, callback) calls callback(watermark) each time `used`
reaches a multiple of `every` below `total`. The Evaluator scores methods
this way, so every method is evaluated at the same env-step counts whatever
its iteration structure.

What is charged: every step a method takes while training, INCLUDING the
simulations used to estimate payoff tables (they are experience the method
needed). What is not: evaluation rollouts done by the Evaluator, which measure
the method rather than train it.
"""

from __future__ import annotations


class BudgetExhausted(Exception):
    """The budget is spent: raised by a hard tracker, or by consume(strict=True)."""


class BudgetTracker:

    def __init__(self, total: int, hard: bool = False):
        if total <= 0:
            raise ValueError(f'Budget must be positive, got {total}.')
        self.total = int(total)
        self.hard = bool(hard)
        self.used = 0
        self._watch = None   # (every, callback, next watermark)

    def watch(self, every: int, callback):
        """Call callback(watermark) whenever `used` reaches a multiple of `every` below total."""
        every = int(every)
        if every <= 0:
            raise ValueError(f'Watermark interval must be positive, got {every}.')
        self._watch = [every, callback, (self.used // every + 1) * every]

    def consume(self, steps: int, strict=False):
        if strict and self.used + steps > self.total:
            raise BudgetExhausted(f'{self.used} + {steps} > {self.total}')
        self.used += int(steps)
        if self._watch is not None:
            every, callback, mark = self._watch
            while mark <= self.used and mark < self.total:
                self._watch[2] = mark + every
                callback(mark)
                mark = self._watch[2]
        if self.hard and self.used >= self.total:
            raise BudgetExhausted(f'budget of {self.total} env steps spent')

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
