"""
The contract every method implements.

main.py builds the shared pieces -- an env factory, the budget, the logger, the
evaluator, the RNG -- and hands them to the method. A method then owns its own
training loop and has three obligations:

    1. charge every training env step to self.budget (pass budget= to
       play_episode / PayoffTable, or call budget.consume() for custom loops);
    2. stop when self.budget.exhausted;
    3. call self.evaluator.maybe_evaluate(self) regularly, so it is scored at the
       same watermarks as every other method.

and must answer policies(): the joint policy (one Policy per seat) that
represents the method's current output. For PPO that is the learned policy in
every seat; for PSRO it is the meta-strategy mixture over each population.

Method-specific hyper-parameters are read from self.args; their defaults live
in the method's argument group in config.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Method(ABC):

    def __init__(self, args, env_fn, spec, budget, logger, evaluator, rng):
        self.args = args
        self.env_fn = env_fn          # () -> MultiAgentEnv; call for each worker env
        self.spec = spec              # EnvSpec, so methods can size networks without an env
        self.budget = budget
        self.logger = logger
        self.evaluator = evaluator
        self.rng = rng

    @abstractmethod
    def train(self):
        """Run until the budget is spent."""

    @abstractmethod
    def policies(self) -> dict:
        """agent -> Policy: the method's current joint policy, for evaluation."""

    def variant(self) -> dict:
        """Settings that make this run a different experimental arm of the same method.

        Stamped on every eval record (next to method / env / seed / budget), so
        two arms of one method -- e.g. PSRO with the pg and the br oracle -- can
        never be averaged together as if they were seeds of one condition.
        """
        return {}

    def save(self, path):
        """Write a checkpoint under `path`. Optional."""

    def load(self, path):
        """Restore from a checkpoint written by save(). Optional."""


class NotImplementedMethod(Method):
    """Placeholder for a method that is registered but not written yet.

    Registering early keeps --method choices, run directories and launch
    scripts stable while the implementation is pending.
    """

    reference = ''

    def train(self):
        raise NotImplementedError(
            f'{type(self).__name__} is not implemented yet. ({self.reference})')

    def policies(self):
        raise NotImplementedError(f'{type(self).__name__} is not implemented yet.')
