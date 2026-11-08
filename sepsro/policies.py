"""
The policy interface every method produces and every evaluator consumes.

A Policy plays ONE seat. It sees one agent's observation and legal-action mask
and returns one discrete action. Neural policies, tabular policies, PSRO
meta-strategy mixtures and fixed opponents all look the same from outside,
which is what lets the rollout and payoff-table code be shared across methods.

reset() is called at the start of every episode. Stateless policies ignore it;
a mixture uses it to sample which member plays this episode (the standard way
a PSRO meta-strategy is executed), and a recurrent policy would clear its
hidden state there.
"""

from __future__ import annotations

import numpy as np


class Policy:

    def reset(self, rng: np.random.Generator):
        pass

    def act(self, obs: np.ndarray, legal_mask: np.ndarray, rng: np.random.Generator) -> int:
        raise NotImplementedError

    def action_probs(self, obs: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Full distribution over actions. Optional; needed by exact-evaluation tools."""
        raise NotImplementedError(f'{type(self).__name__} does not expose action_probs().')


class RandomPolicy(Policy):
    """Uniform over legal actions. The default opponent and PSRO's usual seed policy."""

    def act(self, obs, legal_mask, rng):
        return int(rng.choice(np.flatnonzero(legal_mask)))

    def action_probs(self, obs, legal_mask):
        mask = np.asarray(legal_mask, dtype=float)
        return mask / mask.sum()


class FixedActionPolicy(Policy):
    """Always plays one action (falls back to random if it is illegal). Useful for tests."""

    def __init__(self, action):
        self.action = int(action)

    def act(self, obs, legal_mask, rng):
        if legal_mask[self.action]:
            return self.action
        return int(rng.choice(np.flatnonzero(legal_mask)))

    def action_probs(self, obs, legal_mask):
        if legal_mask[self.action]:
            probs = np.zeros(len(legal_mask))
            probs[self.action] = 1.0
            return probs
        return RandomPolicy().action_probs(obs, legal_mask)


class MixturePolicy(Policy):
    """Samples one member per episode with the given weights, then plays it throughout.

    Per-episode (not per-step) sampling is what a mixed strategy over policies
    means in the empirical game, and is how PSRO meta-strategies are evaluated.
    """

    def __init__(self, policies, weights):
        weights = np.asarray(weights, dtype=float)
        if len(policies) != len(weights):
            raise ValueError('policies and weights differ in length.')
        if weights.min() < 0 or weights.sum() <= 0:
            raise ValueError(f'Invalid mixture weights {weights}.')
        self.policies = list(policies)
        self.weights = weights / weights.sum()
        self._current = None

    def reset(self, rng):
        self._current = self.policies[rng.choice(len(self.policies), p=self.weights)]
        self._current.reset(rng)

    def act(self, obs, legal_mask, rng):
        if self._current is None:
            raise RuntimeError('MixturePolicy.act() before reset().')
        return self._current.act(obs, legal_mask, rng)
