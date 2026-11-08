"""
Populations and the empirical game between them.

Population-based methods (PSRO, GEMS, SESiL-style evolution) all reduce the
real game to an empirical normal-form game: each seat has a population of
policies, and entry [i0, i1, ..., i_{n-1}] of the payoff tensor is the mean
return each seat gets when member i_k plays seat k.

This module only ESTIMATES that tensor. Choosing a meta-strategy over it (Nash,
alpha-rank, PRD, ...) is method logic and lives with the method.

Entries are cached, so growing a population by one member costs only the new
row/column of simulations, not the whole table again.
"""

from __future__ import annotations

import itertools

import numpy as np

from .rollout import average_returns


class Population:
    """An append-only list of policies for one seat, with optional per-member metadata."""

    def __init__(self, policies=None):
        self.policies = []
        self.meta = []
        for policy in policies or []:
            self.add(policy)

    def add(self, policy, **meta):
        self.policies.append(policy)
        self.meta.append(meta)
        return len(self.policies) - 1

    def __len__(self):
        return len(self.policies)

    def __getitem__(self, index):
        return self.policies[index]


class PayoffTable:
    """Empirical payoff tensor over one Population per seat.

    tensor() has shape [len(pop_0), ..., len(pop_{n-1}), n]; the last axis is
    the seat whose payoff it is.
    """

    def __init__(self, env, populations: dict, episodes_per_entry: int):
        self.env = env
        self.agents = env.spec.agents
        missing = set(self.agents) - set(populations)
        if missing:
            raise ValueError(f'No population for seats {sorted(missing)}.')
        self.populations = populations
        self.episodes_per_entry = int(episodes_per_entry)
        self._cache = {}   # index tuple -> np.ndarray of shape [n]

    @property
    def shape(self):
        return tuple(len(self.populations[a]) for a in self.agents)

    def entry(self, index, rng, budget=None):
        index = tuple(int(i) for i in index)
        if index not in self._cache:
            policies = {a: self.populations[a][i] for a, i in zip(self.agents, index)}
            means = average_returns(self.env, policies, rng, self.episodes_per_entry, budget=budget)
            self._cache[index] = np.array([means[a] for a in self.agents])
        return self._cache[index]

    def update(self, rng, budget=None):
        """Simulate every entry not yet in the cache. Returns how many were new."""
        new = 0
        for index in itertools.product(*(range(n) for n in self.shape)):
            if index not in self._cache:
                self.entry(index, rng, budget=budget)
                new += 1
        return new

    def tensor(self):
        out = np.full(self.shape + (len(self.agents),), np.nan)
        for index, value in self._cache.items():
            if all(i < n for i, n in zip(index, self.shape)):
                out[index] = value
        return out
