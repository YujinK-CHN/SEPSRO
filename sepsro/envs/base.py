"""
One multi-agent environment interface for every backend.

PettingZoo (AEC and Parallel) and OpenSpiel disagree on almost everything:
who moves when, how rewards are reported, where the action mask lives, what an
observation is. Every method in this codebase -- PSRO, PPO, GEMS, whatever
comes next -- should be written once against the interface below, never against
a backend directly.

The unifying idea is that every step is a dict of actions keyed by agent, and
every TimeStep says WHICH agents must act next:

    turn-based (OpenSpiel sequential, PettingZoo AEC)   acting == (one agent,)
    simultaneous (OpenSpiel simultaneous, PZ Parallel)  acting == every live agent

so a rollout loop never has to know which kind of game it is playing.

Conventions:
    * Agents are strings ('player_0', 'player_1', ... for OpenSpiel; the
      environment's own names for PettingZoo). spec.agents fixes their order,
      and that order is the 'seat' index used by populations and payoff tables.
    * Actions are discrete ints in [0, spec.num_actions[agent]).
    * TimeStep.rewards holds the reward EVERY agent received from the last
      transition, not only the agents that acted. In turn-based games most
      reward lands on the agent that did not move.
    * Chance nodes are resolved inside the environment with its own RNG, so
      they never show up as an acting agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    """Static description of an environment, fixed for its whole lifetime."""

    name: str                       # preset name, e.g. 'kuhn_poker'
    backend: str                    # 'openspiel' | 'pettingzoo' | 'native'
    agents: tuple                   # seat order
    num_actions: dict               # agent -> int
    obs_shape: dict                 # agent -> tuple
    turn_based: bool                # True if exactly one agent acts per step
    zero_sum: bool | None = None    # None when the backend does not say
    max_episode_steps: int | None = None

    @property
    def num_agents(self):
        return len(self.agents)


@dataclass
class TimeStep:
    """What the environment returns from reset() and step()."""

    observations: dict              # agent -> np.ndarray, for every live agent
    legal_masks: dict               # agent -> bool np.ndarray, for agents in `acting`
    acting: tuple                   # agents that must act in the next step()
    rewards: dict                   # agent -> float, from the last transition
    terminated: bool = False        # the game reached a terminal state
    truncated: bool = False         # stopped by max_episode_steps
    info: dict = field(default_factory=dict)

    @property
    def done(self):
        return self.terminated or self.truncated


class MultiAgentEnv(ABC):
    """Base class for every backend adapter.

    Subclasses implement _reset() and _step(). The public reset()/step() wrap
    them to enforce max_episode_steps identically everywhere, so truncation
    means the same thing whichever backend produced the episode.
    """

    spec: EnvSpec

    def __init__(self):
        self._t = 0
        self._done = True
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------ API
    def reset(self, seed=None) -> TimeStep:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        timestep = self._reset(seed)
        self._done = timestep.done
        return timestep

    def step(self, actions: dict) -> TimeStep:
        if self._done:
            raise RuntimeError('step() called on a finished episode; call reset() first.')
        timestep = self._step(actions)
        self._t += 1
        limit = self.spec.max_episode_steps
        if not timestep.done and limit is not None and self._t >= limit:
            timestep.truncated = True
            timestep.acting = ()
            timestep.legal_masks = {}
        self._done = timestep.done
        return timestep

    def close(self):
        pass

    # ------------------------------------------------------------ backends
    @abstractmethod
    def _reset(self, seed) -> TimeStep:
        ...

    @abstractmethod
    def _step(self, actions: dict) -> TimeStep:
        ...

    # ------------------------------------------------------------- helpers
    def _mask(self, agent, legal_actions):
        mask = np.zeros(self.spec.num_actions[agent], dtype=bool)
        mask[list(legal_actions)] = True
        return mask

    def _zero_rewards(self):
        return {agent: 0.0 for agent in self.spec.agents}
