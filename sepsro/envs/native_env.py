"""
Dependency-free normal-form (matrix) games.

These need neither PettingZoo nor OpenSpiel, which makes them the place to
debug a method: rock-paper-scissors is the textbook PSRO sanity check (its Nash
equilibrium is known exactly), and the tests use these so the core of the
codebase can be checked on any machine.
"""

from __future__ import annotations

import numpy as np

from .base import EnvSpec, MultiAgentEnv, TimeStep

# payoffs[a0, a1] = (reward to player_0, reward to player_1)
MATRIX_GAMES = {
    'rps': np.array([
        [[0, 0], [-1, 1], [1, -1]],
        [[1, -1], [0, 0], [-1, 1]],
        [[-1, 1], [1, -1], [0, 0]],
    ], dtype=float),
    'matching_pennies': np.array([
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]],
    ], dtype=float),
    'prisoners_dilemma': np.array([
        [[-1, -1], [-3, 0]],
        [[0, -3], [-2, -2]],
    ], dtype=float),
}


class MatrixGameEnv(MultiAgentEnv):
    """One simultaneous move, then terminal. Observation is a constant 1."""

    def __init__(self, name, game, max_episode_steps=None):
        super().__init__()
        if game not in MATRIX_GAMES:
            raise ValueError(f'Unknown matrix game {game!r}. Choose one of {sorted(MATRIX_GAMES)}.')
        self._payoffs = MATRIX_GAMES[game]
        agents = ('player_0', 'player_1')
        n0, n1, _ = self._payoffs.shape
        self.spec = EnvSpec(
            name=name,
            backend='native',
            agents=agents,
            num_actions={'player_0': n0, 'player_1': n1},
            obs_shape={a: (1,) for a in agents},
            turn_based=False,
            zero_sum=bool(np.allclose(self._payoffs.sum(axis=-1), 0)),
            max_episode_steps=max_episode_steps,
        )

    @property
    def payoffs(self):
        return self._payoffs

    def _obs(self):
        return {a: np.ones(1, dtype=np.float32) for a in self.spec.agents}

    def _reset(self, seed):
        agents = self.spec.agents
        return TimeStep(
            observations=self._obs(),
            legal_masks={a: np.ones(self.spec.num_actions[a], dtype=bool) for a in agents},
            acting=agents,
            rewards=self._zero_rewards(),
        )

    def _step(self, actions):
        r0, r1 = self._payoffs[int(actions['player_0']), int(actions['player_1'])]
        return TimeStep(
            observations=self._obs(),
            legal_masks={},
            acting=(),
            rewards={'player_0': float(r0), 'player_1': float(r1)},
            terminated=True,
        )
