"""
OpenSpiel adapter.

pyspiel is imported lazily: OpenSpiel ships wheels for Linux and macOS only, so
a Windows checkout can still import the rest of the codebase (and run the
native and PettingZoo environments) without it.

Rewards are taken as the difference of state.returns() before and after a
transition rather than from state.rewards(). Both work for the games we care
about, but the difference also folds in any reward produced while resolving
chance nodes, which state.rewards() alone would drop.
"""

from __future__ import annotations

import numpy as np

from .base import EnvSpec, MultiAgentEnv, TimeStep


def _import_pyspiel():
    try:
        import pyspiel
    except ImportError as exc:
        raise ImportError(
            'OpenSpiel is not installed. `pip install open_spiel` '
            '(Linux / macOS only; on Windows use WSL).'
        ) from exc
    return pyspiel


class OpenSpielEnv(MultiAgentEnv):

    def __init__(self, name, game, game_params=None, obs_type='info_state',
                 max_episode_steps=None):
        super().__init__()
        pyspiel = _import_pyspiel()
        self._pyspiel = pyspiel
        # A full game string such as 'kuhn_poker(players=3)' carries its own
        # parameters; load_game(name, params) would treat it as a bare name.
        self._game = (pyspiel.load_game(game, dict(game_params)) if game_params
                      else pyspiel.load_game(game))
        game_type = self._game.get_type()

        if game_type.dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
            raise ValueError(f'{game!r} is a mean-field game; not supported.')
        self._simultaneous = game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS

        # 'info_state' is the perfect-recall view (what CFR / PSRO best responses
        # usually want); 'observation' is the Markov view. Fall back to whichever
        # the game provides rather than failing on a game that has only one.
        if obs_type == 'info_state' and game_type.provides_information_state_tensor:
            self._obs_fn = 'information_state_tensor'
            shape = tuple(self._game.information_state_tensor_shape())
        elif game_type.provides_observation_tensor:
            self._obs_fn = 'observation_tensor'
            shape = tuple(self._game.observation_tensor_shape())
        elif game_type.provides_information_state_tensor:
            self._obs_fn = 'information_state_tensor'
            shape = tuple(self._game.information_state_tensor_shape())
        else:
            raise ValueError(f'{game!r} provides no observation tensor.')

        agents = tuple(f'player_{i}' for i in range(self._game.num_players()))
        num_actions = self._game.num_distinct_actions()
        self.spec = EnvSpec(
            name=name,
            backend='openspiel',
            agents=agents,
            num_actions={a: num_actions for a in agents},
            obs_shape={a: shape for a in agents},
            turn_based=not self._simultaneous,
            zero_sum=game_type.utility == pyspiel.GameType.Utility.ZERO_SUM,
            max_episode_steps=max_episode_steps,
        )
        self._state = None
        self._returns = None

    @property
    def game(self):
        """The underlying pyspiel game, for OpenSpiel-only tools (e.g. NashConv)."""
        return self._game

    @property
    def state(self):
        return self._state

    def observation_for(self, state, player):
        """The observation this env would give `player` at an arbitrary `state`.

        Used by tree-walking tools (exploitability) so a policy sees exactly
        what it saw during training.
        """
        agent = self.spec.agents[player]
        obs = getattr(state, self._obs_fn)(player)
        return np.asarray(obs, dtype=np.float32).reshape(self.spec.obs_shape[agent])

    def legal_mask_for(self, state, player):
        return self._mask(self.spec.agents[player], state.legal_actions(player))

    # ------------------------------------------------------------ backend
    def _reset(self, seed):
        self._state = self._game.new_initial_state()
        self._returns = np.zeros(self.spec.num_agents)
        self._resolve_chance()
        timestep = self._timestep()
        # Rewards from the initial chance deal belong to no action; drop them.
        timestep.rewards = self._zero_rewards()
        return timestep

    def _step(self, actions):
        state = self._state
        if state.is_simultaneous_node():
            state.apply_actions([int(actions[a]) for a in self.spec.agents])
        else:
            agent = self.spec.agents[state.current_player()]
            state.apply_action(int(actions[agent]))
        self._resolve_chance()
        return self._timestep()

    # ------------------------------------------------------------ helpers
    def _resolve_chance(self):
        state = self._state
        while state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(int(self.rng.choice(outcomes, p=probs)))

    def _timestep(self):
        state = self._state
        agents = self.spec.agents
        returns = np.asarray(state.returns(), dtype=float)
        rewards = {a: float(r) for a, r in zip(agents, returns - self._returns)}
        self._returns = returns

        terminated = state.is_terminal()
        if terminated:
            acting = ()
            # Not every game defines tensors at terminal states, and nothing
            # acts on them; zeros keep the shapes consistent for buffers.
            observations = {a: np.zeros(self.spec.obs_shape[a], dtype=np.float32)
                            for a in agents}
        else:
            if state.is_simultaneous_node():
                acting = tuple(a for i, a in enumerate(agents) if state.legal_actions(i))
            else:
                acting = (agents[state.current_player()],)
            observations = {a: self.observation_for(state, i) for i, a in enumerate(agents)}

        legal_masks = {a: self.legal_mask_for(state, agents.index(a)) for a in acting}
        return TimeStep(
            observations=observations,
            legal_masks=legal_masks,
            acting=acting,
            rewards=rewards,
            terminated=terminated,
        )
