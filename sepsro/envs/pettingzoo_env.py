"""
PettingZoo adapter, for both the AEC (turn-based) and Parallel APIs.

The environment is named by its module path, e.g.
'pettingzoo.classic.connect_four_v3' or 'mpe2.simple_spread_v3', and built with
module.env(**kwargs) for AEC or module.parallel_env(**kwargs) for Parallel. That
keeps this file free of a hard-coded list of games: anything that follows the
PettingZoo module convention works, including third-party packages.

Action masks: classic games put them in the observation (a Dict space with
'observation' and 'action_mask'); others put them in infos[agent]['action_mask'];
most put them nowhere, meaning every action is legal. All three are handled.

AEC reward accounting follows PettingZoo's own aec_to_parallel wrapper: after
each live step, env.rewards for every remaining agent is added up; the "dead
steps" that remove finished agents carry no reward.
"""

from __future__ import annotations

import importlib

import numpy as np

from .base import EnvSpec, MultiAgentEnv, TimeStep


def _discrete_n(space, agent):
    n = getattr(space, 'n', None)
    if n is None:
        raise ValueError(f'Agent {agent!r} has a non-discrete action space {space}; '
                         'only discrete actions are supported.')
    return int(n)


def _obs_shape(space):
    # Dict observation spaces (classic games) carry the real observation under
    # 'observation' next to the mask.
    spaces = getattr(space, 'spaces', None)
    if isinstance(spaces, dict) and 'observation' in spaces:
        space = spaces['observation']
    return tuple(space.shape)


class PettingZooEnv(MultiAgentEnv):

    def __init__(self, name, module, api='aec', env_kwargs=None, max_episode_steps=None):
        super().__init__()
        if api not in ('aec', 'parallel'):
            raise ValueError(f"api must be 'aec' or 'parallel', got {api!r}")
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            raise ImportError(f'Could not import PettingZoo env {module!r}: {exc}') from exc
        factory = mod.env if api == 'aec' else mod.parallel_env
        self._env = factory(**dict(env_kwargs or {}))
        self._api = api

        agents = tuple(self._env.possible_agents)
        self.spec = EnvSpec(
            name=name,
            backend='pettingzoo',
            agents=agents,
            num_actions={a: _discrete_n(self._env.action_space(a), a) for a in agents},
            obs_shape={a: _obs_shape(self._env.observation_space(a)) for a in agents},
            turn_based=api == 'aec',
            max_episode_steps=max_episode_steps,
        )

    @property
    def raw(self):
        """The wrapped PettingZoo environment."""
        return self._env

    def close(self):
        self._env.close()

    # ------------------------------------------------------------ backend
    def _reset(self, seed):
        if self._api == 'aec':
            self._env.reset(seed=seed)
            return self._aec_timestep(self._zero_rewards())
        observations, infos = self._env.reset(seed=seed)
        return self._parallel_timestep(observations, infos, self._zero_rewards(), {}, {})

    def _step(self, actions):
        if self._api == 'aec':
            return self._aec_step(actions)
        observations, rewards, terminations, truncations, infos = self._env.step(
            {a: int(actions[a]) for a in self._env.agents})
        return self._parallel_timestep(observations, infos, rewards, terminations, truncations)

    # ---------------------------------------------------------------- AEC
    def _aec_step(self, actions):
        env = self._env
        env.step(int(actions[env.agent_selection]))
        rewards = self._zero_rewards()
        for agent in env.agents:
            rewards[agent] += float(env.rewards[agent])
        # Read the end condition now: the dead steps below delete the agents'
        # terminations/truncations entries, and with them the information
        # whether the game ended or was cut off (e.g. rps_v2's max_cycles).
        truncated = (any(env.truncations[a] for a in env.agents)
                     and not any(env.terminations[a] for a in env.agents))
        self._skip_dead_agents()
        return self._aec_timestep(rewards, truncated=truncated)

    def _skip_dead_agents(self):
        env = self._env
        while env.agents and (env.terminations[env.agent_selection]
                              or env.truncations[env.agent_selection]):
            env.step(None)

    def _aec_timestep(self, rewards, truncated=False):
        env = self._env
        self._skip_dead_agents()
        observations, legal_masks = {}, {}
        for agent in env.agents:
            obs, mask = self._split(agent, env.observe(agent), env.infos.get(agent, {}))
            observations[agent] = obs
            if agent == env.agent_selection:
                legal_masks[agent] = mask
        done = not env.agents
        return TimeStep(
            observations=observations,
            legal_masks=legal_masks,
            acting=() if done else (env.agent_selection,),
            rewards=rewards,
            terminated=done and not truncated,
            truncated=done and truncated,
        )

    # ----------------------------------------------------------- Parallel
    def _parallel_timestep(self, observations, infos, rewards, terminations, truncations):
        live = tuple(a for a in self._env.agents
                     if not (terminations.get(a) or truncations.get(a)))
        obs_out, legal_masks = {}, {}
        for agent, raw_obs in observations.items():
            obs, mask = self._split(agent, raw_obs, infos.get(agent, {}))
            obs_out[agent] = obs
            if agent in live:
                legal_masks[agent] = mask
        full_rewards = self._zero_rewards()
        for agent, r in rewards.items():
            full_rewards[agent] = float(r)
        done = not live
        truncated = done and bool(truncations) and all(truncations.values())
        return TimeStep(
            observations=obs_out,
            legal_masks=legal_masks,
            acting=live,
            rewards=full_rewards,
            terminated=done and not truncated,
            truncated=truncated,
        )

    # ------------------------------------------------------------- helpers
    def _split(self, agent, raw_obs, info):
        """(observation, action mask) from whichever place the env keeps the mask."""
        n = self.spec.num_actions[agent]
        if isinstance(raw_obs, dict) and 'observation' in raw_obs:
            obs = raw_obs['observation']
            mask = raw_obs.get('action_mask')
        else:
            obs, mask = raw_obs, None
        if mask is None:
            mask = info.get('action_mask') if isinstance(info, dict) else None
        mask = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        return np.asarray(obs, dtype=np.float32), mask
