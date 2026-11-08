"""
Environment construction.

make_env() takes a resolved preset (see ENV_PRESETS in config.py) and returns a
MultiAgentEnv. Backends are imported only when used, so a missing OpenSpiel or
PettingZoo install only matters for the environments that need it.
"""

from __future__ import annotations

from .base import EnvSpec, MultiAgentEnv, TimeStep

BACKENDS = ('native', 'openspiel', 'pettingzoo')


def make_env(preset: dict, max_episode_steps=None, obs_type='info_state') -> MultiAgentEnv:
    """Build an environment from a preset dict.

    Preset keys:
        backend   'native' | 'openspiel' | 'pettingzoo'
        id        matrix-game name / OpenSpiel game name / PettingZoo module path
        name      label used in logs and run directories
        api       PettingZoo only: 'aec' (default) or 'parallel'
        kwargs    backend constructor kwargs (OpenSpiel game params, PZ env kwargs)
        wrappers  optional {wrapper name: kwargs}, applied in order (envs/wrappers.py).
                  max_episode_steps then counts the outermost env's steps.
    """
    wrappers = preset.get('wrappers') or {}
    if wrappers:
        from .wrappers import WRAPPERS
        env = make_env({k: v for k, v in preset.items() if k != 'wrappers'}, obs_type=obs_type)
        for i, (name, kwargs) in enumerate(wrappers.items()):
            last = i == len(wrappers) - 1
            env = WRAPPERS[name](env, **(kwargs or {}),
                                 max_episode_steps=max_episode_steps if last else None)
        return env

    backend = preset['backend']
    name = preset.get('name', preset['id'])
    kwargs = preset.get('kwargs') or {}

    if backend == 'native':
        from .native_env import MatrixGameEnv
        return MatrixGameEnv(name, preset['id'], max_episode_steps=max_episode_steps)
    if backend == 'openspiel':
        from .openspiel_env import OpenSpielEnv
        return OpenSpielEnv(name, preset['id'], game_params=kwargs, obs_type=obs_type,
                            max_episode_steps=max_episode_steps)
    if backend == 'pettingzoo':
        from .pettingzoo_env import PettingZooEnv
        return PettingZooEnv(name, preset['id'], api=preset.get('api', 'aec'),
                             env_kwargs=kwargs, max_episode_steps=max_episode_steps)
    raise ValueError(f'Unknown backend {backend!r}. Choose one of {BACKENDS}.')


__all__ = ['BACKENDS', 'EnvSpec', 'MultiAgentEnv', 'TimeStep', 'make_env']
