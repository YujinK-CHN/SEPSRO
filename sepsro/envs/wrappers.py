"""
Environment wrappers: MultiAgentEnv -> MultiAgentEnv.

AtariPreprocessing is the standard DQN-era pipeline (Mnih et al., 2015;
SuperSuit's max_observation / frame_skip / resize / frame_stack), applied per
agent:

    frame skip   every decision is repeated for `frame_skip` emulator frames,
                 rewards summed (the episode ends early if the game does)
    max-pool     the observation is the pixel-wise max of the last two frames,
                 which removes Atari sprite flicker
    resize       to size x size (area interpolation, via torch, so no OpenCV)
    frame stack  the last `stack` processed frames -> observation [stack, H, W]

Observations stay uint8 (0..255): a replay buffer of 84 x 84 x 4 frames costs a
quarter of the memory in float32. Networks divide by 255 in their image encoder
(sepsro/learners/networks.py).

Budget unit with this wrapper: one env.step() is one DECISION, i.e.
`frame_skip` emulator frames.
"""

from __future__ import annotations

import collections

import numpy as np
import torch
import torch.nn.functional as F

from .base import EnvSpec, MultiAgentEnv, TimeStep


class AtariPreprocessing(MultiAgentEnv):

    def __init__(self, env: MultiAgentEnv, frame_skip=4, stack=4, size=84, max_pool=True,
                 max_episode_steps=None):
        super().__init__()
        self.env = env
        self.frame_skip, self.stack, self.size, self.max_pool = frame_skip, stack, size, max_pool
        inner = env.spec
        self.spec = EnvSpec(
            name=inner.name,
            backend=inner.backend,
            agents=inner.agents,
            num_actions=dict(inner.num_actions),
            obs_shape={a: (stack, size, size) for a in inner.agents},
            turn_based=inner.turn_based,
            zero_sum=inner.zero_sum,
            max_episode_steps=max_episode_steps,
        )
        self._frames = {}

    @property
    def raw(self):
        return getattr(self.env, 'raw', self.env)

    def close(self):
        self.env.close()

    # ------------------------------------------------------------ backend
    def _reset(self, seed):
        timestep = self.env.reset(seed=seed)
        self._frames = {}
        for agent, obs in timestep.observations.items():
            frame = self._process(obs)
            self._frames[agent] = collections.deque([frame] * self.stack, maxlen=self.stack)
        return self._wrap(timestep, timestep.rewards)

    def _step(self, actions):
        rewards = {a: 0.0 for a in self.spec.agents}
        last_two = {}
        timestep = None
        for _ in range(self.frame_skip):
            live = actions if timestep is None else {a: actions[a] for a in timestep.acting}
            timestep = self.env.step(live)
            for agent, r in timestep.rewards.items():
                rewards[agent] += r
            for agent, obs in timestep.observations.items():
                last_two.setdefault(agent, collections.deque(maxlen=2)).append(obs)
            if timestep.done:
                break
        for agent, frames in last_two.items():
            obs = np.maximum(frames[0], frames[-1]) if self.max_pool else frames[-1]
            if agent in self._frames:
                self._frames[agent].append(self._process(obs))
        return self._wrap(timestep, rewards)

    # ------------------------------------------------------------ helpers
    def _process(self, obs):
        """[H, W] or [H, W, C] frame -> [size, size] uint8 (channels averaged to grey)."""
        frame = np.asarray(obs, dtype=np.float32)
        if frame.ndim == 3:
            frame = frame.mean(axis=-1)
        resized = F.interpolate(torch.from_numpy(frame)[None, None], size=(self.size, self.size),
                                mode='area')[0, 0]
        return resized.round().clamp(0, 255).to(torch.uint8).numpy()

    def _wrap(self, timestep, rewards):
        observations = {a: np.stack(self._frames[a]) for a in timestep.observations
                        if a in self._frames}
        return TimeStep(observations=observations, legal_masks=timestep.legal_masks,
                        acting=timestep.acting, rewards=rewards,
                        terminated=timestep.terminated, truncated=timestep.truncated,
                        info=timestep.info)


WRAPPERS = {'atari': AtariPreprocessing}
