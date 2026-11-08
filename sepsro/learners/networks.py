"""
Networks shared by the learners.

make_encoder() picks how an observation becomes a feature vector:
    image  [C, H, W] with H, W >= 36 (e.g. Atari frame stacks): the Nature-DQN
           CNN (Mnih et al., 2015) -> 512 features; uint8 pixels are scaled to
           [0, 1] inside, so observations can stay uint8 in buffers;
    other  flattened as-is, so one MLP serves OpenSpiel information-state vectors
           and PettingZoo board planes exactly as before.
Learners put their MLPs on top of the encoder's features.

Illegal actions are masked by setting their logits to a large negative number
before the softmax, so they get (numerically) zero probability and zero
gradient -- both in acting and in the PPO ratio.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

MASK_VALUE = -1e9


def is_image(obs_shape):
    return len(obs_shape) == 3 and min(obs_shape[1:]) >= 36


class NatureCNN(nn.Module):
    """[B, C, H, W] pixels (0..255) -> [B, 512] features."""

    def __init__(self, obs_shape, features=512):
        super().__init__()
        channels = obs_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, features), nn.ReLU())
        self.out_dim = features

    def forward(self, obs):
        return self.fc(self.conv(obs.float() / 255.0))


class FlattenEncoder(nn.Module):

    def __init__(self, obs_shape):
        super().__init__()
        self.out_dim = int(np.prod(obs_shape))

    def forward(self, obs):
        return obs.float().reshape(obs.shape[0], -1)


def make_encoder(obs_shape):
    obs_shape = tuple(obs_shape)
    return NatureCNN(obs_shape) if is_image(obs_shape) else FlattenEncoder(obs_shape)


def _layer(in_dim, out_dim, gain):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (CleanRL's default layout)."""

    def __init__(self, obs_shape, num_actions, hidden=(128, 128)):
        super().__init__()
        self.obs_dim = int(np.prod(obs_shape))
        self.num_actions = int(num_actions)

        def body():
            layers, last = [], self.obs_dim
            for width in hidden:
                layers += [_layer(last, width, np.sqrt(2)), nn.Tanh()]
                last = width
            return layers, last

        actor, last = body()
        self.actor = nn.Sequential(*actor, _layer(last, self.num_actions, 0.01))
        critic, last = body()
        self.critic = nn.Sequential(*critic, _layer(last, 1, 1.0))

    def forward(self, obs, mask):
        obs = obs.reshape(obs.shape[0], -1)
        logits = self.actor(obs).masked_fill(~mask, MASK_VALUE)
        value = self.critic(obs).squeeze(-1)
        return logits, value
