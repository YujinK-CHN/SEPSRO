"""
DQN learner.

A PyTorch port of OpenSpiel's open_spiel/python/jax/dqn.py, the learner psro_v2
wraps as its 'DQN' oracle (rl_policy.DQNPolicy). Same algorithm and defaults:

  * MLP with ReLU hidden layers, Glorot-uniform weights, zero biases;
  * epsilon-greedy over legal actions, epsilon on an exponential (or linear)
    schedule of the learner's step counter;
  * FIFO replay buffer, uniform sampling; no learning until it holds
    `min_buffer_size_to_learn` transitions;
  * a gradient step every `learn_every` steps, a SOFT target update
    (target <- tau * online + (1 - tau) * target) every
    `update_target_network_every` steps;
  * target r + discount * max over the next state's LEGAL actions of the target
    network, cut at final steps; MSE (0.5 * err^2, optax.l2_loss) or Huber.

Timing: the JAX agent learns from inside step(), and adds a transition at the
agent's NEXT step. Here the step counter, learning and target updates run on
every action exactly as there, and an episode's transitions are added when it
ends (end_episode). For the short games PSRO is run on, the difference is a lag
of at most one episode in what the buffer holds.

Frozen snapshots act greedily, which is what a frozen JAX DQN does (it is
always stepped with is_evaluation=True).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..policies import Policy
from .networks import is_image, make_encoder


@dataclass
class DQNConfig:
    # Defaults are psro_v2_example.py's where it sets them, else the library's.
    hidden: tuple = (256, 256, 256, 256)
    batch_size: int = 32
    replay_buffer_capacity: int = 10_000
    learning_rate: float = 1e-2
    update_target_network_every: int = 1000
    weight_update_coeff: float = 0.995       # tau of the soft target update
    learn_every: int = 10
    discount_factor: float = 1.0
    min_buffer_size_to_learn: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_duration: int = 1_000_000
    epsilon_decay_schedule: str = 'exp'
    optimizer: str = 'adam'
    loss: str = 'mse'
    huber_loss_parameter: float = 1.0
    gradient_clipping: float | None = None

    @classmethod
    def from_args(cls, args):
        return cls(hidden=tuple(args.dqn_hidden), batch_size=args.dqn_batch_size,
                   replay_buffer_capacity=args.dqn_replay_buffer_capacity,
                   learning_rate=args.dqn_lr,
                   update_target_network_every=args.dqn_update_target_network_every,
                   weight_update_coeff=args.dqn_weight_update_coeff,
                   learn_every=args.dqn_learn_every, discount_factor=args.dqn_discount,
                   min_buffer_size_to_learn=args.dqn_min_buffer_size_to_learn,
                   epsilon_start=args.dqn_epsilon_start, epsilon_end=args.dqn_epsilon_end,
                   epsilon_decay_duration=args.dqn_epsilon_decay_duration,
                   epsilon_decay_schedule=args.dqn_epsilon_decay_schedule,
                   optimizer=args.dqn_optimizer, loss=args.dqn_loss,
                   gradient_clipping=args.dqn_gradient_clipping)


class QNet(nn.Module):

    def __init__(self, obs_shape, num_actions, hidden):
        super().__init__()
        # Images go through a CNN first; vectors are flattened, as in the JAX code.
        self.encoder = make_encoder(obs_shape)
        layers, last = [], self.encoder.out_dim
        for width in hidden:
            layers += [self._linear(last, width), nn.ReLU()]
            last = width
        layers.append(self._linear(last, num_actions))
        self.model = nn.Sequential(*layers)
        self.num_actions = num_actions

    @staticmethod
    def _linear(in_dim, out_dim):
        layer = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, obs):
        return self.model(self.encoder(obs))


@torch.no_grad()
def _greedy_probs(net, obs, legal_mask, epsilon):
    q = net(torch.as_tensor(np.asarray(obs)).unsqueeze(0))[0].numpy()
    q = np.where(legal_mask, q, -np.inf)
    probs = (1.0 - epsilon) * np.eye(len(q))[int(q.argmax())]
    probs = probs + epsilon * legal_mask / legal_mask.sum()
    return probs / probs.sum()


class DQNPolicy(Policy):
    """Frozen DQN: greedy over legal actions."""

    def __init__(self, net):
        self.net = net

    def action_probs(self, obs, legal_mask):
        return _greedy_probs(self.net, obs, np.asarray(legal_mask, bool), 0.0)

    def act(self, obs, legal_mask, rng):
        return int(self.action_probs(obs, legal_mask).argmax())


class _ActingPolicy(Policy):
    """Epsilon-greedy policy of a learner that is training; every action is one learner step."""

    def __init__(self, learner):
        self.learner = learner

    def act(self, obs, legal_mask, rng):
        learner = self.learner
        probs = _greedy_probs(learner.net, obs, np.asarray(legal_mask, bool), learner.epsilon())
        action = int(rng.choice(len(probs), p=probs))
        learner.on_step()
        return action


class DQNLearner:

    policy_cls = DQNPolicy

    def __init__(self, obs_shape, num_actions, config: DQNConfig, seed=0):
        torch.manual_seed(seed)
        self.config = config
        self.num_actions = num_actions
        self.obs_shape = tuple(obs_shape)
        self.net = QNet(self.obs_shape, num_actions, config.hidden)
        self.target = copy.deepcopy(self.net)
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f'Unknown optimizer {config.optimizer!r}.')

        cap = int(config.replay_buffer_capacity)
        # Image frames are stored as uint8: 10k Atari transitions take ~0.6 GB
        # instead of ~2.3 GB in float32.
        obs_dtype = np.uint8 if is_image(self.obs_shape) else np.float32
        self._buf = {
            'obs': np.zeros((cap, *self.obs_shape), obs_dtype),
            'action': np.zeros(cap, np.int64),
            'reward': np.zeros(cap, np.float32),
            'next_obs': np.zeros((cap, *self.obs_shape), obs_dtype),
            'final': np.zeros(cap, bool),
            'next_mask': np.zeros((cap, num_actions), bool),
        }
        self._size = 0
        self._index = 0
        self.iteration = 0
        self.last_loss = None
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ policies
    def acting_policy(self) -> Policy:
        return _ActingPolicy(self)

    def policy(self) -> DQNPolicy:
        return DQNPolicy(copy.deepcopy(self.net).eval())

    def load_from(self, policy, sigma=0.0):
        self.net.load_state_dict(policy.net.state_dict())
        if sigma > 0:
            with torch.no_grad():
                for param in self.net.parameters():
                    param.add_(torch.randn_like(param) * sigma)
        self.target.load_state_dict(self.net.state_dict())

    def epsilon(self):
        cfg = self.config
        t = self.iteration
        if cfg.epsilon_decay_schedule == 'exp':
            return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * np.exp(
                -t / cfg.epsilon_decay_duration)
        slope = (cfg.epsilon_end - cfg.epsilon_start) / cfg.epsilon_decay_duration
        return max(slope * t + cfg.epsilon_start, cfg.epsilon_end)

    # ------------------------------------------------------------ training
    def on_step(self):
        cfg = self.config
        self.iteration += 1
        if self.iteration % cfg.learn_every == 0:
            self.last_loss = self.learn()
        if self.iteration % cfg.update_target_network_every == 0:
            tau = cfg.weight_update_coeff
            with torch.no_grad():
                for online, target in zip(self.net.parameters(), self.target.parameters()):
                    target.mul_(1 - tau).add_(online, alpha=tau)

    def end_episode(self, trajectory):
        """Store the episode's transitions; the final step() call of the JAX agent is one more step."""
        for t in trajectory:
            final = t.terminated or t.truncated or t.next_obs is None
            next_obs = 0 if t.next_obs is None else np.asarray(t.next_obs).reshape(self.obs_shape)
            next_mask = np.zeros(self.num_actions, bool) if t.next_legal_mask is None \
                else np.asarray(t.next_legal_mask, bool)
            self._add(np.asarray(t.obs).reshape(self.obs_shape), t.action, t.reward,
                      next_obs, final, next_mask)
        if trajectory:
            self.on_step()

    def _add(self, obs, action, reward, next_obs, final, next_mask):
        i = self._index
        buf = self._buf
        buf['obs'][i], buf['action'][i], buf['reward'][i] = obs, action, reward
        buf['next_obs'][i], buf['final'][i], buf['next_mask'][i] = next_obs, final, next_mask
        self._index = (i + 1) % len(buf['action'])
        self._size = min(self._size + 1, len(buf['action']))

    def learn(self):
        cfg = self.config
        if self._size < cfg.batch_size or self._size < cfg.min_buffer_size_to_learn:
            return None
        idx = self._np_rng.integers(0, self._size, size=cfg.batch_size)
        b = {k: torch.as_tensor(v[idx]) for k, v in self._buf.items()}

        q = self.net(b['obs']).gather(1, b['action'][:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.target(b['next_obs']).masked_fill(~b['next_mask'], -torch.inf)
            next_q = torch.where(b['final'], torch.zeros(()), next_q.max(1).values)
            targets = b['reward'] + cfg.discount_factor * next_q
        if cfg.loss == 'mse':
            loss = (0.5 * (q - targets).pow(2)).mean()
        else:
            loss = F.huber_loss(q, targets, delta=cfg.huber_loss_parameter)

        self.optimizer.zero_grad()
        loss.backward()
        if cfg.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.gradient_clipping)
        self.optimizer.step()
        return loss.item()
