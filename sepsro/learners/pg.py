"""
Policy gradient learner: QPG / RPG / RM / A2C.

A PyTorch port of OpenSpiel's open_spiel/python/jax/policy_gradient.py, the
learner psro_v2 wraps as its 'PG' oracle (rl_policy.PGPolicy). Same algorithm,
same defaults, same update schedule:

  * one network: an MLP torso (ReLU, final activation on) with a policy head and
    either a Q head (qpg/rpg/rm) or a V head (a2c);
  * two optimizers over the SAME parameters: the critic loss updates the whole
    network after every episode, the policy loss every `num_critic_before_pi`
    episodes, both on that single episode;
  * critic target: SARSA(lambda) for Q, TD(lambda) for V, with lambda = 1 by
    default (Monte-Carlo returns);
  * losses from Srinivasan et al., 2018 (QPG, RPG, RM) and standard A2C.

Deliberate differences from the JAX code, each a correction rather than a
change of setting:

  * illegal actions are masked out of the policy LOSS as well as out of acting.
    The JAX loss softmaxes over every action, so Q-values of actions that are
    never taken (never trained) pull probability towards illegal actions.
  * the policy loss is computed over the episode's decision states only; the
    JAX batch also includes the terminal information state.

Like the JAX code, an episode's end is always treated as terminal (no
bootstrap), so this learner assumes episodes are not truncated -- true for the
OpenSpiel games psro_v2 targets.

Entropy follows the JAX sign (rlax.entropy_loss = -entropy, so the bonus RAISES
entropy). OpenSpiel's PyTorch rl_losses.py adds +entropy instead, which would
lower it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..policies import Policy
from .networks import MASK_VALUE, make_encoder

LOSSES = ('qpg', 'rpg', 'rm', 'a2c')


@dataclass
class PGConfig:
    # Defaults are psro_v2_example.py's (which override some library defaults).
    loss: str = 'qpg'
    hidden: tuple = (256, 256, 256, 256)
    lambda_: float = 1.0
    critic_lr: float = 1e-2
    pi_lr: float = 1e-3
    entropy_cost: float = 0.001
    num_critic_before_pi: int = 8
    optimizer: str = 'adam'
    max_grad_norm: float | None = None

    @classmethod
    def from_args(cls, args):
        return cls(loss=args.pg_loss, hidden=tuple(args.pg_hidden), lambda_=args.pg_lambda,
                   critic_lr=args.pg_critic_lr, pi_lr=args.pg_pi_lr,
                   entropy_cost=args.pg_entropy_cost,
                   num_critic_before_pi=args.pg_num_critic_before_pi,
                   optimizer=args.pg_optimizer, max_grad_norm=args.pg_max_grad_norm)


def haiku_linear(in_dim, out_dim):
    """nn.Linear initialised like haiku.Linear: truncated normal, std 1/sqrt(fan_in), zero bias."""
    layer = nn.Linear(in_dim, out_dim)
    std = 1.0 / np.sqrt(in_dim)
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    nn.init.zeros_(layer.bias)
    return layer


class PGNet(nn.Module):
    """NetPG (policy + Q heads) or NetA2C (policy + V head) from the JAX code."""

    def __init__(self, obs_shape, num_actions, hidden, a2c):
        super().__init__()
        # Images go through a CNN first (not in the JAX code, which only sees
        # OpenSpiel vectors); vectors are flattened, as there.
        self.encoder = make_encoder(obs_shape)
        layers, last = [], self.encoder.out_dim
        for width in hidden:
            layers += [haiku_linear(last, width), nn.ReLU()]
            last = width
        self.torso = nn.Sequential(*layers)
        self.policy_head = haiku_linear(last, num_actions)
        self.value_head = haiku_linear(last, 1 if a2c else num_actions)
        self.num_actions = num_actions

    def forward(self, obs):
        hidden = self.torso(self.encoder(obs))
        return self.policy_head(hidden), self.value_head(hidden)


class PGPolicy(Policy):
    """Acts by sampling the masked softmax (generate_act_func in the JAX code)."""

    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def action_probs(self, obs, legal_mask):
        obs_t = torch.as_tensor(np.asarray(obs)).unsqueeze(0)
        logits, _ = self.net(obs_t)
        probs = torch.softmax(logits, dim=-1)[0].double().numpy() * legal_mask
        return probs / probs.sum()

    def act(self, obs, legal_mask, rng):
        probs = self.action_probs(obs, legal_mask)
        return int(rng.choice(len(probs), p=probs))


def _optimizer(name, params, lr):
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"optimizer must be 'adam' or 'sgd', got {name!r}")


class PGLearner:
    """One seat's policy-gradient learner. Fed one finished episode at a time."""

    policy_cls = PGPolicy

    def __init__(self, obs_shape, num_actions, config: PGConfig, seed=0):
        if config.loss not in LOSSES:
            raise ValueError(f'Unknown PG loss {config.loss!r}. Choose one of {LOSSES}.')
        torch.manual_seed(seed)
        self.config = config
        self.net = PGNet(obs_shape, num_actions, config.hidden,
                         a2c=config.loss == 'a2c')
        self._critic_opt = _optimizer(config.optimizer, self.net.parameters(), config.critic_lr)
        self._pi_opt = _optimizer(config.optimizer, self.net.parameters(), config.pi_lr)
        self.episodes = 0
        self.last_losses = {}

    # ------------------------------------------------------------ policies
    def acting_policy(self) -> Policy:
        """The live policy used while training (shares weights with the learner)."""
        return PGPolicy(self.net)

    def policy(self) -> PGPolicy:
        """A frozen snapshot, safe to put in a population."""
        return PGPolicy(copy.deepcopy(self.net).eval())

    def load_from(self, policy, sigma=0.0):
        """Start from an existing PG policy's weights, plus N(0, sigma) noise."""
        self.net.load_state_dict(policy.net.state_dict())
        if sigma > 0:
            with torch.no_grad():
                for param in self.net.parameters():
                    param.add_(torch.randn_like(param) * sigma)

    # ------------------------------------------------------------ training
    def end_episode(self, trajectory):
        """One critic update, and every num_critic_before_pi episodes one policy update."""
        if not trajectory:
            return
        batch = self._batch(trajectory)
        self.last_losses['critic'] = self._step(self._critic_opt, self._critic_loss(batch))
        self.episodes += 1
        if self.episodes % self.config.num_critic_before_pi == 0:
            self.last_losses['pi'] = self._step(self._pi_opt, self._pi_loss(batch))

    def _step(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
        optimizer.step()
        return loss.item()

    def _batch(self, trajectory):
        # Observations keep their dtype (uint8 frames stay uint8 until the encoder).
        obs = torch.as_tensor(np.stack([np.asarray(t.obs) for t in trajectory]))
        mask = torch.as_tensor(np.stack([t.legal_mask for t in trajectory]), dtype=torch.bool)
        actions = torch.as_tensor([t.action for t in trajectory], dtype=torch.long)
        rewards = torch.as_tensor([t.reward for t in trajectory], dtype=torch.float32)
        # rl_environment's discount: 1 within the episode, 0 on the final step.
        discounts = torch.ones(len(trajectory))
        discounts[-1] = 0.0
        return obs, mask, actions, rewards, discounts

    def _lambda_returns(self, rewards, discounts, next_values):
        """rlax.lambda_returns: G_t = r_t + d_t ((1 - lambda) v_{t+1} + lambda G_{t+1}), G_T = 0."""
        lam = self.config.lambda_
        returns = torch.zeros_like(rewards)
        g = torch.zeros(())
        for t in reversed(range(len(rewards))):
            g = rewards[t] + discounts[t] * ((1 - lam) * next_values[t] + lam * g)
            returns[t] = g
        return returns

    def _critic_loss(self, batch):
        obs, mask, actions, rewards, discounts = batch
        _, values = self.net(obs)
        if self.config.loss == 'a2c':
            v = values.squeeze(-1)
            next_v = torch.cat([v[1:], torch.zeros(1)]).detach()
            targets = self._lambda_returns(rewards, discounts, next_v)
            return (targets - v).pow(2).mean()
        # SARSA(lambda) on the Q head.
        q_taken = values.gather(1, actions[:, None]).squeeze(1)
        next_q = torch.cat([q_taken[1:], torch.zeros(1)]).detach()
        targets = self._lambda_returns(rewards, discounts, next_q)
        return (targets - q_taken).pow(2).mean()

    def _pi_loss(self, batch):
        obs, mask, actions, rewards, discounts = batch
        logits, values = self.net(obs)
        logits = logits.masked_fill(~mask, MASK_VALUE)
        policy = F.softmax(logits, dim=1)
        log_policy = F.log_softmax(logits, dim=1)
        entropy = -(policy * torch.where(mask, log_policy, torch.zeros_like(log_policy))).sum(1).mean()

        loss_name = self.config.loss
        if loss_name == 'a2c':
            v = values.squeeze(-1)
            next_v = torch.cat([v[1:], torch.zeros(1)]).detach()
            advantages = (self._lambda_returns(rewards, discounts, next_v) - v).detach()
            pi_loss = (F.cross_entropy(logits, actions, reduction='none') * advantages).mean()
        else:
            q = torch.where(mask, values, torch.zeros_like(values)).detach()
            baseline = (policy * q).sum(1, keepdim=True)
            advantages = q - baseline
            if loss_name == 'rpg':
                # Regret policy gradient: total positive regret, gradient through the baseline.
                pi_loss = F.relu(advantages).masked_fill(~mask, 0.0).sum(1).mean()
            else:
                if loss_name == 'rm':
                    advantages = F.relu(advantages)
                pi_loss = -(policy * advantages.detach()).sum(1).mean()
        return pi_loss - self.config.entropy_cost * entropy
