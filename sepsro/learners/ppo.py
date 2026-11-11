"""
PPO for one seat, against opponents held fixed.

Used as the best-response learner of the post-hoc approximate NashConv
(posthoc.py exploit), which must be the same for every method, and -- through
PPOOracleLearner at the end of this file -- as PSRO's `--psro-oracle ppo`, so
that PSRO and GEMS (whose ABR-TR is PPO) can be compared with the same kind of
best-response learner. The update follows CleanRL's ppo.py
(clipped surrogate, GAE, advantage normalisation, value loss, entropy bonus,
global gradient clipping), with two multi-agent changes:

  * experience comes from rollout.play_episode with record=True, so in a
    turn-based game each Transition spans from this seat's decision to its next
    one, with every reward it received in between;
  * the bootstrap at an episode's end distinguishes termination (value 0) from
    truncation (value of the last observation), using the flags the env
    adapters report.

Rewards are multiplied by `reward_scale` before GAE (a fixed constant, so the
objective -- the expected return -- is unchanged, unlike CleanRL's Atari sign
clipping, which would make the learner maximise hits instead of score and so
not be a best response in the game's own payoffs). It keeps the value targets
O(1) on games with large raw scores.

Acting happens on a CPU copy of the network (`actor`), synced after every
update: per-step single-observation forwards are faster on CPU, and the update
itself can run on GPU.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from ..policies import Policy
from ..rollout import play_episode
from .networks import ActorCritic


@dataclass
class PPOConfig:
    lr: float = 2.5e-4
    rollout_steps: int = 1024        # learner transitions per update
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: tuple = (128, 128)
    reward_scale: float = 1.0        # rewards are multiplied by this for training only

    @classmethod
    def from_args(cls, args):
        return cls(lr=args.ppo_lr, rollout_steps=args.ppo_rollout_steps,
                   epochs=args.ppo_epochs, minibatches=args.ppo_minibatches,
                   gamma=args.ppo_gamma, gae_lambda=args.ppo_gae_lambda,
                   clip=args.ppo_clip, ent_coef=args.ppo_ent_coef,
                   vf_coef=args.ppo_vf_coef, max_grad_norm=args.ppo_max_grad_norm,
                   hidden=tuple(args.ppo_hidden), reward_scale=args.ppo_reward_scale)


def psro_ppo_config(args):
    """PSRO's PPO oracle: the ppo group (+ ENV_OVERRIDES), with any --psro-ppo-* set
    replacing it. The ppo group is also the post-hoc scorer's learner, so tuning the
    oracle through --psro-ppo-* leaves the scoring protocol unchanged."""
    config = PPOConfig.from_args(args)
    for field in ('lr', 'rollout_steps', 'epochs', 'minibatches', 'clip', 'ent_coef'):
        value = getattr(args, f'psro_ppo_{field}', None)
        if value is not None:
            setattr(config, field, value)
    return config


class NetworkPolicy(Policy):
    """A Policy backed by an ActorCritic on CPU. Acting never builds a graph."""

    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def evaluate(self, obs, legal_mask):
        """(action probabilities, value) for one observation."""
        obs_t = torch.as_tensor(np.asarray(obs)).unsqueeze(0)   # uint8 frames stay uint8
        mask_t = torch.as_tensor(np.asarray(legal_mask, dtype=bool)).unsqueeze(0)
        logits, value = self.net(obs_t, mask_t)
        probs = torch.softmax(logits, dim=-1)[0].double().numpy()
        probs = probs * legal_mask
        return probs / probs.sum(), float(value[0])

    def action_probs(self, obs, legal_mask):
        return self.evaluate(obs, legal_mask)[0]

    def act(self, obs, legal_mask, rng):
        probs = self.action_probs(obs, legal_mask)
        return int(rng.choice(len(probs), p=probs))


class _RecordingPolicy(NetworkPolicy):
    """Acts like NetworkPolicy and remembers log-prob and value for the rollout hook."""

    last = None

    def act(self, obs, legal_mask, rng):
        probs, value = self.evaluate(obs, legal_mask)
        action = int(rng.choice(len(probs), p=probs))
        self.last = {'logp': float(np.log(probs[action])), 'value': value}
        return action


class PPOLearner:

    def __init__(self, obs_shape, num_actions, config: PPOConfig, device='cpu', seed=0):
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.net = ActorCritic(obs_shape, num_actions, config.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)
        self.actor = copy.deepcopy(self.net).to('cpu').eval()
        self._torch_rng = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------ policies
    def policy(self) -> NetworkPolicy:
        """A frozen snapshot of the current policy (safe to put in a population)."""
        return NetworkPolicy(copy.deepcopy(self.actor))

    def load_policy(self, policy: NetworkPolicy):
        """Warm-start from an existing snapshot (PSRO's --psro-br-init latest)."""
        self.net.load_state_dict(policy.net.state_dict())
        self._sync_actor()

    def _sync_actor(self):
        self.actor.load_state_dict({k: v.detach().cpu() for k, v in self.net.state_dict().items()})

    # ------------------------------------------------------------ training
    def train(self, env, seat, opponents, rng, budget, steps, on_update=None):
        """Train `seat` against `opponents` for `steps` env steps (or until budget runs out).

        opponents   agent -> Policy for every other seat (a MixturePolicy samples a
                    member per episode, which is how a meta-strategy is played)
        on_update   optional callback(stats: dict) after every PPO update
        """
        start = budget.used
        updates = 0
        while budget.used - start < steps and not budget.exhausted:
            episodes = self.collect(env, seat, opponents, rng, budget)
            if not episodes:
                break
            stats = self.update(episodes)
            updates += 1
            stats.update(updates=updates, br_steps=budget.used - start)
            if on_update:
                on_update(stats)
        return updates

    def collect(self, env, seat, opponents, rng, budget):
        """Play episodes until `rollout_steps` transitions of `seat` are collected."""
        recorder = _RecordingPolicy(self.actor)
        policies = dict(opponents)
        policies[seat] = recorder

        def hook(agent, obs, mask, action):
            return recorder.last if agent == seat else None

        episodes, count = [], 0
        while count < self.config.rollout_steps and not budget.exhausted:
            ep = play_episode(env, policies, rng, budget=budget, record=True, on_action=hook)
            trajectory = ep.trajectories[seat]
            if trajectory:
                episodes.append(trajectory)
                count += len(trajectory)
        return episodes

    def update(self, episodes):
        cfg = self.config
        obs, masks, actions, logps, advantages, returns = self._batch(episodes)
        n = len(actions)
        minibatch = max(1, n // cfg.minibatches)
        clip_fracs, stats = [], {}

        for _ in range(cfg.epochs):
            order = torch.randperm(n, generator=self._torch_rng)
            for start in range(0, n, minibatch):
                idx = order[start:start + minibatch].to(self.device)
                logits, values = self.net(obs[idx], masks[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(actions[idx])
                ratio = (new_logp - logps[idx]).exp()

                adv = advantages[idx]
                if len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio,
                                    -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * (values - returns[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                clip_fracs.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                stats = {'pg_loss': pg_loss.item(), 'v_loss': v_loss.item(),
                         'entropy': entropy.item()}

        self._sync_actor()
        stats.update(transitions=n, episodes=len(episodes), clip_frac=float(np.mean(clip_fracs)),
                     mean_return=float(np.mean([sum(t.reward for t in ep) for ep in episodes])))
        return stats

    # ------------------------------------------------------------- helpers
    def _batch(self, episodes):
        """Flatten episodes into tensors, computing GAE per episode."""
        cfg = self.config
        obs, masks, actions, logps, advantages, returns = [], [], [], [], [], []
        for ep in episodes:
            values = np.array([t.info['value'] for t in ep])
            # Training reward only: the critic learns scaled returns, while
            # mean_return in the stats and every evaluation stay in raw units.
            rewards = np.array([t.reward for t in ep]) * cfg.reward_scale
            last = ep[-1]
            if last.terminated or last.next_obs is None:
                bootstrap = 0.0
            else:
                # Truncated: the episode was cut off, not finished, so the
                # remaining return is estimated from the last observation.
                mask = last.next_legal_mask
                if mask is None:
                    mask = np.ones(self.actor.num_actions, dtype=bool)
                bootstrap = NetworkPolicy(self.actor).evaluate(last.next_obs, mask)[1]
            next_values = np.append(values[1:], bootstrap)
            deltas = rewards + cfg.gamma * next_values - values
            adv = np.zeros_like(deltas)
            running = 0.0
            for t in reversed(range(len(ep))):
                running = deltas[t] + cfg.gamma * cfg.gae_lambda * running
                adv[t] = running
            advantages.append(adv)
            returns.append(adv + values)
            for t in ep:
                obs.append(np.asarray(t.obs))   # shape and dtype kept (CNN for images)
                masks.append(t.legal_mask)
                actions.append(t.action)
                logps.append(t.info['logp'])

        def tensor(x, dtype):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

        obs = torch.as_tensor(np.stack(obs), device=self.device)
        return (obs, tensor(masks, torch.bool),
                tensor(actions, torch.long), tensor(logps, torch.float32),
                tensor(np.concatenate(advantages), torch.float32),
                tensor(np.concatenate(returns), torch.float32))


class PPOOracleLearner:
    """PPOLearner behind the episode-at-a-time interface of PSRO's RLOracle.

    RLOracle plays the episodes and hands this learner its seat's trajectory
    after each one (end_episode); every `rollout_steps` collected transitions
    trigger one PPO update (epochs x minibatches), as in PPOLearner.train. The
    acting policy is the CPU actor, which changes only at an update, so the
    log-probs and values PPO needs are recomputed exactly from the actor when
    an episode ends. finish() spends the last partial rollout on one more
    update, so no experience the budget paid for is thrown away.
    """

    policy_cls = NetworkPolicy

    def __init__(self, obs_shape, num_actions, config: PPOConfig, seed=0):
        self.ppo = PPOLearner(obs_shape, num_actions, config, seed=seed)
        self._buffer, self._count = [], 0
        self.updates = 0
        self.last_losses = {}

    def acting_policy(self) -> Policy:
        """The live actor (not a copy): it follows every update."""
        return NetworkPolicy(self.ppo.actor)

    def policy(self) -> NetworkPolicy:
        return self.ppo.policy()

    def load_from(self, policy, sigma=0.0):
        """Start from an existing PPO policy's weights, plus N(0, sigma) noise."""
        self.ppo.net.load_state_dict(policy.net.state_dict())
        if sigma > 0:
            with torch.no_grad():
                for param in self.ppo.net.parameters():
                    param.add_(torch.randn_like(param) * sigma)
        self.ppo._sync_actor()

    def end_episode(self, trajectory):
        if not trajectory:
            return
        obs = torch.as_tensor(np.stack([np.asarray(t.obs) for t in trajectory]))
        masks = torch.as_tensor(np.stack([t.legal_mask for t in trajectory]), dtype=torch.bool)
        actions = torch.as_tensor([t.action for t in trajectory])
        with torch.no_grad():
            logits, values = self.ppo.actor(obs, masks)
        logp = torch.log_softmax(logits.double(), -1).gather(1, actions[:, None]).squeeze(1)
        for t, lp, v in zip(trajectory, logp.tolist(), values.tolist()):
            t.info['logp'], t.info['value'] = lp, v
        self._buffer.append(trajectory)
        self._count += len(trajectory)
        if self._count >= self.ppo.config.rollout_steps:
            self._update()

    def finish(self):
        if self._buffer:
            self._update()

    def _update(self):
        stats = self.ppo.update(self._buffer)
        self._buffer, self._count = [], 0
        self.updates += 1
        self.last_losses = {k: stats[k] for k in ('pg_loss', 'v_loss', 'entropy', 'clip_frac')}
