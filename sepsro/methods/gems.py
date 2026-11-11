"""
GEMS -- Generative Evolutionary Meta-Solver (Sharma et al., TMLR 2026).

Each seat p keeps (App. F / H, the per-player form of Algorithm 1):
    anchors   Z_p, latent codes z in R^d, initialised z ~ N(0, I)
    generator G_p: z -> policy (sepsro/learners/generator.py), plus a critic
    sigma_p   meta-strategy over Z_p, initialised uniform

One iteration (Algorithm 1):
  1. Meta-game estimation (MC). For every anchor i of seat p, play
     N_MC opponents sampled from the other seats' sigma, M_MC episodes each:
     v_p[i] = mean normalised return. No payoff table is ever built.
  2. Meta-strategy update (optimistic MWU):
        sigma_p(i) ~ sigma_p(i) * exp(eta_t * (2 v_p[i] - v_prev_p[i] - r_p))
  3. Population expansion (EB-UCB oracle). Candidates: |mut| codes mutated from
     anchors (z + N(0, mut_std^2)) and |rand| fresh N(0, I) codes. Each plays
     N_oracle x M_oracle episodes against sigma_{-p}; the one with the highest
        mean + sqrt(2 var ln(3/delta_t) / n) + 3 ln(3/delta_t) / (n - 1) - lambda_J ||J G(z)||^2
     joins Z_p (delta_t = delta_0 / t^2).
  4. Amortized best response with trust region (ABR-TR). K_ABR steps on G_p,
     each on fresh episodes of a batch of anchors against sigma_{-p}, trained
     with PPO ("PPO with GAE" in the paper): several epochs of minibatch
     updates of the clipped surrogate plus the trust region,
        maximise  clip-surrogate(A_hat) - beta_KL * KL(pi_theta || pi_theta_old) - lambda_J ||J G(z)||^2
     with GAE advantages from the critic; pi_theta_old is G_p at the start of
     the phase.

Defaults are the paper's Kuhn-poker reference settings (App. Table 3).

Decisions the paper leaves open, made here and listed in the README:
  * rewards are normalised to [0, 1] per seat (the paper assumes they are):
    OpenSpiel min/max utility, matrix-game payoff range, else the range seen in
    random play at start-up;
  * r_hat (the mixture value) is computed as sigma . v_hat instead of from
    B = 128 extra joint samples: it is the same estimate in expectation, and
    in the OMWU update it is a constant across anchors that the
    normalisation removes, so it changes nothing but the log;
  * a new anchor enters sigma with mass 1 / (k + 1) (OMWU cannot raise a
    zero), and its v_prev starts at 0 like every anchor's;
  * mutation parents are drawn proportionally to sigma ("top anchors");
  * the "new fraction" q_new = 0.25 is read as q_t in Eq. 11: the probability
    that a training anchor in ABR-TR is the newly added code;
  * at the anchor cap (k_max = 32) the anchor with the least meta mass
    ('least_mass', or lowest value with 'worst_ev') is replaced;
  * the generator architecture, critic, GAE gamma / lambda, critic learning
    rate and the PPO epochs / minibatches / clip are not given in the paper;
    see config.py for the values used.
For more than two players the per-anchor estimate of step 1 is used for every
seat (unbiased, like the paper's importance-weighted shared-batch estimator,
with lower variance and more episodes).

Budget: every episode of every phase is charged, including the random-play
episodes that set the reward range for PettingZoo envs (played at the start
of train(), so construction spends nothing). Training stops exactly when the
budget is spent (main.py's tracker is hard), mid-iteration if need be, or
after --gems-iterations if that optional cap is set. What is evaluated at any
moment is the output of the last COMPLETED iteration: its anchors, sigma and a
frozen copy of each generator (ABR-TR trains the live one in place). The
learners run on CPU whatever --device says (rollouts are one step at a time).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..learners.generator import (Critic, GeneratorPolicy, jacobian_penalty, make_generator,
                                  masked_logits)
from ..policies import MixturePolicy, RandomPolicy
from ..registry import METHODS
from ..rollout import play_episode
from .base import Method


def omwu_step(sigma, v, v_prev, eta, logit_cap=50.0):
    """Optimistic MWU (Eq. 4 / 17): sigma(i) ~ sigma(i) exp(eta (2 v_i - v_prev_i - r))."""
    r = float(sigma @ v)
    logits = np.log(np.maximum(sigma, 1e-300)) + eta * (2 * v - v_prev - r)
    logits = np.maximum(logits - logits.max(), -logit_cap)
    weights = np.exp(logits)
    return weights / weights.sum()


def eb_ucb(samples, delta):
    """Empirical-Bernstein UCB (Eq. 7, without the Jacobian term) for rewards in [0, 1]."""
    y = np.asarray(samples, dtype=float)
    n = len(y)
    var = y.var(ddof=1) if n > 1 else 0.0
    log_term = math.log(3.0 / delta)
    return y.mean() + math.sqrt(2 * var * log_term / n) + 3 * log_term / max(n - 1, 1)


def return_range(env, rng, budget=None, episodes=200):
    """Per-seat (low, high) of episode returns, used to normalise rewards to [0, 1]."""
    spec = env.spec
    if spec.backend == 'openspiel':
        game = env.game
        return {a: (float(game.min_utility()), float(game.max_utility())) for a in spec.agents}
    if spec.backend == 'native':
        payoffs = env.payoffs
        return {a: (float(payoffs[..., i].min()), float(payoffs[..., i].max()))
                for i, a in enumerate(spec.agents)}
    lows = {a: np.inf for a in spec.agents}
    highs = {a: -np.inf for a in spec.agents}
    joint = {a: RandomPolicy() for a in spec.agents}
    for _ in range(episodes):
        returns = play_episode(env, joint, rng, budget=budget).returns
        for a, r in returns.items():
            lows[a], highs[a] = min(lows[a], r), max(highs[a], r)
    return {a: (lows[a], highs[a] if highs[a] > lows[a] else lows[a] + 1.0) for a in spec.agents}


@METHODS.register('gems')
class GEMS(Method):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        a = self.args
        torch.manual_seed(a.seed)
        self.env = self.env_fn()
        self.agents = self.spec.agents
        self.range = None   # per-seat return range; train() measures it (it costs budget)
        d = a.gems_latent_dim

        self.gen, self.critic, self.gen_opt, self.critic_opt = {}, {}, {}, {}
        self.anchors, self.sigma, self.v_prev, self.v_ema = {}, {}, {}, {}
        for p in self.agents:
            obs_shape = self.spec.obs_shape[p]
            gen = make_generator(a.gems_generator, obs_shape, self.spec.num_actions[p], d,
                                 tuple(a.gems_hyper_hidden), a.gems_policy_hidden)
            critic = Critic(obs_shape, d, tuple(a.gems_hyper_hidden))
            self.gen[p], self.critic[p] = gen, critic
            self.gen_opt[p] = torch.optim.Adam(gen.parameters(), lr=a.gems_abr_lr)
            self.critic_opt[p] = torch.optim.Adam(critic.parameters(), lr=a.gems_critic_lr)
            k0 = a.gems_init_anchors
            self.anchors[p] = self.rng.standard_normal((k0, d)).astype(np.float32)
            self.sigma[p] = np.full(k0, 1.0 / k0)
            self.v_prev[p] = np.zeros(k0)
            self.v_ema[p] = np.full(k0, np.nan)
        self.iteration = 0
        self._phase = self._phase_seconds = None   # progress logging only
        self._members = {}
        self._refresh()
        self._commit()

    # ------------------------------------------------------------------ loop
    def train(self):
        a = self.args
        if a.gems_progress_interval > 0 and self.logger is not None:
            self.budget.watch(a.gems_progress_interval, self._heartbeat)
        self._phase_begin('return_range')
        self.range = return_range(self.env, self.rng, budget=self.budget)
        self._phase_end()
        cap = a.gems_iterations
        while not self.budget.exhausted and (cap is None or self.iteration < cap):
            self.iteration += 1
            record = {'iteration': self.iteration}
            self._phase_seconds = record['phase_seconds'] = {}

            # Phase 1: every seat against the others' sigma_{t-1}.
            values = {}
            for p in self.agents:
                self._phase_begin('estimate', p)
                values[p] = self._estimate(p)
                self._phase_end()
            # The mixture's normalised value under sigma_{t-1} -- logged now, since
            # sigma changes length once the new anchor is added.
            record['mixture_value'] = {p: float(self.sigma[p] @ values[p]) for p in self.agents}
            # Phase 2.
            for p in self.agents:
                self._omwu(p, values[p])
            self._refresh()
            # Phase 3: candidates scored against the updated sigma.
            chosen = {}
            for p in self.agents:
                self._phase_begin('oracle', p)
                chosen[p] = self._oracle(p)
                self._phase_end()
            for p, (z, info) in chosen.items():
                self._add_anchor(p, z)
                record.setdefault('oracle', {})[p] = info
            self._refresh()
            # Phase 4: one seat at a time, against the others' current generators.
            for p in self.agents:
                self._phase_begin('abr', p)
                record.setdefault('abr', {})[p] = self._abr_tr(p, chosen[p][0])
                self._phase_end()
                self._refresh()

            record.update(budget_used=self.budget.used,
                          values={p: values[p] for p in self.agents},
                          meta_strategy={p: self.sigma[p] for p in self.agents},
                          anchors={p: len(self.anchors[p]) for p in self.agents})
            self.logger.train('gems_iteration', record)
            self._commit()
            self._progress(f'iteration {self.iteration} done at {self.budget.used} steps; '
                           f'phase seconds {record["phase_seconds"]}')
        self.env.close()

    def policies(self):
        return self._output

    def variant(self):
        return {'generator': self.args.gems_generator}

    # ------------------------------------------------------------- progress
    # Logging only: timestamped lines in stdout (the Slurm log) saying which
    # phase the run is in, how many episodes it played, the longest one, and
    # how long each phase and the ABR updates took. Nothing here changes the run.
    def _progress(self, message):
        if self.logger is not None and self.args.gems_progress_interval > 0:
            self.logger.progress(f'[gems] {message}')

    def _phase_begin(self, name, seat=None):
        self._phase = {'name': name if seat is None else f'{name}/{seat}', 'start': time.time(),
                       'steps0': self.budget.used, 'episodes': 0, 'longest': 0, 'update_s': 0.0}

    def _phase_end(self):
        ph = self._phase
        seconds = time.time() - ph['start']
        if ph['name'] == 'return_range':
            self._progress(f'return range {self.range}: {self.budget.used - ph["steps0"]} steps, '
                           f'{seconds:.0f}s')
            return
        if self._phase_seconds is not None:
            self._phase_seconds[ph['name']] = round(seconds, 1)
        self._progress(f'it {self.iteration} {ph["name"]}: {ph["episodes"]} episodes, '
                       f'{self.budget.used - ph["steps0"]} steps, longest {ph["longest"]}, '
                       f'{seconds:.0f}s' + (f' (updates {ph["update_s"]:.0f}s)' if ph['update_s'] else ''))

    def _heartbeat(self, mark):
        ph = self._phase or {'name': '?', 'start': time.time(), 'episodes': 0,
                                               'longest': 0}
        self._progress(f'{mark} steps: it {self.iteration} {ph["name"]}, '
                       f'{ph["episodes"]} episodes so far, longest {ph["longest"]}, '
                       f'{time.time() - ph["start"]:.0f}s in phase')

    def _play(self, joint, record=False):
        ep = play_episode(self.env, joint, self.rng, budget=self.budget, record=record)
        ph = self._phase
        if ph is not None:
            ph['episodes'] += 1
            ph['longest'] = max(ph['longest'], ep.length)
        return ep

    # ------------------------------------------------------------- helpers
    def _refresh(self):
        """Rebuild the anchor policies after the generator or the anchor set changed."""
        tau = self.args.gems_temperature
        self._members = {p: [GeneratorPolicy(self.gen[p], z, tau) for z in self.anchors[p]]
                         for p in self.agents}

    def _commit(self):
        """Freeze the current output: what policies() returns until the next iteration ends."""
        tau = self.args.gems_temperature
        self._output = {}
        for p in self.agents:
            frozen = copy.deepcopy(self.gen[p]).eval()
            self._output[p] = MixturePolicy([GeneratorPolicy(frozen, z, tau) for z in self.anchors[p]],
                                            self.sigma[p].copy())

    def _norm(self, p, value):
        low, high = self.range[p]
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    def _opponents(self, p):
        """One member per other seat, sampled from its meta-strategy."""
        return {q: self._members[q][self.rng.choice(len(self.sigma[q]), p=self.sigma[q])]
                for q in self.agents if q != p}

    def _returns_vs_sigma(self, p, policy, opponents, rollouts):
        out = []
        for _ in range(opponents):
            joint = self._opponents(p)
            joint[p] = policy
            for _ in range(rollouts):
                ep = self._play(joint)
                out.append(self._norm(p, ep.returns[p]))
        return np.asarray(out)

    # Phase 1 --------------------------------------------------------------
    def _estimate(self, p):
        a = self.args
        v = np.array([self._returns_vs_sigma(p, member, a.gems_mc_opponents,
                                             a.gems_mc_rollouts).mean()
                      for member in self._members[p]])
        if a.gems_ema > 0:
            prev = self.v_ema[p]
            v = np.where(np.isnan(prev), v, (1 - a.gems_ema) * prev + a.gems_ema * v)
            self.v_ema[p] = v
        return v

    # Phase 2 --------------------------------------------------------------
    def _eta(self):
        a = self.args
        t = self.iteration
        if a.gems_eta_schedule == 'sqrt':
            return a.gems_eta / math.sqrt(t)
        if a.gems_eta_schedule == 'harmonic':
            return a.gems_eta / (1 + a.gems_eta_alpha * t)
        return a.gems_eta

    def _omwu(self, p, v):
        self.sigma[p] = omwu_step(self.sigma[p], v, self.v_prev[p], self._eta(),
                                  self.args.gems_logit_cap)
        self.v_prev[p] = v

    # Phase 3 --------------------------------------------------------------
    def _oracle(self, p):
        a = self.args
        d = a.gems_latent_dim
        k = len(self.anchors[p])
        parents = self.anchors[p][self.rng.choice(k, size=a.gems_mutation_pool, p=self.sigma[p])]
        mutated = parents + a.gems_mutation_std * self.rng.standard_normal(parents.shape)
        fresh = self.rng.standard_normal((a.gems_random_pool, d))
        candidates = np.concatenate([mutated, fresh]).astype(np.float32)

        delta = a.gems_ucb_delta / self.iteration ** 2
        scores, means = [], []
        for z in candidates:
            policy = GeneratorPolicy(self.gen[p], z, a.gems_temperature)
            y = self._returns_vs_sigma(p, policy, a.gems_oracle_opponents, a.gems_oracle_rollouts)
            score = eb_ucb(y, delta)
            if a.gems_jacobian_coef > 0:
                with torch.enable_grad():
                    jac = jacobian_penalty(self.gen[p], torch.as_tensor(z).reshape(1, -1))
                score -= a.gems_jacobian_coef * float(jac)
            scores.append(score)
            means.append(y.mean())
        best = int(np.argmax(scores))
        return candidates[best], {'ucb': scores[best], 'mean': means[best],
                                  'from': 'mutation' if best < len(mutated) else 'random'}

    def _add_anchor(self, p, z):
        a = self.args
        if len(self.anchors[p]) >= a.gems_max_anchors:
            drop = int(np.argmin(self.sigma[p] if a.gems_replacement == 'least_mass'
                                 else self.v_prev[p]))
            keep = np.arange(len(self.anchors[p])) != drop
            self.anchors[p] = self.anchors[p][keep]
            self.v_prev[p], self.v_ema[p] = self.v_prev[p][keep], self.v_ema[p][keep]
            self.sigma[p] = self.sigma[p][keep] / self.sigma[p][keep].sum()
        k = len(self.anchors[p])
        self.anchors[p] = np.concatenate([self.anchors[p], z[None].astype(np.float32)])
        self.sigma[p] = np.append(self.sigma[p] * k / (k + 1), 1.0 / (k + 1))
        self.v_prev[p] = np.append(self.v_prev[p], 0.0)
        self.v_ema[p] = np.append(self.v_ema[p], np.nan)

    # Phase 4 --------------------------------------------------------------
    def _abr_tr(self, p, z_new):
        a = self.args
        gen, critic = self.gen[p], self.critic[p]
        old = copy.deepcopy(gen).eval()
        scale = self.range[p][1] - self.range[p][0]
        stats = {}
        for _ in range(a.gems_abr_steps):
            # Training anchors: the new code with probability q_new, else any anchor.
            codes = [z_new if self.rng.random() < a.gems_new_fraction
                     else self.anchors[p][self.rng.integers(len(self.anchors[p]))]
                     for _ in range(a.gems_abr_batch_anchors)]
            batch = []
            for z in codes:
                joint = self._opponents(p)
                joint[p] = GeneratorPolicy(gen, z, a.gems_temperature)
                ep = self._play(joint, record=True)
                if ep.trajectories[p]:
                    batch.append((z, ep.trajectories[p]))
            if batch:
                started = time.time()
                stats = self._abr_update(p, gen, critic, old, batch, scale)
                if self._phase is not None:
                    self._phase['update_s'] += time.time() - started
        return stats

    def _abr_update(self, p, gen, critic, old, batch, scale):
        a = self.args
        obs, zs, masks, actions, advantages, returns = [], [], [], [], [], []
        for z, traj in batch:
            # Observations keep their dtype and shape (uint8 frames go to the CNN as is).
            o = torch.as_tensor(np.stack([np.asarray(t.obs) for t in traj]))
            zz = torch.as_tensor(np.repeat(z[None], len(traj), 0), dtype=torch.float32)
            rewards = np.array([t.reward for t in traj]) / scale
            with torch.no_grad():
                values = critic(o, zz).numpy()
                last = traj[-1]
                bootstrap = 0.0
                if last.truncated and last.next_obs is not None:
                    nxt = torch.as_tensor(np.asarray(last.next_obs))[None]
                    bootstrap = float(critic(nxt, zz[:1]))
            next_values = np.append(values[1:], bootstrap)
            deltas = rewards + a.gems_gamma * next_values - values
            adv, running = np.zeros_like(deltas), 0.0
            for t in reversed(range(len(traj))):
                running = deltas[t] + a.gems_gamma * a.gems_gae_lambda * running
                adv[t] = running
            obs.append(o)
            zs.append(zz)
            masks.append(torch.as_tensor(np.stack([t.legal_mask for t in traj]), dtype=torch.bool))
            actions.append(torch.as_tensor([t.action for t in traj]))
            advantages.append(adv)
            returns.append(adv + values)

        obs, zs, masks, actions = (torch.cat(x) for x in (obs, zs, masks, actions))
        adv_all = torch.as_tensor(np.concatenate(advantages), dtype=torch.float32)
        ret_all = torch.as_tensor(np.concatenate(returns), dtype=torch.float32)
        if len(adv_all) > 1:
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        tau = a.gems_temperature
        with torch.no_grad():
            # Behaviour policy: the generator that collected the batch (PPO ratio),
            # and the generator at the start of this ABR-TR phase (trust region).
            behaviour = F.log_softmax(masked_logits(gen(obs, zs), masks, tau), -1)
            behaviour = behaviour.gather(1, actions[:, None]).squeeze(1)
            anchor_logp = F.log_softmax(masked_logits(old(obs, zs), masks, tau), -1)

        n = len(adv_all)
        size = max(1, -(-n // max(1, a.gems_abr_minibatches)))
        clip = a.gems_abr_clip
        totals = {'pg_loss': 0.0, 'kl': 0.0, 'v_loss': 0.0, 'clip_frac': 0.0}
        updates = 0
        for _ in range(max(1, a.gems_abr_epochs)):
            order = torch.as_tensor(self.rng.permutation(n))
            for start in range(0, n, size):
                idx = order[start:start + size]
                o, z, m, act = obs[idx], zs[idx], masks[idx], actions[idx]
                adv, ret = adv_all[idx], ret_all[idx]
                logp = F.log_softmax(masked_logits(gen(o, z), m, tau), -1)
                probs = logp.exp()
                kl = (probs * torch.where(m, logp - anchor_logp[idx],
                                          torch.zeros_like(logp))).sum(-1).mean()
                ratio = (logp.gather(1, act[:, None]).squeeze(1) - behaviour[idx]).exp()
                if clip > 0:
                    pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                    clip_frac = ((ratio - 1).abs() > clip).float().mean().item()
                else:
                    pg = -(ratio * adv).mean()
                    clip_frac = 0.0
                loss = pg + a.gems_kl_coef * kl
                if a.gems_jacobian_coef > 0:
                    unique = torch.as_tensor(np.unique(z.numpy(), axis=0))
                    loss = loss + a.gems_jacobian_coef * jacobian_penalty(gen, unique)

                self.gen_opt[p].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), a.gems_grad_clip)
                self.gen_opt[p].step()

                v_loss = (critic(o, z) - ret).pow(2).mean()
                self.critic_opt[p].zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), a.gems_grad_clip)
                self.critic_opt[p].step()

                for key, value in (('pg_loss', pg.item()), ('kl', kl.item()),
                                   ('v_loss', v_loss.item()), ('clip_frac', clip_frac)):
                    totals[key] += value
                updates += 1
        stats = {k: v / updates for k, v in totals.items()}
        stats.update(transitions=n, updates=updates)
        return stats
