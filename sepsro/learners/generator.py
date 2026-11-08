"""
GEMS's amortized generator: a latent code z -> a policy.

The paper (Sharma et al., TMLR 2026, Sec. 3.1) defines G_theta as a network
mapping z to policy PARAMETERS, phi = G_theta(z), and calls it a hypernetwork
(Sec. 2); beyond "standard MLP architectures" (Sec. 6) it does not specify the
architecture. Two forms are provided:

    hyper        (default, as the paper describes) an MLP maps z to the full
                 weight vector of a small policy MLP  obs -> hidden -> actions.
    conditional  one MLP over [obs, z] -> action logits (NeuPL-style). Cheaper
                 when observations are large; the induced policy family is the
                 same in spirit but parameters are shared, not generated.

Observations first pass through an encoder (networks.make_encoder): flattened
for vectors, a shared Nature-DQN CNN for image stacks such as Atari frames. With
images, the hypernetwork therefore generates the policy HEAD on top of shared
CNN features -- generating a whole CNN per code is not practical, and the paper
does not say how it handled pixels.

Logits are divided by a temperature and illegal actions are masked out.

Also here: the critic V(obs, z) used for GAE advantages in ABR-TR training (the
paper trains with "PPO with GAE", App. Part III, but does not describe the
critic), and GeneratorPolicy, the Policy that one anchor induces.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..policies import Policy
from .networks import MASK_VALUE, make_encoder


def _mlp(in_dim, hidden, out_dim, activation=nn.ReLU):
    layers, last = [], in_dim
    for width in hidden:
        layers += [nn.Linear(last, width), activation()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class HyperGenerator(nn.Module):
    """z -> weights of a one-hidden-layer policy MLP (tanh) over encoded obs -> logits."""

    def __init__(self, obs_shape, num_actions, latent_dim, hyper_hidden=(64, 64),
                 policy_hidden=64):
        super().__init__()
        self.encoder = make_encoder(obs_shape)
        obs_dim = self.encoder.out_dim
        self.obs_dim, self.num_actions, self.h = obs_dim, num_actions, policy_hidden
        self.shapes = [(policy_hidden, obs_dim), (policy_hidden,),
                       (num_actions, policy_hidden), (num_actions,)]
        self.num_params = sum(int(np.prod(s)) for s in self.shapes)
        self.hyper = _mlp(latent_dim, hyper_hidden, self.num_params)
        # Generated weights start at the usual per-layer scale (std 1/sqrt(fan_in))
        # with a small output layer, so every z starts near the uniform policy;
        # the small z-dependent part (weight * 0.1) gives anchors their spread.
        with torch.no_grad():
            self.hyper[-1].weight.mul_(0.1)
            bias, i = self.hyper[-1].bias, 0
            for shape, std in zip(self.shapes, (1 / np.sqrt(obs_dim), 0.0,
                                                0.01 / np.sqrt(policy_hidden), 0.0)):
                n = int(np.prod(shape))
                bias[i:i + n].normal_(0.0, std) if std else bias[i:i + n].zero_()
                i += n

    def params(self, z):
        """z: [B, d] -> flat policy parameters [B, P]."""
        return self.hyper(z)

    def logits_from_params(self, params, obs):
        """params [B, P] (or [1, P], shared), obs [B, *obs_shape] -> logits [B, A]."""
        obs = self.encoder(obs)
        if params.shape[0] == 1 and obs.shape[0] > 1:
            params = params.expand(obs.shape[0], -1)
        chunks, i = [], 0
        for shape in self.shapes:
            n = int(np.prod(shape))
            chunks.append(params[:, i:i + n].reshape(-1, *shape))
            i += n
        w1, b1, w2, b2 = chunks
        hidden = torch.tanh(torch.einsum('bho,bo->bh', w1, obs) + b1)
        return torch.einsum('bah,bh->ba', w2, hidden) + b2

    def forward(self, obs, z):
        return self.logits_from_params(self.params(z), obs)


class ConditionalGenerator(nn.Module):
    """One MLP over [encoded obs, z] -> logits."""

    def __init__(self, obs_shape, num_actions, latent_dim, hidden=(64, 64)):
        super().__init__()
        self.encoder = make_encoder(obs_shape)
        self.net = _mlp(self.encoder.out_dim + latent_dim, hidden, num_actions, nn.Tanh)

    def params(self, z):
        return z   # the "parameters" of an induced policy are its code

    def logits_from_params(self, params, obs):
        obs = self.encoder(obs)
        if params.shape[0] == 1 and obs.shape[0] > 1:
            params = params.expand(obs.shape[0], -1)
        return self.net(torch.cat([obs, params], dim=-1))

    def forward(self, obs, z):
        return self.logits_from_params(z, obs)


class Critic(nn.Module):
    """V(obs, z) for GAE, with its own encoder."""

    def __init__(self, obs_shape, latent_dim, hidden=(64, 64)):
        super().__init__()
        self.encoder = make_encoder(obs_shape)
        self.net = _mlp(self.encoder.out_dim + latent_dim, hidden, 1, nn.Tanh)

    def forward(self, obs, z):
        return self.net(torch.cat([self.encoder(obs), z], dim=-1)).squeeze(-1)


def make_generator(kind, obs_shape, num_actions, latent_dim, hyper_hidden, policy_hidden):
    if kind == 'hyper':
        return HyperGenerator(obs_shape, num_actions, latent_dim, hyper_hidden, policy_hidden)
    if kind == 'conditional':
        return ConditionalGenerator(obs_shape, num_actions, latent_dim, hyper_hidden)
    raise ValueError(f"generator must be 'hyper' or 'conditional', got {kind!r}")


def masked_logits(logits, mask, temperature):
    return (logits / temperature).masked_fill(~mask, MASK_VALUE)


def jacobian_penalty(generator, z):
    """Mean squared Frobenius norm of d G(z) / d z over a batch of codes z [B, d]."""
    def single(code):
        return generator.params(code.unsqueeze(0)).squeeze(0)
    total = 0.0
    for code in z:
        jac = torch.autograd.functional.jacobian(single, code, create_graph=True)
        total = total + jac.pow(2).sum()
    return total / len(z)


class GeneratorPolicy(Policy):
    """The policy one anchor z induces. Parameters are generated once and cached,
    so acting costs only the small policy MLP; build a new object after the
    generator is updated."""

    def __init__(self, generator, z, temperature=1.0):
        self.generator = generator
        self.z = torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)
        self.temperature = temperature
        with torch.no_grad():
            self._params = generator.params(self.z)

    @torch.no_grad()
    def action_probs(self, obs, legal_mask):
        obs_t = torch.as_tensor(np.asarray(obs))[None]
        mask_t = torch.as_tensor(np.asarray(legal_mask, bool)).reshape(1, -1)
        logits = self.generator.logits_from_params(self._params, obs_t)
        probs = torch.softmax(masked_logits(logits, mask_t, self.temperature), -1)[0]
        probs = probs.double().numpy() * legal_mask
        return probs / probs.sum()

    def act(self, obs, legal_mask, rng):
        probs = self.action_probs(obs, legal_mask)
        return int(rng.choice(len(probs), p=probs))
