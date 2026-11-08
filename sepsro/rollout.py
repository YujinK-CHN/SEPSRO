"""
Playing episodes with one Policy per seat.

This is the only loop in the codebase that drives an environment. Training
(PPO collection, PSRO best responses), payoff estimation and evaluation all go
through play_episode(), so budget accounting and per-agent credit assignment are
defined once.

Credit assignment (record=True). In a turn-based game an agent's action is
usually rewarded later, on someone else's turn. Each agent's Transition
therefore accumulates every reward that agent receives from the moment it acts
until the moment it next acts (or the episode ends). next_obs is the agent's
own observation at its next decision, which is the right bootstrap target for
an agent-centric value function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .envs.base import MultiAgentEnv


@dataclass
class Transition:
    obs: np.ndarray
    legal_mask: np.ndarray
    action: int
    reward: float = 0.0
    next_obs: np.ndarray | None = None
    next_legal_mask: np.ndarray | None = None
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)   # per-step extras, e.g. log-prob, value


@dataclass
class Episode:
    returns: dict                 # agent -> total reward
    length: int                   # env steps
    terminated: bool
    truncated: bool
    trajectories: dict | None = None   # agent -> list[Transition], when record=True


def play_episode(env: MultiAgentEnv, policies: dict, rng: np.random.Generator,
                 seed=None, budget=None, record=False, on_action=None) -> Episode:
    """Play one episode.

    policies   agent -> Policy, one for every agent in env.spec.agents
    budget     BudgetTracker to charge one unit per env step, or None (evaluation)
    record     keep per-agent Transitions
    on_action  optional hook(agent, obs, legal_mask, action) -> dict, whose result
               is stored in Transition.info (e.g. log-prob and value for PPO)
    """
    agents = env.spec.agents
    missing = set(agents) - set(policies)
    if missing:
        raise ValueError(f'No policy for agents {sorted(missing)}.')

    for agent in agents:
        policies[agent].reset(rng)

    # Seeding every reset from the caller's RNG makes chance outcomes (card
    # deals, PettingZoo's own randomness) reproducible from the run seed.
    if seed is None:
        seed = int(rng.integers(2**31 - 1))
    timestep = env.reset(seed=seed)
    returns = {a: 0.0 for a in agents}
    trajectories = {a: [] for a in agents} if record else None
    pending = {}   # agent -> its open Transition, waiting for its next decision
    length = 0

    while not timestep.done:
        actions = {}
        for agent in timestep.acting:
            obs = timestep.observations[agent]
            mask = timestep.legal_masks[agent]
            action = policies[agent].act(obs, mask, rng)
            actions[agent] = action
            if record:
                if agent in pending:
                    prev = pending.pop(agent)
                    prev.next_obs, prev.next_legal_mask = obs, mask
                    trajectories[agent].append(prev)
                info = on_action(agent, obs, mask, action) if on_action else {}
                pending[agent] = Transition(obs=obs, legal_mask=mask, action=action,
                                            info=info or {})

        timestep = env.step(actions)
        length += 1
        if budget is not None:
            budget.consume(1)

        for agent, r in timestep.rewards.items():
            returns[agent] += r
            if record and agent in pending:
                pending[agent].reward += r

    if record:
        for agent, transition in pending.items():
            transition.next_obs = timestep.observations.get(agent)
            transition.terminated = timestep.terminated
            transition.truncated = timestep.truncated
            trajectories[agent].append(transition)

    return Episode(returns=returns, length=length, terminated=timestep.terminated,
                   truncated=timestep.truncated, trajectories=trajectories)


def average_returns(env, policies, rng, episodes, budget=None):
    """Mean return per agent over `episodes` episodes."""
    totals = {a: 0.0 for a in env.spec.agents}
    for _ in range(episodes):
        result = play_episode(env, policies, rng, budget=budget)
        for agent, value in result.returns.items():
            totals[agent] += value
    return {a: v / episodes for a, v in totals.items()}
