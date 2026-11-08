"""
PPO -- Proximal Policy Optimization, for multi-agent self-play. Placeholder.

Things to settle before writing it: shared vs. per-seat parameters, IPPO vs.
MAPPO (centralised critic), and how turn-based credit is fed in (rollout.py
already accumulates each agent's reward between its own decisions, and its
on_action hook can store log-probs and values in Transition.info).
"""

from __future__ import annotations

from ..registry import METHODS
from .base import NotImplementedMethod


@METHODS.register('ppo')
class PPO(NotImplementedMethod):
    reference = 'Schulman et al., 2017, "Proximal Policy Optimization Algorithms"'
