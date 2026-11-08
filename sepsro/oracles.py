"""
Best-response oracles, after OpenSpiel's psro_v2.

One call trains a new policy for EVERY seat against the other seats' current
meta-strategy (psro_v2's update_agents -> oracle(training_parameters)) and
returns them frozen. The PSRO loop never knows how a response was computed.

    RLOracle            rl_oracle.RLOracle, with a PG or DQN learner per seat
    BestResponseOracle  best_response_oracle.BestResponseOracle: exact, by
                        walking the game tree (OpenSpiel turn-based games only)

RLOracle schedule (as in rl_oracle.py):
  * each episode trains ONE seat; the seat is drawn with probability
    proportional to 1 / (episodes it has had + 1), so seats stay balanced;
  * the other seats play their meta-strategy mixtures, a member sampled per
    episode (sample_strategy_marginal); with `self_play_proportion`, a seat may
    instead play its own currently-training policy, which then also learns;
  * the call ends once every new policy has MORE than
    `number_training_episodes` episodes -- or when the budget runs out.

Budget: RLOracle charges every env step it plays. BestResponseOracle plays no
episodes, so it charges nothing; payoff-table simulation is then the only cost.
"""

from __future__ import annotations

import numpy as np

from .policies import Policy, RandomPolicy
from .rollout import play_episode


def random_count_weighted_choice(counts, rng):
    """Index i with probability proportional to 1 / (counts[i] + 1)."""
    p = 1.0 / (np.asarray(counts, dtype=float) + 1.0)
    return int(rng.choice(len(p), p=p / p.sum()))


def make_learner_factory(kind, spec, args):
    """seat, seed -> fresh learner, for --psro-oracle pg | dqn."""
    if kind == 'pg':
        from .learners.pg import PGConfig, PGLearner
        config = PGConfig.from_args(args)
        return lambda seat, seed: PGLearner(spec.obs_shape[seat], spec.num_actions[seat],
                                            config, seed=seed)
    if kind == 'dqn':
        from .learners.dqn import DQNConfig, DQNLearner
        config = DQNConfig.from_args(args)
        return lambda seat, seed: DQNLearner(spec.obs_shape[seat], spec.num_actions[seat],
                                             config, seed=seed)
    raise ValueError(f'No learner for oracle {kind!r}.')


class RLOracle:

    def __init__(self, spec, learner_factory, number_training_episodes,
                 self_play_proportion=0.0, sigma=0.0, seed=0):
        self.agents = spec.agents
        self.learner_factory = learner_factory
        self.number_training_episodes = int(number_training_episodes)
        self.self_play_proportion = float(self_play_proportion)
        self.sigma = float(sigma)
        self.seed = seed
        self.calls = 0

    def __call__(self, env, mixtures, rng, budget, start_from=None, on_episode=None):
        """Train one new policy per seat.

        mixtures     agent -> MixturePolicy: each seat's meta-strategy over its population
        start_from   agent -> Policy or None: copy that policy's weights (+ sigma noise)
                     instead of starting from scratch, when it is a learner snapshot
        on_episode   optional callback(seat, episode_index, learner, episode)

        Returns agent -> (frozen Policy or None, episodes trained).
        """
        self.calls += 1
        learners, episodes = {}, {}
        for i, seat in enumerate(self.agents):
            learner = self.learner_factory(seat, seed=self.seed * 1_000_003 + self.calls * 101 + i)
            source = (start_from or {}).get(seat)
            # As in rl_oracle.generate_new_policies: copy only a policy of the
            # learner's own kind; anything else (e.g. the random seed) -> scratch.
            if source is not None and isinstance(source, learner.policy_cls):
                learner.load_from(source, sigma=self.sigma)
            learners[seat], episodes[seat] = learner, 0

        while not self._done(episodes) and not budget.exhausted:
            seat = self.agents[random_count_weighted_choice(
                [episodes[a] for a in self.agents], rng)]
            joint, live = {}, [seat]
            for agent in self.agents:
                if agent == seat:
                    joint[agent] = learners[agent].acting_policy()
                elif self.self_play_proportion and rng.binomial(1, self.self_play_proportion):
                    joint[agent] = learners[agent].acting_policy()
                    live.append(agent)
                else:
                    joint[agent] = mixtures[agent]
            episode = play_episode(env, joint, rng, budget=budget, record=True)
            for agent in live:
                learners[agent].end_episode(episode.trajectories[agent])
                episodes[agent] += 1
                if on_episode:
                    on_episode(agent, episodes[agent], learners[agent], episode)

        return {seat: (learners[seat].policy() if episodes[seat] else None, episodes[seat])
                for seat in self.agents}

    def _done(self, episodes):
        return all(n > self.number_training_episodes for n in episodes.values())


class TabularObsPolicy(Policy):
    """A policy stored as observation bytes -> action probabilities.

    OpenSpiel's exact best response is keyed by information-state strings; the
    rest of this codebase sees observation tensors. With info-state
    observations (perfect recall) the two are in one-to-one correspondence, so
    the best response can be stored against the tensor its user will see.
    """

    def __init__(self, table):
        self.table = table

    def action_probs(self, obs, legal_mask):
        probs = self.table.get(np.asarray(obs, np.float32).tobytes())
        if probs is None:   # never reached in the tree walk; cannot happen with info states
            return RandomPolicy().action_probs(obs, legal_mask)
        return probs

    def act(self, obs, legal_mask, rng):
        probs = self.action_probs(obs, legal_mask)
        return int(rng.choice(len(probs), p=probs))


class BestResponseOracle:
    """Exact best response to the aggregated opponent meta-strategy (OpenSpiel only)."""

    def __init__(self, spec):
        if spec.backend != 'openspiel' or not spec.turn_based:
            raise ValueError("--psro-oracle br needs an OpenSpiel turn-based game.")
        self.agents = spec.agents

    def __call__(self, env, mixtures, rng, budget, start_from=None, on_episode=None):
        from open_spiel.python.algorithms import best_response

        from .exploitability import aggregate

        out = {}
        for p, seat in enumerate(self.agents):
            joint = {a: (RandomPolicy() if a == seat else mixtures[a]) for a in self.agents}
            responder = best_response.BestResponsePolicy(env.game, p, aggregate(env, joint))
            out[seat] = (TabularObsPolicy(self._table(env, p, responder)), 0)
        return out

    @staticmethod
    def _table(env, player, responder):
        table = {}

        def visit(state):
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    visit(state.child(action))
                return
            if state.current_player() == player:
                key = env.observation_for(state, player).tobytes()
                if key not in table:
                    probs = np.zeros(env.game.num_distinct_actions())
                    for action, prob in responder.action_probabilities(state).items():
                        probs[action] = prob
                    table[key] = probs
            for action in state.legal_actions():
                visit(state.child(action))

        visit(env.game.new_initial_state())
        return table
