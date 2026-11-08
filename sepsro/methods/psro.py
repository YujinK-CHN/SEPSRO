"""
PSRO -- Policy-Space Response Oracles (Lanctot et al., 2017).

Follows OpenSpiel's psro_v2 (PSROSolver + AbstractMetaTrainer + its oracles)
in structure and defaults; it is a re-implementation on this codebase's env
interface, not a wrapper, so it runs on PettingZoo as well as OpenSpiel.

Setup (PSROSolver.__init__):
    populations      one per seat, seeded with the uniform random policy
    empirical game   every entry simulated --psro-sims-per-entry times
    meta-strategy    META_SOLVERS[--psro-meta-solver] on the payoff tensor

One iteration (AbstractMetaTrainer.iteration):
    1. update_agents               the oracle trains one new policy per seat
                                   against the others' meta-strategy mixtures
                                   (marginals sampled per episode);
    2. update_empirical_gamestate  simulate only the new payoff entries;
    3. update_meta_strategies      re-solve.

Oracles (--psro-oracle): 'pg' and 'dqn' are psro_v2's RL oracle with its two
learners; 'br' is its exact best response (OpenSpiel turn-based games only).

Training-strategy selector: in psro_v2 it picks which existing policy each
new response is trained FROM. A response is a copy of it only when the learner
supports copy_with_noise; OpenSpiel's current JAX PG / DQN do not, so every
response starts from scratch. That is the default here (--psro-br-init
scratch). --psro-br-init copy restores the copy-plus-noise behaviour of the
older TensorFlow learners (noise scale --psro-sigma).

Stops after --psro-iterations iterations (psro_v2_example's gpsro_iterations)
or when the env-step budget runs out, whichever is first. An iteration that has
started is completed, so the budget can be overshot by one payoff-table update;
the final eval record logs the true budget_used.
"""

from __future__ import annotations

import numpy as np

from ..meta_game import PayoffTable, Population
from ..meta_solvers import META_SOLVERS
from ..oracles import BestResponseOracle, RLOracle, make_learner_factory
from ..policies import MixturePolicy, RandomPolicy
from ..registry import METHODS
from .base import Method

# Log the full payoff tensor while it is small enough to be readable.
_LOG_PAYOFFS_MAX_ENTRIES = 400


@METHODS.register('psro')
class PSRO(Method):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        a = self.args
        self.env = self.env_fn()
        self.agents = self.spec.agents
        self.populations = {agent: Population() for agent in self.agents}
        for pop in self.populations.values():
            pop.add(RandomPolicy(), kind='random', iteration=0)
        self.table = PayoffTable(self.env, self.populations, a.psro_sims_per_entry)
        self.solver = META_SOLVERS.get(a.psro_meta_solver)

        if a.psro_oracle == 'br':
            if a.obs_type != 'info_state':
                raise ValueError("--psro-oracle br needs --obs-type info_state.")
            self.oracle = BestResponseOracle(self.spec)
        else:
            self.oracle = RLOracle(self.spec, make_learner_factory(a.psro_oracle, self.spec, a),
                                   number_training_episodes=a.psro_number_training_episodes,
                                   self_play_proportion=a.psro_self_play_proportion,
                                   sigma=a.psro_sigma, seed=a.seed)
        self.meta = None
        self.iteration = 0

    # -------------------------------------------------------------- loop
    def train(self):
        self._update_meta_game()
        while not self.budget.exhausted and self.iteration < self.args.psro_iterations:
            self.evaluator.maybe_evaluate(self)
            self.iteration += 1

            mixtures = {agent: MixturePolicy(self.populations[agent].policies, self.meta[agent])
                        for agent in self.agents}
            responses = self.oracle(self.env, mixtures, self.rng, self.budget,
                                    start_from=self._select_start_policies(),
                                    on_episode=self._log_episode)
            added = 0
            for seat, (policy, episodes) in responses.items():
                if policy is not None:
                    self.populations[seat].add(policy, kind=self.args.psro_oracle,
                                               iteration=self.iteration, episodes=episodes,
                                               budget_used=self.budget.used)
                    added += 1
            if not added:
                break   # the budget ran out before any response was trained
            self._update_meta_game()
        self.env.close()

    def policies(self):
        return {agent: MixturePolicy(self.populations[agent].policies, self.meta[agent])
                for agent in self.agents}

    def variant(self):
        a = self.args
        out = {'oracle': a.psro_oracle, 'meta_solver': a.psro_meta_solver}
        if a.psro_oracle == 'pg':
            out['pg_loss'] = a.pg_loss
        return out

    # ----------------------------------------------------------- helpers
    def _update_meta_game(self):
        new_entries = self.table.update(self.rng, budget=self.budget)
        payoffs = self.table.tensor()
        a = self.args
        distributions = self.solver(payoffs, iterations=a.psro_prd_iterations,
                                    dt=a.psro_prd_dt, gamma=a.psro_prd_gamma)
        self.meta = dict(zip(self.agents, distributions))

        record = {'iteration': self.iteration, 'budget_used': self.budget.used,
                  'new_entries': new_entries,
                  'population_sizes': {agent: len(pop) for agent, pop in self.populations.items()},
                  'meta_strategy': dict(self.meta)}
        if int(np.prod(payoffs.shape[:-1])) <= _LOG_PAYOFFS_MAX_ENTRIES:
            record['payoffs'] = payoffs
        self.logger.train('psro_iteration', record)

    def _select_start_policies(self):
        """psro_v2's training_strategy_selector, with number_policies_selected = 1."""
        if self.args.psro_br_init != 'copy':
            return None
        selector = self.args.psro_training_selector
        chosen = {}
        for agent in self.agents:
            weights = self.meta[agent]
            n = len(weights)
            if selector == 'probabilistic':
                index = self.rng.choice(n, p=weights)
            elif selector == 'top_k_probabilities':
                index = int(np.argmax(weights))
            else:   # uniform
                index = self.rng.integers(n)
            chosen[agent] = self.populations[agent][int(index)]
        return chosen

    def _log_episode(self, seat, episode_index, learner, episode):
        every = self.args.psro_log_every_episodes
        if every and episode_index % every == 0:
            losses = getattr(learner, 'last_losses', None) or {'q': getattr(learner, 'last_loss', None)}
            self.logger.train('oracle_episode', {'iteration': self.iteration, 'seat': seat,
                                                 'episode': episode_index,
                                                 'budget_used': self.budget.used,
                                                 'return': episode.returns[seat], **losses})
