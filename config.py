"""
Single configuration file for every method in SEPSRO.

Only a few things are meant to change from the command line, because only they
change between runs of an experiment suite:

    --method    which method       (see sepsro/methods; `--help` lists them)
    --env       which environment  (a key of ENV_PRESETS, or backend:id)
    --budget    total training budget, in ENV STEPS (see sepsro/budget.py)
    --seed      random seed

    python main.py --method random --env kuhn_poker --budget 100000 --seed 0

Everything else -- evaluation protocol, every method's hyper-parameters -- is
set by the defaults below, adjusted per environment by ENV_OVERRIDES. Edit them
here. They are still CLI flags so a one-off sweep can override one without
editing the file (a flag given explicitly always wins over ENV_OVERRIDES), but
run.sh / submit.sh do not pass them, so this file stays the single source of
truth.

Groups:
    run         the knobs above, plus where output goes
    env         environment options shared by all backends
    eval        evaluation protocol (identical for every method)
    psro        PSRO loop (names and defaults follow OpenSpiel's psro_v2)
    pg / dqn    the RL oracle's learners (psro_v2_example.py's settings)
    ppo         PPO learner, for the future self-play `ppo` method
    gems        GEMS (defaults from the paper's Kuhn-poker reference settings)
    exploit     post-hoc approximate NashConv (posthoc.py), with a PPO best response
"""

import argparse
import json
import os

from sepsro.meta_solvers import META_SOLVERS
from sepsro.registry import METHODS, METRICS
import sepsro.methods  # noqa: F401  (registers methods)
import sepsro.evaluation  # noqa: F401  (registers metrics)
import sepsro.exploitability  # noqa: F401  (registers nash_conv)


# --------------------------------------------------------------------------- #
# Environments.  Keyed by --env.
#
#   backend   'native' | 'openspiel' | 'pettingzoo'
#   id        matrix-game name | OpenSpiel game name | PettingZoo module path
#   api       PettingZoo only: 'aec' (turn-based) or 'parallel' (simultaneous)
#   kwargs    OpenSpiel game parameters / PettingZoo env kwargs
#
# Any environment not listed can still be used ad hoc as --env backend:id, e.g.
#   --env openspiel:liars_dice
#   --env pettingzoo:pettingzoo.classic.go_v5
#   --env pettingzoo:mpe2.simple_tag_v3@parallel
# Add it here once it is used for real, so its options are recorded.
# --------------------------------------------------------------------------- #
ENV_PRESETS = {
    # native -- no dependencies; for debugging methods
    'rps':              {'backend': 'native', 'id': 'rps'},
    'matching_pennies': {'backend': 'native', 'id': 'matching_pennies'},
    'prisoners_dilemma': {'backend': 'native', 'id': 'prisoners_dilemma'},

    # OpenSpiel -- imperfect-information benchmarks of the PSRO literature
    'kuhn_poker':       {'backend': 'openspiel', 'id': 'kuhn_poker'},
    'leduc_poker':      {'backend': 'openspiel', 'id': 'leduc_poker'},
    'goofspiel':        {'backend': 'openspiel', 'id': 'goofspiel',
                         'kwargs': {'num_cards': 5, 'points_order': 'descending'}},
    'liars_dice':       {'backend': 'openspiel', 'id': 'liars_dice'},
    'os_tic_tac_toe':   {'backend': 'openspiel', 'id': 'tic_tac_toe'},

    # PettingZoo
    'connect_four':     {'backend': 'pettingzoo', 'id': 'pettingzoo.classic.connect_four_v3'},
    'tic_tac_toe':      {'backend': 'pettingzoo', 'id': 'pettingzoo.classic.tictactoe_v3'},
    'pz_rps':           {'backend': 'pettingzoo', 'id': 'pettingzoo.classic.rps_v2'},

    # PettingZoo Atari (needs `pip install "pettingzoo[atari]" autorom` + `AutoROM`).
    # Two players, both scoring against the invaders (not zero-sum). Standard
    # Atari preprocessing: 4-frame skip with max-pooling, 84x84, 4-frame stack;
    # one env step = one decision = 4 emulator frames. Sticky actions 0.25
    # (Machado et al., 2018): PettingZoo turns ALE's own off, which leaves the
    # game deterministic.
    'space_invaders':   {'backend': 'pettingzoo', 'id': 'pettingzoo.atari.space_invaders_v2',
                         'api': 'parallel', 'kwargs': {'obs_type': 'grayscale_image'},
                         'wrappers': {'atari': {'frame_skip': 4, 'stack': 4, 'size': 84,
                                                'sticky_actions': 0.25}}},
}


def parse_env(value):
    """--env value -> preset dict. Accepts a preset key or 'backend:id[@api]'."""
    if value in ENV_PRESETS:
        return {'name': value, **ENV_PRESETS[value]}
    if ':' not in value:
        raise ValueError(f'Unknown env {value!r}. Use one of {sorted(ENV_PRESETS)} '
                         'or the form backend:id (e.g. openspiel:liars_dice).')
    backend, ident = value.split(':', 1)
    preset = {'backend': backend, 'id': ident}
    if '@' in ident:
        preset['id'], preset['api'] = ident.split('@', 1)
    preset['name'] = preset['id'].rsplit('.', 1)[-1]
    return preset


# --------------------------------------------------------------------------- #
# Per-environment settings.  Keyed like ENV_PRESETS.
#
# The defaults below are each method's published settings for small games
# (psro_v2_example.py for PSRO, the paper's Kuhn table for GEMS). Kuhn and Leduc
# use them unchanged (only the post-hoc check's episode count differs). Space
# Invaders episodes are ~150x longer, so a few settings that are counts of
# EPISODES (or that assume such counts) are scaled
# to fit a 30M-step budget; everything not listed keeps its default. Starting
# points for the pilot; the README lists the reasoning for each.
#
# Applied by parse_args() to every setting NOT given explicitly on the command
# line, and recorded in config.json under 'env_overrides'.
# --------------------------------------------------------------------------- #
ENV_OVERRIDES = {
    # Poker: the methods' published settings. Only the post-hoc approximate
    # NashConv (a check against the exact one there) needs more evaluation
    # episodes: NashConv is ~0.01-0.3 while returns vary by ~1 per episode.
    'kuhn_poker': {'exploit_eval_episodes': 20_000},
    'leduc_poker': {'exploit_eval_episodes': 20_000},
    'space_invaders': {
        # evaluation: 20 points x 100 episodes per metric (~1.5 h of a ~20 h run)
        'eval_episodes': 100,
        # PSRO: ~2.7M steps per best response pair, ~50 x (2n+1) episodes of
        # payoff simulation at iteration n -> ~9 iterations in 30M steps
        'psro_number_training_episodes': 3000,
        'psro_sims_per_entry': 50,
        # GEMS: the Kuhn values barely move on Atari (si_main: ABR KL ~1e-4 per
        # iteration, sigma stayed uniform). Same oracle episodes (1024 per seat)
        # spent on 16 candidates x 64 instead of 64 x 16, so the choice is not
        # noise; a step size suited to ~20 iterations; a faster ABR-TR.
        'gems_mutation_pool': 8,
        'gems_random_pool': 8,
        'gems_oracle_opponents': 16,
        'gems_oracle_rollouts': 4,
        'gems_eta': 0.5,
        'gems_abr_lr': 1e-3,
        # approximate NashConv: PPO best response per seat, Atari PPO clip
        'exploit_steps': 5_000_000,
        'ppo_clip': 0.1,
    },
}


def _add_run_config(parser):
    group = parser.add_argument_group('run')
    group.add_argument('--method', type=str, default='random', choices=METHODS.names())
    group.add_argument('--env', type=str, default='kuhn_poker',
                       help=f'One of {sorted(ENV_PRESETS)}, or backend:id.')
    group.add_argument('--budget', type=int, default=100_000,
                       help='Total training budget in ENV STEPS (calls to env.step). '
                            'The common currency that makes methods comparable.')
    group.add_argument('--seed', type=int, default=0)
    group.add_argument('--exp-name', type=str, default='check',
                       help='Groups runs under results/<exp-name>/.')
    group.add_argument('--results-dir', type=str, default='results')
    group.add_argument('--overwrite', action='store_true',
                       help='Replace the logs of an earlier run in the same run directory '
                            '(otherwise refused).')
    group.add_argument('--device', type=str, default='cpu',
                       help="'auto' picks cuda when available. Default cpu: PSRO's learners "
                            'update on one episode at a time, which a GPU does not speed up.')


def _add_env_config(parser):
    group = parser.add_argument_group('env')
    group.add_argument('--max-episode-steps', type=int, default=None,
                       help='Truncate episodes after this many env steps (None = game length).')
    group.add_argument('--obs-type', type=str, default='info_state',
                       choices=('info_state', 'observation'),
                       help='OpenSpiel only: perfect-recall information state or Markov observation.')


def _add_eval_config(parser):
    group = parser.add_argument_group('eval')
    group.add_argument('--eval-interval', type=int, default=None,
                       help='Evaluate every this many env steps. Default: budget / 20.')
    group.add_argument('--eval-episodes', type=int, default=200)
    group.add_argument('--eval-metrics', type=str, nargs='+',
                       default=['vs_random', 'self_play'], choices=METRICS.names())


def _add_psro_config(parser):
    # Names and defaults follow OpenSpiel's psro_v2 / psro_v2_example.py.
    group = parser.add_argument_group('psro')
    group.add_argument('--psro-oracle', type=str, default='pg', choices=('pg', 'dqn', 'br'),
                       help="Best-response oracle (psro_v2's oracle_type): RL with policy "
                            "gradient or DQN, or exact best response (OpenSpiel only; the "
                            "example's default).")
    group.add_argument('--psro-iterations', type=int, default=None,
                       help='Optional cap on PSRO iterations (psro_v2_example: gpsro_iterations=100). '
                            'Default: none -- the env-step budget is the only stopping rule, so '
                            'every method trains on the same experience.')
    group.add_argument('--psro-meta-solver', type=str, default='auto',
                       choices=META_SOLVERS.names(),
                       help="'auto' = nash for two-player zero-sum tables, else prd. "
                            '(psro_v2 defaults to alpharank, not ported yet.)')
    group.add_argument('--psro-sims-per-entry', type=int, default=1000,
                       help='Episodes simulated per payoff-table entry (charged to budget).')
    group.add_argument('--psro-number-training-episodes', type=int, default=10_000,
                       help='RL oracle: every new policy trains for more than this many episodes.')
    group.add_argument('--psro-self-play-proportion', type=float, default=0.0,
                       help="RL oracle: chance that another seat plays its own training policy.")
    group.add_argument('--psro-br-init', type=str, default='scratch', choices=('scratch', 'copy'),
                       help="RL oracle: new policies start from scratch (current OpenSpiel), or as "
                            'a copy of the policy the training selector picks (older TF learners).')
    group.add_argument('--psro-training-selector', type=str, default='probabilistic',
                       choices=('probabilistic', 'uniform', 'top_k_probabilities'),
                       help='Which existing policy a new one is copied from (--psro-br-init copy).')
    group.add_argument('--psro-sigma', type=float, default=0.0,
                       help='Gaussian noise added to copied weights (--psro-br-init copy).')
    group.add_argument('--psro-prd-iterations', type=int, default=50_000)
    group.add_argument('--psro-prd-dt', type=float, default=1e-3)
    group.add_argument('--psro-prd-gamma', type=float, default=1e-10,
                       help='PRD exploration floor: minimum mass on every member.')
    group.add_argument('--psro-log-every-episodes', type=int, default=500,
                       help='Write an oracle_episode record to train.jsonl every N episodes per '
                            'seat (0 = never).')


def _add_pg_config(parser):
    # psro_v2_example.py's PG settings for open_spiel/python/jax/policy_gradient.py.
    group = parser.add_argument_group('pg')
    group.add_argument('--pg-loss', type=str, default='qpg', choices=('qpg', 'rpg', 'rm', 'a2c'))
    group.add_argument('--pg-hidden', type=int, nargs='+', default=[256, 256, 256, 256])
    group.add_argument('--pg-lambda', type=float, default=1.0)
    group.add_argument('--pg-critic-lr', type=float, default=1e-2)
    group.add_argument('--pg-pi-lr', type=float, default=1e-3)
    group.add_argument('--pg-entropy-cost', type=float, default=0.001)
    group.add_argument('--pg-num-critic-before-pi', type=int, default=8)
    group.add_argument('--pg-optimizer', type=str, default='adam', choices=('adam', 'sgd'))
    group.add_argument('--pg-max-grad-norm', type=float, default=None)


def _add_dqn_config(parser):
    # psro_v2_example.py's DQN settings for open_spiel/python/jax/dqn.py; the
    # rest are that file's defaults.
    group = parser.add_argument_group('dqn')
    group.add_argument('--dqn-hidden', type=int, nargs='+', default=[256, 256, 256, 256])
    group.add_argument('--dqn-batch-size', type=int, default=32)
    group.add_argument('--dqn-lr', type=float, default=1e-2)
    group.add_argument('--dqn-update-target-network-every', type=int, default=1000)
    group.add_argument('--dqn-weight-update-coeff', type=float, default=0.995,
                       help='tau of the soft target update.')
    group.add_argument('--dqn-learn-every', type=int, default=10)
    group.add_argument('--dqn-replay-buffer-capacity', type=int, default=10_000)
    group.add_argument('--dqn-min-buffer-size-to-learn', type=int, default=1000)
    group.add_argument('--dqn-discount', type=float, default=1.0)
    group.add_argument('--dqn-epsilon-start', type=float, default=1.0)
    group.add_argument('--dqn-epsilon-end', type=float, default=0.1)
    group.add_argument('--dqn-epsilon-decay-duration', type=int, default=1_000_000)
    group.add_argument('--dqn-epsilon-decay-schedule', type=str, default='exp',
                       choices=('exp', 'linear'))
    group.add_argument('--dqn-optimizer', type=str, default='adam',
                       choices=('adam', 'sgd', 'rmsprop'))
    group.add_argument('--dqn-loss', type=str, default='mse', choices=('mse', 'huber'))
    group.add_argument('--dqn-gradient-clipping', type=float, default=None)


def _add_ppo_config(parser):
    # For the (future) self-play `ppo` method; PSRO does not use PPO.
    # Defaults are CleanRL's, except a smaller rollout for short board/card games.
    group = parser.add_argument_group('ppo')
    group.add_argument('--ppo-lr', type=float, default=2.5e-4)
    group.add_argument('--ppo-rollout-steps', type=int, default=1024,
                       help="Learner transitions (the learning seat's decisions) per update.")
    group.add_argument('--ppo-epochs', type=int, default=4)
    group.add_argument('--ppo-minibatches', type=int, default=4)
    group.add_argument('--ppo-gamma', type=float, default=0.99)
    group.add_argument('--ppo-gae-lambda', type=float, default=0.95)
    group.add_argument('--ppo-clip', type=float, default=0.2)
    group.add_argument('--ppo-ent-coef', type=float, default=0.01)
    group.add_argument('--ppo-vf-coef', type=float, default=0.5)
    group.add_argument('--ppo-max-grad-norm', type=float, default=0.5)
    group.add_argument('--ppo-hidden', type=int, nargs='+', default=[128, 128])


def _add_gems_config(parser):
    # Defaults: the paper's Kuhn-poker reference settings (Sharma et al., TMLR
    # 2026, App. Table 3). Marked "not in paper" where the paper is silent.
    group = parser.add_argument_group('gems')
    # budget / population
    group.add_argument('--gems-iterations', type=int, default=None,
                       help='Optional cap on outer iterations T (the paper: 40). Default: none -- '
                            'the env-step budget is the only stopping rule.')
    group.add_argument('--gems-init-anchors', type=int, default=1, help='k0 per seat.')
    group.add_argument('--gems-max-anchors', type=int, default=32, help='k_max per seat.')
    group.add_argument('--gems-replacement', type=str, default='least_mass',
                       choices=('least_mass', 'worst_ev'),
                       help='Which anchor is replaced at k_max (App. P ablations).')
    # generator / latent
    group.add_argument('--gems-latent-dim', type=int, default=8, help='d_z.')
    group.add_argument('--gems-temperature', type=float, default=1.0, help='Policy temperature tau.')
    group.add_argument('--gems-generator', type=str, default='hyper', choices=('hyper', 'conditional'),
                       help="'hyper': z -> policy weights, as described in the paper; "
                            "'conditional': MLP([obs, z]).")
    group.add_argument('--gems-hyper-hidden', type=int, nargs='+', default=[64, 64],
                       help='Generator (and critic) hidden sizes. Not in paper.')
    group.add_argument('--gems-policy-hidden', type=int, default=64,
                       help='Hidden width of each generated policy. Not in paper.')
    # phase 1: MC meta-game estimation
    group.add_argument('--gems-mc-opponents', type=int, default=8, help='N_MC.')
    group.add_argument('--gems-mc-rollouts', type=int, default=2, help='M_MC.')
    group.add_argument('--gems-ema', type=float, default=0.0, help='EMA of v_hat; 0 disables.')
    # phase 2: OMWU
    group.add_argument('--gems-eta', type=float, default=0.03, help='OMWU step size eta.')
    group.add_argument('--gems-eta-schedule', type=str, default='const',
                       choices=('const', 'sqrt', 'harmonic'))
    group.add_argument('--gems-eta-alpha', type=float, default=0.5, help="'harmonic' schedule alpha.")
    group.add_argument('--gems-logit-cap', type=float, default=50.0)
    # phase 3: EB-UCB expansion
    group.add_argument('--gems-oracle-opponents', type=int, default=8, help='N_oracle.')
    group.add_argument('--gems-oracle-rollouts', type=int, default=2, help='M_oracle.')
    group.add_argument('--gems-mutation-pool', type=int, default=32, help='|Lambda_mut|.')
    group.add_argument('--gems-random-pool', type=int, default=32, help='|Lambda_rand|.')
    group.add_argument('--gems-mutation-std', type=float, default=0.2, help='sigma_mut.')
    group.add_argument('--gems-ucb-delta', type=float, default=0.5,
                       help='delta_0; confidence delta_t = delta_0 / t^2.')
    group.add_argument('--gems-jacobian-coef', type=float, default=0.0,
                       help='lambda_J. Not in Table 3 or the ablations; 0 = off.')
    # phase 4: ABR-TR
    group.add_argument('--gems-abr-steps', type=int, default=30, help='K_ABR gradient steps.')
    group.add_argument('--gems-abr-batch-anchors', type=int, default=16)
    group.add_argument('--gems-abr-lr', type=float, default=2e-4)
    group.add_argument('--gems-kl-coef', type=float, default=0.05, help='beta_KL.')
    group.add_argument('--gems-new-fraction', type=float, default=0.25,
                       help='q_new: chance a training anchor is the newly added one.')
    group.add_argument('--gems-grad-clip', type=float, default=0.5)
    group.add_argument('--gems-critic-lr', type=float, default=1e-3, help='Not in paper.')
    group.add_argument('--gems-gamma', type=float, default=0.99, help='GAE gamma. Not in paper.')
    group.add_argument('--gems-gae-lambda', type=float, default=0.95,
                       help='GAE lambda. Not in paper.')


def _add_exploit_config(parser):
    # Post-hoc approximate NashConv (posthoc.py exploit). The same protocol for
    # every method, seed and env: a fresh PPO learner (ppo group above) trains a
    # best response for each seat against the run's final joint policy.
    group = parser.add_argument_group('exploit')
    group.add_argument('--exploit-steps', type=int, default=1_000_000,
                       help='Env steps of best-response training per seat.')
    group.add_argument('--exploit-eval-episodes', type=int, default=1000,
                       help='Episodes to value the best response and the final policy.')


def get_config():
    # allow_abbrev=False: parse_args() detects explicit flags by their full name.
    parser = argparse.ArgumentParser(
        description='SEPSRO', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    _add_run_config(parser)
    _add_env_config(parser)
    _add_eval_config(parser)
    _add_psro_config(parser)
    _add_pg_config(parser)
    _add_dqn_config(parser)
    _add_ppo_config(parser)
    _add_gems_config(parser)
    _add_exploit_config(parser)
    return parser


def explicit_dests(parser, argv):
    """Names of the settings given on the command line in argv."""
    given = {token.split('=', 1)[0] for token in argv if token.startswith('--')}
    return {action.dest for action in parser._actions
            if any(option in given for option in action.option_strings)}


def apply_env_overrides(args, explicit):
    """ENV_OVERRIDES for args.env, except for settings given explicitly."""
    applied = {}
    for key, value in ENV_OVERRIDES.get(args.env_name, {}).items():
        if not hasattr(args, key):
            raise KeyError(f'ENV_OVERRIDES[{args.env_name!r}] sets unknown setting {key!r}.')
        if key not in explicit:
            setattr(args, key, value)
            applied[key] = value
    args.env_overrides = applied
    return args


def parse_args(argv=None, parser=None):
    """argv -> resolved args with per-env overrides; what main.py runs on."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = parser or get_config()
    args = parser.parse_args(argv)
    args.env_preset = parse_env(args.env)
    args.env_name = args.env_preset['name']
    return resolve(apply_env_overrides(args, explicit_dests(parser, argv)))


def resolve(args):
    """Fill in derived fields. Everything downstream reads only the result."""
    args.env_preset = parse_env(args.env)
    args.env_name = args.env_preset['name']
    if args.eval_interval is None:
        args.eval_interval = max(1, args.budget // 20)
    if args.device == 'auto':
        try:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.device = 'cpu'
    args.run_dir = os.path.join(args.results_dir, args.exp_name, args.env_name,
                                args.method, f'seed{args.seed}')
    return args


def save_config(args, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)
