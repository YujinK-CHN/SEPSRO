# SEPSRO

Multi-agent RL research on SESiL, over **PettingZoo** and **OpenSpiel**, with
room for several methods (PSRO, PPO, GEMS, ...) compared under one budget and
one evaluation protocol. Developed for Linux (native, a remote server, or WSL).

Status:
- **`psro`** works. It follows OpenSpiel's `psro_v2` in structure and defaults
  (re-implemented on this codebase's env interface, so it also runs on
  PettingZoo), with psro_v2's three oracles: RL with policy gradient (`pg`),
  RL with DQN (`dqn`), and exact best response (`br`, OpenSpiel only), plus
  RL with PPO (`ppo`, not in psro_v2) for comparisons where PSRO and GEMS use
  the same kind of best-response learner.
- **`gems`** works: the Generative Evolutionary Meta-Solver (Sharma et al.,
  TMLR 2026), per-player form, with the paper's Kuhn-poker defaults. See below.
- `ppo` (self-play) is a registered placeholder that raises
  `NotImplementedError`.
- `random` is a pipeline check, not a method.

## Installation (Linux / WSL)

```bash
conda create -n sepsro python=3.11 -y && conda activate sepsro
pip install torch --index-url https://download.pytorch.org/whl/cpu    # or a CUDA build
pip install -r requirements.txt
pytest -q
```

Without conda, `python3 -m venv .venv && source .venv/bin/activate` works the
same. OpenSpiel ships Linux and macOS wheels only, so on a Windows machine run
everything inside WSL.

## Alliance clusters (Slurm)

On a login node:

```bash
cd ~/projects/def-yourpi && git clone https://github.com/YujinK-CHN/SEPSRO.git && cd SEPSRO
bash setup_cluster.sh                 # venv at ~/venvs/sepsro (python/3.11 module)
source ~/venvs/sepsro/bin/activate && AutoROM   # only for space_invaders; accept the ROM licence
bash submit.sh --account def-yourpi --exp-name si_gems --method gems \
    --env space_invaders --budget 30000000 --seeds 0,1,2,3,4 --time 36:00:00
```

`setup_cluster.sh` installs from the Alliance wheelhouse where possible and
from PyPI otherwise (compute nodes have no internet, so everything is installed
up front). One `submit.sh` call is one task of one [round](#rounds): exactly
one `--method`, a required `--exp-name`, one Slurm array task per seed,
CPU-only by default (`--cpus 4 --mem 16G --time 24:00:00`; `--gpus 1` if ever
needed; `--exclude fc30564,...` keeps tasks off broken nodes); every other flag goes to `main.py`. Before submitting it builds the
env once on the login node and refuses what `main.py` would refuse: seeds that
already hold logs (unless `--overwrite`) and settings that differ from the
round's. `--dry-run` writes and prints the job script only. Job scripts and
Slurm logs go to `<round>/slurm/`.

`--task exploit` runs the approximate NashConv on the round's finished seeds
(one task per seed and checkpoint: `--checkpoints final` by default, e.g.
`--checkpoints 0,0.25,0.5,0.75,1` for a 5-point curve), `--task crossplay` the
round's cross-play (one task; `--against <round> ...` adds other rounds). Neither
takes `--method`: the round has it. See [Experiment protocol](#experiment-protocol).

```bash
bash submit.sh --account def-yourpi --exp-name si_gems --seeds 0,1,2,3,4 --task exploit \
    --checkpoints 0,0.25,0.5,0.75,1 --time 12:00:00
bash submit.sh --account def-yourpi --exp-name si_gems --task crossplay --against si_psro_ppo
```

## How to run

```bash
python main.py --method random --env kuhn_poker --budget 100000 --seed 0

# PSRO as in psro_v2_example.py, with its three oracles
python main.py --method psro --env kuhn_poker --psro-oracle br  --budget 100000000
python main.py --method psro --env kuhn_poker --psro-oracle pg  --budget 100000000
python main.py --method psro --env kuhn_poker --psro-oracle dqn --budget 100000000

bash run.sh --method psro --env leduc_poker --budget 100000000 --seeds 0,1,2 --exp-name leduc_psro

# GEMS with the paper's defaults (no iteration cap: it runs until the budget is spent)
python main.py --method gems --env kuhn_poker --budget 100000000
python main.py --method gems --env connect_four --budget 100000000

# post-hoc metrics on finished runs
python posthoc.py exploit results/si_gems/stats/seed0                       # final policy
python posthoc.py exploit results/si_gems/stats/seed0 --checkpoint 0,0.5,1  # a curve
python posthoc.py crossplay results/si_gems --against si_psro_ppo            # within + against
```

`--eval-metrics nash_conv` gives exact NashConv on OpenSpiel turn-based games
and on the native matrix games; on Kuhn and Leduc it is on by default
(`nash_conv vs_random self_play`, from `config.ENV_OVERRIDES`). PettingZoo games have no exact equivalent; use
`vs_random` / `self_play` there.

### Rounds

A **round** is the complete result of one method with one configuration in
one env: all its seeds, in one folder named by `--exp-name`. Name it after what
defines it (e.g. `si_gems`, `kuhn_psro_br`); another method or configuration is
another round.

```
results/<round>/
  round.json              the round's settings: every setting except the seed, paths
                          and logging-only ones (device, progress intervals)
  stats/                  the statistics -- copy this back to plot
    seed<N>/
      config.json         every setting, including the per-env overrides applied
      eval.jsonl          one record before training, one every --eval-interval env
                          steps, one at the end; same schema for every method; each
                          names the checkpoint it scored
      train.jsonl         method internals: psro_iteration / oracle_episode records,
                          gems_iteration records (meta-strategies, values, oracle
                          picks, ABR losses)
      exploit/<checkpoint>.json   posthoc.py exploit
    crossplay/within.json, vs_<round>.json   posthoc.py crossplay
  checkpoints/seed<N>/    the policies -- only if needed (large on Atari)
    step_<N>.pkl          the joint policy of the evaluation at env step N (every
                          --checkpoint-every-th point; default 1 = all, 0 = none)
    final.pkl             the final one (always)
    objects/              their networks
  slurm/                  job scripts and Slurm logs (submit.sh)
```

The first seed writes `round.json`; a later seed whose settings differ is
refused, so a round's seeds can be averaged. The name can take a new
configuration only once no seed of the old one is left (or with `--overwrite`
for a round of one seed).

`posthoc.py` finds a seed's checkpoints next to its statistics (wherever the
round now lives) and writes only into `stats/`. Each network is stored
once in `objects/`, named by a hash of its contents (`sepsro/checkpoints.py`):
a PSRO population member is written when it first appears, not again in every
later checkpoint, and a GEMS generator once per iteration however many anchors
use it. So a run's checkpoints cost about one copy of every network it ever
committed; `step_<N>.pkl` files themselves are small. Copy the checkpoint
directory whole (the `.pkl` files need `objects/`).

Every record in both JSONL files carries `wall_time` and `elapsed_s`, and the
`[eval]` lines in stdout are timestamped, so a run's speed can be read from
its files. GEMS also prints timestamped progress lines to stdout (the Slurm
log): one per phase and seat (episodes, env steps, longest episode, seconds,
and for ABR-TR the time spent in updates), one per finished iteration, and a
heartbeat every `--gems-progress-interval` env steps (default 100k, 0 = off)
naming the current phase. `gems_iteration` records hold `phase_seconds`. None
of this changes the run.

The budget is a hard stop: training ends at exactly `--budget` env steps,
mid-iteration if need be, and every evaluation scores the method's last
*committed* output (PSRO: the meta-strategy after the last solve; GEMS: the
anchors, sigma and generator after the last completed iteration).

Every eval record carries `method`, `env`, `seed`, `budget` and the method's
variant (PSRO: `oracle`, `meta_solver`, `pg_loss`; GEMS: `generator`). A seed
that already holds logs is refused; use `--overwrite` to replace it.

OpenSpiel games take parameters inline: `--env "openspiel:kuhn_poker(players=3)"`.

## Plotting

```bash
python plot_results.py results --list-metrics                       # every round, its seeds, metrics
python plot_results.py results/kuhn_psro results/kuhn_gems          # nash_conv, else approx_nash_conv, else vs_random/mean
python plot_results.py results/si_psro --metric vs_random/mean --metric self_play/mean --train
python plot_results.py results --filter env=space_invaders --band minmax --out fig.pdf
python plot_results.py results/kuhn_psro --table                    # the numbers instead of a figure
```

Give it rounds (or `results/` for all). One series per round, named after it;
seeds are averaged with a std / min-max / IQR band on the env-step axis,
interpolated onto a shared grid where seeds' evaluation points differ. Only
`stats/` is read, so it runs on a copy without checkpoints.
Scored checkpoints (`exploit/*.json`) join their run as the metric
`approx_nash_conv`, one point per checkpoint, so `posthoc.py exploit
--checkpoint 0,0.25,0.5,0.75,1` on every run gives an approximate-NashConv curve.
`--train` adds `<out>_train.png`: PSRO's best-response training return and
GEMS's mixture value and chosen-candidate value per iteration and seat.
Default output: `results/plots/<round>_vs_<round>__<metric>.png`, outside every
round.

## Space Invaders (PettingZoo Atari)

```bash
pip install "pettingzoo[atari]" autorom
AutoROM            # accept the ROM licence
python main.py --method gems --env space_invaders --budget 100000000
```

The `space_invaders` preset is `pettingzoo.atari.space_invaders_v2` (Parallel
API, grey-scale) behind `AtariPreprocessing` (`sepsro/envs/wrappers.py`):
4-frame skip with max-pooling of the last two frames, 84x84, 4-frame stack,
uint8 observations of shape `[4, 84, 84]`, and sticky actions with probability
0.25 (Machado et al., 2018): at every emulator frame each player repeats its
previously executed action instead of its chosen one with that probability.
PettingZoo sets ALE's own `repeat_action_probability` to 0, and without it the
game is deterministic: one action sequence always replays the same episode, so
policies can memorise moves instead of reacting. One env step = one decision = 4
emulator frames, so `--budget` and `--max-episode-steps` count decisions. Two
players, 6 actions each, both scoring against the invaders (not zero-sum: use
PRD / GEMS's OMWU, not the Nash LP). No exact NashConv exists; evaluate with
`vs_random` / `self_play`.

Image observations go through a Nature-DQN CNN (`learners/networks.py`) in PG, PPO,
DQN and GEMS (for GEMS's `hyper` generator the CNN torso is shared and the
policy head is generated from z). The hypernetwork generates each distinct
code's weights once per batch, not once per transition, which on Space Invaders
halves both the time and the memory of a GEMS ABR-TR update.

Measured cost (WSL, CPU, 8 threads): random play ~380 decisions and 0.6 s per
episode; CNN policies ~450-500 decisions and 1.1-1.2 s per episode; a PG
update ~0.25 s per episode. Measured on fir (4 CPUs per task, 5M-step test
run with the Kuhn defaults, before sticky actions and the generator change):
GEMS ~1.3M env steps per hour and ~9.4 GB peak memory, PSRO-PG ~1.4-2.1M
steps per hour and ~4.8 GB.

## PSRO and psro_v2

Defaults are `psro_v2_example.py`'s: PG oracle with QPG loss, a 4x256 ReLU MLP,
Adam, critic lr 1e-2, policy lr 1e-3, entropy cost 0.001, 8 critic updates per
policy update; 10,000 training episodes per response; 1000 simulations per
payoff entry; PRD with 50,000 iterations and gamma 1e-10; at most 100
iterations. The DQN oracle uses the example's 4x256 network, batch 32, lr 1e-2,
target update every 1000 steps and learning every 10.

`--psro-oracle ppo` (not in psro_v2) is the same RL oracle and schedule with
the repo's PPO learner (CleanRL-style, Nature CNN on frames): each finished
episode's transitions go into a rollout buffer, every `--ppo-rollout-steps`
transitions trigger one PPO update (`--ppo-epochs` x `--ppo-minibatches`,
clipped surrogate, GAE), and the last partial rollout of a response is spent
on one more update. It is there so PSRO can be compared with GEMS, whose
ABR-TR is PPO, with the same kind of learner. It takes the `ppo` group's
settings (with `ENV_OVERRIDES`); `--psro-ppo-lr`, `-rollout-steps`, `-epochs`,
`-minibatches`, `-clip`, `-ent-coef` replace them for the oracle only, so
tuning the oracle never changes the post-hoc scorer, which uses the `ppo`
group.

Where this differs from psro_v2, on purpose:

| | psro_v2 | here | why |
|---|---|---|---|
| Learners | JAX (`open_spiel/python/jax`) | PyTorch ports, same algorithms | one framework for the codebase |
| Default meta-solver | `alpharank` | `auto` (Nash LP for 2-player zero-sum, else PRD) | alpharank not ported yet |
| Default oracle | `BR` | `pg` | BR exists only for OpenSpiel; `pg` runs everywhere |
| PPO oracle | -- | `--psro-oracle ppo` | same learner family as GEMS's ABR-TR, for oracle-aligned comparisons |
| PG policy loss | softmax over all actions | illegal actions masked | untrained Q of illegal actions otherwise pulls probability to them |
| Stopping | 100 iterations | the env-step budget only, exactly (`--psro-iterations` is an optional cap) | the budget is the cross-method currency |

Not ported yet: alpharank, `rm` / `uniform_biased` meta-solvers, rectified
training, symmetric-game mode, `number_policies_selected > 1`.

**Checked against psro_v2.** Kuhn poker, PG oracle, PRD meta-solver, example
defaults otherwise, 10 iterations, NashConv after each iteration (OpenSpiel
2.0.2, its JAX learners):

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| psro_v2_example | 0.586 | 0.395 | 0.374 | 0.364 | 0.335 | 0.343 | 0.331 | 0.323 | 0.324 | 0.317 |
| SEPSRO | 0.570 | 0.289 | 0.390 | 0.370 | 0.365 | 0.341 | 0.316 | 0.326 | 0.328 | 0.311 |

(The example's `nash` meta-solver could not be compared: its cvxpy LP returns
no solution on the initial 1x1 game with current cvxpy.)

## GEMS

`sepsro/methods/gems.py`, after Algorithm 1 of the paper, in its per-player form
(App. F/H): every seat has its own generator, anchor set and meta-strategy.
Each iteration: (1) Monte-Carlo value of every anchor against the other seats'
mixtures, (2) one optimistic MWU step, (3) an EB-UCB bandit over mutated and
random latent codes adds one anchor, (4) ABR-TR fine-tunes the generator
(policy gradient with GAE plus a KL trust region). No payoff table is kept.

Defaults are the paper's Kuhn reference settings (App. Table 3), except that
T=40 is not imposed (`--gems-iterations` is an optional cap; the budget stops
it, as for every method): k0=1,
k_max=32, d_z=8, tau=1; N_MC=8 x M_MC=2; OMWU eta=0.03 const, logit cap 50;
N_oracle=8 x M_oracle=2, 32 mutated (std 0.2) + 32 random candidates, delta_0=0.5;
K_ABR=30 steps of 16 anchors, lr 2e-4, beta_KL=0.05, q_new=0.25, grad clip 0.5.

Where the paper is silent or ambiguous, the choice made here:

| | paper | here |
|---|---|---|
| Generator | "hypernetwork", "standard MLP" | MLP z -> weights of a 1-hidden-layer (64, tanh) policy (`--gems-generator hyper`); `conditional` = MLP([obs, z]) |
| Generator initialisation | not given | PyTorch's default (an earlier version shrank the z-dependent weights x0.1, so all anchors started as near-copies) |
| MC opponents (Phase 1) | one batch {j_s} ~ sigma_{t-1} (Algorithm 1) | the same: one batch per seat and iteration, shared by every anchor |
| Critic for GAE | not described | V(obs, z) MLP, lr 1e-3, gamma 0.99, lambda 0.95 |
| ABR-TR optimiser | "PPO with GAE", no settings | each of the K_ABR steps: one fresh batch, 4 epochs x 4 minibatches of the clipped surrogate (clip 0.2) + beta_KL x KL to the generator at the start of the phase (`--gems-abr-epochs/-minibatches/-clip`; `1 1 0` = one plain PG step) |
| Reward scale | assumed in [0, 1] | normalised per seat: OpenSpiel min/max utility, matrix payoff range, else range seen in random play |
| Mixture value r_hat | B=128 joint samples | sigma . v_hat (same in expectation; a constant OMWU's normalisation removes) |
| New anchor's mass | not stated | 1/(k+1) (OMWU cannot grow a zero) |
| "Top anchors" to mutate | not defined | parents drawn proportionally to sigma |
| q_new = 0.25 | "new-opponent fraction" | probability a training anchor in ABR-TR is the new code (q_t of Eq. 11) |
| lambda_J (Jacobian) | not in Table 3 | 0 (implemented, off) |
| n > 2 players | importance-weighted shared batch | per-anchor estimates for every seat (also unbiased) |

**Kuhn poker, paper defaults, one seed, before the PPO update** (with one plain PG step per batch; NashConv; the paper reports
exploitability ~0.18 at iteration 40, which may be NashConv / 2):
0.895 at start, best 0.332, 0.484 after 40 iterations (exploitability 0.24),
347k env steps. Training anchor codes only on the new code (`--gems-new-fraction 1`)
ended at 0.388. On rock-paper-scissors GEMS drifts away from the uniform Nash it
starts near: ABR-TR pushes all anchors of the shared generator towards the same
best response faster than the KL term holds them.

## Experiment protocol

The paper compares PSRO and GEMS as baselines on three environments: Kuhn and
Leduc poker (the standard benchmarks, exact NashConv) and two-player Space
Invaders (the harder one). 5 seeds (0-4) per method and environment; each run
is a single Slurm job.

**Fairness rules**, enforced by the code:

1. *Equal experience.* Every method trains for exactly `--budget` env steps,
   payoff-table simulation included; the hard stop means nobody finishes an
   iteration on extra steps.
2. *Equal measurement.* Every method is evaluated at the same env-step counts
   (0, every `--eval-interval`, the budget), on its last committed output,
   with the same metrics, episodes and evaluation RNG stream.
3. *No iteration caps.* The budget is the only stopping rule (psro_v2's 100
   iterations and GEMS's T=40 would stop either method at a different amount
   of experience).
4. *Published settings where they exist.* Kuhn and Leduc use psro_v2_example's
   and the GEMS paper's settings unchanged. Space Invaders changes only the
   settings listed in `config.ENV_OVERRIDES` (below), applied to both methods
   in the same way, with the same tuning effort for each.
5. *Same network torso.* Both methods use the Nature-DQN CNN on image
   observations.

**Metrics**

| | Kuhn, Leduc | Space Invaders |
|---|---|---|
| headline | exact NashConv (on by default for Kuhn and Leduc), the full curve | approximate NashConv (`posthoc.py exploit`): at the end, or a curve over checkpoints |
| secondary | -- | cross-play (`posthoc.py crossplay`) |
| learning curves | NashConv | `vs_random`, `self_play` |

Space Invaders is general-sum and partly competitive (the players compete for
the same invaders; in the test run one trained player's score fell while the
other's rose), so there is no winner and raw score in self-play says little
about equilibrium play. *Approximate NashConv* is the same quantity as the
poker headline: for each seat a fresh PPO learner (config groups `exploit`
and `ppo`; 5M steps per seat, Atari PPO clip 0.1, rewards scaled by 0.01
for training) trains a best response to
the run's final policy of the other seat, and the gains in return over the
final policy are summed. It is a lower bound (an RL best response can be
weaker than the exact one) and is only comparable because every run gets the
identical protocol; on Kuhn and Leduc `exploit/<checkpoint>.json` also holds the exact
value, which checks the protocol. *Cross-play* (per round) pairs each seed's
final policy with co-players it never trained with: the round's other seeds
(`within.json`), and every seed of each `--against` round in both seat
assignments (`vs_<round>.json`); it reports each seed's mean return and the
mean and std over seeds. Methods are compared by running each round against
the same other rounds.

**Space Invaders settings** (`config.ENV_OVERRIDES['space_invaders']`;
starting points for the pilot, every other setting is the published default)

| setting | default | Space Invaders | why |
|---|---|---|---|
| `--budget` | -- | 30M env steps (120M frames) | ~15-23 h per run on fir, inside one 36 h job |
| `--eval-episodes` | 200 | 100 | 20 evaluation points x 3 metrics stay ~1.5 h of a run |
| `--psro-number-training-episodes` | 10,000 | 3,000 | episodes are ~150x longer than Kuhn's; ~2.7M steps per response pair -> ~9 iterations |
| `--psro-sims-per-entry` | 1,000 | 50 | payoff simulation costs (2n+1) x sims episodes at iteration n |
| `--gems-mutation-pool` / `--gems-random-pool` | 32 / 32 | 8 / 8 | the same 1,024 oracle episodes per seat, spent on 16 candidates x 64 episodes instead of 64 x 16, so the pick is not noise |
| `--gems-oracle-opponents` x `--gems-oracle-rollouts` | 8 x 2 | 16 x 4 | (same) |
| `--gems-eta` | 0.03 | 0.5 | with ~20 iterations and value gaps of ~0.05, 0.03 left sigma uniform in the test run |
| `--gems-abr-lr` | 2e-4 | 1e-3 | the test run's ABR-TR moved the policy by KL ~1e-4 per iteration |
| `--ppo-clip` | 0.2 | 0.1 | CleanRL's Atari PPO value |
| `--ppo-reward-scale` | 1 | 0.01 | raw scores are 5-200 per hit (~500 per episode), which left the PPO critic's loss in the hundreds; a constant factor keeps the objective -- the score -- unchanged (sign clipping, CleanRL's Atari choice, would make the learner maximise hits, so it would not be a best response in the game's payoffs) |

The two `ppo` rows apply to the post-hoc best response and to `--psro-oracle
ppo` alike; GEMS normalises its own rewards to [0, 1] (see GEMS above).

**Pilot (equal tuning effort).** Before the main runs: per method, the same
number of short runs (1 seed, outside 0-4) varying only the settings above,
judged by approximate NashConv; the chosen values replace the starting points
in `ENV_OVERRIDES`, and everything tried is listed here.

## Layout

```
config.py                 single source of truth: ENV_PRESETS + every hyper-parameter
main.py                   single entry point; builds shared pieces, runs --method
posthoc.py                approximate NashConv (PPO best responses) and cross-play on finished runs
run.sh                    launcher over the seeds of one round (local)
setup_cluster.sh          one-time venv setup on an Alliance login node
submit.sh                 Slurm job-array submission: one round, one task per seed [x checkpoint]
plot_results.py           learning curves of rounds, seeds averaged (+ training diagnostics)
sepsro/
  checkpoints.py          policy checkpoints that store each network once (objects/<hash>.pkl)
  envs/
    base.py               MultiAgentEnv / EnvSpec / TimeStep -- the one interface
    openspiel_env.py      OpenSpiel adapter (sequential + simultaneous, chance resolved)
    pettingzoo_env.py     PettingZoo adapter (AEC + Parallel, any module path)
    native_env.py         dependency-free matrix games (rps, ...) for debugging
    wrappers.py           AtariPreprocessing (skip, max-pool, resize, stack)
  policies.py             Policy interface, RandomPolicy, MixturePolicy
  rollout.py              play_episode(): the only env loop; per-agent credit assignment
  meta_game.py            Population + cached empirical PayoffTable
  meta_solvers.py         payoff tensor -> meta-strategy (uniform, nash, prd, auto)
  oracles.py              RLOracle (pg / dqn / ppo) and exact BestResponseOracle
  learners/
    pg.py                 QPG / RPG / RM / A2C (port of OpenSpiel's JAX policy_gradient)
    dqn.py                DQN (port of OpenSpiel's JAX dqn)
    generator.py          GEMS generator (hyper / conditional), critic, GeneratorPolicy
    networks.py           encoders (Nature-DQN CNN for images, flatten otherwise)
    ppo.py                PPO: post-hoc best response, and the PSRO `ppo` oracle
  exploitability.py       exact NashConv; mixture -> behaviour policy by reach
  budget.py               BudgetTracker, in env steps
  evaluation.py           Evaluator + metric registry (vs_random, self_play)
  logging.py              JSONL eval/train streams
  registry.py             METHODS / METRICS registries
  methods/
    base.py               Method contract
    random_baseline.py    pipeline check
    psro.py               PSRO
    gems.py               GEMS
    ppo.py                placeholder
tests/test_core.py        env adapters, rollout, budget, payoff table, pipeline
tests/test_psro.py        meta-solvers, oracles, PSRO, NashConv (closed-form checks)
tests/test_gems.py        OMWU, EB-UCB, generator, GEMS end to end
tests/test_images.py      Atari wrapper, CNN learners, Space Invaders preset (if ROMs)
tests/test_plot.py        plot_results grouping, naming, seed aggregation, CLI
tests/test_protocol.py    hard budget, evaluation points, overrides, saved policies, post-hoc metrics
```

## Design rules

- **One env interface.** Every step takes `{agent: action}`; every `TimeStep`
  says which agents act next (`acting`). Turn-based and simultaneous games are
  the same loop. Rewards are reported for *all* agents on every transition.
- **Budget = env steps, hard stop.** Every training step a method takes is
  charged, including payoff-table simulation, and training stops at exactly
  the budget. Evaluation is not charged.
- **Same evaluation for everyone.** Methods expose `policies()` (one `Policy`
  per seat, their last committed output); the `Evaluator` scores it at the
  same env-step counts with the same metrics, so `eval.jsonl` files from
  different methods line up.
- **Registries, not edits.** A new method is a file in `sepsro/methods/` with
  `@METHODS.register('name')`, imported in `methods/__init__.py`, plus its
  argument group in `config.py`. A new metric is `@METRICS.register('name')`.
- **Environments by preset.** Add frequently used envs to `ENV_PRESETS`; try
  others ad hoc with `--env openspiel:<game>` or
  `--env pettingzoo:<module>[@parallel]`.
