"""
Post-hoc metrics on finished runs, from the policy checkpoints main.py saves.

    python posthoc.py exploit   <run_dir> [--checkpoint WHICH] [config flags]
    python posthoc.py crossplay <round> [--against <round> ...] [--episodes N]

A round is results/<exp-name>/: stats/seed<N>/ (a <run_dir>) and
checkpoints/seed<N>/. Both tools read the checkpoints and write only into
stats/, so copying stats/ back is enough.

exploit -- approximate NashConv (the paper's Atari metric; Timbers et al., 2022)
    For each seat i, a FRESH PPO learner trains a best response BR_i against
    the run's final policies of the other seats for --exploit-steps env steps.
    Then, over --exploit-eval-episodes episodes each,

        gain_i            = V_i(BR_i, pi_-i) - V_i(pi)
        approx_nash_conv  = sum_i gain_i

    The protocol -- learner, settings, steps, episodes -- is the same for every
    method, seed and env (its settings are config.py's 'exploit' and 'ppo'
    groups, with ENV_OVERRIDES), so the numbers are comparable across runs. It
    is a LOWER bound on the true NashConv: an RL best response can be weaker
    than the exact one, and a gain can even come out negative. On OpenSpiel
    and native games the exact NashConv is written alongside, which checks the
    protocol.

    --checkpoint picks which saved policy is scored (default final):
        final            the policy of the final evaluation
        step_<N>         the one saved at env step N
        <fraction>       e.g. 0.25: the checkpoint nearest that fraction of
                         the budget (0 = before training, 1 = final), so every
                         run is scored at the same points of its budget
        all              every checkpoint, one after the other
    and a comma list of these. A curve over training = several checkpoints.
    Output: <run_dir>/exploit/<checkpoint>.json, with the checkpoint's env step.

crossplay -- returns against co-players never trained with (two-player games)
    With the final policies (final.pkl) of the round's seeds, --episodes
    episodes per pairing:
        within   seed i's seat 0 with seed j's seat 1, for all i != j
                 (needs two finished seeds)
        against  for each --against round: every seed of this round with every
                 seed of that one, in both seat assignments
    Per seed: the mean return as seat 0, as seat 1, and their mean; then mean
    and std over the seeds. Output: <round>/stats/crossplay/within.json and
    vs_<round>.json. Comparing methods = each round against the same others.

Neither spends training budget; both only read the runs. Load pickles you
wrote yourself only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from config import STATS_DIR, checkpoint_dir_of, parse_args, parse_env
from main import FINAL_CHECKPOINT, checkpoint_path, load_policies
from sepsro.budget import BudgetTracker
from sepsro.envs import make_env
from sepsro.rollout import play_episode


def _read_config(run_dir):
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as handle:
        cfg = json.load(handle)
    # From where the run is now, not where it was trained: a round can be moved.
    cfg['checkpoint_dir'] = checkpoint_dir_of(run_dir)
    return cfg


def checkpoints(cfg):
    """[(env step, name)] of a run's saved policies, in training order."""
    directory = cfg['checkpoint_dir']
    found = []
    for f in os.listdir(directory) if os.path.isdir(directory) else ():
        name, ext = os.path.splitext(f)
        if ext != '.pkl':
            continue
        if name == FINAL_CHECKPOINT:
            found.append((int(cfg['budget']), name))
        elif name.startswith('step_') and name[5:].isdigit():
            found.append((int(name[5:]), name))
    return sorted(found, key=lambda item: (item[0], item[1] == FINAL_CHECKPOINT))


def select_checkpoints(cfg, which):
    """--checkpoint (final | step_<N> | fraction | all, or a comma list) -> [(step, name)]."""
    saved = checkpoints(cfg)
    if not saved:
        raise SystemExit(f'No checkpoints in {cfg["checkpoint_dir"]}: train the run first.')
    by_name = {name: step for step, name in saved}
    chosen = []
    for item in str(which).split(','):
        item = item.strip()
        if item == 'all':
            chosen += saved
        elif item in by_name:
            chosen.append((by_name[item], item))
        else:
            try:
                fraction = float(item)
            except ValueError:
                raise SystemExit(f'Unknown checkpoint {item!r}; saved: '
                                 f'{", ".join(name for _, name in saved)}.') from None
            if not 0.0 <= fraction <= 1.0:
                raise SystemExit(f'A checkpoint fraction must be in [0, 1], got {item}.')
            if fraction == 1.0 and FINAL_CHECKPOINT in by_name:
                chosen.append((by_name[FINAL_CHECKPOINT], FINAL_CHECKPOINT))
            else:
                target = fraction * int(cfg['budget'])
                chosen.append(min(saved, key=lambda c: abs(c[0] - target)))
    return list(dict.fromkeys(chosen))


def _env_for(cfg):
    return make_env(parse_env(cfg['env']), max_episode_steps=cfg.get('max_episode_steps'),
                    obs_type=cfg.get('obs_type', 'info_state'))


def _returns(env, joint, rng, episodes):
    """Per-seat (mean, standard error) of episode returns."""
    out = {a: [] for a in env.spec.agents}
    for _ in range(episodes):
        for agent, value in play_episode(env, joint, rng).returns.items():
            out[agent].append(value)
    return {a: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
            for a, v in out.items()}


def _exact_nash_conv(env, policies, rng):
    """Exact NashConv where it exists (OpenSpiel turn-based, native matrix games), else None."""
    from sepsro.exploitability import nash_conv
    try:
        return nash_conv(env, policies, rng, 0)['nash_conv']
    except Exception:
        return None


# --------------------------------------------------------------------- exploit
def exploit_path(run_dir, name):
    return os.path.join(run_dir, 'exploit', f'{name}.json')


def exploit(run_dir, argv, checkpoint=FINAL_CHECKPOINT):
    """Approximate NashConv of each selected checkpoint; returns the list of results."""
    cfg = _read_config(run_dir)
    # The run's env and seed, this protocol's settings (+ ENV_OVERRIDES, + flags).
    args = parse_args(['--env', cfg['env'], '--seed', str(cfg['seed']), *argv])
    chosen = select_checkpoints(cfg, checkpoint)
    done = [exploit_path(run_dir, name) for _, name in chosen
            if os.path.exists(exploit_path(run_dir, name))]
    if done and not args.overwrite:
        raise SystemExit(f'{", ".join(done)} exist(s). Use --overwrite to replace.')
    return [_exploit_one(run_dir, cfg, args, step, name) for step, name in chosen]


def _exploit_one(run_dir, cfg, args, step, name):
    import torch

    from sepsro.learners.ppo import PPOConfig, PPOLearner

    out_path = exploit_path(run_dir, name)
    policies = load_policies(checkpoint_path(cfg['checkpoint_dir'], name))
    env = _env_for(cfg)
    agents = env.spec.agents
    rng = np.random.default_rng([cfg['seed'], 2])
    torch.manual_seed(cfg['seed'])
    ppo = PPOConfig.from_args(args)
    print(f'[exploit] {run_dir} checkpoint {name} (env step {step}): {cfg["method"]} on '
          f'{cfg["env"]}, {args.exploit_steps} PPO steps per seat', flush=True)

    start = time.time()
    baseline = _returns(env, policies, rng, args.exploit_eval_episodes)
    seats = {}
    for index, seat in enumerate(agents):
        learner = PPOLearner(env.spec.obs_shape[seat], env.spec.num_actions[seat], ppo,
                             seed=cfg['seed'] * 1000 + index)
        opponents = {a: policies[a] for a in agents if a != seat}
        budget = BudgetTracker(args.exploit_steps)
        curve = []

        def on_update(stats, seat=seat, budget=budget):
            curve.append({'steps': budget.used, 'mean_return': stats['mean_return']})
            if len(curve) % 50 == 0:
                print(f'[exploit] {seat} {budget.used:>10d} steps  '
                      f'return {stats["mean_return"]:.3f}', flush=True)

        learner.train(env, seat, opponents, rng, budget, args.exploit_steps, on_update=on_update)
        joint = dict(opponents)
        joint[seat] = learner.policy()
        br_value, br_se = _returns(env, joint, rng, args.exploit_eval_episodes)[seat]
        value, se = baseline[seat]
        gain_se = float(np.hypot(br_se, se))
        seats[seat] = {'br_value': br_value, 'br_se': br_se, 'policy_value': value,
                       'policy_se': se, 'gain': br_value - value, 'gain_se': gain_se,
                       'curve': curve}
        print(f'[exploit] {seat}: best response {br_value:.3f} vs policy {value:.3f} '
              f'-> gain {br_value - value:.3f} +- {gain_se:.3f}', flush=True)

    result = {'run_dir': run_dir, 'method': cfg['method'], 'env': cfg['env'], 'seed': cfg['seed'],
              'exp_name': cfg.get('exp_name'), 'budget': cfg.get('budget'),
              'checkpoint': name, 'budget_used': step,
              'approx_nash_conv': float(sum(s['gain'] for s in seats.values())),
              # Standard error from the evaluation episodes only (the best
              # response's own training randomness is not in it).
              'approx_nash_conv_se': float(np.sqrt(sum(s['gain_se'] ** 2 for s in seats.values()))),
              'exact_nash_conv': _exact_nash_conv(env, policies, rng),
              'seats': seats, 'exploit_steps': args.exploit_steps,
              'eval_episodes': args.exploit_eval_episodes, 'ppo': vars(ppo),
              'seconds': time.time() - start}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    exact = result['exact_nash_conv']
    print(f'[exploit] approx NashConv {result["approx_nash_conv"]:.4f}'
          f' +- {result["approx_nash_conv_se"]:.4f}'
          + (f'  (exact {exact:.4f})' if exact is not None else '') + f'  -> {out_path}',
          flush=True)
    return result


# ------------------------------------------------------------------- crossplay
def resolve_round(name, near=None):
    """A round given as a directory, or as a name next to `near` (another round) / in results/."""
    if os.path.isdir(os.path.join(name, STATS_DIR)):
        return os.path.normpath(name)
    for base in ([os.path.dirname(os.path.normpath(near))] if near else []) + ['results']:
        candidate = os.path.join(base, name)
        if os.path.isdir(os.path.join(candidate, STATS_DIR)):
            return os.path.normpath(candidate)
    raise SystemExit(f'No round {name!r} (a directory holding {STATS_DIR}/).')


def finished_seeds(round_dir):
    """[(run_dir, config)] of a round's seeds whose final.pkl exists, in seed order."""
    out = []
    for run_dir in glob.glob(os.path.join(round_dir, STATS_DIR, 'seed*')):
        try:
            cfg = _read_config(run_dir)
        except (OSError, ValueError):
            continue
        if os.path.exists(checkpoint_path(cfg['checkpoint_dir'], FINAL_CHECKPOINT)):
            out.append((run_dir, cfg))
    return sorted(out, key=lambda item: item[1]['seed'])


def _load_finals(seeds):
    return [load_policies(checkpoint_path(cfg['checkpoint_dir'], FINAL_CHECKPOINT))
            for _, cfg in seeds]


def _summary(values):
    values = list(values)
    return {'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 'n': len(values)}


def _pairing(env, first, second, seat0, seat1, episodes, rng_key):
    """Mean returns (seat 0, seat 1) of first's seat-0 policy with second's seat-1 policy."""
    values = _returns(env, {seat0: first[seat0], seat1: second[seat1]},
                      np.random.default_rng(rng_key), episodes)
    return values[seat0][0], values[seat1][0]


def crossplay(round_dir, against=(), episodes=100, overwrite=False):
    """Cross-play of a round: among its own seeds, and against each round in `against`.

    within.json      seed i's seat 0 with seed j's seat 1, for every i != j (the
                     diagonal, self-play, is recorded but not in the means): how
                     well independently trained runs of the method play together.
    vs_<round>.json  each of this round's seeds with each of the other round's,
                     in both seat assignments: this round's return as seat 0
                     against their seat 1, and as seat 1 against their seat 0.
    Both in <round>/stats/crossplay/, with per-seed means and mean / std over seeds.
    """
    round_dir = resolve_round(round_dir)
    others = [resolve_round(o, near=round_dir) for o in against]
    name = os.path.basename(round_dir)
    mine = finished_seeds(round_dir)
    if not mine:
        raise SystemExit(f'Round {round_dir} has no finished seed (final.pkl).')
    if len(mine) < 2 and not others:
        raise SystemExit(f'Round {round_dir} has one finished seed: within-round cross-play '
                         'needs two; add --against <round>.')
    pools = {os.path.basename(o): finished_seeds(o) for o in others}
    for other, seeds in pools.items():
        if not seeds:
            raise SystemExit(f'Round {other} has no finished seed (final.pkl).')
    envs = {cfg['env'] for _, cfg in mine} | {cfg['env'] for s in pools.values() for _, cfg in s}
    if len(envs) != 1:
        raise SystemExit(f'Rounds come from different envs: {sorted(envs)}.')
    env = _env_for(mine[0][1])
    if env.spec.num_agents != 2:
        raise SystemExit('Cross-play is defined here for two-player games.')
    seat0, seat1 = env.spec.agents

    out_dir = os.path.join(round_dir, STATS_DIR, 'crossplay')
    targets = ([os.path.join(out_dir, 'within.json')] if len(mine) > 1 else []) + \
        [os.path.join(out_dir, f'vs_{other}.json') for other in pools]
    done = [t for t in targets if os.path.exists(t)]
    if done and not overwrite:
        raise SystemExit(f'{", ".join(done)} exist(s). Use --overwrite to replace.')
    os.makedirs(out_dir, exist_ok=True)

    policies = _load_finals(mine)
    seeds = [cfg['seed'] for _, cfg in mine]
    common = {'round': name, 'env': mine[0][1]['env'], 'method': mine[0][1]['method'],
              'episodes': episodes, 'seats': [seat0, seat1], 'seeds': seeds}
    results = {}

    if len(mine) > 1:
        n = len(mine)
        matrix = np.zeros((n, n, 2))   # [seed of seat 0, seed of seat 1, seat]
        for i in range(n):
            for j in range(n):
                matrix[i, j] = _pairing(env, policies[i], policies[j], seat0, seat1, episodes,
                                        [seeds[i], seeds[j], 3])
        per_seed = {}
        for i, seed in enumerate(seeds):
            rest = [j for j in range(n) if j != i]
            as0, as1 = float(matrix[i, rest, 0].mean()), float(matrix[rest, i, 1].mean())
            per_seed[seed] = {f'{seat0}_vs_others': as0, f'{seat1}_vs_others': as1,
                              'mean_vs_others': (as0 + as1) / 2,
                              'self_play': matrix[i, i].tolist()}
        results['within'] = {**common, 'matrix': matrix.tolist(), 'per_seed': per_seed,
                             'summary': _summary(r['mean_vs_others'] for r in per_seed.values())}
        print(f'[crossplay] {name} within: {_fmt(results["within"]["summary"])}', flush=True)

    for other, theirs in pools.items():
        their_policies = _load_finals(theirs)
        their_seeds = [cfg['seed'] for _, cfg in theirs]
        n, m = len(mine), len(theirs)
        as_seat0 = np.zeros((n, m, 2))   # [my seed, their seed, seat]: me as seat 0
        as_seat1 = np.zeros((m, n, 2))   # [their seed, my seed, seat]: me as seat 1
        for i in range(n):
            for k in range(m):
                as_seat0[i, k] = _pairing(env, policies[i], their_policies[k], seat0, seat1,
                                          episodes, [seeds[i], their_seeds[k], 4])
                as_seat1[k, i] = _pairing(env, their_policies[k], policies[i], seat0, seat1,
                                          episodes, [their_seeds[k], seeds[i], 5])
        per_seed = {}
        for i, seed in enumerate(seeds):
            mine0, mine1 = float(as_seat0[i, :, 0].mean()), float(as_seat1[:, i, 1].mean())
            per_seed[seed] = {f'{seat0}_vs_them': mine0, f'{seat1}_vs_them': mine1,
                              'mean_vs_them': (mine0 + mine1) / 2,
                              'their_return': float((as_seat0[i, :, 1].mean()
                                                     + as_seat1[:, i, 0].mean()) / 2)}
        key = f'vs_{other}'
        results[key] = {**common, 'against': other, 'against_method': theirs[0][1]['method'],
                        'against_seeds': their_seeds, 'as_seat0': as_seat0.tolist(),
                        'as_seat1': as_seat1.tolist(), 'per_seed': per_seed,
                        'summary': _summary(r['mean_vs_them'] for r in per_seed.values())}
        print(f'[crossplay] {name} vs {other}: {_fmt(results[key]["summary"])}', flush=True)

    for key, result in results.items():
        path = os.path.join(out_dir, f'{key}.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
    print(f'[crossplay] -> {out_dir}')
    return results


def _fmt(summary):
    return f'mean return {summary["mean"]:.3f} +- {summary["std"]:.3f} over {summary["n"]} seed(s)'


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('exploit', 'crossplay'):
        print(__doc__)
        raise SystemExit(2)
    if argv[0] == 'exploit':
        if len(argv) < 2:
            raise SystemExit('usage: python posthoc.py exploit <run_dir> [--checkpoint WHICH] '
                             '[config flags]')
        rest, which = [], FINAL_CHECKPOINT
        tail = iter(argv[2:])
        for a in tail:
            if a == '--checkpoint':
                which = next(tail, None)
                if which is None:
                    raise SystemExit('--checkpoint needs a value.')
            elif a.startswith('--checkpoint='):
                which = a.split('=', 1)[1]
            else:
                rest.append(a)
        return exploit(argv[1], rest, which)
    parser = argparse.ArgumentParser(prog='posthoc.py crossplay')
    parser.add_argument('round', help='The round: its directory, or its name in results/.')
    parser.add_argument('--against', nargs='+', default=[],
                        help='Other rounds (directories, or names next to the round).')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per pairing.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv[1:])
    return crossplay(args.round, args.against, args.episodes, args.overwrite)


if __name__ == '__main__':
    main()
