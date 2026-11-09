"""
Post-hoc metrics on finished runs, from the final_policies.pkl main.py saves.

    python posthoc.py exploit   <run_dir> [config flags]
    python posthoc.py crossplay <dir> [<dir> ...] [--episodes N] [--out file.json]

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
    protocol. Output: <run_dir>/exploit.json.

crossplay -- returns against co-players never trained with (two-player games)
    Every run's final seat-0 policy plays every run's final seat-1 policy for
    --episodes episodes. For each run: the mean return of its seat 0 against
    the OTHER runs' seat 1, of its seat 1 against the other runs' seat 0, and
    their mean; then mean and std over the runs of each method. Output: the
    full matrix and the summary, in --out (default <first dir>/crossplay.json).

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

from config import parse_args, parse_env
from main import load_policies
from sepsro.budget import BudgetTracker
from sepsro.envs import make_env
from sepsro.rollout import play_episode


def _read_config(run_dir):
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as handle:
        return json.load(handle)


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
def exploit(run_dir, argv):
    import torch

    from sepsro.learners.ppo import PPOConfig, PPOLearner

    cfg = _read_config(run_dir)
    out_path = os.path.join(run_dir, 'exploit.json')
    # The run's env and seed, this protocol's settings (+ ENV_OVERRIDES, + flags).
    args = parse_args(['--env', cfg['env'], '--seed', str(cfg['seed']), *argv])
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f'{out_path} exists. Use --overwrite to replace it.')

    policies = load_policies(os.path.join(run_dir, 'final_policies.pkl'))
    env = _env_for(cfg)
    agents = env.spec.agents
    rng = np.random.default_rng([cfg['seed'], 2])
    torch.manual_seed(cfg['seed'])
    ppo = PPOConfig.from_args(args)
    print(f'[exploit] {run_dir}: {cfg["method"]} on {cfg["env"]}, '
          f'{args.exploit_steps} PPO steps per seat', flush=True)

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
              'approx_nash_conv': float(sum(s['gain'] for s in seats.values())),
              # Standard error from the evaluation episodes only (the best
              # response's own training randomness is not in it).
              'approx_nash_conv_se': float(np.sqrt(sum(s['gain_se'] ** 2 for s in seats.values()))),
              'exact_nash_conv': _exact_nash_conv(env, policies, rng),
              'seats': seats, 'exploit_steps': args.exploit_steps,
              'eval_episodes': args.exploit_eval_episodes, 'ppo': vars(ppo),
              'seconds': time.time() - start}
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    exact = result['exact_nash_conv']
    print(f'[exploit] approx NashConv {result["approx_nash_conv"]:.4f}'
          f' +- {result["approx_nash_conv_se"]:.4f}'
          + (f'  (exact {exact:.4f})' if exact is not None else '') + f'  -> {out_path}',
          flush=True)
    return result


# ------------------------------------------------------------------- crossplay
def _find_runs(paths):
    runs = []
    for path in paths:
        hits = ([path] if os.path.exists(os.path.join(path, 'final_policies.pkl')) else
                sorted(os.path.dirname(f) for f in
                       glob.glob(os.path.join(path, '**', 'final_policies.pkl'), recursive=True)))
        runs += hits
    if len(runs) < 2:
        raise SystemExit(f'Cross-play needs at least two finished runs; found {len(runs)} in {paths}.')
    return runs


def crossplay(paths, episodes, out=None):
    runs = _find_runs(paths)
    configs = [_read_config(r) for r in runs]
    envs = {c['env'] for c in configs}
    if len(envs) != 1:
        raise SystemExit(f'Runs come from different envs: {sorted(envs)}.')
    env = _env_for(configs[0])
    if env.spec.num_agents != 2:
        raise SystemExit('Cross-play is defined here for two-player games.')
    seat0, seat1 = env.spec.agents
    policies = [load_policies(os.path.join(r, 'final_policies.pkl')) for r in runs]
    root = os.path.commonpath(runs)
    several_exps = len({c.get('exp_name') for c in configs}) > 1
    labels = [(f'{c.get("exp_name")}/' if several_exps else '') + f'{c["method"]}/seed{c["seed"]}'
              for c in configs]
    if len(set(labels)) < len(labels):   # e.g. two arms of one method: fall back to paths
        labels = [os.path.relpath(r, root) for r in runs]

    n = len(runs)
    matrix = np.zeros((n, n, 2))   # [run of seat 0, run of seat 1, seat]
    for i in range(n):
        for j in range(n):
            joint = {seat0: policies[i][seat0], seat1: policies[j][seat1]}
            values = _returns(env, joint, np.random.default_rng([i, j, 3]), episodes)
            matrix[i, j] = values[seat0][0], values[seat1][0]
        print(f'[crossplay] {labels[i]} done ({i + 1}/{n})', flush=True)

    per_run = {}
    for i, label in enumerate(labels):
        others = [j for j in range(n) if j != i]
        as0 = float(matrix[i, others, 0].mean())
        as1 = float(matrix[others, i, 1].mean())
        per_run[label] = {'method': configs[i]['method'], 'seed': configs[i]['seed'],
                          f'{seat0}_vs_others': as0, f'{seat1}_vs_others': as1,
                          'mean_vs_others': (as0 + as1) / 2,
                          'self_play': matrix[i, i].tolist()}
    per_method = {}
    for method in sorted({c['method'] for c in configs}):
        values = [r['mean_vs_others'] for r in per_run.values() if r['method'] == method]
        per_method[method] = {'mean': float(np.mean(values)),
                              'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                              'runs': len(values)}

    print(f'\n{"run":40s} {seat0 + " vs others":>18s} {seat1 + " vs others":>18s} {"mean":>10s}')
    for label, r in per_run.items():
        print(f'{label:40s} {r[f"{seat0}_vs_others"]:18.2f} {r[f"{seat1}_vs_others"]:18.2f} '
              f'{r["mean_vs_others"]:10.2f}')
    print()
    for method, r in per_method.items():
        print(f'{method:12s} mean vs others {r["mean"]:.2f} +- {r["std"]:.2f} ({r["runs"]} runs)')

    out = out or os.path.join(paths[0], 'crossplay.json')
    with open(out, 'w', encoding='utf-8') as handle:
        json.dump({'env': configs[0]['env'], 'episodes': episodes, 'seats': [seat0, seat1],
                   'runs': labels, 'root': root, 'matrix': matrix.tolist(),
                   'per_run': per_run, 'per_method': per_method}, handle, indent=2)
    print(f'[crossplay] -> {out}')
    return per_run, per_method


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('exploit', 'crossplay'):
        print(__doc__)
        raise SystemExit(2)
    if argv[0] == 'exploit':
        if len(argv) < 2:
            raise SystemExit('usage: python posthoc.py exploit <run_dir> [config flags]')
        return exploit(argv[1], argv[2:])
    parser = argparse.ArgumentParser(prog='posthoc.py crossplay')
    parser.add_argument('paths', nargs='+', help='Run directories, or directories holding runs.')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per pairing.')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv[1:])
    return crossplay(args.paths, args.episodes, args.out)


if __name__ == '__main__':
    main()
