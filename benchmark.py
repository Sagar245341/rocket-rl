"""Paired, reproducible controller benchmarks and robustness experiments."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from checkpoints import load_checkpoint
from rocket_env import ENV_VERSION, PhysicsConfig, RewardConfig, RocketEnv, guidance

ROOT = Path(__file__).resolve().parent


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['phase'],row['scenario'],row['controller']),[]).append(row)
    summaries = []
    for (phase,scenario,controller), group in sorted(groups.items()):
        n = len(group)
        successes = [r for r in group if r['outcome']=='LANDED']
        rate = len(successes)/n
        z = 1.959963984540054
        center = (rate+z*z/(2*n))/(1+z*z/n)
        half = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        summaries.append(dict(phase=phase,scenario=scenario,controller=controller,episodes=n,
            successes=len(successes),success_rate=rate,success_ci_low=center-half,success_ci_high=center+half,
            mean_reward=float(np.mean([r['reward'] for r in group])),
            std_reward=float(np.std([r['reward'] for r in group])),
            mean_fuel=float(np.mean([r['fuel_used'] for r in group])),
            successful_mean_fuel=float(np.mean([r['fuel_used'] for r in successes])) if successes else None,
            mean_duration=float(np.mean([r['duration_seconds'] for r in group]))))
    return summaries


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/benchmark.json')
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--phases',nargs='+',type=int,choices=(1,2,3))
    parser.add_argument('--controllers',nargs='+',choices=('ppo','guidance','random'),default=['ppo','guidance','random'])
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/benchmark')
    parser.add_argument('--update-readme',action='store_true')
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        config['episodes'] = args.episodes if args.episodes is not None else config['episodes']
        config['phases'] = args.phases or config['phases']
        if type(config['episodes']) is not int or config['episodes'] <= 0 or config['seed'] < 0:
            raise ValueError('episodes must be positive and seed nonnegative')
        if not config['phases'] or any(phase not in (1,2,3) for phase in config['phases']):
            raise ValueError('phases must contain one or more values from 1, 2, 3')
        if config['action_seed'] < 0:
            raise ValueError('action_seed must be nonnegative')
        scenarios = {name:PhysicsConfig(**values) for name,values in config['scenarios'].items()}
        if 'nominal' not in scenarios:
            raise ValueError('A nominal scenario is required')
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.error(f'Invalid benchmark configuration: {error}')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    rows, checkpoints = [], {}
    for phase in config['phases']:
        model, rewards = None, RewardConfig()
        if 'ppo' in args.controllers:
            path = args.model_dir/f'rocket_phase{phase}.zip'
            try:
                model,rewards = load_checkpoint(path,phase)
            except ValueError as error:
                parser.error(str(error))
            checkpoints[str(phase)] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8')))
        for scenario,physics in scenarios.items():
            if phase != 3 and scenario != 'nominal':
                continue
            for controller in args.controllers:
                env = RocketEnv(phase,rewards=rewards,physics=physics)
                try:
                    for seed in range(config['seed'],config['seed']+config['episodes']):
                        obs,_ = env.reset(seed=seed)
                        # Random policy RNG is independent of environment/sensor RNG.
                        action_rng = np.random.default_rng(np.random.SeedSequence([config['action_seed'],phase,seed]))
                        score = 0.0
                        while True:
                            if controller == 'ppo':
                                assert model is not None
                                action,_ = model.predict(obs,deterministic=True)
                            elif controller == 'guidance':
                                action = guidance(obs,phase)
                            else:
                                action = action_rng.uniform(-1,1,3).astype(np.float32)
                            obs,reward,terminated,truncated,info = env.step(action)
                            score += reward
                            if terminated or truncated:
                                rows.append(dict(phase=phase,scenario=scenario,controller=controller,seed=seed,
                                    reward=score,outcome=info['outcome'],fuel_used=info['fuel_used'],
                                    duration_seconds=env.steps*env.dt,steps=env.steps,
                                    touchdown_vy=env.vy,touchdown_vx=env.vx,pad_offset=env.x,
                                    **asdict(physics),landing_reward=rewards.landing,
                                    crash_penalty=rewards.crash,fuel_penalty=rewards.fuel))
                                break
                finally:
                    env.close()
                latest = summarize([r for r in rows if r['phase']==phase and r['scenario']==scenario and r['controller']==controller])[0]
                print(f"Phase {phase} / {scenario} / {controller}: {latest['successes']}/{latest['episodes']} safe, reward {latest['mean_reward']:+.1f}",flush=True)
                write_csv(args.output/'episodes.csv',rows)
    summaries = summarize(rows)
    write_csv(args.output/'summary.csv',summaries)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    manifest = dict(environment=ENV_VERSION,config=config,controllers=args.controllers,
        checkpoints=checkpoints,elapsed_seconds=round(time.perf_counter()-started,2),
        python=platform.python_version(),platform=platform.platform(),
        versions={name:importlib.metadata.version(name) for name in ('numpy','gymnasium','stable-baselines3','torch')},
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                       ('rocket_env.py','benchmark.py','train.py','checkpoints.py')})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from report import build_report
    build_report(args.output,args.model_dir,args.update_readme)
    print(f'Benchmark saved: {args.output.resolve()}',flush=True)


if __name__ == '__main__':
    main()
