"""Evaluate custom rocket guidance or PPO on fixed consecutive seeds."""
import argparse
import csv
from pathlib import Path
import numpy as np
from checkpoints import load_checkpoint
from rocket_env import RocketEnv, RewardConfig, SimulationClosed, guidance

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--controller',choices=('guidance','ppo','random'),default=None)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--csv', metavar='FILE', type=Path,
                        help='Save completed flights to a CSV file, e.g. results/evaluation.csv')
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error('--episodes must be positive')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if args.controller in ('guidance', 'random') and args.model:
        parser.error('--model requires PPO control; omit --controller or use --controller ppo.')
    args.controller = args.controller or ('ppo' if args.model else 'guidance')
    model, rewards = None, RewardConfig()
    if args.controller == 'ppo':
        path = args.model or ROOT / f'models/rocket_phase{args.phase}.zip'
        try:
            model, rewards = load_checkpoint(path, args.phase)
        except ValueError as error:
            parser.error(str(error))
        import torch
        torch.set_num_threads(1)
    csv_file = None
    if args.csv:
        try:
            args.csv = args.csv.expanduser()
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            # Check the destination before starting potentially lengthy visual flights.
            csv_file = args.csv.open('w', newline='', encoding='utf-8')
        except OSError as error:
            parser.error(f'Cannot write CSV to {args.csv}: {error}')
    env = RocketEnv(phase=args.phase,rewards=rewards,render_mode=None if args.headless else 'human')
    rows = []
    writer = None
    try:
        if csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['seed','phase','controller','reward','outcome','fuel_used','steps'])
            writer.writeheader()
        for episode in range(args.episodes):
            seed = args.seed + episode
            obs, info = env.reset(seed=seed)
            action_rng = np.random.default_rng(np.random.SeedSequence([98765,args.phase,seed]))
            done, score = False, 0.0
            while not done:
                if model is not None:
                    action, _ = model.predict(obs,deterministic=True)
                elif args.controller == 'random':
                    action = action_rng.uniform(-1,1,3).astype(np.float32)
                else:
                    action = guidance(obs,args.phase)
                obs,reward,terminated,truncated,info = env.step(action)
                score += reward
                done = terminated or truncated
            rows.append(dict(seed=seed,phase=args.phase,controller=args.controller,reward=round(score,3),
                             outcome=info['outcome'],fuel_used=round(info['fuel_used'],3),steps=env.steps))
            if writer is not None and csv_file is not None:
                writer.writerow(rows[-1])
                csv_file.flush()
            print(f"Seed {seed}: {score:+.2f} | {info['outcome']} | fuel {info['fuel_used']:.1f}")
    except (SimulationClosed, KeyboardInterrupt):
        print('Evaluation stopped. Only completed flights are included.')
    except OSError as error:
        parser.error(f'Cannot save evaluation CSV: {error}')
    finally:
        env.close()
        if csv_file:
            csv_file.close()
    if not rows:
        print('No completed flights to summarize.')
        return
    scores = [r['reward'] for r in rows]
    print(f'Mean reward: {np.mean(scores):.2f} +/- {np.std(scores):.2f}')
    print(f"Safe landings: {sum(r['outcome'] == 'LANDED' for r in rows)}/{len(rows)}")
    if args.csv:
        print(f'CSV saved: {args.csv.resolve()}')


if __name__ == '__main__':
    main()
