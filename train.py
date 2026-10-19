"""Reproducible PPO training with held-out validation and checkpoint selection."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rocket_env import ENV_VERSION, RocketEnv, RewardConfig, guidance

ROOT = Path(__file__).resolve().parent


class ValidationCallback(BaseCallback):
    """Select by landing rate, breaking ties with mean reward on validation seeds."""
    def __init__(self, phase, rewards, output, frequency, episodes):
        super().__init__()
        self.phase, self.rewards, self.output = phase, rewards, output
        self.frequency, self.episodes = frequency, episodes
        self.best = (-1.0, -float('inf'))
        self.best_steps = 0
        self.records = []

    def _on_training_start(self):
        self._on_step()

    def _on_step(self):
        if self.num_timesteps % self.frequency:
            return True
        env = RocketEnv(self.phase, rewards=self.rewards)
        scores, successes = [], []
        try:
            for seed in range(5000, 5000+self.episodes):
                obs, _ = env.reset(seed=seed)
                score = 0.0
                while True:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    score += reward
                    if terminated or truncated:
                        scores.append(score)
                        successes.append(info['is_success'])
                        break
        finally:
            env.close()
        rate, mean = float(np.mean(successes)), float(np.mean(scores))
        self.records.append(dict(timesteps=self.num_timesteps, success_rate=rate, mean_reward=mean))
        self.output.with_suffix('.validation.json').write_text(json.dumps(self.records, indent=2),encoding='utf-8')
        self.logger.record('validation/safe_landing_rate',rate)
        self.logger.record('validation/mean_reward',mean)
        if (rate, mean) > self.best:
            self.best = (rate,mean)
            self.best_steps = self.num_timesteps
            self.model.save(self.output)
        print(f'Validation {self.num_timesteps}: safe landings {rate:.0%}, mean reward {mean:+.2f}',flush=True)
        return True


def pretrain_actor(model, phase, rewards, episodes, epochs, augmentation, seed):
    """Optional, explicitly logged behavior-cloning warm start from reference flights."""
    env = RocketEnv(phase,rewards=rewards)
    observations, targets = [], []
    try:
        for seed in range(1000,1000+episodes):
            obs,_ = env.reset(seed=seed)
            while True:
                action = guidance(obs,phase)
                observations.append(obs.copy())
                targets.append(action)
                obs,_,terminated,truncated,_ = env.step(action)
                if terminated or truncated:
                    break
        # Decorrelate previous throttle from the target action. Flight-only data
        # otherwise encourages an identity shortcut and fails off trajectory.
        rng = np.random.default_rng(2000+seed)
        for index in range(augmentation):
            env.x = 0.0 if phase == 1 else float(rng.uniform(-80,80))
            env.y = float(rng.uniform(0,35 if index%2 else 200))
            env.vx = 0.0 if phase == 1 else float(rng.uniform(-12,12))
            env.vy = float(rng.uniform(-25,5))
            env.angle = 0.0 if phase == 1 else float(rng.uniform(-.4,.4))
            env.omega = 0.0 if phase == 1 else float(rng.uniform(-1,1))
            env.fuel = float(rng.uniform(2,40))
            env.throttle = float(rng.uniform(0,1))
            obs = env._observation()
            observations.append(obs)
            targets.append(guidance(obs,phase))
    finally:
        env.close()
    x = torch.as_tensor(np.stack(observations),device=model.device)
    y = torch.as_tensor(np.stack(targets),device=model.device)
    optimizer = torch.optim.Adam(model.policy.parameters(),lr=.001)
    loss_value = 0.0
    for _ in range(epochs):
        indices = torch.randperm(len(x),device=model.device)
        for batch in indices.split(512):
            features = model.policy.extract_features(x[batch])
            assert isinstance(features,torch.Tensor)
            latent = model.policy.mlp_extractor.forward_actor(features)
            prediction = model.policy.action_net(latent)
            loss = torch.nn.functional.mse_loss(prediction,y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
    with torch.no_grad():
        model.policy.log_std.fill_(-2.0)
    print(f'Behavior-cloning warm start: {episodes} reference flights, {len(x)} samples, {epochs} epochs, loss {loss_value:.6f}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/training.json')
    parser.add_argument('--phase',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--timesteps',type=int,default=300000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--learning-rate',type=float,default=.0003)
    parser.add_argument('--gamma',type=float,default=.995)
    parser.add_argument('--landing-reward',type=float,default=100)
    parser.add_argument('--crash-penalty',type=float,default=-100)
    parser.add_argument('--fuel-penalty',type=float,default=-2)
    parser.add_argument('--eval-freq',type=int,default=50000)
    parser.add_argument('--eval-episodes',type=int,default=10)
    parser.add_argument('--pretrain-episodes',type=int,default=0)
    parser.add_argument('--pretrain-epochs',type=int,default=20)
    parser.add_argument('--pretrain-augmentation',type=int,default=0)
    preliminary,_ = parser.parse_known_args()
    try:
        config = json.loads(preliminary.config.read_text(encoding='utf-8'))
        unknown = set(config)-{action.dest for action in parser._actions}
        if unknown:
            raise ValueError(f'Unknown training settings: {sorted(unknown)}')
        parser.set_defaults(**config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(f'Cannot read training configuration: {error}')
    args = parser.parse_args()
    if min(args.timesteps,args.threads,args.eval_freq,args.eval_episodes) <= 0 or args.seed < 0:
        parser.error('Budgets, threads, validation frequency/episodes must be positive; seed must be nonnegative.')
    if not 0 < args.gamma <= 1 or args.learning_rate <= 0:
        parser.error('gamma must be in (0,1] and learning-rate must be positive.')
    if args.pretrain_episodes < 0 or args.pretrain_episodes > 1000 or args.pretrain_epochs <= 0 or args.pretrain_augmentation < 0:
        parser.error('pretrain-episodes must be in [0,1000], pretrain-epochs positive.')
    torch.set_num_threads(args.threads)
    output = (args.output or ROOT/f'models/rocket_phase{args.phase}').with_suffix('.zip')
    output.parent.mkdir(parents=True,exist_ok=True)
    rewards = RewardConfig(args.landing_reward,args.crash_penalty,args.fuel_penalty)
    env = Monitor(RocketEnv(args.phase,rewards=rewards),filename=str(output.with_suffix('.monitor.csv')))
    callback = ValidationCallback(args.phase,rewards,output,args.eval_freq,args.eval_episodes)
    started = time.perf_counter()
    try:
        if args.resume:
            from checkpoints import load_checkpoint
            metadata = json.loads(args.resume.with_suffix('.json').read_text(encoding='utf-8'))
            model,_ = load_checkpoint(args.resume,metadata['phase'])
            model.set_env(env)
            model.set_random_seed(args.seed)
            model.learning_rate = args.learning_rate
            model.lr_schedule = lambda _: args.learning_rate
            model.gamma = args.gamma
            model.rollout_buffer.gamma = args.gamma
            model.tensorboard_log = str(ROOT/'logs')
        else:
            model = PPO('MlpPolicy',env,learning_rate=args.learning_rate,n_steps=2048,batch_size=64,
                        gamma=args.gamma,verbose=0,seed=args.seed,tensorboard_log=str(ROOT/'logs'),device='cpu')
        if args.pretrain_episodes:
            pretrain_actor(model,args.phase,rewards,args.pretrain_episodes,args.pretrain_epochs,args.pretrain_augmentation,args.seed)
        model.learn(total_timesteps=args.timesteps,callback=callback,reset_num_timesteps=True,
                    tb_log_name=f'rocket_phase{args.phase}')
        # Always validate the final policy, including short smoke runs.
        callback.frequency = 1
        callback.num_timesteps = model.num_timesteps
        callback._on_step()
        final_path = output.with_name(output.stem+'_final.zip')
        model.save(final_path)
        metadata = dict(environment=ENV_VERSION,phase=args.phase,seed=args.seed,
            total_timesteps=model.num_timesteps,checkpoint_timesteps=callback.best_steps,
            rewards=asdict(rewards),physics=asdict(RocketEnv(args.phase).physics),
            selection='validation landing rate, then mean reward',validation_seed_start=5000,
            validation_episodes=args.eval_episodes,validation_success_rate=callback.best[0],
            elapsed_seconds=round(time.perf_counter()-started,2),resume=str(args.resume) if args.resume else None,
            method='behavior cloning + PPO' if args.pretrain_episodes else 'PPO',
            demonstration_seed_start=1000 if args.pretrain_episodes else None,
            training_config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
            versions={name:importlib.metadata.version(name) for name in ('gymnasium','stable-baselines3','numpy','torch')})
        output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        final_metadata = dict(metadata,checkpoint_timesteps=model.num_timesteps,selection='final policy')
        final_path.with_suffix('.json').write_text(json.dumps(final_metadata,indent=2),encoding='utf-8')
    finally:
        env.close()
    print(f'Training complete: {output} | selected validation success {callback.best[0]:.0%}',flush=True)


if __name__ == '__main__':
    main()
