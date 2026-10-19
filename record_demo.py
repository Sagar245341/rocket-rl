"""Record a reproducible 75-second portfolio demo without desktop capture."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import numpy as np
import pygame
import torch
from checkpoints import load_checkpoint
from rocket_env import RocketEnv, guidance
from rocket_visuals import draw_world

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--seed',type=int,default=10000)
    parser.add_argument('--duration',type=int,default=75)
    parser.add_argument('--fps',type=int,default=25)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/apogee_demo.mp4')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models')
    args = parser.parse_args()
    if args.duration < 30 or args.fps <= 0 or args.seed < 0:
        parser.error('duration must be >=30 seconds, fps positive, and seed nonnegative')
    try:
        import cv2
    except ImportError:
        parser.error('Install optional recording dependencies: python -m pip install -r requirements-demo.txt')
    try:
        model,rewards = load_checkpoint(args.model_dir/f'rocket_phase{args.phase}.zip',args.phase)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    pygame.font.init()
    font = pygame.font.SysFont('segoeui',22)
    heading = pygame.font.SysFont('segoeui',36,bold=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    writer = cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(900,690))
    if not writer.isOpened():
        parser.error('Could not open MP4 encoder/output. Choose a writable .mp4 path.')
    records = []
    frames_written = 0

    def encode(surface):
        nonlocal frames_written
        rgb = np.transpose(pygame.surfarray.array3d(surface),(1,0,2))
        writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        frames_written += 1

    def title(lines):
        frame = pygame.Surface((900,690))
        frame.fill('#0b1424')
        for i,line in enumerate(lines):
            frame.blit((heading if i==0 else font).render(line,True,'#54cbbb' if i==0 else '#d5e6f6'),(55,160+i*62))
        return frame

    try:
        intro = title(['APOGEE / ROCKET LANDING LAB','Custom Gymnasium physics + PPO training',
                       'Reference guidance / learned policy / random baseline',
                       f'Phase {args.phase} | paired seed {args.seed}',
                       'Flight replay is time-compressed. Outcomes are unedited.'])
        for _ in range(5*args.fps):
            encode(intro)
        scene_budget = (args.duration-10)*args.fps
        for scene_index,controller in enumerate(('guidance','ppo','random')):
            env = RocketEnv(args.phase,rewards=rewards)
            actions,score = [],0.0
            obs,_ = env.reset(seed=args.seed)
            rng = np.random.default_rng(np.random.SeedSequence([98765,args.phase,args.seed]))
            while True:
                if controller=='ppo':
                    action,_ = model.predict(obs,deterministic=True)
                elif controller=='guidance':
                    action = guidance(obs,args.phase)
                else:
                    action = rng.uniform(-1,1,3).astype(np.float32)
                actions.append(np.asarray(action).copy())
                obs,reward,done,truncated,info = env.step(action)
                score += reward
                if done or truncated:
                    break
            records.append(dict(controller=controller,phase=args.phase,seed=args.seed,outcome=info['outcome'],
                                reward=score,fuel_used=info['fuel_used'],steps=env.steps))
            count = scene_budget//3 + int(scene_index < scene_budget%3)
            flight_frames = count-3*args.fps
            indices = np.linspace(0,len(actions)-1,flight_frames).astype(int)
            env.reset(seed=args.seed)
            step = 0
            last = None
            for index in indices:
                while step <= index:
                    env.step(actions[step])
                    step += 1
                frame = pygame.Surface((900,690))
                frame.blit(draw_world(env),(0,0))
                pygame.draw.rect(frame,'#101e30',(0,590,900,100))
                frame.blit(heading.render(f'{controller.upper()} / PHASE {args.phase}',True,'#54cbbb'),(24,596))
                label = f'Altitude {env.y:.1f}m | speed {env.vy:+.1f}m/s | fuel {env.fuel:.1f} | seed {args.seed}'
                frame.blit(font.render(label,True,'#d5e6f6'),(24,642))
                frame.blit(font.render('TIME-COMPRESSED REPLAY',True,'#b7c8dc'),(560,18))
                encode(frame)
                last = frame
            assert last is not None
            overlay = pygame.Surface((900,130))
            overlay.fill('#101e30')
            overlay.blit(heading.render(info['outcome'],True,'#54cbbb' if info['is_success'] else '#ff9b8c'),(30,15))
            overlay.blit(font.render(f'Return {score:+.1f} | fuel used {info["fuel_used"]:.1f}',True,'white'),(30,72))
            last.blit(overlay,(0,210))
            for _ in range(3*args.fps):
                encode(last)
            env.close()
        outro = title(['MEASURE, COMPARE, REPRODUCE','100 held-out episodes per controller / condition',
            'Open docs/benchmark/dashboard.html for measured results',
            'Single demonstration flights do not establish reliability.',
            'Simplified educational 2D physics / not a flight-certified model'])
        for _ in range(5*args.fps):
            encode(outro)
    finally:
        writer.release()
        pygame.font.quit()
    args.output.with_suffix('.json').write_text(json.dumps(dict(fps=args.fps,frames=frames_written,
        duration_seconds=frames_written/args.fps,flights=records),indent=2),encoding='utf-8')
    print(f'Saved {args.output} ({frames_written/args.fps:.1f} seconds)')
    # A small animated thumbnail is easy to preview in a repository README.
    from PIL import Image
    capture = cv2.VideoCapture(str(args.output))
    previews = []
    for second in np.linspace(7,args.duration-7,36):
        capture.set(cv2.CAP_PROP_POS_MSEC,float(second)*1000)
        ok,frame = capture.read()
        if ok:
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            image.thumbnail((450,345))
            previews.append(image)
    capture.release()
    if previews:
        previews[0].save(args.output.with_name('demo_preview.gif'),save_all=True,append_images=previews[1:],duration=160,loop=0)


if __name__ == '__main__':
    main()
