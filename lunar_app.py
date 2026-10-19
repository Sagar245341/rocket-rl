"""Interactive PPO flight demo: python app.py."""
import csv
import os
from datetime import datetime
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent


def main():
    pygame.init()
    screen = pygame.display.set_mode((1100, 740))
    pygame.display.set_caption('Rocket AI | Flight Lab')
    font = pygame.font.SysFont('consolas', 18)
    heading = pygame.font.SysFont('consolas', 30, bold=True)
    clock = pygame.time.Clock()
    path = ROOT / 'models/rocket_ai.zip'
    model = PPO.load(path, device='cpu') if path.exists() else None
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    mode, seed, speed = 'AI' if model else 'MANUAL', 42, 1
    results = []
    message = 'Ready' if model else 'Model missing. Manual flight available.'
    buttons = [(pygame.Rect(24 + i * 180, 648, 168, 38), label, key)
               for i, (label, key) in enumerate([
                   ('Pause [P]', pygame.K_p), ('Restart [R]', pygame.K_r),
                   ('Mode [M]', pygame.K_m), ('Next seed [N]', pygame.K_n),
                   ('Export [E]', pygame.K_e)])]

    def reset():
        observation, _ = env.reset(seed=seed)
        return observation, 0.0, 0, False, False, 0

    obs, score, steps, finished, paused, action = reset()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                command = event.key if event.type == pygame.KEYDOWN else None
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, _, key in buttons:
                        if rect.collidepoint(event.pos):
                            command = key
                if command == pygame.K_ESCAPE:
                    running = False
                elif command == pygame.K_p:
                    paused = not paused
                elif command == pygame.K_r:
                    obs, score, steps, finished, paused, action = reset()
                elif command == pygame.K_n:
                    seed += 1
                    obs, score, steps, finished, paused, action = reset()
                elif command == pygame.K_m:
                    if model:
                        mode = 'MANUAL' if mode == 'AI' else 'AI'
                        obs, score, steps, finished, paused, action = reset()
                    else:
                        message = 'Train a model to unlock AI mode.'
                elif command == pygame.K_s:
                    speed = {1: 2, 2: 4, 4: 1}[speed]
                elif command == pygame.K_e:
                    if results:
                        folder = ROOT / 'results'
                        folder.mkdir(exist_ok=True)
                        output = folder / f'flights_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
                        with output.open('w', newline='', encoding='utf-8') as file:
                            writer = csv.DictWriter(file, fieldnames=['mode', 'seed', 'reward', 'steps', 'outcome'])
                            writer.writeheader()
                            writer.writerows(results)
                        message = f'Saved {output.name}'
                    else:
                        message = 'Finish a flight before exporting.'
            if running and not paused and not finished:
                for _ in range(speed if mode == 'AI' else 1):
                    keys = pygame.key.get_pressed()
                    action = int(model.predict(obs, deterministic=True)[0]) if mode == 'AI' else (
                        2 if keys[pygame.K_UP] or keys[pygame.K_w] else
                        1 if keys[pygame.K_LEFT] or keys[pygame.K_a] else
                        3 if keys[pygame.K_RIGHT] or keys[pygame.K_d] else 0)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    score += float(reward)
                    steps += 1
                    finished = terminated or truncated
                    if finished:
                        outcome = ('TIME LIMIT' if truncated else
                                   'LANDED' if not env.unwrapped.lander.awake else 'CRASH / OUT OF BOUNDS')
                        results.append(dict(mode=mode, seed=seed, reward=round(score, 2), steps=steps, outcome=outcome))
                        message = f'{outcome} | R: retry  N: next seed'
                        break
            screen.fill('#0c1324')

            def text(value, x, y, color='#b5c6dc', large=False):
                screen.blit((heading if large else font).render(str(value), True, color), (x, y))

            text('ROCKET AI / FLIGHT LAB', 24, 20, '#55e5c0', True)
            text(f'PPO AUTOPILOT / {mode} / SEED {seed}', 24, 64)
            frame = env.render()
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(pygame.transform.smoothscale(surface, (720, 480)), (24, 100))
            text('LIVE TELEMETRY', 770, 104, '#55e5c0')
            values = [
                ('Reward', f'{score:+.1f}'), ('Steps', steps),
                ('Position X/Y', f'{obs[0]:+.2f} / {obs[1]:+.2f}'),
                ('Velocity X/Y', f'{obs[2]:+.2f} / {obs[3]:+.2f}'),
                ('Angle (rad)', f'{obs[4]:+.2f}'),
                ('Leg contact', f'L:{int(obs[6])} R:{int(obs[7])}'),
                ('Engine', ['COAST', 'LEFT ENGINE', 'MAIN ENGINE', 'RIGHT ENGINE'][action]),
                ('AI playback', f'{speed}x'),
                ('State', 'FINISHED' if finished else 'PAUSED' if paused else 'FLYING')]
            for i, (label, value) in enumerate(values):
                text(label, 770, 144 + i * 44)
                text(value, 770, 164 + i * 44, '#ffffff')
            text('RECENT FLIGHTS', 24, 592, '#55e5c0')
            recent = ' | '.join(f"{r['mode']} #{r['seed']}: {r['reward']:+.0f}" for r in results[-4:])
            text(recent or 'Complete a flight to record a score.', 24, 619)
            for rect, label, _ in buttons:
                pygame.draw.rect(screen, '#243752', rect, border_radius=6)
                text(label, rect.x + 8, rect.y + 9, '#ffffff')
            text('Fly: arrows / WASD    S: AI speed    Esc: quit', 24, 695)
            text(message, 24, 719, '#55e5c0')
            pygame.display.flip()
            clock.tick(50)
    finally:
        env.close()
        pygame.quit()


if __name__ == '__main__':
    main()
