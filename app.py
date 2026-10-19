"""Interactive custom rocket simulator. Run: python app.py."""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stable_baselines3 import PPO

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import numpy as np
import pygame
from rocket_env import RocketEnv, RewardConfig, guidance
from rocket_visuals import draw_world
from checkpoints import load_checkpoint

ROOT = Path(__file__).resolve().parent
BG, PANEL, MUTED, WHITE, TEAL = '#080f1d', '#111f31', '#8297ae', '#edf4fa', '#52e0c4'
DESIGN_SIZE = (1400, 970)


def viewport_for(size):
    """Fit the dashboard inside the display without stretching it."""
    scale = min(size[0] / DESIGN_SIZE[0], size[1] / DESIGN_SIZE[1])
    width, height = max(1, round(DESIGN_SIZE[0] * scale)), max(1, round(DESIGN_SIZE[1] * scale))
    return pygame.Rect((size[0] - width) // 2, (size[1] - height) // 2, width, height)


def dashboard_event(event, viewport):
    """Translate mouse positions into dashboard coordinates after window resizing."""
    if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
        return event
    attributes = event.dict.copy()
    if viewport.collidepoint(event.pos):
        attributes['pos'] = (round((event.pos[0] - viewport.x) * DESIGN_SIZE[0] / viewport.w),
                             round((event.pos[1] - viewport.y) * DESIGN_SIZE[1] / viewport.h))
    else:
        attributes['pos'] = (-100, -100)
    return pygame.event.Event(event.type, attributes)


class Slider:
    def __init__(self, label: str, low: float, high: float, value: float, y: int):
        self.label = label
        self.low: float = float(low)
        self.high: float = float(high)
        self.value: float = float(value)
        self.rect = pygame.Rect(1000, y + 33, 350, 16)
        self.y = y
        self.dragging = False

    def event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.dragging = self.rect.inflate(14, 18).collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if self.dragging and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            fraction = np.clip((event.pos[0] - self.rect.x) / self.rect.w, 0, 1)
            self.value = round(float(self.low + fraction * (self.high - self.low)))

    def draw(self, screen, text):
        text(self.label, 1000, self.y, MUTED)
        text(f'{self.value:+.0f}', 1295, self.y, TEAL)
        pygame.draw.rect(screen, '#293b50', self.rect.move(0, 6).inflate(0, -12), border_radius=2)
        x = self.rect.x + int((self.value-self.low)/(self.high-self.low)*self.rect.w)
        pygame.draw.line(screen, TEAL, (self.rect.x,self.rect.centery), (x,self.rect.centery), 4)
        pygame.draw.circle(screen, '#e5fff8', (x,self.rect.centery), 7)


def main():
    pygame.init()
    desktop = pygame.display.Info()
    scale = min(1.0, max(1, desktop.current_w - 80) / DESIGN_SIZE[0],
                max(1, desktop.current_h - 100) / DESIGN_SIZE[1])
    display = pygame.display.set_mode(tuple(max(1, round(value * scale)) for value in DESIGN_SIZE), pygame.RESIZABLE)
    screen = pygame.Surface(DESIGN_SIZE)
    pygame.display.set_caption('APOGEE | Rocket Landing Lab')
    fonts = {size: pygame.font.SysFont('segoeui', size, bold=size >= 28) for size in (14,16,18,22,28,38)}
    clock = pygame.time.Clock()
    phase, seed, speed, mode = 1, 42, 1, 'GUIDANCE'
    env = RocketEnv(phase=phase)
    model: PPO | None = None
    results, history = [], []
    message = 'Reference guidance is active. Train PPO to unlock learned control.'
    sliders = [Slider('Safe landing reward',0,300,100,525),
               Slider('Crash penalty',-300,0,-100,595), Slider('Fuel penalty / unit',-10,0,-2,665)]
    buttons = []
    for i, (label,key) in enumerate([('Pause  P',pygame.K_p),('Retry  R',pygame.K_r),('New seed  N',pygame.K_n),
                                   ('Control  M',pygame.K_m),('Speed  F',pygame.K_f),('Export  X',pygame.K_x)]):
        buttons.append((pygame.Rect(28+i*150,718,138,40),label,key))
    for i in range(3):
        buttons.append((pygame.Rect(28+i*300,102,286,38),f'0{i+1}  '+['VERTICAL','PRECISION','ROBUSTNESS'][i],pygame.K_1+i))
    apply_rect = pygame.Rect(1000,851,350,38)
    throttle = 0.0

    def reset():
        nonlocal history, throttle
        history, throttle = [], 0.0
        obs, _ = env.reset(seed=seed)
        return obs, 0.0, False, False

    def load_model():
        nonlocal model, message
        model = None
        path = ROOT / f'models/rocket_phase{phase}.zip'
        if not path.exists():
            message = f'No PPO checkpoint for phase {phase}. Reference guidance and manual control are available.'
            return
        try:
            model, env.rewards = load_checkpoint(path, phase)
            for slider, value in zip(sliders, (env.rewards.landing, env.rewards.crash, env.rewards.fuel)):
                slider.value = value
                slider.low = min(slider.low, value)
                slider.high = max(slider.high, value)
            message = f'PPO checkpoint available for phase {phase}. M changes control.'
        except ValueError as error:
            model = None
            message = f'Could not load compatible checkpoint: {error}'

    load_model()
    obs, score, finished, paused = reset()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    display = pygame.display.set_mode((max(1,event.w), max(1,event.h)), pygame.RESIZABLE)
                event = dashboard_event(event, viewport_for(display.get_size()))
                for slider in sliders:
                    slider.event(event)
                key = event.key if event.type == pygame.KEYDOWN else None
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, _, command in buttons:
                        if rect.collidepoint(event.pos):
                            key = command
                    if apply_rect.collidepoint(event.pos):
                        env.rewards = RewardConfig(*(s.value for s in sliders))
                        obs, score, finished, paused = reset()
                        message = 'Rewards applied to this simulation. Existing PPO weights are unchanged.'
                if key == pygame.K_ESCAPE:
                    running = False
                elif key == pygame.K_p:
                    paused = not paused
                elif key in (pygame.K_r, pygame.K_n):
                    seed += int(key == pygame.K_n)
                    obs, score, finished, paused = reset()
                elif key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    phase = key - pygame.K_1 + 1
                    rewards = env.rewards
                    env.close()
                    env = RocketEnv(phase=phase, rewards=rewards)
                    mode = 'GUIDANCE'
                    load_model()
                    obs, score, finished, paused = reset()
                elif key == pygame.K_m:
                    modes = ['GUIDANCE','MANUAL'] + (['PPO'] if model else [])
                    mode = modes[(modes.index(mode)+1) % len(modes)]
                    obs, score, finished, paused = reset()
                    message = f'{mode} control. '+('Hold W/S to adjust throttle.' if mode == 'MANUAL' else 'Same seed restarted.')
                elif key == pygame.K_f:
                    speed = {1:2,2:4,4:1}[speed]
                elif key == pygame.K_x:
                    if results:
                        try:
                            folder = ROOT / 'results'
                            folder.mkdir(parents=True, exist_ok=True)
                            path = folder / f'rocket_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
                            with path.open('w', newline='', encoding='utf-8') as file:
                                writer = csv.DictWriter(file, fieldnames=list(results[0]))
                                writer.writeheader()
                                writer.writerows(results)
                            message = f'Exported {len(results)} flights to results/{path.name}'
                        except OSError as error:
                            message = f'Could not export flights: {error}'
                    else:
                        message = 'Complete a flight before exporting.'
            if running and not finished and not paused:
                for _ in range(1 if mode == 'MANUAL' else speed):
                    if mode == 'PPO' and model is not None:
                        action, _ = model.predict(obs, deterministic=True)
                    elif mode == 'GUIDANCE':
                        action = guidance(obs, phase)
                    else:
                        keys = pygame.key.get_pressed()
                        throttle = float(np.clip(throttle + .015*(int(keys[pygame.K_w] or keys[pygame.K_UP])-int(keys[pygame.K_s] or keys[pygame.K_DOWN])),0,1))
                        if keys[pygame.K_SPACE]:
                            throttle = 0
                        turn = int(keys[pygame.K_d] or keys[pygame.K_RIGHT])-int(keys[pygame.K_a] or keys[pygame.K_LEFT])
                        lateral = int(keys[pygame.K_e])-int(keys[pygame.K_q])
                        action = np.array([2*throttle-1, turn, lateral], dtype=np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    score += reward
                    if env.steps % 5 == 0:
                        history.append((env.y,env.vy))
                    finished = terminated or truncated
                    if finished:
                        results.append(dict(phase=phase,mode=mode,seed=seed,reward=round(score,2),
                                            steps=env.steps,outcome=info['outcome'],fuel_used=round(info['fuel_used'],2),
                                            landing_reward=env.rewards.landing,crash_penalty=env.rewards.crash,
                                            fuel_penalty=env.rewards.fuel))
                        message = f"{info['outcome']} | reward {score:+.1f} | R retries the same scenario."
                        break
            screen.fill(BG)

            def text(value,x,y,color=WHITE,size=18):
                screen.blit(fonts[size].render(str(value),True,color),(x,y))

            def panel(rect):
                pygame.draw.rect(screen,PANEL,rect,border_radius=12)
                pygame.draw.rect(screen,'#20334a',rect,1,border_radius=12)

            text('A P O G E E',28,18,WHITE,38)
            text('REUSABLE ROCKET / LANDING LAB',300,37,MUTED,16)
            text(f'PHASE 0{phase}   /   SEED {seed}   /   {mode}',970,27,TEAL,18)
            text('SIMPLIFIED 2D PHYSICS   |   EARTH GRAVITY',970,58,MUTED,14)
            screen.blit(draw_world(env, (900,550)), (28,154))
            text(f'{env.outcome if finished else "PAUSED" if paused else "LIVE FLIGHT"}    {1 if mode == "MANUAL" else speed}x',48,171,TEAL,16)
            text(f'T+ {env.steps*env.dt:05.1f} s',785,171,WHITE,16)
            panel(pygame.Rect(970,102,405,365))
            text('FLIGHT TELEMETRY',995,122,TEAL,16)
            text(f'{env.y:05.1f}',995,150,WHITE,38)
            text('METERS / ALTITUDE',1170,169,MUTED,14)
            metrics = [('Vertical speed',f'{env.vy:+.1f} m/s'),('Pad offset',f'{env.x:+.1f} m'),
                       ('Lateral speed',f'{env.vx:+.1f} m/s'),('Tilt',f'{np.degrees(env.angle):+.1f} deg'),
                       ('Wind acceleration',f'{env.wind:+.2f} m/s2')]
            for i,(label,value) in enumerate(metrics):
                text(label,995,205+i*29,MUTED,16)
                text(value,1230,205+i*29,WHITE,16)
            for y,label,fraction,color in [(367,'FUEL',env.fuel/env.initial_fuel,TEAL),(409,'THRUST',env.throttle,'#ffbd69')]:
                text(label,995,y,MUTED,14)
                text(f'{fraction*100:3.0f}%',1290,y,WHITE,14)
                pygame.draw.rect(screen,'#293b50',(1060,y+6,215,7),border_radius=3)
                pygame.draw.rect(screen,color,(1060,y+6,int(215*fraction),7),border_radius=3)
            panel(pygame.Rect(970,483,405,420))
            text('REWARD DESIGN',995,490,TEAL,16)
            for slider in sliders:
                slider.draw(screen,text)
            landing,crash,fuel = [s.value for s in sliders]
            text('ILLUSTRATION / 8 FUEL UNITS USED',1000,720,MUTED,14)
            text(f'Landing  {landing+8*fuel:+.0f}',1000,746,TEAL,28)
            text(f'Crash     {crash+8*fuel:+.0f}',1000,782,'#ff9b8c',28)
            text('Example excludes flight shaping rewards.',1000,825,MUTED,14)
            pygame.draw.rect(screen,'#22463f',apply_rect,border_radius=6)
            text('Apply to simulation + restart',1024,860,TEAL,18)
            # Altitude history with a consistent 0..260 meter scale.
            panel(pygame.Rect(28,776,900,97))
            text('ALTITUDE HISTORY',45,786,MUTED,14)
            text(f'RETURN  {score:+.1f}',720,786,TEAL,16)
            if len(history)>1:
                points = [(48+i*860/max(1,len(history)-1),860-np.clip(row[0]/260,0,1)*48) for i,row in enumerate(history)]
                pygame.draw.lines(screen,TEAL,False,points,2)
            for rect,label,key in buttons:
                active = key in (pygame.K_1,pygame.K_2,pygame.K_3) and phase == key-pygame.K_1+1
                pygame.draw.rect(screen,'#22463f' if active else '#1c2e44',rect,border_radius=7)
                text(label,rect.x+12,rect.y+9,TEAL if active else WHITE,16)
            text('W/S: throttle   A/D: tilt   Q/E: lateral   Space: cut engine   1/2/3: phase',28,886,MUTED,16)
            text('Sliders do not retrain PPO. Apply changes scoring; training uses CLI reward settings.',28,912,MUTED,14)
            text(message[:140],28,933,TEAL,14)
            viewport = viewport_for(display.get_size())
            display.fill(BG)
            display.blit(pygame.transform.smoothscale(screen, viewport.size), viewport)
            pygame.display.flip()
            clock.tick(50)
    finally:
        env.close()
        pygame.quit()


if __name__ == '__main__':
    main()


