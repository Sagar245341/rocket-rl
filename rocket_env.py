"""A simplified 2D reusable rocket landing environment (SI units)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TYPE_CHECKING, TypedDict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pygame

ENV_VERSION = 'RocketEnv-v2'


class SimulationClosed(Exception):
    """The user closed the visual evaluation window or pressed Escape."""


@dataclass
class RewardConfig:
    landing: float = 100.0
    crash: float = -100.0
    fuel: float = -2.0


@dataclass
class PhysicsConfig:
    wind_scale: float = 1.0
    delay_scale: float = 1.0
    noise_scale: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in
               (self.wind_scale, self.delay_scale, self.noise_scale)):
            raise ValueError('Physics scales must be finite and nonnegative')


class RocketInfo(TypedDict):
    outcome: str
    is_success: bool
    fuel_used: float
    phase: int
    rewards: dict[str, float]


class RocketEnv(gym.Env):
    """Continuous throttle, gimbal and side thrusters; simplified rigid-body dynamics.

    Actions: throttle [-1, 1] maps to [0, 1], gimbal [-1, 1], lateral [-1, 1].
    Observations: x, altitude, vx, vy, tilt, angular rate, fuel fraction,
    actual throttle, wind, gravity, max thrust acceleration (normalized).
    Altitude measures landing-foot clearance. Positive tilt points thrust right.
    """
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 50}
    dt = 0.02
    initial_fuel = 40.0
    dry_mass = 1000.0
    pad_half_width = 12.0
    max_steps = 2500

    def __init__(self, phase: int = 1, render_mode: str | None = None,
                 rewards: RewardConfig | None = None, physics: PhysicsConfig | None = None):
        super().__init__()
        if phase not in (1, 2, 3):
            raise ValueError('phase must be 1, 2 or 3')
        if render_mode not in (None, 'rgb_array', 'human'):
            raise ValueError('Unsupported render mode')
        self.phase, self.render_mode = phase, render_mode
        self.rewards = rewards or RewardConfig()
        self.physics = physics or PhysicsConfig()
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), dtype=np.float32)
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self.reset()

    def reset(self, *, seed=None, options=None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        self.x = 0.0 if self.phase == 1 else float(rng.uniform(-55, 55))
        self.y = float(rng.uniform(130, 180))
        self.vx = float(rng.uniform(-4, 4)) if self.phase == 3 else 0.0
        self.vy = float(rng.uniform(-18, -10))
        self.angle = float(rng.uniform(-0.12, 0.12)) if self.phase == 3 else 0.0
        self.omega = self.throttle = self.lateral = self.gimbal = 0.0
        self.fuel = self.initial_fuel
        self.gravity = float(rng.uniform(9.3, 10.3)) if self.phase == 3 else 9.81
        self.max_thrust = float(rng.uniform(20500, 23500)) if self.phase == 3 else 22000.0
        self.wind = float(rng.uniform(-1.5, 1.5)) * self.physics.wind_scale if self.phase == 3 else 0.0
        self.engine_delay = float(rng.uniform(0.15, 0.4)) * self.physics.delay_scale if self.phase == 3 else 0.0
        self.steps = 0
        self.outcome = 'FLYING'
        self.trajectory = [(self.x, self.y)]
        self.previous_potential = self._potential()
        return self._observation(), dict(self._info())

    def _potential(self):
        return -(0.22 * abs(self.x) + 0.08 * self.y +
                 0.6 * abs(self.vx) + 0.7 * abs(self.vy) + 6 * abs(self.angle))

    def _observation(self) -> NDArray[np.float32]:
        values = np.array([self.x / 100, self.y / 200, self.vx / 20, self.vy / 30,
                           self.angle, self.omega / 2, self.fuel / self.initial_fuel,
                           self.throttle, self.wind / 2, self.gravity / 10,
                           self.max_thrust / (self.dry_mass + self.fuel) / 25], dtype=np.float32)
        if self.phase == 3:
            values[:6] += self.np_random.normal(0, 0.002 * self.physics.noise_scale, 6).astype(np.float32)
        return values

    def _info(self) -> RocketInfo:
        return {
            'outcome': self.outcome,
            'is_success': self.outcome == 'LANDED',
            'fuel_used': self.initial_fuel - self.fuel,
            'phase': self.phase,
            'rewards': {
                'landing': self.rewards.landing,
                'crash': self.rewards.crash,
                'fuel': self.rewards.fuel,
            },
        }

    def step(self, action) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.outcome != 'FLYING':
            raise RuntimeError('Episode has ended; call reset() before step().')
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('Expected three finite action values')
        action = np.clip(action, -1, 1)
        target = (action[0] + 1) / 2
        response = 1.0 if not self.engine_delay else 1 - math.exp(-self.dt / self.engine_delay)
        self.throttle += (target - self.throttle) * response
        self.gimbal = float(action[1]) * 0.18 if self.phase > 1 else 0.0
        self.lateral = float(action[2]) if self.phase > 1 else 0.0
        requested = (2.0 * self.throttle + 0.25 * abs(self.lateral)) * self.dt
        used = min(self.fuel, requested)
        fraction = used / requested if requested > 0 else 0.0
        self.fuel = max(0.0, self.fuel - used)
        self.throttle *= fraction
        self.lateral *= fraction
        mass = self.dry_mass + self.fuel
        force = self.max_thrust * self.throttle
        if self.phase > 1:
            ax = force * math.sin(self.angle + self.gimbal) / mass + 3 * self.lateral + self.wind
            # Simplified gimbal torque and passive rotational damping.
            self.omega += (3.0 * self.gimbal * self.throttle - 0.7 * self.omega) * self.dt
            self.angle += self.omega * self.dt
            self.vx += ax * self.dt
            self.x += self.vx * self.dt
        ay = force * math.cos(self.angle + self.gimbal) / mass - self.gravity
        self.vy += ay * self.dt
        self.y += self.vy * self.dt
        self.steps += 1
        if self.steps % 5 == 0:
            self.trajectory.append((self.x, max(0.0, self.y)))
        terminated = False
        if self.y <= 0:
            self.y = 0.0
            safe = (abs(self.x) <= self.pad_half_width - 3 and abs(self.vx) <= 2
                    and abs(self.vy) <= 3 and abs(self.angle) <= 0.12 and abs(self.omega) <= 0.2)
            self.outcome = 'LANDED' if safe else 'CRASHED'
            terminated = True
        elif abs(self.x) > 155 or self.y > 260 or abs(self.angle) > 1.4:
            self.outcome = 'OUT OF BOUNDS'
            terminated = True
        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            self.outcome = 'TIME LIMIT'
        potential = self._potential()
        shaping = potential - self.previous_potential
        self.previous_potential = potential
        reward = shaping + self.rewards.fuel * used - 0.01
        if terminated or truncated:
            reward += self.rewards.landing if self.outcome == 'LANDED' else self.rewards.crash
        if self.render_mode == 'human':
            self.render()
        return self._observation(), float(reward), terminated, truncated, dict(self._info())

    def render(self):
        import pygame
        from rocket_visuals import draw_world
        surface = draw_world(self)
        if self.render_mode == 'human':
            if self._window is None:
                pygame.init()
                self._window = pygame.display.set_mode(surface.get_size())
                pygame.display.set_caption('Rocket Landing Simulator')
            if self._clock is None:
                self._clock = pygame.time.Clock()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    raise SimulationClosed()
            self._window.blit(surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata['render_fps'])
            return None
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def close(self):
        if self._window is not None:
            import pygame
            pygame.display.quit()
            self._window = None


def guidance(obs, phase=1):
    """Transparent PD reference controller, not a learned AI policy."""
    x, y, vx, vy = obs[:4] * np.array([100, 200, 20, 30])
    angle, omega = obs[4], obs[5] * 2
    desired_vy = -min(16.0, max(0.6, y * 0.32))
    ay = np.clip((desired_vy - vy) * 1.5, -7, 10)
    gravity, thrust_accel = obs[9] * 10, obs[10] * 25
    throttle = np.clip((gravity + ay) / max(1, thrust_accel * math.cos(angle)), 0, 1)
    desired_angle = np.clip(-0.012 * x - 0.07 * vx, -0.3, 0.3) if phase > 1 else 0
    gimbal = np.clip((5 * (desired_angle - angle) - 3 * omega) / 0.54, -1, 1)
    lateral = np.clip(-0.07 * x - 0.4 * vx, -1, 1) if phase > 1 else 0
    return np.array([throttle * 2 - 1, gimbal, lateral], dtype=np.float32)
