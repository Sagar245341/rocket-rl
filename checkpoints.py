"""Validate custom rocket checkpoints before loading a policy."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import BadZipFile

from gymnasium import spaces
import numpy as np

from rocket_env import ENV_VERSION, RewardConfig

if TYPE_CHECKING:
    from stable_baselines3 import PPO


def load_checkpoint(path: str | Path, phase: int) -> tuple[PPO, RewardConfig]:
    """Return (PPO policy, rewards), or a readable ValueError for invalid files."""
    path = Path(path).expanduser()
    if not path.suffix:
        path = path.with_suffix('.zip')
    if not path.is_file():
        raise ValueError(f'Model not found: {path}. Train phase {phase} first.')
    metadata = path.with_suffix('.json')
    try:
        config = json.loads(metadata.read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        raise ValueError(f'Missing metadata: {metadata}. Use the JSON saved by train.py.') from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read checkpoint metadata: {metadata}: {error}') from error
    if not isinstance(config, dict) or config.get('environment') != ENV_VERSION:
        raise ValueError(f'Expected {ENV_VERSION} metadata. Retrain older rocket/LunarLander policies.')
    if type(config.get('phase')) is not int or config['phase'] != phase:
        raise ValueError(f"Checkpoint phase {config.get('phase')} does not match phase {phase}.")
    try:
        weights = config['rewards']
        if not isinstance(weights, dict) or set(weights) != {'landing', 'crash', 'fuel'}:
            raise ValueError('Expected landing, crash, and fuel weights')
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in weights.values()):
            raise ValueError('Reward weights must be finite numbers')
        rewards = RewardConfig(**weights)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid checkpoint reward settings: {error}') from error
    try:
        from stable_baselines3 import PPO
        model = PPO.load(path, device='cpu')
    except ImportError as error:
        raise ValueError('PPO dependencies are missing. Install requirement.txt using the project venv.') from error
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError, EOFError, BadZipFile) as error:
        raise ValueError(f'Cannot load PPO checkpoint {path}: {error}') from error
    if not isinstance(model.observation_space, spaces.Box) or model.observation_space.shape != (11,):
        raise ValueError('Checkpoint must have 11 RocketEnv observations.')
    action_space = model.action_space
    if (not isinstance(action_space, spaces.Box) or action_space.shape != (3,)
            or not np.allclose(action_space.low, -1) or not np.allclose(action_space.high, 1)):
        raise ValueError('Checkpoint must have three continuous actions in [-1, 1].')
    return model, rewards
