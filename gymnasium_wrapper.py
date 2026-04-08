# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gymnasium wrapper for the Adaptive Cyber Defense OpenEnv environment.

Bridges the OpenEnv API (step/reset/state) to the standard Gymnasium
interface, making the environment immediately usable with:
    - Stable Baselines 3 (PPO, DQN, A2C, SAC)
    - RLlib
    - CleanRL
    - Any other Gymnasium-compatible RL library

Usage — connect PPO in ~5 lines:
    from gymnasium_wrapper import CyberDefenseGymEnv
    from stable_baselines3 import PPO

    env = CyberDefenseGymEnv(task="hard_stealth_defense")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500_000)
    model.save("cyber_defense_ppo")

Observation space:
    Flat float32 vector of 11 normalised signal values.
    All values in [0.0, 1.0] — compatible with any MLP policy.

Action space:
    Discrete(5) — maps to [monitor, isolate, block, escalate, do_nothing]

Episode structure:
    - Each reset() draws a fresh episode with a randomized seed.
    - Fixed seeds (task default) are used only when seed= is passed explicitly.
    - For RL training, leave seed=None so each episode is unique.
    - For evaluation/reproducibility, pass seed=TASK_CONFIG["seed"].

Why the non-stationary attacker matters for RL:
    The attacker's policy vector updates every step based on the defender's
    action (EMA, alpha=0.3). A passive defender sees accelerating attacks.
    An aggressive defender sees a retreating, hiding attacker.
    This forces the RL agent to learn a ROBUST general policy, not just
    memorize a fixed sequence — the core research value of this environment.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        gym = None
        spaces = None

try:
    from .models import CyberDefenseAction
    from .server.cyber_defense_environment import CyberDefenseEnvironment, TASK_CONFIGS
except ImportError:
    try:
        from models import CyberDefenseAction
        from server.cyber_defense_environment import CyberDefenseEnvironment, TASK_CONFIGS
    except ImportError:
        from cyber_defense.models import CyberDefenseAction
        from cyber_defense.server.cyber_defense_environment import (
            CyberDefenseEnvironment, TASK_CONFIGS,
        )


# Action index → action name mapping (Discrete(5))
ACTION_MAP: List[str] = ["monitor", "isolate", "block", "escalate", "do_nothing"]

# Observation field names (in order) — useful for feature importance analysis
OBS_FIELDS: List[str] = [
    "login_failure_rate",       # [0, 1]
    "alert_count_norm",         # alerts / 10 → [0, 1]
    "network_anomaly_score",    # [0, 1]
    "port_scan_detected",       # 0 or 1
    "cpu_spike",                # 0 or 1
    "lateral_movement_flag",    # 0 or 1
    "scrutiny_active",          # 0 or 1
    "anomaly_trend",            # [0, 1]  — key zero-day signal
    "packet_entropy",           # [0, 1]  — key zero-day signal
    "steps_remaining_norm",     # steps_remaining / max_steps → [0, 1]
    "consecutive_escalates_norm",  # consecutive_escalates / 5 → [0, 1]
]


class CyberDefenseGymEnv:
    """
    Gymnasium-compatible wrapper around CyberDefenseEnvironment.

    Inherits from gym.Env when gymnasium/gym is available, otherwise
    provides the same interface so the class can be inspected without
    a Gymnasium installation.

    Args:
        task:       Task name. One of the four TASK_CONFIGS keys.
                    Default: "easy_breach_prevention"
        seed:       Episode seed. None = random seed each episode (for training).
                    Set to TASK_CONFIGS[task]["seed"] for reproducible evaluation.
        render_mode: "human" prints step info to stdout. None = silent.
    """

    # Gymnasium metadata
    metadata = {"render_modes": ["human"], "render_fps": 1}

    # Action index → name (for rendering/logging)
    ACTION_NAMES = ACTION_MAP

    def __init__(
        self,
        task: str = "easy_breach_prevention",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        if not GYM_AVAILABLE:
            raise ImportError(
                "gymnasium (or gym) is required for the Gymnasium wrapper.\n"
                "Install with: pip install gymnasium\n"
                "Or: pip install gym"
            )

        if task not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task}'. "
                f"Valid tasks: {sorted(TASK_CONFIGS.keys())}"
            )

        self._task         = task
        self._seed         = seed
        self.render_mode   = render_mode
        self._env          = CyberDefenseEnvironment(task_name=task)
        self._episode_seed = seed
        self._last_obs     = None
        self._step_count   = 0

        # ---- Gymnasium spaces ----
        # Observation: 11-dim normalised float32 vector, all in [0, 1]
        self.observation_space = spaces.Box(
            low   = np.zeros(len(OBS_FIELDS), dtype=np.float32),
            high  = np.ones(len(OBS_FIELDS),  dtype=np.float32),
            dtype = np.float32,
        )

        # Action: 5 discrete choices
        # 0=monitor, 1=isolate, 2=block, 3=escalate, 4=do_nothing
        self.action_space = spaces.Discrete(len(ACTION_MAP))

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed:    Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed:    Episode seed. Overrides the instance seed if provided.
                     For RL training, leave None to get a unique seed each episode.
            options: Unused (Gymnasium API compliance).

        Returns:
            observation: float32 array of shape (11,)
            info:        Dict with task metadata
        """
        # Determine episode seed
        if seed is not None:
            self._episode_seed = seed
        elif self._seed is None:
            # Random seed each episode — essential for RL generalisation
            self._episode_seed = random.randint(0, 999_999)
        else:
            self._episode_seed = self._seed

        obs_raw = self._env.reset(seed=self._episode_seed)
        self._step_count = 0
        self._last_obs   = obs_raw

        obs_vec = self._obs_to_vector(obs_raw)
        info    = self._build_info(obs_raw)

        if self.render_mode == "human":
            self._render_reset(info)

        return obs_vec, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Integer in [0, 4] mapping to ACTION_MAP.

        Returns:
            observation:  float32 array of shape (11,)
            reward:       shaped per-step reward (float)
            terminated:   True if episode ended (breach or time limit)
            truncated:    Always False (no truncation, only termination)
            info:         Dict with hidden state details for logging
        """
        action_name = ACTION_MAP[int(action)]
        action_obj  = CyberDefenseAction(action_type=action_name)

        obs_raw    = self._env.step(action_obj)
        reward     = float(obs_raw.reward or 0.0)
        terminated = bool(obs_raw.done)
        truncated  = False  # We use termination, not truncation

        self._step_count += 1
        self._last_obs    = obs_raw

        obs_vec = self._obs_to_vector(obs_raw)
        info    = self._build_info(obs_raw)

        if terminated:
            # Attach final episode grade to info for SB3 callbacks
            info["episode_score"] = self._env.grade()
            info["episode_steps"] = self._step_count

        if self.render_mode == "human":
            self._render_step(action_name, reward, terminated, info)

        return obs_vec, reward, terminated, truncated, info

    def render(self) -> None:
        """Print current state to stdout (human mode only)."""
        if self.render_mode != "human" or self._last_obs is None:
            return
        obs = self._last_obs
        print(
            f"  Step {self._step_count} | "
            f"login={obs.login_failure_rate:.2f} "
            f"alerts={obs.alert_count} "
            f"anomaly={obs.network_anomaly_score:.2f} "
            f"trend={getattr(obs, 'anomaly_trend', 0.0):.2f}"
        )

    def close(self) -> None:
        """Clean up resources."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_to_vector(self, obs) -> np.ndarray:
        """
        Convert CyberDefenseObservation to a normalised float32 vector.

        All values are clipped to [0, 1] so the observation space bounds
        are always satisfied. This is important for neural network stability.
        """
        max_steps = TASK_CONFIGS[self._task]["max_steps"]
        vec = np.array([
            float(obs.login_failure_rate),
            float(obs.alert_count) / 10.0,         # normalise 0-10 → 0-1
            float(obs.network_anomaly_score),
            1.0 if obs.port_scan_detected else 0.0,
            1.0 if obs.cpu_spike else 0.0,
            1.0 if obs.lateral_movement_flag else 0.0,
            1.0 if obs.scrutiny_active else 0.0,
            float(getattr(obs, "anomaly_trend",  0.0)),
            float(getattr(obs, "packet_entropy", 0.0)),
            float(obs.steps_remaining) / max(max_steps, 1),  # normalise
            float(getattr(obs, "consecutive_escalates", 0)) / 5.0,  # normalise
        ], dtype=np.float32)
        return np.clip(vec, 0.0, 1.0)

    def _build_info(self, obs) -> Dict[str, Any]:
        """Build the info dict returned with each step."""
        return {
            "task":              self._task,
            "step":              self._step_count,
            "login_rate":        obs.login_failure_rate,
            "alert_count":       obs.alert_count,
            "anomaly_score":     obs.network_anomaly_score,
            "anomaly_trend":     getattr(obs, "anomaly_trend",  0.0),
            "packet_entropy":    getattr(obs, "packet_entropy", 0.0),
            "scrutiny_active":   obs.scrutiny_active,
            "steps_remaining":   obs.steps_remaining,
            "last_action":       obs.last_action,
            "episode_seed":      self._episode_seed,
        }

    def _render_reset(self, info: Dict[str, Any]) -> None:
        print(f"\n[CyberDefense] New episode — task={self._task} seed={self._episode_seed}")

    def _render_step(self, action: str, reward: float, done: bool, info: Dict[str, Any]) -> None:
        print(
            f"  step={self._step_count:2d} | action={action:10s} | "
            f"reward={reward:+.3f} | done={done} | "
            f"alerts={info['alert_count']} login={info['login_rate']:.2f}"
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def task_config(self) -> Dict[str, Any]:
        """Return the active task configuration dict."""
        return TASK_CONFIGS[self._task]

    @property
    def hidden_state(self):
        """Return the full hidden state (for evaluation/logging, NOT for the agent)."""
        return self._env.state

    @property
    def current_grade(self) -> float:
        """Return the deterministic grade for the current episode so far."""
        return self._env.grade()


# ------------------------------------------------------------------
# Register with Gymnasium if available
# ------------------------------------------------------------------

def register_gymnasium_envs() -> None:
    """
    Register all four tasks as named Gymnasium environments.

    After calling this, environments can be created with:
        env = gym.make("CyberDefense-easy-v0")
        env = gym.make("CyberDefense-medium-v0")
        env = gym.make("CyberDefense-hard-v0")
        env = gym.make("CyberDefense-zeroday-v0")
    """
    if not GYM_AVAILABLE:
        return

    task_ids = {
        "CyberDefense-easy-v0":    "easy_breach_prevention",
        "CyberDefense-medium-v0":  "medium_detection_balance",
        "CyberDefense-hard-v0":    "hard_stealth_defense",
        "CyberDefense-zeroday-v0": "zero_day_detection",
    }

    for gym_id, task_name in task_ids.items():
        try:
            gym.register(
                id             = gym_id,
                entry_point    = "gymnasium_wrapper:CyberDefenseGymEnv",
                kwargs         = {"task": task_name},
                max_episode_steps = TASK_CONFIGS[task_name]["max_steps"],
            )
        except Exception:
            pass  # Already registered


# ------------------------------------------------------------------
# Quick start example (run this file directly)
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Cyber Defense Gymnasium Wrapper — Quick Test ===")
    print()

    if not GYM_AVAILABLE:
        print("Install gymnasium to use this wrapper:")
        print("  pip install gymnasium stable-baselines3")
        exit(1)

    for task_name in TASK_CONFIGS:
        env = CyberDefenseGymEnv(task=task_name, seed=42)
        obs, info = env.reset()
        print(f"Task: {task_name}")
        print(f"  obs shape: {obs.shape}  dtype: {obs.dtype}")
        print(f"  obs min/max: {obs.min():.3f} / {obs.max():.3f}")

        total_reward = 0.0
        for _ in range(TASK_CONFIGS[task_name]["max_steps"]):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break

        grade = env.current_grade
        print(f"  Random agent: total_reward={total_reward:.3f}  grade={grade:.3f}")
        assert env.observation_space.contains(obs), "Obs out of space bounds!"
        print(f"  ✅ Observation space bounds satisfied")
        print()

    print("PPO connection (requires stable-baselines3):")
    print()
    print("  from gymnasium_wrapper import CyberDefenseGymEnv")
    print("  from stable_baselines3 import PPO")
    print()
    print("  env = CyberDefenseGymEnv(task='hard_stealth_defense')")
    print("  model = PPO('MlpPolicy', env, verbose=1)")
    print("  model.learn(total_timesteps=500_000)")
    print("  model.save('cyber_defense_ppo')")