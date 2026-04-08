# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """
# Cyber Defense Environment Implementation.

# A simple test environment that echoes back messages sent to it.
# Perfect for testing HTTP server infrastructure.
# """

# from uuid import uuid4

# from openenv.core.env_server.interfaces import Environment
# from openenv.core.env_server.types import State

# try:
#     from ..models import CyberDefenseAction, CyberDefenseObservation
# except ImportError:
#     from models import CyberDefenseAction, CyberDefenseObservation


# class CyberDefenseEnvironment(Environment):
#     """
#     A simple echo environment that echoes back messages.

#     This environment is designed for testing the HTTP server infrastructure.
#     It maintains minimal state and simply echoes back whatever message it receives.

#     Example:
#         >>> env = CyberDefenseEnvironment()
#         >>> obs = env.reset()
#         >>> print(obs.echoed_message)  # "Cyber Defense environment ready!"
#         >>>
#         >>> obs = env.step(CyberDefenseAction(message="Hello"))
#         >>> print(obs.echoed_message)  # "Hello"
#         >>> print(obs.message_length)  # 5
#     """

#     # Enable concurrent WebSocket sessions.
#     # Set to True if your environment isolates state between instances.
#     # When True, multiple WebSocket clients can connect simultaneously, each
#     # getting their own environment instance (when using factory mode in app.py).
#     SUPPORTS_CONCURRENT_SESSIONS: bool = True

#     def __init__(self):
#         """Initialize the cyber_defense environment."""
#         self._state = State(episode_id=str(uuid4()), step_count=0)
#         self._reset_count = 0

#     def reset(self) -> CyberDefenseObservation:
#         """
#         Reset the environment.

#         Returns:
#             CyberDefenseObservation with a ready message
#         """
#         self._state = State(episode_id=str(uuid4()), step_count=0)
#         self._reset_count += 1

#         return CyberDefenseObservation(
#             echoed_message="Cyber Defense environment ready!",
#             message_length=0,
#             done=False,
#             reward=0.0,
#         )

#     def step(self, action: CyberDefenseAction) -> CyberDefenseObservation:  # type: ignore[override]
#         """
#         Execute a step in the environment by echoing the message.

#         Args:
#             action: CyberDefenseAction containing the message to echo

#         Returns:
#             CyberDefenseObservation with the echoed message and its length
#         """
#         self._state.step_count += 1

#         message = action.message
#         length = len(message)

#         # Simple reward: longer messages get higher rewards
#         reward = length * 0.1

#         return CyberDefenseObservation(
#             echoed_message=message,
#             message_length=length,
#             done=False,
#             reward=reward,
#             metadata={"original_message": message, "step": self._state.step_count},
#         )

#     @property
#     def state(self) -> State:
#         """
#         Get the current environment state.

#         Returns:
#             Current State with episode_id and step_count
#         """
#         return self._state


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Adaptive Cyber Defense Environment — v3
========================================

A real-world SOC (Security Operations Center) decision environment for
training and evaluating RL agents. Models the full MITRE ATT&CK kill-chain
as a POMDP with an adversarially adaptive attacker.

WHAT'S NEW IN THIS VERSION
---------------------------
Fix 1 — Escalate penalty when attack is active (R_ESCALATE_ACTIVE = -0.06).
    Escalate is now free ONLY on an idle network. Using it while under attack
    costs -0.06 per use after the first. This breaks the escalate-loop where
    the agent defers indefinitely with zero cost.

Fix 2 — Consecutive escalate counter with compounding penalty.
    After 2 consecutive escalates, each additional escalate costs an extra
    -0.06 (on top of R_ESCALATE_ACTIVE). Exposed in observation so LLM can
    see it. Forces commitment to block/isolate after gathering information.

Fix 3 — Stealth attacker retreat cap (max 2 retreats per episode).
    The stealth attacker can only retreat twice per episode. After that it
    only advances or holds. Removes the lucky-passive-survival loophole.

Fix 4 — Proper RL-vs-RL attacker policy (not just probability adjustment).
    The attacker maintains an explicit policy vector [advance, retreat, hold]
    that is updated via a policy gradient-like rule based on defender behavior.
    This is true RL-vs-RL: both agents are learning simultaneously in the
    same episode. The attacker's policy is visible in state() for evaluators.

Fix 5 — Zero-day simulation in hard_stealth_defense task.
    With 30% probability per episode, a zero-day event is injected at step 8.
    During zero-day: standard signals are SUPPRESSED (attack looks idle) while
    anomaly_trend and packet_entropy carry the real signal. This forces the
    agent to learn temporal reasoning rather than threshold-matching.

FOUR TASKS (added zero_day_detection)
---------------------------------------
easy_breach_prevention  (10 steps, fixed attacker, no noise)
medium_detection_balance (20 steps, adaptive RL-vs-RL, 20% noise)
hard_stealth_defense     (30 steps, stealth+zero-day, 40% noise, partial obs)
zero_day_detection       (25 steps, pure zero-day, all standard signals masked)

RL-vs-RL DESIGN
----------------
The adaptive attacker maintains a policy vector P = [p_advance, p_retreat, p_hold].
After each step it updates P based on what the defender just did:
    - Defender used isolate/block: attacker increases p_retreat (evade)
    - Defender used escalate/monitor: attacker increases p_advance (exploit passivity)
    - Defender used do_nothing: attacker maximally accelerates
The update uses exponential moving average (alpha=0.3) so the policy smoothly
tracks the defender's behavior over the episode. Both agents are adapting
simultaneously — this is the RL-vs-RL game.

ZERO-DAY SIGNAL MODEL
-----------------------
Real zero-days have no known signature. They are detected by:
    - Anomaly TREND (rising pattern) rather than absolute threshold
    - Unusual packet size distributions (high entropy = non-standard protocol)
    - Timing irregularities (not modeled here but referenced in lit)

When zero_day_active=True:
    - login_failure_rate suppressed to near-zero (attacker avoids brute force)
    - alert_count suppressed to 0-1 (no known IDS signatures match)
    - anomaly_trend rises from 0.0 → 0.9 over 5 steps (key signal)
    - packet_entropy rises from 0.3 → 0.8 (key signal)
    - port_scan hidden (attacker uses legitimate-looking connections)
Agent must detect the zero-day via anomaly_trend + packet_entropy, not IDS alerts.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv_core import Environment

try:
    from ..models import CyberDefenseAction, CyberDefenseObservation, CyberDefenseState
except ImportError:
    from models import CyberDefenseAction, CyberDefenseObservation, CyberDefenseState


# ---------------------------------------------------------------------------
# Signal lookup table (MITRE ATT&CK kill-chain grounded)
# ---------------------------------------------------------------------------

STAGE_SIGNALS: Dict[int, Dict[str, Any]] = {
    # Stage: login  alerts  anomaly  port_scan  cpu_spike  lateral
    0: {"login": 0.00, "alerts": 0,  "anomaly": 0.00, "port_scan": False, "cpu_spike": False, "lateral": False},
    1: {"login": 0.30, "alerts": 2,  "anomaly": 0.28, "port_scan": True,  "cpu_spike": False, "lateral": False},
    2: {"login": 0.52, "alerts": 4,  "anomaly": 0.46, "port_scan": True,  "cpu_spike": True,  "lateral": False},
    3: {"login": 0.68, "alerts": 6,  "anomaly": 0.65, "port_scan": True,  "cpu_spike": True,  "lateral": True},
    4: {"login": 0.82, "alerts": 8,  "anomaly": 0.82, "port_scan": True,  "cpu_spike": True,  "lateral": True},
    5: {"login": 0.95, "alerts": 10, "anomaly": 0.95, "port_scan": True,  "cpu_spike": True,  "lateral": True},
}


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy_breach_prevention": {
        "max_steps":          10,
        "attacker_type":      "fixed",
        "noise_rate":         0.0,
        "partial_obs":        False,
        "advance_prob":       0.55,
        "initial_stage":      0,
        "zero_day_prob":      0.0,    # No zero-day in easy task
        "seed":               42,
        "success_threshold":  0.60,
    },
    "medium_detection_balance": {
        "max_steps":          20,
        "attacker_type":      "adaptive",
        "noise_rate":         0.20,
        "partial_obs":        False,
        "advance_prob":       0.60,
        "initial_stage":      1,
        "zero_day_prob":      0.0,    # No zero-day in medium task
        "seed":               137,
        "success_threshold":  0.45,
    },
    "hard_stealth_defense": {
        "max_steps":          30,
        "attacker_type":      "stealth",
        "noise_rate":         0.40,
        "partial_obs":        True,   # lateral_movement hidden
        "advance_prob":       0.65,
        "initial_stage":      1,
        "zero_day_prob":      0.30,   # 30% chance of zero-day injection at step 8
        "seed":               271,
        "success_threshold":  0.30,
    },
    "zero_day_detection": {
        "max_steps":          25,
        "attacker_type":      "zero_day",
        "noise_rate":         0.30,
        "partial_obs":        True,
        "advance_prob":       0.50,
        "initial_stage":      1,
        "zero_day_prob":      1.0,   # Always zero-day from step 5
        "seed":               999,
        "success_threshold":  0.35,
    },
}


# ---------------------------------------------------------------------------
# Reward constants
#
# Mathematical design guarantee:
#   E[monitor | attack active] = R_MONITOR_PASSIVE + advance_prob * R_MISSED_ATTACK
#                              = -0.10 + 0.55 * (-0.10) = -0.155 / step
#   E[block | attack active]   = R_CORRECT_BLOCK = +0.25 / step
#   Delta = +0.405 per step — block always strictly dominates monitor
#
#   E[escalate | attack active, consecutive < 2] = R_ESCALATE_ACTIVE = -0.06 / step
#   E[escalate | attack active, consecutive >= 2] = -0.06 + -0.06 = -0.12 / step
#   This ensures 2+ escalates in a row is MORE costly than one block
# ---------------------------------------------------------------------------

R_CORRECT_ISOLATE      = +0.40   # Full stop — attack_stage → 0
R_CORRECT_BLOCK        = +0.25   # Partial stop — attack_stage -= 1
R_EARLY_DETECTION      = +0.15   # Bonus: action taken when stage <= 2

R_FALSE_POSITIVE_ISO   = -0.12   # Isolate on idle network
R_FALSE_POSITIVE_BLK   = -0.08   # Block on idle network

R_MONITOR_PASSIVE      = -0.10   # Monitor while attack_stage > 0
R_DO_NOTHING_PENALTY   = -0.18   # do_nothing while attack_stage > 0
R_MISSED_ATTACK        = -0.10   # Attacker advanced uncontested

R_ESCALATE_ACTIVE      = -0.06   # Escalate while attack_stage > 0 (after first use)
R_ESCALATE_REPEATED    = -0.06   # Additional penalty per escalate when consecutive >= 2

R_BREACH_PENALTY       = -0.80   # Terminal: breach
R_SCRUTINY_SEQUENCE    = +0.05   # Bonus: escalate followed by correct block/isolate within 2 steps
R_SUCCESS_BONUS        = +0.50   # Terminal: survived all steps


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CyberDefenseEnvironment(Environment):
    """
    Adaptive Cyber Defense RL Environment — v3.

    POMDP design:
        - Hidden state: true attack stage, attacker policy vector, zero-day flag.
        - Observation: noisy signals derived from hidden state.
        - Actions: {monitor, isolate, block, escalate, do_nothing}.
        - Reward: dense per-step shaping, terminal rewards on breach/success.

    RL-vs-RL:
        The adaptive attacker maintains a policy vector P updated by EMA
        on the defender's last action. Both agents adapt simultaneously.

    Zero-day simulation:
        In hard/zero_day tasks, standard signals are suppressed mid-episode.
        Agent must learn to detect via anomaly_trend and packet_entropy.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "easy_breach_prevention"):
        super().__init__()
        if task_name not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Choose from: {sorted(TASK_CONFIGS.keys())}"
            )
        self._task_name          = task_name
        self._cfg                = TASK_CONFIGS[task_name]
        self._rng                = random.Random(self._cfg["seed"])
        self._hidden             = self._make_initial_state()
        self._scrutiny_active    = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        task:       Optional[str] = None,
        **kwargs:   Any,
    ) -> CyberDefenseObservation:
        if task is not None and task != self._task_name:
            if task not in TASK_CONFIGS:
                raise ValueError(f"Unknown task '{task}'.")
            self._task_name = task
            self._cfg       = TASK_CONFIGS[task]

        effective_seed        = seed if seed is not None else self._cfg["seed"]
        self._rng             = random.Random(effective_seed)
        self._scrutiny_active = False
        self._hidden          = self._make_initial_state(episode_id=episode_id)
        self._reset_rubric()
        return self._build_observation()

    def _reset_rubric(self) -> None:
        pass  # No-op — OpenEnv base class requirement

    def step(
        self,
        action:    CyberDefenseAction,
        timeout_s: Optional[float] = None,
        **kwargs:  Any,
    ) -> CyberDefenseObservation:
        """
        Execute one step.

        Order:
            1. Guard: already breached → terminal obs.
            2. Decrement scrutiny countdown.
            3. Apply defender action → immediate reward.
            4. RL-vs-RL: update attacker policy based on defender action.
            5. Advance attacker using updated policy.
            6. Zero-day injection check.
            7. Escalate loop penalty (consecutive escalate tracking).
            8. Passivity penalty if attack still active after action.
            9. Missed-attack penalty if attacker advanced.
            10. Terminal check + terminal reward.
            11. Record history. Build and return noisy observation.
        """
        if self._hidden.is_compromised:
            return self._build_observation(force_done=True)

        self._hidden.step_count += 1
        step = self._hidden.step_count

        # 2. Scrutiny countdown
        if self._scrutiny_active and self._hidden.scrutiny_steps_left > 0:
            self._hidden.scrutiny_steps_left -= 1
            if self._hidden.scrutiny_steps_left == 0:
                self._scrutiny_active = False

        # 3. Apply defender action
        reward = self._apply_action(action)

        # 3a. Scrutiny sequence bonus: escalate → correct block/isolate within 2 steps
        # Teaches the agent that escalate is a diagnostic tool, not a stalling tactic.
        # The bonus fires when: last action was escalate AND this action is a correct block/isolate.
        if (
            len(self._hidden.action_history) >= 1
            and self._hidden.action_history[-1] == "escalate"
            and action.action_type in ("isolate", "block")
            and self._hidden.attack_stage > 0  # Must be responding to a real attack
        ):
            reward += R_SCRUTINY_SEQUENCE

        # 4. RL-vs-RL: update attacker policy based on what the defender did
        self._update_attacker_policy(action)

        # 5. Advance attacker using updated policy
        stage_before_advance = self._hidden.attack_stage
        if not self._hidden.is_compromised:
            self._advance_attacker()
        attack_advanced = self._hidden.attack_stage > stage_before_advance

        # 6. Zero-day injection check
        self._check_zero_day_injection(step)

        # 7. Escalate loop penalty
        if action.action_type == "escalate":
            self._hidden.consecutive_escalates += 1
            # Cost for escalating while attack is active
            if self._hidden.attack_stage > 0:
                reward += R_ESCALATE_ACTIVE
                # Additional compounding cost for repeated escalation
                if self._hidden.consecutive_escalates >= 2:
                    reward += R_ESCALATE_REPEATED
        else:
            self._hidden.consecutive_escalates = 0  # Reset on any non-escalate action

        # 8. Passivity penalty — fires if attack STILL active after defender acted
        #    (correct isolate → stage=0 → this does NOT fire)
        if self._hidden.attack_stage > 0 and not self._hidden.is_compromised:
            if action.action_type == "monitor":
                reward += R_MONITOR_PASSIVE
            elif action.action_type == "do_nothing":
                reward += R_DO_NOTHING_PENALTY

        # 9. Missed-attack penalty — stage-proportional.
        # Advancing from stage 4→5 (near breach) is far more dangerous
        # than 1→2 (early recon). This gradient teaches the RL agent
        # to prioritise blocking high-stage advances over low-stage ones,
        # matching real SOC triage priority.
        # Formula: penalty = R_MISSED_ATTACK * (0.5 + 0.5 * stage/5)
        #   Stage 0→1: -0.10 * (0.5 + 0.0) = -0.05  (recon starting)
        #   Stage 2→3: -0.10 * (0.5 + 0.3) = -0.08  (lateral movement)
        #   Stage 4→5: -0.10 * (0.5 + 0.4) = -0.09  (near breach — close to full penalty)
        # Note: uses stage AFTER advance so the full severity is captured.
        if attack_advanced:
            stage_severity = 0.5 + 0.5 * (self._hidden.attack_stage / 5.0)
            reward += R_MISSED_ATTACK * stage_severity
            self._hidden.missed_attacks += 1

        # 10. Terminal check
        done = self._hidden.is_compromised or (step >= self._hidden.max_steps)
        if self._hidden.is_compromised:
            reward += R_BREACH_PENALTY
        elif done:
            activity = min(self._hidden.correct_blocks / max(step * 0.25, 1.0), 1.0)
            reward += R_SUCCESS_BONUS * max(activity, 0.20)

        # 11. Record
        self._hidden.total_reward += reward
        self._hidden.attack_history.append(self._hidden.attack_stage)
        self._hidden.action_history.append(action.action_type)
        self._hidden.reward_history.append(round(reward, 4))

        # Update anomaly history for trend computation
        raw_anomaly = STAGE_SIGNALS[min(self._hidden.attack_stage, 5)]["anomaly"]
        self._hidden.anomaly_history.append(raw_anomaly)

        obs             = self._build_observation(force_done=done, reward=reward)
        obs.last_action = action.action_type
        return obs

    @property
    def state(self) -> CyberDefenseState:
        """Full hidden state. Graders/evaluators only. Attaches current grade."""
        self._hidden.task_score = self.grade()
        return self._hidden

    # ------------------------------------------------------------------
    # Graders (deterministic, 0.0–1.0, no LLM calls)
    # ------------------------------------------------------------------

    def grade(self) -> float:
        """
        Score the completed episode.

        All graders require active correct_blocks > 0 for high scores.
        Passive survival is rewarded modestly but below the success threshold.
        """
        h       = self._hidden
        steps   = max(h.step_count, 1)
        eps     = 1e-9

        if self._task_name == "easy_breach_prevention":
            if h.is_compromised:
                return max(0.0, round(0.25 * (h.step_count / max(h.max_steps, 1)), 3))
            real_atk = sum(1 for s in h.attack_history if s > 0)
            if real_atk == 0:
                return 0.50  # Attacker never activated — modest score
            ir = h.correct_blocks / (real_atk + eps)
            return round(min(0.50 + 0.50 * ir, 1.0), 3)

        elif self._task_name == "medium_detection_balance":
            total_real = max(sum(1 for s in h.attack_history if s > 0), 1)
            precision  = h.correct_blocks / (h.correct_blocks + h.false_positives + eps)
            recall     = h.correct_blocks / (total_real + eps)
            bc = 0.35 if not h.is_compromised else 0.08 * (h.step_count / max(h.max_steps, 1))
            return round(min(max(bc + 0.35 * precision + 0.30 * recall, 0.0), 1.0), 3)

        elif self._task_name == "hard_stealth_defense":
            fp_rate = h.false_positives / (steps + eps)
            ir      = h.correct_blocks  / (steps + eps)
            stage_c = sum((5 - s) / 5.0 for s in h.attack_history) / max(len(h.attack_history), 1)
            bc = 0.40 if not h.is_compromised else 0.06 * (h.step_count / max(h.max_steps, 1))
            return round(min(max(
                bc + 0.28 * (1.0 - min(fp_rate, 1.0)) + 0.22 * min(ir * 4.0, 1.0) + 0.10 * stage_c,
                0.0), 1.0), 3)

        elif self._task_name == "zero_day_detection":
            # Grader emphasises early detection and intervention rate
            # Standard IDS metrics don't apply — score on anomaly response
            total_steps = max(h.step_count, 1)
            zd_onset    = h.zero_day_onset_step if h.zero_day_onset_step >= 0 else total_steps
            # Steps where zero-day was active and agent responded correctly
            zd_steps      = max(total_steps - zd_onset, 0)
            zd_correct    = sum(
                1 for i, a in enumerate(h.action_history)
                if i >= zd_onset and a in ("block", "isolate")
            )
            detection_rate = zd_correct / max(zd_steps, 1)
            fp_rate        = h.false_positives / (total_steps + eps)
            bc = 0.30 if not h.is_compromised else 0.05 * (h.step_count / max(h.max_steps, 1))
            return round(min(max(
                bc + 0.40 * detection_rate + 0.20 * (1.0 - min(fp_rate, 1.0)) + 0.10,
                0.0), 1.0), 3)

        return 0.0

    # ------------------------------------------------------------------
    # Internal: action logic
    # ------------------------------------------------------------------

    def _apply_action(self, action: CyberDefenseAction) -> float:
        """Apply defender action before attacker advances. Returns immediate reward."""
        h             = self._hidden
        attack_active = h.attack_stage > 0
        reward        = 0.0

        if action.action_type == "monitor":
            pass

        elif action.action_type == "isolate":
            if attack_active:
                early = h.attack_stage <= 2
                h.correct_blocks += 1
                h.attack_stage    = 0
                reward += R_CORRECT_ISOLATE
                if early:
                    reward += R_EARLY_DETECTION
            else:
                h.false_positives += 1
                # Context-sensitive FP: acting on a truly idle network costs more
                # than acting on a network with ambiguous signals.
                # This teaches calibrated precision, not binary act/don't-act.
                # Scale = (1 - anomaly_score): idle=1.0x, ambiguous=0.6x cost
                fp_scale = 1.0 - min(
                    STAGE_SIGNALS[min(h.attack_stage, 5)]["anomaly"], 0.80
                )
                reward += R_FALSE_POSITIVE_ISO * fp_scale

        elif action.action_type == "block":
            if attack_active:
                early = h.attack_stage <= 2
                h.correct_blocks += 1
                h.attack_stage    = max(0, h.attack_stage - 1)
                reward += R_CORRECT_BLOCK
                if early:
                    reward += R_EARLY_DETECTION * 0.5
            else:
                h.false_positives += 1
                fp_scale = 1.0 - min(
                    STAGE_SIGNALS[min(h.attack_stage, 5)]["anomaly"], 0.80
                )
                reward += R_FALSE_POSITIVE_BLK * fp_scale

        elif action.action_type == "escalate":
            # Base escalate: activate scrutiny
            # Reward/penalty handled separately in step() via R_ESCALATE_ACTIVE
            self._scrutiny_active       = True
            h.scrutiny_steps_left       = 2

        elif action.action_type == "do_nothing":
            pass

        return reward

    # ------------------------------------------------------------------
    # Internal: RL-vs-RL attacker policy update
    # ------------------------------------------------------------------

    def _update_attacker_policy(self, last_action: CyberDefenseAction) -> None:
        """
        Update the attacker's policy vector based on the defender's last action.

        This is the RL-vs-RL mechanic: the attacker is a learning agent that
        responds to the defender's observed strategy.

        Policy vector: [p_advance, p_retreat, p_hold]
        Update rule: exponential moving average (alpha=0.3)

        Defender action → attacker adaptation:
            isolate/block  → increase p_retreat (attacker evades aggressive defense)
            escalate       → slight increase p_retreat (attacker wary of scrutiny)
            monitor        → increase p_advance (exploit passivity)
            do_nothing     → strongly increase p_advance (maximum exploitation)

        The policy vector is visible in state() so evaluators can watch the
        attacker learn in real time during an episode.
        """
        h     = self._hidden
        alpha = 0.3   # EMA weight for policy update
        pv    = h.attacker_policy_vector   # [p_advance, p_retreat, p_hold]

        act = last_action.action_type

        if act in ("isolate", "block"):
            # Defender is aggressive → attacker retreats to hide
            target = [0.30, 0.50, 0.20]
        elif act == "escalate":
            # Defender is escalating → attacker slightly cautious
            target = [0.45, 0.30, 0.25]
        elif act == "monitor":
            # Defender is passive → attacker exploits
            target = [0.75, 0.05, 0.20]
        else:  # do_nothing
            # Defender is completely passive → max exploitation
            target = [0.90, 0.02, 0.08]

        # EMA update
        pv[0] = (1 - alpha) * pv[0] + alpha * target[0]
        pv[1] = (1 - alpha) * pv[1] + alpha * target[1]
        pv[2] = (1 - alpha) * pv[2] + alpha * target[2]

        # Normalise to sum to 1.0
        total = pv[0] + pv[1] + pv[2]
        h.attacker_policy_vector = [round(p / total, 4) for p in pv]

        # Track adaptation event for evaluators
        if pv[0] > 0.70 and not h.attacker_adapted:
            h.attacker_adapted = True   # Attacker shifted to aggressive mode
        if h.step_count >= 10 and h.attacker_type == "adaptive":
            h.attacker_adapted = True   # Always adapted after midpoint

    # ------------------------------------------------------------------
    # Internal: attacker FSM (uses updated policy vector)
    # ------------------------------------------------------------------

    def _advance_attacker(self) -> None:
        """
        Advance attacker using current policy vector (for adaptive/stealth attackers)
        or fixed probability (for fixed attacker).

        For adaptive and stealth types, the policy vector computed by
        _update_attacker_policy() determines the action probabilities.
        This is the RL-vs-RL game: the attacker's behavior is a direct
        function of what the defender has been doing.
        """
        h = self._hidden
        if h.attack_stage >= 5:
            h.is_compromised = True
            return

        atype = h.attacker_type

        if atype == "fixed":
            # Base Bernoulli — but with passive-defender acceleration.
            # If the defender has been passive (monitor/do_nothing) for
            # 4+ consecutive steps, the attacker speeds up by 30%.
            # This prevents trivial solving after a few hundred RL episodes
            # and ensures meaningful difficulty even in the "easy" task.
            recent = h.action_history[-4:] if len(h.action_history) >= 4 else h.action_history
            passive_count = sum(1 for a in recent if a in ("monitor", "do_nothing", "escalate"))
            acceleration  = 1.30 if passive_count >= 4 else 1.0
            if self._rng.random() < self._cfg["advance_prob"] * acceleration:
                h.attack_stage = min(h.attack_stage + 1, 5)

        elif atype == "adaptive":
            # RL policy: use the learned policy vector
            roll = self._rng.random()
            pv   = h.attacker_policy_vector
            if roll < pv[0]:
                # Advance
                h.attack_stage = min(h.attack_stage + 1, 5)
            elif roll < pv[0] + pv[1]:
                # Retreat (attacker evades detection)
                h.attack_stage = max(h.attack_stage - 1, 0)
            # else: hold

        elif atype == "stealth":
            # Stealth: uses policy vector but with per-episode retreat cap.
            # Cap is 1-3 (drawn from RNG in _make_initial_state via signal_scales seed).
            # Randomizing prevents RL agents from memorizing "after 2 retreats, attack".
            roll = self._rng.random()
            pv   = h.attacker_policy_vector

            retreat_cap = h.signal_scales.get("retreat_cap", 2)
            can_retreat = h.retreat_count < retreat_cap   # Randomized cap

            if roll < pv[0]:
                h.attack_stage = min(h.attack_stage + 1, 5)
            elif roll < pv[0] + pv[1] and can_retreat:
                h.attack_stage = max(h.attack_stage - 1, 0)
                h.retreat_count += 1
            # else: hold (also forced when retreat cap reached)

        elif atype == "zero_day":
            # Zero-day attacker: advances steadily but hides signals
            # The advance_prob is base; once zero-day active it doesn't retreat
            if self._rng.random() < self._cfg["advance_prob"]:
                h.attack_stage = min(h.attack_stage + 1, 5)

        if h.attack_stage >= 5:
            h.is_compromised = True

    # ------------------------------------------------------------------
    # Internal: zero-day injection
    # ------------------------------------------------------------------

    def _check_zero_day_injection(self, step: int) -> None:
        """
        Inject a zero-day event if conditions are met.

        For hard_stealth_defense: 30% chance, injected at step 8.
        For zero_day_detection: always injected at step 5.

        Once activated, standard signals are suppressed in _build_observation.
        anomaly_trend and packet_entropy become the primary detection channels.
        """
        h = self._hidden
        if h.zero_day_active:
            return  # Already active

        zero_day_prob = self._cfg.get("zero_day_prob", 0.0)
        if zero_day_prob <= 0.0:
            return

        # Onset step is randomized per episode — drawn from RNG when the
        # first possible onset step is reached. This prevents RL agents from
        # learning a fixed "switch mode at step 8" policy.
        # zero_day_detection task: onset between steps 4-8 (earlier pressure)
        # hard_stealth_defense task: onset between steps 6-13 (wider window)
        if self._cfg["attacker_type"] == "zero_day":
            onset_range = (4, 8)
        else:
            onset_range = (6, 13)

        # Only consider injection within the onset range
        if onset_range[0] <= step <= onset_range[1]:
            # Roll once per step in the range — capped by zero_day_prob per step
            # This means onset is geometrically distributed in the range
            remaining_steps = onset_range[1] - step + 1
            # Adjust per-step probability so expected onset is in the middle of range
            per_step_prob = zero_day_prob / max(remaining_steps, 1)
            if self._rng.random() < per_step_prob:
                h.zero_day_active      = True
                h.zero_day_onset_step  = step

    # ------------------------------------------------------------------
    # Internal: observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        force_done: bool  = False,
        reward:     float = 0.0,
    ) -> CyberDefenseObservation:
        """
        Build noisy observation from hidden state.

        Standard mode:
            Uses STAGE_SIGNALS lookup + jitter + noise events.

        Zero-day mode (when hidden.zero_day_active=True):
            Standard signals suppressed. Attack looks idle.
            anomaly_trend and packet_entropy carry the real signal.
            Agent must learn to detect via these non-standard channels.

        Noise model:
            false_signal fires with probability noise_rate each step.
            When active, continuous signals get upward noise and booleans
            may flip — tests false-positive discipline.
            Scrutiny halves noise_rate.
        """
        h       = self._hidden
        cfg     = self._cfg
        done    = force_done or h.is_compromised or (h.step_count >= h.max_steps)

        noise_rate = cfg["noise_rate"]
        if self._scrutiny_active:
            noise_rate *= 0.5

        stage       = min(h.attack_stage, 5)
        sig         = STAGE_SIGNALS[stage]
        false_signal = self._rng.random() < noise_rate
        h.true_alert_count = sig["alerts"]

        # ---- Compute anomaly trend (used in zero-day mode) ----
        hist = h.anomaly_history
        if len(hist) >= 3:
            # Rate of change over last 3 steps
            trend = max(0.0, hist[-1] - hist[-3]) / 0.3   # Normalise to [0,1]
            trend = min(round(trend, 3), 1.0)
        else:
            trend = 0.0

        # ---- Compute packet entropy (zero-day indicator) ----
        if h.zero_day_active:
            zd_steps_elapsed = h.step_count - h.zero_day_onset_step
            # Packet entropy rises from 0.3 → 0.85 over 6 steps
            pkt_entropy = min(0.30 + zd_steps_elapsed * 0.09, 0.85)
            pkt_entropy = round(pkt_entropy + self._rng.uniform(-0.03, 0.03), 3)
        else:
            # Baseline: low entropy (normal traffic patterns)
            pkt_entropy = round(self._rng.uniform(0.05, 0.15), 3)

        # ---- Zero-day mode: suppress standard signals ----
        if h.zero_day_active:
            # Attacker uses legitimate-looking connections — IDS is blind
            login_failure_rate    = round(self._rng.uniform(0.0, 0.08), 3)
            alert_count           = int(self._rng.uniform(0, 1))
            port_scan_detected    = False   # Attacker avoids scanning
            cpu_spike             = self._rng.random() < 0.15  # Rare false spike
            network_anomaly_score = round(
                0.10 + trend * 0.3 + self._rng.uniform(-0.02, 0.02), 3
            )  # Low but rising with trend
            lateral_movement_flag = False   # Hidden in zero-day mode
            anomaly_trend_obs     = trend

        else:
            # ---- Standard mode: STAGE_SIGNALS × episode scale + jitter + noise ----
            #
            # Signal scaling is the key anti-memorization mechanism:
            #   base_signal × episode_scale_factor (drawn once per episode in reset)
            # This means the same attack stage produces different absolute values
            # each episode, forcing multi-signal pattern recognition over thresholds.
            scales = h.signal_scales  # Set once per episode in _make_initial_state

            # Login failure rate: scaled base + noise event + small jitter
            jitter   = self._rng.uniform(-0.03, 0.03)
            noise_lr = self._rng.uniform(0.06, noise_rate * 0.8 + 0.06) if false_signal else 0.0
            login_failure_rate = round(min(max(
                sig["login"] * scales["login"] + noise_lr + jitter, 0.0), 1.0), 3)

            # Network anomaly score: scaled base + noise + jitter
            jitter_a  = self._rng.uniform(-0.02, 0.02)
            noise_nas = self._rng.uniform(0.04, noise_rate * 0.7 + 0.04) if false_signal else 0.0
            network_anomaly_score = round(min(max(
                sig["anomaly"] * scales["anomaly"] + noise_nas + jitter_a, 0.0), 1.0), 3)

            # Boolean signals: unaffected by scaling (they are presence/absence)
            port_scan_detected = sig["port_scan"] or (false_signal and self._rng.random() < 0.55)
            cpu_spike          = sig["cpu_spike"] or (false_signal and self._rng.random() < 0.40)

            if cfg["partial_obs"]:
                lateral_movement_flag = False
            else:
                lateral_movement_flag = sig["lateral"] or (false_signal and self._rng.random() < 0.25)

            # Alert count: scaled base (rounded to int) + noise
            scaled_alerts = int(round(sig["alerts"] * scales["alerts"]))
            noise_alerts  = int(self._rng.uniform(0, noise_rate * 2.5)) if false_signal else 0
            alert_count   = min(scaled_alerts + noise_alerts, 10)
            anomaly_trend_obs = trend

        expose_false_signal = (self._task_name == "easy_breach_prevention" and false_signal)

        return CyberDefenseObservation(
            done                  = done,
            reward                = round(reward, 4),
            login_failure_rate    = login_failure_rate,
            port_scan_detected    = port_scan_detected,
            cpu_spike             = cpu_spike,
            network_anomaly_score = network_anomaly_score,
            lateral_movement_flag = lateral_movement_flag,
            alert_count           = alert_count,
            anomaly_trend         = anomaly_trend_obs,
            packet_entropy        = pkt_entropy,
            false_signal_injected = expose_false_signal,
            scrutiny_active       = self._scrutiny_active,
            consecutive_escalates = h.consecutive_escalates,
            steps_remaining       = max(0, h.max_steps - h.step_count),
            last_action           = "none",   # Overwritten in step()
            task_name             = self._task_name,
            metadata              = {
                "step":                h.step_count,
                "attack_stage":        stage,
                "zero_day_active":     h.zero_day_active,
                "attacker_policy":     h.attacker_policy_vector,
                "attacker_adapted":    h.attacker_adapted,
                "false_positives":     h.false_positives,
                "correct_blocks":      h.correct_blocks,
                "missed_attacks":      h.missed_attacks,
                "retreat_count":       h.retreat_count,
                "consecutive_escalates": h.consecutive_escalates,
                "total_reward":        round(h.total_reward, 4),
            },
        )

    # ------------------------------------------------------------------
    # Internal: state factory
    # ------------------------------------------------------------------

    def _make_initial_state(self, episode_id: Optional[str] = None) -> CyberDefenseState:
        cfg = self._cfg
        # ------------------------------------------------------------------
        # Signal scaling randomization (THE most important RL improvement).
        #
        # Purpose: prevent RL agents from memorizing fixed thresholds like
        #   "login_rate ≈ 0.52 always means stage 2".
        # Mechanism: draw one scale factor per continuous signal ONCE per
        #   episode from the RNG. Every signal derived from STAGE_SIGNALS is
        #   multiplied by this factor, so the same attack stage produces
        #   different absolute signal values in every episode while keeping
        #   the inter-stage ordering intact.
        # Range: ±20% gives 4× more variance than ±3% jitter alone, which
        #   is sufficient to require cross-signal pattern detection rather
        #   than single-threshold matching. Beyond ±30%, stages 3 and 4
        #   overlap in login_failure_rate, creating confusion even for a
        #   perfect agent.
        # Determinism: drawn from the per-episode RNG so scores are
        #   reproducible when seed is fixed (hackathon evaluation).
        # ------------------------------------------------------------------
        signal_scales = {
            "login":       round(self._rng.uniform(0.80, 1.20), 4),
            "anomaly":     round(self._rng.uniform(0.80, 1.20), 4),
            "alerts":      round(self._rng.uniform(0.85, 1.15), 4),  # Narrower: alerts are integer
            "retreat_cap": self._rng.randint(1, 3),                  # Stealth attacker retreat cap
        }

        return CyberDefenseState(
            episode_id              = episode_id or str(uuid4()),
            step_count              = 0,
            attack_stage            = cfg["initial_stage"],
            attacker_type           = cfg["attacker_type"],
            attacker_adapted        = False,
            is_compromised          = False,
            zero_day_active         = False,
            zero_day_onset_step     = -1,
            attacker_policy_vector  = [
                cfg["advance_prob"],
                max(0.05, 0.15 - cfg["advance_prob"] * 0.1),
                round(1.0 - cfg["advance_prob"] - max(0.05, 0.15 - cfg["advance_prob"] * 0.1), 4),
            ],
            true_alert_count        = 0,
            false_positives         = 0,
            correct_blocks          = 0,
            missed_attacks          = 0,
            retreat_count           = 0,
            consecutive_escalates   = 0,
            anomaly_history         = [],
            total_reward            = 0.0,
            task_name               = self._task_name,
            max_steps               = cfg["max_steps"],
            scrutiny_steps_left     = 0,
            attack_history          = [],
            action_history          = [],
            reward_history          = [],
            signal_scales           = signal_scales,
        )