# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """
# Data models for the Adaptive Cyber Defense Environment.

# Defines typed Pydantic models for Actions, Observations, and State
# following the OpenEnv specification. Separate from the echo template —
# this models a real SOC (Security Operations Center) decision environment.

# Action Space:
#     The agent chooses one of five defensive actions per step.
#     - monitor     : Observe without intervention (passive, no cost)
#     - isolate     : Isolate the suspected host from the network
#     - block       : Block suspicious IP/port at the firewall level
#     - escalate    : Escalate alert to senior analyst (increases scrutiny next step)
#     - do_nothing  : Take no action (penalized if attack is active)

# Observation Space:
#     Noisy signals from the network monitoring layer. Does NOT expose
#     the true hidden attack state — the agent must infer from signals.

# State Space:
#     Full hidden state, used by graders and evaluators only.
#     Not exposed to the agent during inference.
# """

# from typing import Any, Dict, List, Literal, Optional

# from openenv_core import Action, Observation, State
# from pydantic import Field


# # ---------------------------------------------------------------------------
# # Action
# # ---------------------------------------------------------------------------

# class CyberDefenseAction(Action):
#     """
#     Defensive action the agent can take each step.

#     The agent must choose exactly one action per step. Actions have
#     different effects on attack progression, false-positive rate, and reward.

#     Attributes:
#         action_type: One of five possible defensive actions.
#             - "monitor"    : Watch and log. No effect on attack. No FP cost.
#             - "isolate"    : Quarantine the host. Stops active attack if correct.
#                              High FP penalty if host is benign.
#             - "block"      : Block traffic at firewall. Partially slows attack.
#                              Medium FP penalty if benign.
#             - "escalate"   : Flag for human review. Adds scrutiny_bonus next step.
#                              No direct FP penalty.
#             - "do_nothing" : Explicit inaction. Penalized if attack is active.
#         reasoning: Optional free-text reasoning from the agent (logged, not used
#                    by the environment, useful for debugging agent behavior).
#     """

#     action_type: Literal["monitor", "isolate", "block", "escalate", "do_nothing"] = Field(
#         ...,
#         description=(
#             "Defensive action to take. One of: monitor, isolate, block, "
#             "escalate, do_nothing."
#         ),
#     )
#     reasoning: Optional[str] = Field(
#         default=None,
#         description="Optional agent reasoning for this action (not used by env logic).",
#     )


# # ---------------------------------------------------------------------------
# # Observation  (what the agent sees — noisy, partial)
# # ---------------------------------------------------------------------------

# class CyberDefenseObservation(Observation):
#     """
#     Noisy network monitoring signals observed by the agent.

#     These are the ONLY signals available to the agent. They do not reveal
#     the true attack stage. The agent must reason about whether an attack
#     is occurring and choose an appropriate defensive action.

#     Signal semantics:
#         - login_failure_rate   : Normalised failed login attempts [0.0, 1.0].
#                                  High values suggest brute-force or credential stuffing.
#         - port_scan_detected   : Whether a port scan was detected this step.
#         - cpu_spike            : Whether anomalous CPU usage was recorded.
#         - network_anomaly_score: Composite network anomaly score [0.0, 1.0].
#                                  Derived from traffic volume, timing, and entropy.
#         - lateral_movement_flag: True if suspicious east-west traffic was observed.
#         - alert_count          : Number of IDS/IPS alerts fired this step (0-10).
#         - false_signal_injected: True if a synthetic noise signal was injected.
#                                  Only visible in Easy task; hidden in Medium/Hard.
#         - scrutiny_active      : True if a prior "escalate" action is in effect,
#                                  giving the agent sharper (less noisy) signals.
#         - steps_remaining      : Steps left in the episode (helps agent manage urgency).
#         - last_action          : Echo of the last action taken (or "none" at reset).
#         - task_name            : Active task identifier for multi-task scripts.
#     """

#     # --- Network signals ---
#     login_failure_rate: float = Field(
#         default=0.0,
#         ge=0.0,
#         le=1.0,
#         description="Normalised failed login rate [0.0, 1.0].",
#     )
#     port_scan_detected: bool = Field(
#         default=False,
#         description="True if a port scan was detected this step.",
#     )
#     cpu_spike: bool = Field(
#         default=False,
#         description="True if anomalous CPU usage was recorded.",
#     )
#     network_anomaly_score: float = Field(
#         default=0.0,
#         ge=0.0,
#         le=1.0,
#         description="Composite network anomaly score [0.0, 1.0].",
#     )
#     lateral_movement_flag: bool = Field(
#         default=False,
#         description="True if suspicious east-west traffic was observed.",
#     )
#     alert_count: int = Field(
#         default=0,
#         ge=0,
#         le=10,
#         description="Number of IDS/IPS alerts fired this step.",
#     )

#     # --- Meta signals ---
#     false_signal_injected: bool = Field(
#         default=False,
#         description=(
#             "True if a synthetic noise event was injected. "
#             "Visible in Easy only; always False in Medium/Hard."
#         ),
#     )
#     scrutiny_active: bool = Field(
#         default=False,
#         description="True if escalate is in effect, reducing signal noise.",
#     )
#     steps_remaining: int = Field(
#         default=0,
#         ge=0,
#         description="Steps remaining in the episode.",
#     )
#     last_action: str = Field(
#         default="none",
#         description="Last action taken by the agent, or 'none' at reset.",
#     )
#     task_name: str = Field(
#         default="easy_breach_prevention",
#         description="Active task identifier.",
#     )


# # ---------------------------------------------------------------------------
# # State  (full hidden state — graders and evaluators only, not the agent)
# # ---------------------------------------------------------------------------

# class CyberDefenseState(State):
#     """
#     Full hidden environment state, separate from the agent's observation.

#     Used by:
#         - Graders: to compute deterministic task scores after an episode.
#         - Evaluators: to verify the environment is behaving correctly.
#         - state() API endpoint: returned verbatim.

#     NOT exposed to the agent during inference. The agent sees only the
#     noisy CyberDefenseObservation signals.

#     Attributes:
#         attack_stage      : True attacker progression (0 = dormant, 5 = full breach).
#         attacker_type     : Current attacker strategy ("fixed", "adaptive", "stealth").
#         attacker_adapted  : True if the attacker has changed strategy this episode.
#         is_compromised    : True if the system has been fully breached.
#         true_alert_count  : Ground-truth alert count (before noise injection).
#         false_positives   : Cumulative false positive actions taken by the agent.
#         correct_blocks    : Cumulative correct block/isolate actions.
#         missed_attacks    : Cumulative steps where an attack advanced uncontested.
#         total_reward      : Cumulative reward accumulated across the episode.
#         task_name         : Active task identifier.
#         max_steps         : Maximum episode length for this task.
#         scrutiny_steps_left: Steps remaining for the scrutiny bonus from escalate.
#         attack_history    : List of attack stages at each step (for grader analysis).
#     """

#     # --- True attack state ---
#     attack_stage: int = Field(
#         default=0,
#         ge=0,
#         le=5,
#         description="True attacker progression (0=dormant, 5=breach).",
#     )
#     attacker_type: str = Field(
#         default="fixed",
#         description="Attacker strategy: 'fixed', 'adaptive', or 'stealth'.",
#     )
#     attacker_adapted: bool = Field(
#         default=False,
#         description="True if attacker changed strategy during this episode.",
#     )
#     is_compromised: bool = Field(
#         default=False,
#         description="True if the system has been fully breached (attack_stage == 5).",
#     )

#     # --- Ground truth counters ---
#     true_alert_count: int = Field(
#         default=0,
#         ge=0,
#         description="Ground-truth alert count before noise injection.",
#     )
#     false_positives: int = Field(
#         default=0,
#         ge=0,
#         description="Cumulative false-positive defensive actions this episode.",
#     )
#     correct_blocks: int = Field(
#         default=0,
#         ge=0,
#         description="Cumulative correct block/isolate actions this episode.",
#     )
#     missed_attacks: int = Field(
#         default=0,
#         ge=0,
#         description="Cumulative steps where attack advanced uncontested.",
#     )

#     # --- Reward tracking ---
#     total_reward: float = Field(
#         default=0.0,
#         description="Cumulative reward accumulated across the episode.",
#     )

#     # --- Episode metadata ---
#     task_name: str = Field(
#         default="easy_breach_prevention",
#         description="Active task identifier.",
#     )
#     max_steps: int = Field(
#         default=10,
#         ge=1,
#         description="Maximum episode length for the active task.",
#     )
#     scrutiny_steps_left: int = Field(
#         default=0,
#         ge=0,
#         description="Remaining steps for escalate scrutiny bonus.",
#     )

#     # --- History (for graders) ---
#     attack_history: List[int] = Field(
#         default_factory=list,
#         description="Attack stage at each step; used by graders.",
#     )
#     action_history: List[str] = Field(
#         default_factory=list,
#         description="Agent actions at each step; used by graders.",
#     )
#     reward_history: List[float] = Field(
#         default_factory=list,
#         description="Per-step rewards; used by graders.",
#     )
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Adaptive Cyber Defense Environment.

Observation, Action, and State Pydantic models following the OpenEnv spec.

New in this version:
    - consecutive_escalates added to Observation (exposed to agent)
    - retreat_count added to State (enforces retreat cap for stealth attacker)
    - zero_day_active added to State (tracks zero-day injection event)
    - task_score added to State (attached by grade() for inference.py)
    - attacker_policy_vector added to State (RL-vs-RL policy fingerprint)
"""

from typing import Any, Dict, List, Literal, Optional

from openenv_core import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class CyberDefenseAction(Action):
    """
    Defensive action chosen by the agent each step.

    action_type choices:
        monitor    : Observe only. Free on idle network. Costs -0.10 if attack active.
        isolate    : Quarantine host. Full stop if attack present (+0.40).
                     False positive penalty (-0.12) if idle.
        block      : Firewall rule. Partial stop if attack present (+0.25, stage-1).
                     False positive penalty (-0.08) if idle.
        escalate   : Activate scrutiny for 2 steps (halves noise). Neutral reward.
                     COSTS -0.06 if attack is active and consecutive_escalates >= 2.
        do_nothing : Explicit inaction. Costs -0.18 if attack active.
    """

    action_type: Literal["monitor", "isolate", "block", "escalate", "do_nothing"] = Field(
        ...,
        description="Defensive action. One of: monitor, isolate, block, escalate, do_nothing.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional agent reasoning (logged only, not used by env logic).",
    )


# ---------------------------------------------------------------------------
# Observation  (what the agent sees — noisy, partial)
# ---------------------------------------------------------------------------

class CyberDefenseObservation(Observation):
    """
    Noisy network monitoring signals observed by the agent.

    Does NOT reveal true attack stage. Agent must infer from signals.

    New field: consecutive_escalates — how many escalate actions in a row.
    Agent should commit to block/isolate after seeing this >= 2.

    Zero-day signals: when zero_day_active=True (hard task only), the normal
    signal-to-stage correlation breaks down. Signals are deceptively low while
    the attack is at stage 3+. The agent must rely on anomaly_trend and
    packet_entropy instead of the usual login/alert thresholds.
    """

    # --- Standard network signals ---
    login_failure_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Normalised failed login rate [0,1]. High = brute force likely.",
    )
    port_scan_detected: bool = Field(
        default=False,
        description="True if port scan was detected this step.",
    )
    cpu_spike: bool = Field(
        default=False,
        description="True if anomalous CPU usage was recorded.",
    )
    network_anomaly_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Composite network anomaly score [0,1].",
    )
    lateral_movement_flag: bool = Field(
        default=False,
        description="True if suspicious east-west traffic observed. Hidden in hard task.",
    )
    alert_count: int = Field(
        default=0, ge=0, le=10,
        description="Number of IDS/IPS alerts fired this step.",
    )

    # --- Zero-day specific signals (only meaningful in hard/zero-day tasks) ---
    anomaly_trend: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Rate of change in anomaly score over last 3 steps [0,1]. "
            "Key zero-day indicator — rises even when absolute score is low."
        ),
    )
    packet_entropy: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Normalised entropy of network packet sizes [0,1]. "
            "Zero-day attacks often use unusual packet distributions."
        ),
    )

    # --- Meta signals ---
    false_signal_injected: bool = Field(
        default=False,
        description="True if synthetic noise was injected. Visible in easy task only.",
    )
    scrutiny_active: bool = Field(
        default=False,
        description="True if escalate is in effect (noise halved for 2 steps).",
    )
    consecutive_escalates: int = Field(
        default=0, ge=0,
        description=(
            "How many escalate actions taken in a row. "
            "If >= 2, commit to block or isolate — more escalation costs points."
        ),
    )
    steps_remaining: int = Field(
        default=0, ge=0,
        description="Steps remaining in the episode.",
    )
    last_action: str = Field(
        default="none",
        description="Last action taken by the agent, or 'none' at reset.",
    )
    task_name: str = Field(
        default="easy_breach_prevention",
        description="Active task identifier.",
    )


# ---------------------------------------------------------------------------
# State  (full hidden state — graders / evaluators only, NOT the agent)
# ---------------------------------------------------------------------------

class CyberDefenseState(State):
    """
    Full hidden environment state. Used by graders and evaluators.
    Never exposed to the agent during inference.

    New fields vs previous version:
        retreat_count          : Stealth attacker retreat counter (capped at 2).
        consecutive_escalates  : Mirrors observation field for grader analysis.
        zero_day_active        : True when zero-day injection is in effect.
        attacker_policy_vector : RL-vs-RL fingerprint of attacker's current policy.
        task_score             : Pre-computed grade() result (attached by state property).
    """

    # --- True attack state ---
    attack_stage: int = Field(default=0, ge=0, le=5)
    attacker_type: str = Field(default="fixed")
    attacker_adapted: bool = Field(default=False)
    is_compromised: bool = Field(default=False)

    # --- Zero-day state ---
    zero_day_active: bool = Field(
        default=False,
        description="True when a zero-day injection is active this episode.",
    )
    zero_day_onset_step: int = Field(
        default=-1,
        description="Step at which zero-day was injected (-1 = not injected).",
    )

    # --- RL-vs-RL attacker policy ---
    attacker_policy_vector: List[float] = Field(
        default_factory=lambda: [0.5, 0.5, 0.0],
        description=(
            "Attacker's internal policy state [advance_prob, retreat_prob, hold_prob]. "
            "Updated each step based on defender behavior (RL-vs-RL mechanic). "
            "Visible in state() so evaluators can see the adaptation."
        ),
    )

    # --- Ground truth counters ---
    true_alert_count: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    correct_blocks: int = Field(default=0, ge=0)
    missed_attacks: int = Field(default=0, ge=0)
    retreat_count: int = Field(
        default=0, ge=0,
        description="Cumulative retreats taken by stealth attacker this episode.",
    )
    consecutive_escalates: int = Field(
        default=0, ge=0,
        description="Consecutive escalate actions taken by the agent.",
    )

    # --- Anomaly trend history (for zero-day signal) ---
    anomaly_history: List[float] = Field(
        default_factory=list,
        description="Raw anomaly scores per step for trend computation.",
    )

    # --- Reward tracking ---
    total_reward: float = Field(default=0.0)

    # --- Episode metadata ---
    task_name: str = Field(default="easy_breach_prevention")
    max_steps: int = Field(default=10, ge=1)
    scrutiny_steps_left: int = Field(default=0, ge=0)

    # --- History ---
    attack_history: List[int] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)

    # --- Signal scaling (per-episode anti-memorization factors) ---
    signal_scales: Dict[str, Any] = Field(
        default_factory=lambda: {
            "login": 1.0, "anomaly": 1.0, "alerts": 1.0, "retreat_cap": 2
        },
        description=(
            "Per-episode signal scale factors drawn once at reset(). "
            "Prevents RL threshold memorization. login/anomaly in [0.80,1.20], "
            "alerts in [0.85,1.15], retreat_cap in {1,2,3}. "
            "Exposed in state() so evaluators can verify scaling is active."
        ),
    )

    # --- Grade (attached by state property) ---
    task_score: Optional[float] = Field(
        default=None,
        description="Pre-computed grade(). Attached by state property, not stored in env.",
    )