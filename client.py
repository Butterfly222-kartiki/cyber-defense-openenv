# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """Cyber Defense Environment Client."""

# from typing import Dict

# from openenv.core import EnvClient
# from openenv.core.client_types import StepResult
# from openenv.core.env_server.types import State

# from .models import CyberDefenseAction, CyberDefenseObservation


# class CyberDefenseEnv(
#     EnvClient[CyberDefenseAction, CyberDefenseObservation, State]
# ):
#     """
#     Client for the Cyber Defense Environment.

#     This client maintains a persistent WebSocket connection to the environment server,
#     enabling efficient multi-step interactions with lower latency.
#     Each client instance has its own dedicated environment session on the server.

#     Example:
#         >>> # Connect to a running server
#         >>> with CyberDefenseEnv(base_url="http://localhost:8000") as client:
#         ...     result = client.reset()
#         ...     print(result.observation.echoed_message)
#         ...
#         ...     result = client.step(CyberDefenseAction(message="Hello!"))
#         ...     print(result.observation.echoed_message)

#     Example with Docker:
#         >>> # Automatically start container and connect
#         >>> client = CyberDefenseEnv.from_docker_image("cyber_defense-env:latest")
#         >>> try:
#         ...     result = client.reset()
#         ...     result = client.step(CyberDefenseAction(message="Test"))
#         ... finally:
#         ...     client.close()
#     """

#     def _step_payload(self, action: CyberDefenseAction) -> Dict:
#         """
#         Convert CyberDefenseAction to JSON payload for step message.

#         Args:
#             action: CyberDefenseAction instance

#         Returns:
#             Dictionary representation suitable for JSON encoding
#         """
#         return {
#             "message": action.message,
#         }

#     def _parse_result(self, payload: Dict) -> StepResult[CyberDefenseObservation]:
#         """
#         Parse server response into StepResult[CyberDefenseObservation].

#         Args:
#             payload: JSON response data from server

#         Returns:
#             StepResult with CyberDefenseObservation
#         """
#         obs_data = payload.get("observation", {})
#         observation = CyberDefenseObservation(
#             echoed_message=obs_data.get("echoed_message", ""),
#             message_length=obs_data.get("message_length", 0),
#             done=payload.get("done", False),
#             reward=payload.get("reward"),
#             metadata=obs_data.get("metadata", {}),
#         )

#         return StepResult(
#             observation=observation,
#             reward=payload.get("reward"),
#             done=payload.get("done", False),
#         )

#     def _parse_state(self, payload: Dict) -> State:
#         """
#         Parse server response into State object.

#         Args:
#             payload: JSON response from state request

#         Returns:
#             State object with episode_id and step_count
#         """
#         return State(
#             episode_id=payload.get("episode_id"),
#             step_count=payload.get("step_count", 0),
#         )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cyber Defense Environment Client.

Wraps the WebSocket API exposed by app.py into a typed Python client.

Key facts about EnvClient (confirmed from source):
    - Communicates exclusively over WebSocket /ws (not HTTP REST).
    - reset(**kwargs) sends {type:'reset', data:kwargs} over WS.
      The server calls session_env.reset(**valid_kwargs) — so any kwarg
      accepted by CyberDefenseEnvironment.reset() (seed, task, episode_id)
      is forwarded transparently.
    - step(action) sends {type:'step', data:_step_payload(action)}.
    - state() sends {type:'state'} and calls _parse_state() on the response.
    - StepResult is a dataclass with ONLY: observation, reward, done.
      There is no 'info' field — do NOT pass info= to StepResult().
    - reset() returns StepResult, not a raw observation.
      Caller must use result.observation to get the CyberDefenseObservation.
"""

from typing import Any, Dict, Optional

try:
    from openenv_core import EnvClient
    from openenv_core.client_types import StepResult
except ImportError:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult

try:
    from .models import (
        CyberDefenseAction,
        CyberDefenseObservation,
        CyberDefenseState,
    )
except ModuleNotFoundError:
    from models import (
        CyberDefenseAction,
        CyberDefenseObservation,
        CyberDefenseState,
    )


class CyberDefenseEnv(
    EnvClient[CyberDefenseAction, CyberDefenseObservation, CyberDefenseState]
):
    """
    Client for the Adaptive Cyber Defense Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance owns one server-side environment session. Task
    switching is done by calling reset(task=<name>) — the same WS session
    handles all 3 tasks in sequence without reconnecting.

    Usage (async):
        async with CyberDefenseEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task="easy_breach_prevention")
            obs = result.observation
            while not obs.done:
                result = await env.step(CyberDefenseAction(action_type="monitor"))
                obs = result.observation

    Usage (Docker):
        env = await CyberDefenseEnv.from_docker_image("cyber_defense-env:latest")
        result = await env.reset(task="hard_stealth_defense")
        obs = result.observation
    """

    # ------------------------------------------------------------------
    # _step_payload — CyberDefenseAction → JSON dict for WS step message
    # ------------------------------------------------------------------

    def _step_payload(self, action: CyberDefenseAction) -> Dict[str, Any]:
        """
        Serialise CyberDefenseAction to the dict sent as WS step data.

        The server's action deserialiser reads action_type (required) and
        reasoning (optional). Only include fields the server expects.
        """
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        return payload

    # ------------------------------------------------------------------
    # _parse_result — WS response dict → StepResult[CyberDefenseObservation]
    # ------------------------------------------------------------------

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[CyberDefenseObservation]:
        """
        Parse the WS response data into a typed StepResult.

        The server serialises the Observation via serialize_observation(),
        which calls obs.model_dump() and returns the fields at the top
        level of the 'data' dict (not nested under 'observation').

        StepResult only has: observation, reward, done.
        Do NOT pass info= — it does not exist on StepResult.
        """
        observation = CyberDefenseObservation(
            # Network signals
            login_failure_rate=payload.get("login_failure_rate", 0.0),
            port_scan_detected=payload.get("port_scan_detected", False),
            cpu_spike=payload.get("cpu_spike", False),
            network_anomaly_score=payload.get("network_anomaly_score", 0.0),
            lateral_movement_flag=payload.get("lateral_movement_flag", False),
            alert_count=payload.get("alert_count", 0),
            # Zero-day specific signals (critical for hard/zero-day tasks)
            anomaly_trend=payload.get("anomaly_trend", 0.0),
            packet_entropy=payload.get("packet_entropy", 0.0),
            # Meta signals
            false_signal_injected=payload.get("false_signal_injected", False),
            scrutiny_active=payload.get("scrutiny_active", False),
            consecutive_escalates=payload.get("consecutive_escalates", 0),
            steps_remaining=payload.get("steps_remaining", 0),
            last_action=payload.get("last_action", "none"),
            task_name=payload.get("task_name", "easy_breach_prevention"),
            # OpenEnv base fields
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=payload.get("metadata", {}),
        )

        # StepResult dataclass: only observation, reward, done
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    # ------------------------------------------------------------------
    # _parse_state — WS state response dict → CyberDefenseState
    # ------------------------------------------------------------------

    def _parse_state(self, payload: Dict[str, Any]) -> CyberDefenseState:
        """
        Parse the WS {type:'state'} response into CyberDefenseState.

        The server sends session_env.state.model_dump() as the data dict.
        All fields are at the top level — no nesting.

        task_score is attached by state property (grade() result) and
        travels here as an extra field (State.extra='allow').
        """
        return CyberDefenseState(
            # True attack state
            attack_stage=payload.get("attack_stage", 0),
            attacker_type=payload.get("attacker_type", "fixed"),
            attacker_adapted=payload.get("attacker_adapted", False),
            is_compromised=payload.get("is_compromised", False),
            # Ground truth counters
            true_alert_count=payload.get("true_alert_count", 0),
            false_positives=payload.get("false_positives", 0),
            correct_blocks=payload.get("correct_blocks", 0),
            missed_attacks=payload.get("missed_attacks", 0),
            # Reward tracking
            total_reward=payload.get("total_reward", 0.0),
            # Episode metadata
            task_name=payload.get("task_name", "easy_breach_prevention"),
            max_steps=payload.get("max_steps", 10),
            scrutiny_steps_left=payload.get("scrutiny_steps_left", 0),
            # History lists
            attack_history=payload.get("attack_history", []),
            action_history=payload.get("action_history", []),
            reward_history=payload.get("reward_history", []),
            # Signal scaling factors (for evaluator verification)
            signal_scales=payload.get("signal_scales", {
                "login": 1.0, "anomaly": 1.0, "alerts": 1.0, "retreat_cap": 2
            }),
            # OpenEnv base fields
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )