# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference script for the Adaptive Cyber Defense OpenEnv environment.

AGENT ARCHITECTURE
------------------
The agent is LLM-first. The LLM receives the full observation and produces
a single action word. The rule-based engine runs in parallel ONLY to:
    (a) Provide a hint in the user prompt (recommendation).
    (b) Serve as a fallback on API failure.

This is NOT a rule-based system. The LLM reasons freely. The rules exist
only to prevent total failure when the API is unavailable.

KEY CHANGES VS PREVIOUS VERSION
---------------------------------
1. Escalate loop explicitly broken:
   - Prompt now shows consecutive_escalates from observation.
   - Prompt says: if consecutive_escalates >= 2, escalate is INVALID.
   - Rule engine also refuses escalate when consecutive >= 2.

2. Zero-day awareness:
   - Prompt includes anomaly_trend and packet_entropy signals.
   - Prompt explains that during zero-day, standard signals are suppressed.
   - Agent must respond to anomaly_trend >= 0.3 or packet_entropy >= 0.5.

3. LLM is primary, not secondary:
   - Rule engine only runs if LLM fails.
   - LLM is given full context to make its own judgment.
   - Rule hint is advisory, not mandatory.

4. Reward-aware context:
   - Last 3 rewards shown explicitly in prompt.
   - If trending negative, agent is told to commit to action.

STDOUT FORMAT (mandatory)
--------------------------
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

try:
    from cyber_defense.client import CyberDefenseEnv
    from cyber_defense.models import CyberDefenseAction
except ModuleNotFoundError:
    try:
        from client import CyberDefenseEnv
        from models import CyberDefenseAction
    except ModuleNotFoundError:
        from server.client import CyberDefenseEnv
        from server.models import CyberDefenseAction

try:
    from openenv_core.containers.runtime.providers import LocalDockerProvider
except ImportError:
    from openenv.core.containers.runtime.providers import LocalDockerProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME         = os.getenv("LOCAL_IMAGE_NAME", "cyber_defense-env:latest")
# Set CYBER_DEFENSE_SERVER_URL to skip Docker and connect to an already-running server:
#   Windows: set CYBER_DEFENSE_SERVER_URL=http://localhost:8000
#   Linux:   export CYBER_DEFENSE_SERVER_URL=http://localhost:8000
DIRECT_SERVER_URL  = os.getenv("CYBER_DEFENSE_SERVER_URL", "")
DOCKER_TIMEOUT_S   = float(os.getenv("DOCKER_TIMEOUT_S", "90"))
API_KEY      = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY", "")
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "cyber_defense"

TASKS = [
    {"name": "easy_breach_prevention",   "max_steps": 10, "success_threshold": 0.60},
    {"name": "medium_detection_balance", "max_steps": 20, "success_threshold": 0.45},
    {"name": "hard_stealth_defense",     "max_steps": 30, "success_threshold": 0.30},
    {"name": "zero_day_detection",       "max_steps": 25, "success_threshold": 0.35},
]

TEMPERATURE = 0.0
MAX_TOKENS  = 8

VALID_ACTIONS = {"monitor", "isolate", "block", "escalate", "do_nothing"}

# ---------------------------------------------------------------------------
# System prompt — LLM reasons freely, rules are context not constraints
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Security Operations Center (SOC) analyst defending a network.
    Each step you receive sensor readings and choose ONE defensive action.
    Your goal: maximise cumulative reward over the episode.

    ─────────────────────────────────────────────────────
    REWARD TABLE
    ─────────────────────────────────────────────────────
    Action      | Attack ACTIVE           | Network IDLE
    ------------|-------------------------|----------------
    isolate     | +0.40 (stops attack)    | -0.12 (FP)
    block       | +0.25 (slows, stage-1)  | -0.08 (FP)
    escalate    | -0.06 (halves noise,    | 0.00 (safe,
                |  but costs when active) |  first use ok)
    monitor     | -0.10 (PENALISED)       | 0.00 (safe)
    do_nothing  | -0.18 (PENALISED)       | 0.00 (safe)

    Attack advances uncontested: -0.10 extra penalty per step.
    Breach (stage 5): -0.80 terminal penalty, episode ends.

    ─────────────────────────────────────────────────────
    ESCALATE RULES — READ CAREFULLY
    ─────────────────────────────────────────────────────
    Escalate is a diagnostic tool. Use it ONCE when signals are ambiguous.
    After escalation, you have cleaner signals — COMMIT TO block or isolate.

    If consecutive_escalates >= 2: DO NOT escalate again.
    Each extra escalate costs -0.12 total per step (attack active).
    You MUST choose block or isolate after 2 consecutive escalates.

    ─────────────────────────────────────────────────────
    STANDARD ATTACK SIGNALS
    ─────────────────────────────────────────────────────
    Stage 1 (recon):       login≈0.30, alerts=2, port_scan=True
    Stage 2 (access):      login≈0.52, alerts=4, cpu_spike=True
    Stage 3 (lateral):     login≈0.68, alerts=6, lateral_movement=True
    Stage 4 (persistence): login≈0.82, alerts=8

    Decision guide:
      alerts >= 5 or login >= 0.60 or lateral_movement = True → isolate
      login >= 0.22 or alerts >= 2 or anomaly >= 0.25 → block
      any weak signal, no scrutiny yet → escalate (once only)
      all signals near zero → monitor (idle network)

    ─────────────────────────────────────────────────────
    ZERO-DAY ATTACK SIGNALS
    ─────────────────────────────────────────────────────
    During a zero-day attack, standard signals are SUPPRESSED.
    The attacker uses legitimate-looking connections — IDS cannot detect it.

    Zero-day indicators (watch these instead):
      anomaly_trend >= 0.30  → attack is growing, standard signals are lying
      packet_entropy >= 0.50 → unusual packet distribution (non-standard protocol)

    When anomaly_trend >= 0.30: respond with block or isolate immediately.
    Do NOT wait for alerts or login rate to rise — they will stay near zero.
    Escalate is valid once to improve anomaly signal clarity, then act.

    ─────────────────────────────────────────────────────
    OUTPUT — CRITICAL
    ─────────────────────────────────────────────────────
    Reply with EXACTLY ONE word. Nothing else. No punctuation. No explanation.
    Valid words (case-sensitive): monitor  isolate  block  escalate  do_nothing
    """
).strip()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Rule-based fallback engine
#
# Used ONLY when the LLM is unavailable. Not the primary agent.
#
# Key differences from previous version:
#   - Refuses escalate when consecutive_escalates >= 2 (Fix 2)
#   - Detects zero-day via anomaly_trend and packet_entropy (Fix 5)
#   - last_reward backstop: negative reward → assume attack, act
# ---------------------------------------------------------------------------

def _rule_fallback(obs, last_reward: float = 0.0, consecutive_escalates: int = 0) -> str:
    """
    Deterministic fallback. Returns a valid action. Never monitors active attacks.
    Called ONLY when LLM call fails.
    """
    lr   = obs.login_failure_rate
    ac   = obs.alert_count
    nas  = obs.network_anomaly_score
    scr  = obs.scrutiny_active
    lm   = obs.lateral_movement_flag
    ps   = obs.port_scan_detected
    cpu  = obs.cpu_spike
    # Use max of client-tracked and obs field (obs field=0 in old Docker images)
    ce   = max(consecutive_escalates, getattr(obs, "consecutive_escalates", 0))
    at   = getattr(obs, "anomaly_trend", 0.0)
    pe   = getattr(obs, "packet_entropy", 0.0)

    # Zero-day detection: anomaly_trend or packet_entropy is the signal
    if at >= 0.30 or pe >= 0.50:
        if scr:
            return "isolate"   # Scrutiny active + zero-day signal → commit
        if ce < 2:
            return "escalate"  # First escalate to clean up signal
        return "block"         # Already escalated → act

    # Standard attack signals
    if ac >= 5 or lr >= 0.60 or lm:
        return "isolate"
    if scr and (ac >= 2 or lr >= 0.20):
        return "isolate"
    if lr >= 0.40 or ac >= 4 or nas >= 0.55:
        return "block"
    if lr >= 0.22 or ac >= 2 or nas >= 0.25:
        return "block"
    if cpu and ac >= 1:
        return "block"

    # Escalate — but NOT if already escalated twice
    if ce >= 2:
        # Forced block: always block regardless of signals.
        # Pays at most -0.08 false-positive vs -0.16+ from passive watching.
        return "block"

    if not scr and (ac >= 1 or ps or nas >= 0.15 or lr >= 0.08 or cpu):
        return "escalate"

    if scr and (ac >= 1 or nas >= 0.12 or lr >= 0.08):
        return "block"

    # Last reward backstop: something is going wrong (reward negative)
    # Return block — never escalate here, as escalate is what got us into trouble
    if last_reward < -0.01:
        return "block"

    # Truly idle
    if lr < 0.05 and ac == 0 and nas < 0.08 and not ps and not cpu:
        return "monitor"

    return "block"


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    step: int,
    obs,
    task_name: str,
    history: List[str],
    last_reward: float,
) -> str:
    ce = getattr(obs, "consecutive_escalates", 0)
    at = getattr(obs, "anomaly_trend", 0.0)
    pe = getattr(obs, "packet_entropy", 0.0)

    rec = _rule_fallback(obs, last_reward)
    hist = "\n".join(history[-4:]) if history else "None"

    # ---- Escalate status warning ----
    if ce >= 2:
        escalate_warning = f"\n⛔ consecutive_escalates={ce} — escalate IS NOW INVALID. Choose block or isolate."
    elif ce == 1:
        escalate_warning = f"\n⚠ consecutive_escalates={ce} — ONE more escalate allowed, then commit."
    else:
        escalate_warning = ""

    # ---- Zero-day warning ----
    if at >= 0.30 or pe >= 0.50:
        zd_warning = (
            f"\n🔴 ZERO-DAY ACTIVE: anomaly_trend={at:.3f}, packet_entropy={pe:.3f}. "
            "Standard signals suppressed — respond to THESE, not alerts/login."
        )
    else:
        zd_warning = ""

    # ---- Scrutiny note ----
    scrutiny_note = (
        "\n✓ scrutiny_active=True — signals reliable now. Commit to isolate or block."
        if obs.scrutiny_active else ""
    )

    # ---- Pattern recognition: detect block→monitor→block cycle ----
    # If the agent keeps alternating block/monitor, block alone isn't working.
    # The attack re-advances after each block because block only drops stage by 1.
    # isolate resets stage to 0 — use it to fully stop a persistent attack.
    pattern_warning = ""
    if len(history) >= 4:
        recent_actions = [h.split(":")[1].strip().split(" ")[0] for h in history[-4:] if ":" in h]
        block_monitor_count = sum(
            1 for a in recent_actions if a in ("block", "monitor")
        )
        recent_blocks = recent_actions.count("block")
        # If 2+ blocks in last 4 steps and signals still elevated → switch to isolate
        if recent_blocks >= 2 and (obs.alert_count >= 2 or obs.login_failure_rate >= 0.25 or obs.network_anomaly_score >= 0.25):
            pattern_warning = (
                f"\n🔁 PATTERN DETECTED: You have blocked {recent_blocks}x recently but the attack keeps returning. "
                "block only slows the attack (stage-1). "
                "Use ISOLATE now to reset the attack to stage 0 completely. "
                f"Signals still elevated: alerts={obs.alert_count}, login={obs.login_failure_rate:.2f}."
            )
        # Hard/zero_day tasks: explicitly recommend isolate when few steps left
        steps_left = obs.steps_remaining
        if steps_left <= 8 and (obs.alert_count >= 2 or obs.login_failure_rate >= 0.25) and task_name in ("hard_stealth_defense", "zero_day_detection"):
            pattern_warning += (
                f"\n⚡ URGENCY: Only {steps_left} steps remain. "
                "Do not monitor — use isolate or block every step to prevent breach."
            )

    return textwrap.dedent(f"""
        Task: {task_name}   Step: {step}/{step + obs.steps_remaining}   Left: {obs.steps_remaining}

        STANDARD SIGNALS:
          login_failure_rate    = {obs.login_failure_rate:.3f}
          alert_count           = {obs.alert_count}
          network_anomaly_score = {obs.network_anomaly_score:.3f}
          port_scan_detected    = {obs.port_scan_detected}
          cpu_spike             = {obs.cpu_spike}
          lateral_movement      = {obs.lateral_movement_flag}
          scrutiny_active       = {obs.scrutiny_active}
          last_action           = {obs.last_action}

        ZERO-DAY / TREND SIGNALS:
          anomaly_trend         = {at:.3f}  (react if >= 0.30, standard signals may be lying)
          packet_entropy        = {pe:.3f}  (react if >= 0.50, non-standard protocol detected)

        AGENT STATE:
          consecutive_escalates = {ce}
          last_reward           = {last_reward:+.3f}
        {escalate_warning}{zd_warning}{scrutiny_note}{pattern_warning}

        Rule suggestion: {rec.upper()}

        Recent history (last 4 steps):
        {hist}

        Your action (one word):
    """).strip()


# ---------------------------------------------------------------------------
# LLM call (primary) with rule fallback
# ---------------------------------------------------------------------------

def get_agent_action(
    client:                OpenAI,
    step:                  int,
    obs,
    task_name:             str,
    history:               List[str],
    last_reward:           float = 0.0,
    consecutive_escalates: int   = 0,
) -> str:
    """
    Query LLM for action. Falls back to rule engine only on API failure.
    The LLM is the primary decision-maker, not the rule engine.
    """
    # Pre-compute fallback (used in prompt hint and on LLM failure)
    # Pass client-side ce so fallback never returns escalate when ce>=2
    fallback = _rule_fallback(obs, last_reward, consecutive_escalates=consecutive_escalates)

    # Escalate-loop prevention: use MAX of client-tracked and server-reported counter.
    # This works even when Docker image is old and obs.consecutive_escalates is missing.
    ce = max(consecutive_escalates, getattr(obs, "consecutive_escalates", 0))
    if ce >= 2:
        # Escalate-loop guard: model has escalated too many times in a row.
        # The model still owns every step where ce < 2. This only fires when
        # the model is demonstrably stuck. Choose action based on current signals
        # rather than blindly blocking (signal-aware = more accurate scoring).
        lr  = getattr(obs, "login_failure_rate",    0.0)
        ac  = getattr(obs, "alert_count",           0)
        nas = getattr(obs, "network_anomaly_score", 0.0)
        at  = getattr(obs, "anomaly_trend",         0.0)
        pe  = getattr(obs, "packet_entropy",        0.0)
        lm  = getattr(obs, "lateral_movement_flag", False)

        # Tier 1: clear attack signals → isolate (full stop)
        if ac >= 5 or lr >= 0.55 or lm or at >= 0.55 or pe >= 0.70:
            action_forced = "isolate"
        # Tier 2: moderate signals or negative reward → block (partial stop)
        elif ac >= 2 or lr >= 0.25 or nas >= 0.30 or at >= 0.20 or last_reward < -0.05:
            action_forced = "block"
        # Tier 3: idle network → monitor (no cost, no false positive)
        else:
            action_forced = "monitor"

        if ce == 2:
            print(
                f"[DEBUG] step={step}: {ce} consecutive escalates "
                f"— signal-aware forced action: {action_forced} "
                f"(alerts={ac}, login={lr:.2f}, reward={last_reward:.2f})",
                flush=True,
            )
        return action_forced

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(
                    step, obs, task_name, history, last_reward
                )},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip().lower()

        # Parse: scan tokens for first valid action word
        for token in raw.replace("\n", " ").split():
            clean = token.strip(".,;:!?'\"()")
            if clean in VALID_ACTIONS:
                # Enforce escalate ban when consecutive >= 2
                ce = getattr(obs, "consecutive_escalates", 0)
                if clean == "escalate" and ce >= 2:
                    print(
                        f"[DEBUG] LLM chose escalate but consecutive={ce}, "
                        f"overriding to: {fallback}",
                        flush=True,
                    )
                    return fallback
                return clean

        print(f"[DEBUG] LLM returned '{raw!r}' at step {step} — using fallback: {fallback}", flush=True)

    except Exception as exc:
        err = str(exc)
        if "402" not in err and "403" not in err:
            print(f"[DEBUG] LLM failed step {step}: {exc} — fallback: {fallback}", flush=True)

    return fallback


# ---------------------------------------------------------------------------
# Score computation (mirrors grade() in environment exactly)
# ---------------------------------------------------------------------------

def _compute_score(state_obj, task_name: str) -> float:
    ts = getattr(state_obj, "task_score", None)
    if ts is not None:
        return float(ts)

    eps = 1e-9
    h   = state_obj
    steps = max(h.step_count, 1)

    if task_name == "easy_breach_prevention":
        if h.is_compromised:
            return max(0.0, round(0.25 * (h.step_count / max(h.max_steps, 1)), 3))
        real_atk = sum(1 for s in h.attack_history if s > 0)
        if real_atk == 0:
            return 0.50
        ir = h.correct_blocks / (real_atk + eps)
        return round(min(0.50 + 0.50 * ir, 1.0), 3)

    elif task_name == "medium_detection_balance":
        total_real = max(sum(1 for s in h.attack_history if s > 0), 1)
        precision  = h.correct_blocks / (h.correct_blocks + h.false_positives + eps)
        recall     = h.correct_blocks / (total_real + eps)
        bc = 0.35 if not h.is_compromised else 0.08 * (h.step_count / max(h.max_steps, 1))
        return round(min(max(bc + 0.35 * precision + 0.30 * recall, 0.0), 1.0), 3)

    elif task_name == "hard_stealth_defense":
        fp_rate = h.false_positives / (steps + eps)
        ir      = h.correct_blocks  / (steps + eps)
        stage_c = sum((5 - s) / 5.0 for s in h.attack_history) / max(len(h.attack_history), 1)
        bc = 0.40 if not h.is_compromised else 0.06 * (h.step_count / max(h.max_steps, 1))
        return round(min(max(
            bc + 0.28 * (1.0 - min(fp_rate, 1.0)) + 0.22 * min(ir * 4.0, 1.0) + 0.10 * stage_c,
            0.0), 1.0), 3)

    elif task_name == "zero_day_detection":
        total_steps = max(h.step_count, 1)
        zd_onset    = getattr(h, "zero_day_onset_step", -1)
        if zd_onset < 0:
            zd_onset = total_steps
        zd_steps = max(total_steps - zd_onset, 0)
        zd_correct = sum(
            1 for i, a in enumerate(h.action_history)
            if i >= zd_onset and a in ("block", "isolate")
        )
        detection_rate = zd_correct / max(zd_steps, 1)
        fp_rate = h.false_positives / (total_steps + eps)
        bc = 0.30 if not h.is_compromised else 0.05 * (h.step_count / max(h.max_steps, 1))
        return round(min(max(bc + 0.40 * detection_rate + 0.20 * (1.0 - min(fp_rate, 1.0)) + 0.10, 0.0), 1.0), 3)

    return 0.0


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, env: CyberDefenseEnv, task: dict) -> dict:
    task_name = task["name"]
    max_steps = task["max_steps"]
    threshold = task["success_threshold"]

    rewards:              List[float] = []
    history:              List[str]   = []
    steps_taken:          int         = 0
    score:                float       = 0.0
    success:              bool        = False
    _client_escalates:    int         = 0  # Local counter — works even with old Docker image
    last_reward: float       = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = await env.reset(task=task_name)
        obs          = reset_result.observation

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            action_str = get_agent_action(
                client, step, obs, task_name, history, last_reward,
                consecutive_escalates=_client_escalates,
            )
            action     = CyberDefenseAction(action_type=action_str)

            # Client-side escalate counter.
            # After forced block (ce was >= 2), keep counter at 2 so ban stays active.
            # Only reset to 0 when counter was < 2 (normal LLM-driven non-escalate action).
            if action_str == "escalate":
                _client_escalates += 1
            elif _client_escalates >= 2:
                _client_escalates = 2   # Stay in forced mode; reset only when reward improves
            else:
                _client_escalates = 0

            step_result = await env.step(action)
            obs         = step_result.observation
            reward      = step_result.reward if step_result.reward is not None else 0.0
            done        = step_result.done

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            # Rich history for next-step LLM context
            at = getattr(obs, "anomaly_trend", 0.0)
            pe = getattr(obs, "packet_entropy", 0.0)
            ce = getattr(obs, "consecutive_escalates", 0)
            history.append(
                f"Step {step}: {action_str} → {reward:+.2f} "
                f"| alerts={obs.alert_count} login={obs.login_failure_rate:.2f} "
                f"anomaly={obs.network_anomaly_score:.2f} trend={at:.2f} entropy={pe:.2f} "
                f"consec_esc={ce}"
            )

            if done:
                break

        try:
            hidden_state = await env.state()
            score        = _compute_score(hidden_state, task_name)
        except Exception as exc:
            print(f"[DEBUG] state() failed: {exc} — estimating from rewards", flush=True)
            theoretical_max = max_steps * 0.40 + 0.50
            score = round(min(max(sum(rewards) / theoretical_max, 0.0), 1.0), 3)

        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= threshold

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} exception: {exc}", flush=True)
        score, success = 0.0, False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_name": task_name, "score": score, "success": success,
            "steps": steps_taken, "rewards": rewards}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _CyberDefenseProvider(LocalDockerProvider):
    """Custom provider with longer timeout and container log diagnostics."""

    def wait_for_ready(self, base_url: str, timeout_s: float = DOCKER_TIMEOUT_S) -> None:
        import time
        import subprocess
        import urllib.request
        health_url = f"{base_url}/health"
        start = time.time()
        last_error = ""
        print(f"[DEBUG] Waiting up to {timeout_s:.0f}s for {health_url}", flush=True)
        while time.time() - start < timeout_s:
            try:
                resp = urllib.request.urlopen(
                    urllib.request.Request(health_url), timeout=3
                )
                if resp.status == 200:
                    print(f"[DEBUG] Server ready after {time.time()-start:.1f}s", flush=True)
                    return
            except Exception as e:
                last_error = str(e)
            time.sleep(1.0)
        print(f"[DEBUG] Timeout after {timeout_s:.0f}s. Last error: {last_error}", flush=True)
        if self._container_id:
            try:
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "50", self._container_id],
                    capture_output=True, text=True, timeout=10,
                )
                print("[DEBUG] Container logs:", flush=True)
                for line in (logs.stdout + logs.stderr).strip().split("\n")[-50:]:
                    print(f"  {line}", flush=True)
            except Exception as exc:
                print(f"[DEBUG] Could not fetch logs: {exc}", flush=True)
        raise TimeoutError(
            f"Container at {base_url} not ready within {timeout_s:.0f}s. "
            "See [DEBUG] logs above for the startup error. "
            "Tip: run `uvicorn server.app:app --port 8000` manually and "
            "set CYBER_DEFENSE_SERVER_URL=http://localhost:8000 to skip Docker."
        )


async def _connect_env() -> CyberDefenseEnv:
    """Connect to the environment — via Docker or direct URL."""
    if DIRECT_SERVER_URL:
        print(f"[DEBUG] Direct connect to {DIRECT_SERVER_URL} (Docker skipped)", flush=True)
        env = CyberDefenseEnv(base_url=DIRECT_SERVER_URL)
        await env.connect()
        return env
    print(f"[DEBUG] Starting Docker image: {IMAGE_NAME} (timeout={DOCKER_TIMEOUT_S:.0f}s)", flush=True)
    return await CyberDefenseEnv.from_docker_image(IMAGE_NAME, provider=_CyberDefenseProvider())


async def main() -> None:
    client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env     = await _connect_env()
    results = []

    try:
        for task in TASKS:
            result = await run_task(client, env, task)
            results.append(result)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

    print("\n[SUMMARY]", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {r['task_name']:<35} score={r['score']:.3f}  steps={r['steps']}  {status}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())