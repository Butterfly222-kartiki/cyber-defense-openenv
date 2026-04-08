---
title: Cyber Defense OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: server/app.py
pinned: false
---

# Adaptive Cyber Defense — OpenEnv Environment


A real-world Security Operations Center (SOC) decision environment for training and evaluating RL agents. Models the full MITRE ATT&CK kill-chain as a POMDP with an adversarially adaptive attacker.

The core research challenge: a defender agent must learn to detect and stop a live adversary using only noisy, partial network monitoring signals — the same challenge faced by human SOC analysts responding to real intrusions.

---

## Motivation

95% of real-world security breaches involve delayed human response. The bottleneck is not detection technology — it is decision-making under uncertainty. This environment trains agents to make fast, precise defensive decisions when signals are noisy, attackers adapt, and false positives carry operational cost. Unlike detection classifiers, this is a **decision intelligence** environment grounded in real network intrusion behavior (NSL-KDD / CICIDS datasets).

---

## Environment Description

The environment is a **Partially Observable Markov Decision Process (POMDP)**:

- The **hidden state** is the true attack stage (0–5), the attacker's current policy, and whether a zero-day event is active. The agent never sees this directly.
- The **observation** is a set of noisy network monitoring signals derived from the hidden state. The agent must infer what is happening from these signals.
- The **attacker** is a built-in adaptive agent with a policy vector `[p_advance, p_retreat, p_hold]` that updates via exponential moving average (alpha=0.3) every step based on what the defender just did. This is the **RL-vs-RL mechanic**: both agents are adapting simultaneously within the episode.

### Attack kill-chain (MITRE ATT&CK grounded)

| Stage | Name | Signals |
|-------|------|---------|
| 0 | Idle | All signals near zero |
| 1 | Reconnaissance | login_rate 0.30, alerts=2, port scan begins |
| 2 | Initial access | login_rate 0.52, alerts=4, CPU spike |
| 3 | Lateral movement | login_rate 0.68, alerts=6, lateral movement flag |
| 4 | Persistence / exfiltration | login_rate 0.82, alerts=8 |
| 5 | Full breach | Episode terminates, -0.80 penalty |

Signal values are scaled by a per-episode random factor (±20%) drawn at `reset()`. This prevents RL agents from memorizing fixed thresholds and forces genuine multi-signal pattern recognition.

---

## Action Space

Five discrete actions. Exactly one must be chosen each step.

| Action | Effect | Reward if correct | Penalty if false positive |
|--------|--------|-------------------|--------------------------|
| `monitor` | Observe only | 0.00 | 0.00 |
| `isolate` | Quarantine host — full stop (stage → 0) | +0.40 | -0.12 |
| `block` | Firewall rule — partial stop (stage -= 1) | +0.25 | -0.08 |
| `escalate` | Activate scrutiny for 2 steps (noise halved) | 0.00 | 0.00 |
| `do_nothing` | Explicit inaction | — | -0.18 if attack active |

Early detection bonus: +0.15 when action taken while stage ≤ 2.

---

## Observation Space

All fields are returned by `step()` and `reset()` as a `CyberDefenseObservation`.

| Field | Type | Description |
|-------|------|-------------|
| `login_failure_rate` | float [0,1] | Normalised failed login rate |
| `port_scan_detected` | bool | True if port scan observed this step |
| `cpu_spike` | bool | True if anomalous CPU usage recorded |
| `network_anomaly_score` | float [0,1] | Composite anomaly score |
| `lateral_movement_flag` | bool | Suspicious east-west traffic (hidden in hard/zero-day) |
| `alert_count` | int [0–10] | IDS/IPS alerts fired this step |
| `anomaly_trend` | float [0,1] | Rate of change in anomaly score over last 3 steps — key zero-day indicator |
| `packet_entropy` | float [0,1] | Packet size entropy — rises under zero-day attack |
| `false_signal_injected` | bool | Synthetic noise event active (visible in easy task only) |
| `scrutiny_active` | bool | True when escalate is in effect |
| `consecutive_escalates` | int | Escalate actions taken in a row — commit to block/isolate after ≥ 2 |
| `steps_remaining` | int | Steps left in the episode |
| `last_action` | str | Last action taken by the agent |
| `task_name` | str | Active task identifier |
| `done` | bool | True when episode has ended |
| `reward` | float | Per-step shaped reward |

---

## Tasks

Four tasks with genuine difficulty progression.

### easy\_breach\_prevention (easy)

Fixed deterministic attacker, no noise, 10-step episodes, starts from stage 0.

The attacker advances with constant probability each step. Signals are clean — no noise injection and no zero-day. The agent must notice when the attack begins and intervene before stage 5. Expected baseline score with a good rule-based agent: ~1.0.

Success threshold: **score ≥ 0.60**

### medium\_detection\_balance (medium)

Adaptive RL-vs-RL attacker, 20% false signal noise, 20-step episodes, starts at stage 1.

The attacker updates its advance/retreat policy in real time based on the defender's last 5 actions. At step 10 the attacker switches into stealth mode regardless. 20% noise means roughly 1 in 5 steps fires a false alarm — the agent must balance recall (catching real attacks) against precision (avoiding false positives that cost -0.08 each). Expected baseline: ~0.65–0.75.

Success threshold: **score ≥ 0.45**

### hard\_stealth\_defense (hard)

Stealth attacker with randomized retreat cap, 40% noise, lateral movement hidden, 30-step episodes, zero-day possible.

The attacker uses the full RL policy vector with a per-episode retreat cap drawn from {1, 2, 3} — preventing the agent from counting retreats. With 30% probability a zero-day event is injected between steps 6–13, suppressing all standard IDS signals while `anomaly_trend` and `packet_entropy` rise. The agent must learn to detect via both standard and non-standard channels. Expected baseline: ~0.55–0.65.

Success threshold: **score ≥ 0.30**

### zero\_day\_detection (hard)

Pure zero-day scenario, 30% noise, 25-step episodes. Standard signals are suppressed from steps 4–8 onwards. All detection must rely on `anomaly_trend` (rising temporal pattern) and `packet_entropy` (unusual packet size distribution). Tests temporal reasoning rather than threshold matching — the hardest task in the set. Expected baseline: ~0.35–0.45.

Success threshold: **score ≥ 0.35**

---

## Reward Function

Dense shaped reward on every step — never sparse.

```
correct isolate     : +0.40   (attack_stage → 0)
correct block       : +0.25   (attack_stage -= 1)
early detection     : +0.15   (bonus when stage ≤ 2)
false positive iso  : -0.12
false positive blk  : -0.08
monitor passive     : -0.10   (when attack_stage > 0)
do_nothing penalty  : -0.18   (when attack_stage > 0)
missed attack       : -0.10 × (0.5 + 0.5 × stage/5)   ← stage-proportional
escalate active     : -0.06   (when attack_stage > 0)
escalate repeated   : -0.06   (additional, per use when consecutive ≥ 2)
breach penalty      : -0.80   (terminal, attack reached stage 5)
success bonus       : +0.50 × intervention_quality   (terminal, survived all steps)
```

Mathematical design guarantee: `E[block | attack active] = +0.25` vs `E[monitor | attack active] ≈ -0.155`. Intervening always strictly dominates watching.

---

## RL-vs-RL Design

The attacker maintains a policy vector `P = [p_advance, p_retreat, p_hold]` updated via exponential moving average (α=0.3) after every step:

- Defender uses `isolate`/`block` → attacker increases `p_retreat` (evades aggressive defense)
- Defender uses `monitor` → attacker increases `p_advance` (exploits passivity)
- Defender uses `do_nothing` → attacker maximally accelerates

This update happens before the attacker acts each step, so the attacker's behaviour in step N is a direct function of what the defender did in steps 1 through N-1. The policy vector is exposed in `state()` under `attacker_policy_vector` so evaluators can watch the adaptation in real time.

The defender does not have a built-in learning mechanism — the inference script uses an LLM for baseline evaluation. To train a true RL defender, connect a PPO/DQN agent via the OpenEnv API. The non-stationary attacker forces the defender to learn robust generalised policies rather than memorising fixed attack patterns.

---

## Zero-Day Simulation

Real zero-days bypass signature-based IDS. In the environment, `zero_day_active=True` suppresses standard signals:

- `login_failure_rate` drops to 0.0–0.08 (attacker uses valid credentials)
- `alert_count` drops to 0–1 (no IDS signatures match)
- `port_scan_detected` becomes False

The real signal is carried by:
- `anomaly_trend`: rate of change in the anomaly score over the last 3 steps — rises from 0.0 to 0.9 over 5 steps while standard signals stay near zero
- `packet_entropy`: normalised entropy of packet sizes — rises from 0.30 to 0.85 as the attacker's non-standard protocol becomes detectable

An agent relying only on `alert_count` will miss the zero-day entirely. An agent that learns to monitor `anomaly_trend ≥ 0.25` will detect it within 3–4 steps of onset.

---

## Setup

### Requirements

```
Python 3.10+
openenv-core >= 0.2.2
pydantic >= 2.0.0
openai >= 1.0.0
python-dotenv >= 1.0.0
fastapi
uvicorn
```

### Install

```bash
git clone <your-repo-url>
cd cyber-defense-openenv
pip install openenv-core
pip install -e .
```

### Run the server locally

```bash
cd cyber_defense
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker

```bash
# Build
docker build -t cyber_defense-env:latest .

# Run
docker run -p 8000:8000 cyber_defense-env:latest

# Verify
curl http://localhost:8000/health
```

### Run the inference script

```bash
# Create .env file at the project root
# MODEL_NAME controls which LLM the agent uses — change this to switch models
cat > .env << EOF
HF_TOKEN=your_hf_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
LOCAL_IMAGE_NAME=cyber_defense-env:latest
EOF

# Run inference
python inference.py
```

**To change the model**, edit `MODEL_NAME` in `.env`. The inference script reads it at startup. Recommended models via the HF router:

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-72B-Instruct` | 72B | Default — best instruction following |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | Strong alternative |
| `Qwen/Qwen2.5-32B-Instruct` | 32B | Faster, slightly lower quality |

No code changes are needed when switching models — only the `.env` file.

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier for inference |
| `HF_TOKEN` | Yes | Hugging Face / API key |
| `LOCAL_IMAGE_NAME` | Yes | Docker image name for from_docker_image() |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset environment. Accepts `task=` and `seed=` kwargs. |
| `POST` | `/step` | Execute one defensive action. |
| `GET` | `/state` | Return full hidden state (graders / evaluators only). |
| `GET` | `/schema` | Return JSON schemas for action, observation, state. |
| `GET` | `/health` | Liveness check — returns 200 OK. |
| `WS` | `/ws` | WebSocket for persistent multi-step sessions (used by EnvClient). |

---

## Inference Script Output Format

The inference script (`inference.py`) emits structured stdout logs parsed by the automated evaluation system. The format is strict — any deviation breaks scoring.

```
[START] task=easy_breach_prevention env=cyber_defense model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=monitor reward=0.05 done=false error=null
[STEP] step=2 action=block reward=0.36 done=false error=null
...
[END] success=true steps=10 score=1.000 rewards=0.05,0.36,...
```

---

## Baseline Scores

Scores produced by the rule-based fallback agent (minimum baseline — LLM agent should score at least this well).

| Task | Score | Steps | Status |
|------|-------|-------|--------|
| easy\_breach\_prevention | 1.000 | 10 | PASS |
| medium\_detection\_balance | 0.867 | 20 | PASS |
| hard\_stealth\_defense | 0.807 | 30 | PASS |
| zero\_day\_detection | 0.912 | 25 | PASS |

Scores above use the rule-based fallback agent (72B model or better). With a
smaller model (7B), the escalate-loop prevention in `inference.py` ensures the
fallback rule engine takes over after 2 consecutive escalates, giving scores in
the range 0.55–0.90 across all tasks.

Scores are deterministic with fixed seeds (42, 137, 271, 999). Running `python inference.py` with the same seed configuration reproduces these exactly.

---

## Project Structure

```
cyber-defense-openenv/
├── inference.py                  ← Inference script (root, required by PS)
├── openenv.yaml                  ← OpenEnv spec metadata
├── Dockerfile                    ← Container build (root)
├── pyproject.toml
├── README.md
└── cyber_defense/
    ├── __init__.py
    ├── models.py                 ← Pydantic models: Action, Observation, State
    ├── client.py                 ← EnvClient subclass for WebSocket communication
    └── server/
        ├── __init__.py
        ├── app.py                ← FastAPI app (create_app factory)
        └── cyber_defense_environment.py  ← Core environment logic
```

---

## Graders

Each task has a deterministic grader producing a score in [0.0, 1.0] with no LLM calls.

**easy\_breach\_prevention**: `1.0` if not breached. If breached: `0.25 × (steps_survived / max_steps)` for partial credit.

**medium\_detection\_balance**: `0.35 × (not_breached) + 0.35 × precision + 0.30 × recall + 0.05 × survival_ratio` where precision and recall are computed from `correct_blocks` and `false_positives` in the hidden state.

**hard\_stealth\_defense**: `0.40 × (not_breached) + 0.28 × (1 - fp_rate) + 0.22 × intervention_rate + 0.10 × stage_quality` where `stage_quality` is the mean `(5 - stage) / 5` over all steps (lower average stage = higher score).

**zero\_day\_detection**: `0.30 × (not_breached) + 0.40 × zero_day_detection_rate + 0.20 × (1 - fp_rate) + 0.10` where detection rate is the fraction of post-onset steps where the agent used `block` or `isolate`.

All graders use the `correct_blocks`, `false_positives`, `is_compromised`, `attack_history`, and `action_history` fields from `state()`. They are called once after the episode ends and are fully reproducible given a fixed seed.

---

## For RL researchers

This environment is immediately usable for PPO, DQN, and other RL algorithms via the included Gymnasium wrapper.

### Install and connect

```bash
pip install gymnasium stable-baselines3
```

```python
from gymnasium_wrapper import CyberDefenseGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Validate the environment (should print no warnings)
env = CyberDefenseGymEnv(task="hard_stealth_defense")
check_env(env)

# Train a PPO agent
model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=64)
model.learn(total_timesteps=500_000)
model.save("cyber_defense_ppo")

# Evaluate
obs, _ = env.reset(seed=271)  # fixed seed for reproducibility
for _ in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if done:
        print(f"Episode grade: {info['episode_score']:.3f}")
        break
```

All four tasks are available as named Gymnasium environments after registration:

```python
from gymnasium_wrapper import register_gymnasium_envs
import gymnasium as gym

register_gymnasium_envs()
env = gym.make("CyberDefense-hard-v0")
```

### Observation space

11-dimensional normalised float32 vector. All values in [0, 1].

| Index | Signal | Notes |
|-------|--------|-------|
| 0 | `login_failure_rate` | Brute-force indicator |
| 1 | `alert_count / 10` | IDS alert rate |
| 2 | `network_anomaly_score` | Composite anomaly |
| 3 | `port_scan_detected` | Binary |
| 4 | `cpu_spike` | Binary |
| 5 | `lateral_movement_flag` | Hidden in hard task |
| 6 | `scrutiny_active` | Escalate bonus active |
| 7 | `anomaly_trend` | Key zero-day indicator |
| 8 | `packet_entropy` | Key zero-day indicator |
| 9 | `steps_remaining / max_steps` | Urgency signal |
| 10 | `consecutive_escalates / 5` | Loop prevention signal |

### Why the non-stationary attacker matters

Standard RL environments have fixed dynamics. The defender can learn by memorising which actions work for which states. Our attacker's policy vector updates every step via exponential moving average (α=0.3) based on the defender's action. A passive defender sees accelerating attacks. An aggressive defender sees a retreating, evading attacker.

This non-stationarity forces the RL agent to learn a robust general policy rather than memorising a fixed action sequence. This is the environment's primary research contribution: an adversarial training ground that prevents overfitting to static attack patterns.

### Expected training curves (PPO, MlpPolicy, 500k steps)

| Task | Steps to first pass | Converged score | Notes |
|------|---------------------|-----------------|-------|
| easy_breach_prevention | ~5,000 | ~0.92 | Trivial for PPO |
| medium_detection_balance | ~30,000 | ~0.78 | FP discipline required |
| hard_stealth_defense | ~150,000 | ~0.65 | Zero-day + partial obs |
| zero_day_detection | ~200,000 | ~0.58 | Hardest — anomaly-only detection |

Per-episode signal scaling (±20%) prevents threshold memorisation and requires pattern-based generalisation, which increases convergence time but produces policies that generalise to unseen attack intensities.

---

## Safety

This environment simulates behavioral network signals only. It does not generate, store, or transmit real attack code, real exploits, actual credentials, or any information that could be used to perform a real intrusion. All signal values are synthetic, derived from published academic datasets (NSL-KDD, CICIDS) for research purposes.