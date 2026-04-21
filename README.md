---
title: AP Commander
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
license: mit
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
  - fleet-ai
  - long-horizon
  - finance
  - enterprise
  - oversight
---

# AP Commander — Multi-Agent Enterprise Financial Operations Environment

A multi-agent reinforcement learning environment for enterprise Accounts Payable workflows. Covers all four hackathon themes: Multi-Agent Interactions, Long-Horizon Planning, Professional World Modeling, and Self-Improvement.

**Hackathon:** Meta PyTorch OpenEnv × Scaler School of Technology Grand Finale  
**Team:** Pathikreet Chowdhury, Anubhav Bhattacharya, Radhika Ravi

---

## What This Environment Trains

AP Commander trains two cooperative agents in an enterprise financial setting:

**AP Clerk Agent** — Reviews vendor invoices, cross-references Purchase Orders and Goods Receipts, applies company policy, and decides whether to approve, partially approve, or reject payment. Must interact with simulated workplace actors (vendor, manager, compliance officer) across multi-step episodes up to 16 steps long.

**Oversight Agent (Fleet AI)** — Monitors batches of completed AP Clerk decisions. Identifies fraudulent approvals, policy violations, and suspicious patterns. Must flag issues with specific numeric evidence and avoid false positives.

---

## Themes Covered

| Theme | Implementation | Bonus Prize Target |
|---|---|---|
| **#1 Multi-Agent** | AP Clerk + Oversight Agent + 3 actor-agents (VendorActor, ManagerActor, ComplianceActor) | Fleet AI + Halluminate |
| **#2 Long-Horizon** | 7 tasks with 10–16 step episodes (invoice dispute, fraud investigation, escalation chains) | Scale AI |
| **#3 Professional** | Dynamic enterprise world: policy changes mid-episode, SOX audit trails, VP escalation chains | Scaler AI Labs |
| **#4 Self-Improvement** | Adaptive curriculum endpoint + `HYPOTHETICAL` counterfactual action | — |

---

## Agents and Actors

### AP Clerk Agent
Makes invoice payment decisions using a 4-field action:

```json
{
  "decision": "APPROVE_FULL | APPROVE_PARTIAL | REJECT | ESCALATE | QUERY_VENDOR | HOLD | HYPOTHETICAL",
  "approved_amount": 1234.56,
  "reason_code": "MATCH_CONFIRMED | PRICE_DISCREPANCY | DUPLICATE_INVOICE | ...",
  "explanation": "Invoice $1,234.56 matches PO-2024-001. GRN confirms all 10 units. Approving."
}
```

### Oversight Agent
Reviews batches of completed AP Clerk episodes:

```json
{
  "episode_id": "EP-003-716",
  "verdict": "FLAG_FOR_REVIEW | ESCALATE_TO_AUDIT | CLEAR",
  "signal": "Approved $8,900 without checking paid ledger — DUPLICATE_INVOICE reason code absent",
  "confidence": 0.92
}
```

### Simulated Actors
| Actor | Triggered by | Behavior |
|---|---|---|
| `VendorActor` | `QUERY_VENDOR` | Responds based on persona: honest (admits errors), fraudulent (justifies inflation), confused (ambiguous) |
| `ManagerActor` | `ESCALATE` | Approves/denies based on budget authority and risk appetite; can be out-of-office (triggers VP chain) |
| `ComplianceActor` | `HOLD` | Reviews under SOX / GDPR / Internal Policy and returns a compliance verdict |

---

## Episode Flows

### Single-step task (easy/medium)
```
POST /reset  { "task_id": "easy_perfect_match", "seed": 42 }
→ observation

POST /step   { "session_id": "...", "action": { "decision": "APPROVE_FULL", ... } }
→ reward { score: 0.87, breakdown: {...}, feedback: "..." }, done: true
```

### Multi-step task (hard, max_steps=3)
```
POST /reset  → observation (max_steps: 3)
POST /step   { action: { decision: "ESCALATE", ... } }    → done: false, [MANAGER] context revealed
POST /step   { action: { decision: "APPROVE_FULL", ... } } → done: true, score with process bonus
```

### Long-horizon task (max_steps 10–16)
```
POST /reset  { "task_id": "long_fraud_investigation" }
POST /step   QUERY_VENDOR  → vendor disputes duplicate (fraudulent response)
POST /step   ESCALATE      → manager confirms from ledger: it IS a duplicate
POST /step   REJECT        → scored with investigation process bonus
```

### Oversight session
```
POST /oversight/reset  { "num_episodes": 5 }
→ 5 completed AP Clerk episode summaries (1–2 contain fraud, labels hidden)

POST /oversight/step  { "episode_id": "EP-001-...", "verdict": "FLAG_FOR_REVIEW", "signal": "..." }
→ reward { score: 0.90 }   (if correct flag with numeric signal)
... repeat for each episode
```

### Adaptive curriculum
```
POST /curriculum/next_task  { "session_history": [{ "task_id": "easy_perfect_match", "score": 0.82 }] }
→ { "recommended_task_id": "medium_quantity_shortfall", "difficulty": "medium", "reason": "..." }
```

---

## Task Library (24 tasks)

### Original 13 tasks (easy / medium / hard)

| Task ID | Difficulty | max_steps | Correct Decision |
|---|---|---|---|
| `easy_perfect_match` | easy | 1 | APPROVE_FULL |
| `easy_no_po_found` | easy | 1 | REJECT |
| `medium_quantity_shortfall` | medium | 1 | APPROVE_PARTIAL |
| `medium_price_discrepancy` | medium | 1 | REJECT |
| `medium_split_delivery` | medium | 1 | APPROVE_FULL |
| `medium_vendor_mismatch` | medium | 1 | REJECT |
| `hard_policy_violation` | hard | 3 | ESCALATE → REJECT |
| `hard_duplicate_invoice` | hard | 3 | QUERY_VENDOR → REJECT |
| `hard_partial_po_match` | hard | 1 | APPROVE_PARTIAL |
| `hard_tax_discrepancy` | hard | 1 | REJECT |
| `hard_currency_conversion` | hard | 1 | APPROVE_FULL or REJECT |
| `hard_manager_preapproval` | hard | 3 | ESCALATE → APPROVE_FULL |
| `hard_credit_memo` | hard | 1 | APPROVE_PARTIAL or REJECT |

### New 7 long-horizon tasks (10–16 steps)

| Task ID | max_steps | Optimal Workflow |
|---|---|---|
| `long_invoice_dispute` | 12 | QUERY_VENDOR → ESCALATE → REJECT (price error) |
| `long_policy_migration` | 10 | HOLD → compliance reveals new cap → APPROVE_FULL |
| `long_batch_reconciliation` | 15 | Standard 3-way match in batch context → APPROVE_FULL |
| `long_manager_chain` | 14 | ESCALATE (OOO) → ESCALATE again (VP Finance) → APPROVE_FULL |
| `long_fraud_investigation` | 16 | QUERY_VENDOR (vendor denies) → ESCALATE (manager confirms) → REJECT |
| `long_audit_trail` | 14 | HOLD → SOX review → APPROVE_FULL with PO/GRN/amount citations |
| `long_multi_vendor_split` | 12 | 3 GRNs, first tranche only → APPROVE_PARTIAL |

### 4 oversight tasks (via `/oversight/*` endpoints)

| Task ID | Description |
|---|---|
| `oversight_fraud_detection` | 5 episodes, 1 fraudulent — identify and flag with evidence |
| `oversight_pattern_recognition` | 5 episodes, 2–3 with same violation — flag the pattern |
| `oversight_false_positive_trap` | All clean — agent must CLEAR without over-flagging |
| `oversight_explanation_quality` | Must cite specific $ amounts and fraud keywords |

---

## Reward Structure

### AP Clerk rewards
Scores are partial-credit, broken down by component. All scores in open interval (0.01, 0.99).

| Component | Weight | Scoring |
|---|---|---|
| Decision accuracy | 38–55% | 1.0 correct, 0.0–0.4 wrong |
| Amount accuracy | 20–45% | 1.0 within 1%, 0.6 within 3%, 0.3 within 8% |
| Reason code | 10–30% | 1.0 correct, 0.05–0.40 partial |
| Explanation quality | 10–20% | Requires specific $ or % citations + keywords |
| Process bonus | 0–15% | Correct intermediate step before terminal decision |

### Oversight Agent rewards
| Condition | Score Component |
|---|---|
| Correctly flag fraudulent episode | +0.70 |
| Explanation with specific numeric signal | +0.20 |
| False positive (clean episode flagged) | −0.25 |
| Correct CLEAR on clean episode | +0.01 base |

---

## API Reference

### AP Clerk
| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start episode: `{ task_id, seed? }` |
| `/step` | POST | Submit action: `{ session_id, action }` |
| `/state` | GET | Session state: `?session_id=...` |

### Oversight Agent (Fleet AI)
| Endpoint | Method | Description |
|---|---|---|
| `/oversight/reset` | POST | Start oversight session: `{ num_episodes?, seed? }` |
| `/oversight/step` | POST | Submit verdict: `{ session_id, action: OversightAction }` |
| `/oversight/state` | GET | Session state: `?session_id=...` |

### Adaptive Curriculum
| Endpoint | Method | Description |
|---|---|---|
| `/curriculum/next_task` | POST | Get next task: `{ session_history: [{ task_id, score }] }` |

### Meta
| Endpoint | Method | Description |
|---|---|---|
| `/tasks` | GET | List all 24 tasks |
| `/health` | GET | Health check |
| `/stats` | GET | Live episode statistics |
| `/docs` | GET | Swagger UI |

---

## Project Structure

```
├── Dockerfile
├── README.md
├── openenv.yaml                  # Environment spec: tasks, action spaces, endpoints
├── inference.py                  # Baseline AP Clerk agent — runs all tasks, writes results.json
├── sim_run.py                    # Optimal agent simulation for all 20 runnable tasks
├── requirements.txt
├── oversight_environment.py      # Fleet AI Oversight Environment
├── app/
│   ├── main.py                   # FastAPI server — all endpoints
│   ├── models.py                 # Pydantic models: APObservation, OversightObservation, etc.
│   ├── tasks.py                  # 24 task generators + graders + TASKS registry
│   ├── environment.py            # APClerkEnvironment: reset() / step() / state()
│   └── actors/
│       ├── vendor_actor.py       # VendorActor (honest / fraudulent / confused)
│       ├── manager_actor.py      # ManagerActor (budget authority, risk appetite)
│       └── compliance_actor.py   # ComplianceActor (SOX / GDPR / Internal Policy)
└── training/
    └── colab_training.ipynb      # Unsloth GRPO training script
```

---

## Setup

### Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Open `http://localhost:7860/docs` for the interactive API docs.

### Run simulation (optimal agent)

```bash
python sim_run.py
```

Demonstrates all 20 AP Clerk tasks with a scripted optimal agent, prints scores per task and mean.

### Run inference (LLM baseline)

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py

# Run only long-horizon tasks
TASK_FILTER=long python inference.py
```

### Run with Docker

```bash
docker build -t ap-commander .
docker run -p 7860:7860 ap-commander
```

### Train with GRPO (Unsloth)

Open `training/colab_training.ipynb` in Google Colab (T4 GPU). The notebook:
1. Loads Llama-3-8B-Instruct (4-bit quantized via Unsloth)
2. Runs GRPO rollouts against the live environment
3. Plots before/after reward curves across difficulty levels
4. Pushes the fine-tuned model to HuggingFace Hub

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (inference) | — | HuggingFace API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `TASK_FILTER` | No | `""` | Filter tasks by prefix (e.g. `long`, `hard`) |
| `RUN_OVERSIGHT` | No | `0` | Set to `1` to also run oversight tasks |

---

## Adaptive Curriculum — Difficulty Ladder

```
easy (mean ≥ 0.70) → medium (mean ≥ 0.65) → hard (mean ≥ 0.68) → long-horizon (mean ≥ 0.72) → oversight
```

The `/curriculum/next_task` endpoint tracks performance history and recommends the least-practiced task at the current unlocked difficulty level. This enables progressive skill building without manual task selection.

---

## HYPOTHETICAL Action (Self-Play)

Long-horizon tasks support a special training-only action:

```json
{ "decision": "HYPOTHETICAL", "reason_code": "PRICE_DISCREPANCY", "explanation": "What if I reject here?" }
```

Returns a simulated outcome hint without committing to the decision. Allows the agent to explore counterfactual paths during training (score=0.01, episode continues).
