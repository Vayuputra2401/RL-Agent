# AP Commander — Technical Documentation

> **AP Commander** is a multi-agent reinforcement learning environment for enterprise Accounts Payable (AP) workflows. It is built on the [OpenEnv](https://meta-pytorch.org/OpenEnv/) standard and covers all four hackathon themes: Multi-Agent (#1), Long-Horizon Planning (#2), Professional World Modeling (#3), and Self-Improvement (#4).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Agent Interaction Flowchart](#3-agent-interaction-flowchart)
4. [The AP Clerk Environment](#4-the-ap-clerk-environment)
5. [Reward Function Design](#5-reward-function-design)
6. [Actor Agents](#6-actor-agents)
7. [Oversight Environment (Fleet AI)](#7-oversight-environment-fleet-ai)
8. [Adaptive Curriculum](#8-adaptive-curriculum)
9. [Task Registry](#9-task-registry)
10. [Training Pipeline (GRPO)](#10-training-pipeline-grpo)
11. [API Reference](#11-api-reference)
12. [Design Principles](#12-design-principles)

---

## 1. System Overview

AP Commander simulates an enterprise finance department where an AI agent acts as an Accounts Payable Clerk. Each episode presents a real-world scenario — an invoice arrives, the agent must cross-reference it against a Purchase Order (PO) and a Goods Receipt Note (GRN), apply company policy, and issue a structured decision.

The environment is adversarial in the sense that some scenarios include fraudulent vendors, duplicate invoices, and policy violations. The agent must detect these and respond correctly.

**Three levels of agents operate in this system:**

| Agent | Role | Trained? |
|---|---|---|
| AP Clerk | Issues invoice decisions | Yes (GRPO) |
| Actor agents (Vendor, Manager, Compliance) | Respond to clerk's intermediate actions | Scripted |
| Oversight Agent | Reviews batches of completed clerk episodes | Separate env |

---

## 2. Architecture

```
RL-Agent/
├── app/
│   ├── main.py              # FastAPI server — all HTTP endpoints
│   ├── environment.py       # APClerkEnvironment: episode state machine
│   ├── tasks.py             # Task generators + graders (27 tasks)
│   ├── models.py            # Pydantic models for all I/O
│   └── actors/
│       ├── vendor_actor.py      # VendorActor (honest / fraudulent / confused)
│       ├── manager_actor.py     # ManagerActor (budget authority)
│       └── compliance_actor.py  # ComplianceActor (SOX / GDPR / Internal)
├── oversight_environment.py # OversightEnvironment: Fleet AI bonus
├── training/
│   ├── train.py             # GRPO training script (TRL + LoRA)
│   └── eval_baseline.py     # Untrained baseline evaluation
└── openenv.yaml             # Environment manifest (v4.0.0)
```

**Two independent HF Spaces:**
- `Pathikreet/ap-clerk-env` — the live environment API (FastAPI)
- `Pathikreet/ap-commander-training` — the training UI (Gradio + training scripts)

---

## 3. Agent Interaction Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          AP CLERK EPISODE LOOP                                  │
│                                                                                 │
│   Training Loop / Inference                                                     │
│          │                                                                      │
│          │  POST /reset  (task_id + seed)                                       │
│          ▼                                                                      │
│   APClerkEnvironment.reset                                                      │
│          │                                                                      │
│          ├─▶ Task Generator  →  invoice + PO + GRN                              │
│          │                                                                      │
│          └─▶ Actor Agents pre-generate context_notes                            │
│                (Vendor / Manager / Compliance)                                  │
│                stored hidden in _context_store                                  │
│          │                                                                      │
│          ▼                                                                      │
│   Observation returned to agent  ◀──────────────────────────┐                  │
│          │                                                   │                  │
│          ▼                                                   │                  │
│   Agent Decision                                             │                  │
│          │                                                   │                  │
│          ├─ ESCALATE / QUERY_VENDOR / HOLD ─▶ Intermediate Step                │
│          │                                    context_note revealed             │
│          │                                    appended to obs.context_notes ───┘│
│          │                                                                      │
│          └─ APPROVE_FULL / APPROVE_PARTIAL / REJECT  ──┐                       │
│             (or step limit reached)                     │                       │
│                                                         ▼                       │
│                                                  Grader called                  │
│                                                  decision_score × weight        │
│                                                  amount_score   × weight        │
│                                                  reason_score   × weight        │
│                                                  expl_score     × weight        │
│                                                  + process_bonus (multi-step)   │
│                                                         │                       │
│                                                  score clamped to 0.01–0.99     │
│                                                         │                       │
│                                                  Reward → GRPO Trainer          │
│                                                         │                       │
│                                              group-relative advantage           │
│                                                         │                       │
│                                              LoRA weight update ──▶ loop        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
│        OVERSIGHT (Fleet AI)          │   │         ADAPTIVE CURRICULUM          │
│                                      │   │                                      │
│  POST /oversight/reset               │   │  GET /curriculum/next_task           │
│         │                            │   │         │                            │
│         ▼                            │   │         ▼                            │
│  OversightEnvironment                │   │  Server-side history                 │
│  3–5 episode summaries               │   │  keyed by X-Run-Id header            │
│  1–2 may be fraudulent               │   │         │                            │
│         │                            │   │  Score thresholds:                   │
│         ▼                            │   │  easy   ≥ 0.70 → unlock medium      │
│  Oversight Agent verdict             │   │  medium ≥ 0.65 → unlock hard        │
│  per episode                         │   │  hard   ≥ 0.60 → unlock long        │
│         │                            │   │         │                            │
│         ▼                            │   │         ▼                            │
│  Oversight Grader:                   │   │  Next task returned                  │
│  +0.70 correct flag                  │   │                                      │
│  +0.20 cited numeric signal          │   │                                      │
│  -0.25 false positive                │   │                                      │
└──────────────────────────────────────┘   └──────────────────────────────────────┘
```

---

## 4. The AP Clerk Environment

### Episode Lifecycle

Every episode follows this state machine:

```
reset(task_id, seed)
    → generates fresh invoice / PO / GRN
    → actors pre-populate context_store
    → returns Observation (context_notes empty)

step(action)  [may repeat up to max_steps]
    if action is ESCALATE / QUERY_VENDOR / HOLD and steps_remaining > 0:
        → pop matching note from context_store
        → append to obs.context_notes
        → return score=0.01, done=False   ← episode continues
    else (terminal action or step limit reached):
        → call grader
        → return scored reward, done=True
```

### Observation Structure

```python
APObservation:
    task_id, task_name, task_description
    invoice:
        invoice_id, vendor_name, invoice_total
        line_items: [{description, quantity, unit_price}]
        freight_charge
    purchase_orders: [{po_number, status, vendor_name, lines: [...]}]
    goods_receipts:  [{grn_id, po_number, lines: [{description, received_quantity}]}]
    context_notes:   []   # revealed progressively on intermediate actions
    action_history:  []   # all decisions taken so far this episode
    paid_invoice_ids: []  # for duplicate detection tasks
    company_policy:  str  # full policy text including freight cap, PO requirement, etc.
    step_count, max_steps
```

### Action Structure

```python
APAction:
    decision:        APPROVE_FULL | APPROVE_PARTIAL | REJECT | ESCALATE | QUERY_VENDOR | HOLD | HYPOTHETICAL
    approved_amount: float
    reason_code:     MATCH_CONFIRMED | QUANTITY_MISMATCH | PRICE_DISCREPANCY | POLICY_VIOLATION |
                     NO_PO_FOUND | DUPLICATE_INVOICE | VENDOR_MISMATCH | TAX_DISCREPANCY |
                     PENDING_CLARIFICATION | MANAGER_REVIEW
    explanation:     str   # must cite specific dollar amounts
```

### HYPOTHETICAL Action (Self-Play)

The agent can issue `HYPOTHETICAL` as a decision type. The environment returns a hint about the likely outcome without committing to the action and without revealing graded context. This enables counterfactual reasoning — the agent can explore "what if I ESCALATE vs REJECT?" before committing. Score returned is always 0.01 for hypothetical steps.

---

## 5. Reward Function Design

### Core Formula

Every grader computes a weighted sum of four independent components:

```
score = w_d × decision_score
      + w_a × amount_score
      + w_r × reason_score
      + w_e × explanation_score
      + process_bonus          (multi-step tasks only)

score = clamp(score, 0.01, 0.99)
```

Weights vary by task difficulty:

| Difficulty | w_decision | w_amount | w_reason | explanation |
|---|---|---|---|---|
| Easy | 0.50 | 0.35 | 0.15 | — |
| Medium | 0.45 | 0.40 | 0.15 | keyword hits |
| Hard | 0.40 | 0.35 | 0.10 | coherence multiplier |
| Long | 0.35 | 0.35 | 0.15 | coherence multiplier |

### Component Details

**Decision score** — Is the decision type correct?
- Full credit (1.0): correct decision for the scenario
- Partial credit (0.15): defensible but suboptimal (e.g. REJECT on a shortfall instead of APPROVE_PARTIAL)
- Zero (0.0): wrong decision type

**Amount score** — Is the approved dollar amount correct?
- Full credit (1.0): within $0.01 of the expected amount
- Sliding credit: degrades linearly with % deviation up to ~20%
- Zero: amount > 20% off or wrong direction

**Reason code score** — Does the reason code match the scenario?
- Full credit (1.0): exact match to expected reason
- Partial (0.05–0.20): plausible but imprecise
- Zero: reason code contradicts the scenario

**Explanation score** — Does the explanation cite specific numbers?
- Counts keyword hits (vendor names, exact dollar amounts, PO numbers)
- Requires ≥ 3 hits for full credit (task-specific)
- Multiplied by `_explanation_coherence()` anti-keyword-salad factor

### Anti-Keyword-Salad (`_explanation_coherence`)

```python
def _explanation_coherence(expl: str, hits: int) -> float:
    words = expl.split()
    if len(words) < 8:       return 0.5   # too short
    if hits/len(words) > 0.4: return 0.6   # >40% keywords = salad penalty
    return 1.0
```

This prevents the model from learning to dump reward-keywords as its explanation. It must write a coherent sentence that cites numbers in context.

### Process Bonus (Multi-Step Tasks)

Hard and long-horizon tasks reward using the correct intermediate action sequence:

```python
process_bonus = 0.10–0.20 if agent used ESCALATE/QUERY_VENDOR before terminal decision
```

Example — `hard_duplicate_invoice`:
- Agent sends `QUERY_VENDOR` → environment reveals `[VENDOR: this invoice was already paid]`
- Agent then sends `REJECT` with `DUPLICATE_INVOICE`
- Grader detects `QUERY_VENDOR` in `action_history` → adds `process_bonus = 0.10`

Without the bonus, a lucky `REJECT` guess scores the same as a reasoned multi-step sequence. The bonus ensures correct process is rewarded even when the terminal decision would be identical.

### Accumulated Discount (Training)

During GRPO training, multi-step episodes use discounted reward accumulation:

```
total_score = Σ (discount^step_n × step_score)
            = 0.01 + 0.9 × 0.99   (QUERY_VENDOR then REJECT)
            = 0.901
```

vs a shortcut single-step `REJECT` ≈ 0.40 — the correct 2-step sequence earns more than twice as much. This is the key incentive for the agent to learn multi-step behavior.

---

## 6. Actor Agents

Three scripted actor agents simulate the workplace humans the AP Clerk interacts with. They are called at episode **reset** time to pre-generate `context_notes`, which are stored hidden in `_context_store` and revealed one at a time when the clerk sends the matching intermediate action.

### VendorActor (`app/actors/vendor_actor.py`)

| Persona | Triggered by | Behavior |
|---|---|---|
| `honest` | QUERY_VENDOR | Accurately confirms or corrects the discrepancy |
| `fraudulent` | QUERY_VENDOR | Tries to justify inflated prices or deny duplicate status |
| `confused` | QUERY_VENDOR | Provides ambiguous, partially-helpful reply |

Note revealed in context: `[VENDOR: ...]`

Correct handling of a `fraudulent` vendor earns `+0.10` process bonus on relevant tasks.

### ManagerActor (`app/actors/manager_actor.py`)

| Property | Values |
|---|---|
| `budget_authority` | $5K / $50K / unlimited (randomized per episode) |
| `risk_appetite` | conservative / moderate / aggressive |

- Triggered by: `ESCALATE`
- Responds: approves if within authority, denies if exceeds, routes to CFO if ambiguous
- Note revealed: `[MANAGER: approved/denied/escalated to CFO]`

### ComplianceActor (`app/actors/compliance_actor.py`)

| Property | Values |
|---|---|
| `regulatory_framework` | SOX / GDPR / INTERNAL_POLICY |

- Triggered by: `HOLD`
- Responds: reviews and returns a compliance verdict with cited regulation
- Note revealed: `[COMPLIANCE: cleared/flagged — regulation reference]`

### How Context Revelation Works

```python
# In APClerkEnvironment.step():
prefix_map = {
    DecisionType.ESCALATE:     "[MANAGER]",
    DecisionType.QUERY_VENDOR: "[VENDOR]",
    DecisionType.HOLD:         "[COMPLIANCE]",
}
# Finds the matching note in _context_store and pops it into obs.context_notes
```

The agent sees the note in the next observation's `context_notes` field and can incorporate it into its final decision.

---

## 7. Oversight Environment (Fleet AI)

The `OversightEnvironment` (in `oversight_environment.py`) implements the Fleet AI Scalable Oversight bonus theme. A second agent — the **Oversight Agent** — monitors batches of completed AP Clerk episodes and must identify fraudulent or policy-violating ones.

### Oversight Observation

```python
OversightObservation:
    episode_summaries: list[EpisodeSummary]   # 3–5 completed clerk episodes
    known_fraud_patterns: list[str]           # injected hints from training
    audit_budget: int                         # max flags allowed this batch
    step_count: int
```

Each `EpisodeSummary` contains: invoice total, vendor name, PO status, clerk decision, clerk explanation, and a hidden `is_fraudulent` flag used only by the grader.

### Fraud Injection

The environment generates episodes with three types of synthetic fraud:

| Fraud Type | Signal |
|---|---|
| `duplicate_fraud` | Invoice ID appears in paid ledger; vendor denies it |
| `price_inflation` | Unit prices 15–40% above agreed PO price |
| `fake_vendor` | Vendor name is a slight variation of a legitimate vendor |

### Oversight Reward

```
+0.70  correctly flag a fraudulent episode (FLAG_FOR_REVIEW or ESCALATE_TO_AUDIT)
+0.20  explanation cites a specific numeric signal (amount, rate, vendor string)
-0.25  false positive — flagging a clean episode
+0.01  base score for correctly clearing a clean episode
```

Score is clamped to `[-0.99, +0.99]` (unlike AP Clerk which is `[0.01, 0.99]`). Negative scores are intentional — the false-positive penalty must reach the model to discourage over-flagging.

### Oversight Endpoints

```
POST /oversight/reset  → { observation: OversightObservation, session_id }
POST /oversight/step   → { observation, reward: OversightReward, done }
GET  /oversight/state  → current session state
```

---

## 8. Adaptive Curriculum

The curriculum system (`GET /curriculum/next_task`) escalates task difficulty as the agent's performance improves.

### Unlock Thresholds

```
easy   → mean reward ≥ 0.70  → unlocks medium
medium → mean reward ≥ 0.65  → unlocks hard
hard   → mean reward ≥ 0.60  → unlocks long-horizon
```

### Server-Side Anti-Spoofing

History is tracked **server-side**, keyed by `X-Run-Id` request header:

```python
_curriculum_history: Dict[str, List[dict]] = {}   # run_id → episode records

# Populated at /step completion:
bucket = _curriculum_history.setdefault(run_id, [])
bucket.append({"task_id": ..., "score": ..., "timestamp": ...})

# Read at /curriculum/next_task — client history used only as fallback:
server_history = _curriculum_history.get(run_id, [])
```

A client cannot forge a high-score history to skip to hard tasks — the server independently verifies what actually happened.

### CurriculumSampler (Training)

During GRPO training, `CurriculumSampler` mirrors this logic client-side:

```python
# gate_task() redirects locked tasks at reward-score time:
def gate_task(self, task_id: str) -> str:
    if _TASK_DIFFICULTY[task_id] in self.unlocked:
        return task_id
    return random.choice([t for t, d in _TASK_DIFFICULTY.items() if d == 'easy'])
```

The dataset always contains all 17 tasks — gating happens at reward time, not at dataset-build time. This prevents the critical bug where the dataset only contains easy tasks.

---

## 9. Task Registry

27 total tasks across four difficulty tiers.

### Easy (2 tasks, max_steps=1)

| Task ID | Scenario | Expected Decision |
|---|---|---|
| `easy_perfect_match` | Invoice matches PO and GRN exactly | APPROVE_FULL |
| `easy_no_po_found` | Invoice arrives with no matching PO | REJECT |

### Medium (4 tasks, max_steps=2–3)

| Task ID | Scenario | Expected Decision |
|---|---|---|
| `medium_quantity_shortfall` | GRN shows fewer units received than invoiced | APPROVE_PARTIAL |
| `medium_price_discrepancy` | Unit price exceeds agreed PO price | REJECT |
| `medium_split_delivery` | Partial delivery; only received tranche should be approved | APPROVE_PARTIAL |
| `medium_vendor_mismatch` | Invoice vendor name doesn't match PO vendor | REJECT |

### Hard (6 tasks, max_steps=3)

| Task ID | Scenario | Key Challenge |
|---|---|---|
| `hard_policy_violation` | Freight charge exceeds policy cap | Must ESCALATE to discover pre-approval |
| `hard_duplicate_invoice` | Invoice already paid | Must QUERY_VENDOR to confirm |
| `hard_partial_po_match` | Multi-line invoice; only some lines match GRN | Arithmetic on partial approval |
| `hard_tax_discrepancy` | Tax rate applied incorrectly | Must compute expected tax |
| `hard_currency_conversion` | Invoice in foreign currency; FX rate applied wrong | FX arithmetic |
| `hard_manager_preapproval` | Above-threshold purchase needs pre-approval | Must ESCALATE |

### Long-Horizon (7 tasks, max_steps=10–16)

| Task ID | Steps | Scenario |
|---|---|---|
| `long_invoice_dispute` | 12 | Vendor disputes REJECT; multi-round negotiation |
| `long_policy_migration` | 10 | Freight cap policy changes mid-episode |
| `long_batch_reconciliation` | 15 | 4 invoices arrive; 2 valid, 1 duplicate, 1 missing PO |
| `long_manager_chain` | 14 | Manager OOO; must escalate to VP then CFO |
| `long_fraud_investigation` | 16 | Duplicate chain A→B→C; must trace and reject |
| `long_audit_trail` | 14 | SOX audit requires HOLD for compliance review |
| `long_multi_vendor_split` | 12 | Single PO, 3 vendor deliveries; partial per tranche |

### Oversight (4 tasks, via `/oversight/*` endpoints)

`oversight_fraud_detection`, `oversight_pattern_recognition`, `oversight_false_positive_trap`, `oversight_explanation_quality`

---

## 10. Training Pipeline (GRPO)

### Algorithm

**GRPO (Group Relative Policy Optimization)** — a variant of PPO that eliminates the value network by using group-relative advantages:

```
For each prompt:
  1. Generate G=8 completions
  2. Score each with reward functions
  3. Advantage_i = (score_i - mean(group)) / std(group)
  4. Update policy with clipped surrogate objective
```

This makes GRPO memory-efficient (no critic network) and well-suited to text-generation tasks where reward is sparse.

### Two Independent Reward Functions

```python
reward_funcs = [env_reward_fn, format_reward_fn]
```

**`env_reward_fn`** — calls the AP Commander environment:
- Runs a full multi-step episode with accumulated discounted reward
- Applies curriculum gating (locked tasks redirected to easy)
- Records result to `CURRICULUM` for unlock tracking
- Writes structured per-episode JSON line to `episodes.jsonl`

**`format_reward_fn`** — checks JSON validity:
- `+0.05` if output is valid JSON with all required fields and valid enum values
- `-0.05` otherwise

Keeping them separate follows the GRPO guide recommendation — multiple independent signals give the optimizer more gradient information than a single combined signal.

### LoRA Configuration

```python
LoraConfig(
    r=16, lora_alpha=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_dropout=0, bias='none',
)
```

Only ~0.5% of parameters are trainable. The base model is frozen in 4-bit NF4 quantization. Adapters are saved separately and pushed to `Pathikreet/ap-commander-adapter`.

### Metrics Tracked Per Run

All figures saved to `runs/grpo/MODEL-NEP-DATETIME/`:

| Figure | What it shows |
|---|---|
| `fig1_reward_curve.png` | Mean episode return vs training step (raw + EMA) |
| `fig2_difficulty_curves.png` | Per-difficulty learning curves (easy/medium/hard/long) |
| `fig3_episode_lengths.png` | Episode length histogram + mean by difficulty |
| `fig4_format_compliance.png` | JSON format rate over time |
| `fig5_decision_distribution.png` | Stacked bar: decision mix over 20 training checkpoints |
| `fig6_per_task_means.png` | Horizontal bar: per-task mean reward sorted by difficulty |
| `results.png` | Before/After GRPO comparison + delta per task |
| `episodes.jsonl` | Structured log: every episode, every step, actor responses, scores |

---

## 11. API Reference

All endpoints at `https://pathikreet-ap-clerk-env.hf.space`

### AP Clerk

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode. Body: `{task_id, seed?}` |
| `POST` | `/step` | Submit action. Body: `{session_id, action: APAction}` |
| `GET` | `/state` | Current episode state |

### Oversight

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/oversight/reset` | Start oversight episode. Body: `{seed?, num_episodes?, fraud_rate?}` |
| `POST` | `/oversight/step` | Submit verdict. Body: `{session_id, action: OversightAction}` |
| `GET` | `/oversight/state` | Current oversight session state |

### Curriculum

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/curriculum/next_task` | Get next task based on performance history |

### Meta

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/tasks` | List all registered tasks with metadata |
| `GET` | `/health` | Server health + task count |
| `GET` | `/stats` | Session statistics |
| `GET` | `/docs` | Swagger UI (auto-generated) |

---

## 12. Design Principles

### 1. Reward Hacking Resistance

Every reward component is independently validated to prevent the model from exploiting a single signal:

- **Decision** and **amount** are checked independently — scoring APPROVE_FULL with the wrong amount gets partial credit, not full
- **Explanation coherence** penalizes keyword stuffing — the model must write sentences, not lists of reward words
- **Process bonus** rewards correct procedure even when the terminal decision would be the same regardless

### 2. Grounded, Non-Gameable Scores

All scores are clamped to `(0.01, 0.99)`. No episode scores exactly 0 or 1. This matches the OpenEnv evaluator's expectation and prevents gradient collapse at the boundaries.

### 3. Deterministic Reproducibility

Every task generator accepts a `seed` parameter. The same `(task_id, seed)` pair always produces the same invoice amounts, vendor names, discrepancies, and actor personas. This makes episodes fully reproducible for debugging and evaluation comparison across training runs.

### 4. Server-Side State Only

The environment server holds all episode state. Clients are stateless — they pass a `session_id` cookie. This prevents clients from modifying observed state between steps.

### 5. Separation of Concerns

- Task **generators** are pure functions: `(seed) → APObservation`
- Task **graders** are pure functions: `(obs, action) → APReward`
- The **environment** is the state machine that sequences them
- The **server** handles HTTP, session management, and curriculum history
- **Actors** are separate classes injected at generation time

This makes each component independently testable and replaceable.

### 6. Multi-Step Incentive

Intermediate actions (`ESCALATE`, `QUERY_VENDOR`, `HOLD`) never terminate the episode and return `score=0.01`. The only way to earn meaningful reward is through a correct terminal decision. But the process bonus and accumulated discount mean the *sequence* `QUERY_VENDOR → REJECT` earns more than the shortcut `REJECT` alone — the optimizer is incentivized to learn the correct investigative procedure, not just the correct label.
