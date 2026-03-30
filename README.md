---
title: AP Clerk Environment
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
license: mit
tags:
  - openenv
  - reinforcement-learning
  - finance
  - enterprise
---

# AP Clerk Environment

**OpenEnv Hackathon Submission — AI Accounts Payable Clerk: Three-Way Invoice Matching**

An RL training environment where an agent learns to act as a corporate Accounts Payable clerk. Given a vendor invoice, Purchase Orders, and Goods Receipt Notes, the agent must cross-reference all documents, apply free-text company policy, and decide whether to approve, partially approve, or reject the invoice for payment.

Three-way invoice matching is a mandatory control in every company that purchases goods. It prevents overpayments, duplicate payments, and policy violations before money leaves the organisation. This environment models that process faithfully across ten distinct task types with four major environment upgrades over the baseline.

---

## Environment Overview

### Episode Flow — Single-step tasks

```
POST /reset  { "task_id": "...", "seed": 42 }
  → { "observation": { invoice, purchase_orders, goods_receipts, company_policy, ... }, "session_id": "uuid" }

POST /step   { "session_id": "uuid", "action": { decision, approved_amount, reason_code, explanation } }
  → { "reward": { "score": 0.0–1.0, "breakdown": {...}, "feedback": "..." }, "done": true }
```

### Episode Flow — Multi-step tasks (hard_policy_violation, hard_duplicate_invoice)

```
POST /reset  → observation (max_steps: 3)
POST /step   { "action": { "decision": "ESCALATE", ... } }  → done: false, context_notes revealed
POST /step   { "action": { "decision": "REJECT", ... } }    → done: true, score: 1.0 + process bonus
```

Intermediate actions (ESCALATE, QUERY_VENDOR, HOLD) are optional. The agent can go straight to a terminal decision — it just misses the process bonus for proper verification protocol.

---

## What Makes This Environment Hard (v2 Improvements)

### 1. Randomised Policy Per Episode
Freight caps ($30–$100) and price tolerance thresholds (0.5%–3%) vary with each seed. The agent cannot hardcode "$50" — it must read the `company_policy` field every time and extract the actual threshold. This tests genuine policy comprehension, not memorisation.

### 2. Distractor Documents
Every episode includes 1–2 CLOSED historical POs from other vendors and occasionally GRNs referencing different POs. The agent must identify the relevant documents (OPEN status, matching vendor, matching PO number) and ignore the noise — exactly as a real AP clerk does in enterprise ERP systems.

### 3. Multi-Step Episodes
Hard tasks support up to 3 steps. An ESCALATE or QUERY_VENDOR action reveals pre-generated manager/vendor responses in `context_notes`. The grader awards a small process bonus for agents that correctly verify before deciding — incentivising proper AP procedure over lucky guessing.

### 4. Harder Graders
- Stricter partial credit: wrong decisions score lower (REJECT on a shortfall: 0.15 not 0.35)
- Tighter amount tolerance curves (1% for full credit, 3% for partial, 8% near-zero)
- Higher explanation keyword thresholds (3–4 hits required vs 2)
- Dynamic keyword matching: graders check for the episode-specific cap value, not a hardcoded "$50"

---

## Action Space

**Type:** `APAction`

| Field | Type | Constraint | Description |
|---|---|---|---|
| `decision` | enum | see below | Payment verdict or intermediate step |
| `approved_amount` | float | ≥ 0.0 | Dollar amount to pay. Must be 0.0 if decision is REJECT. |
| `reason_code` | enum | see below | Diagnostic classification of the decision |
| `explanation` | string | 10–600 chars | Plain-English justification the agent must provide |

**Terminal decisions** (end the episode):

| Decision | When to use |
|---|---|
| `APPROVE_FULL` | All documents match; pay the full invoice total |
| `APPROVE_PARTIAL` | Partial match; pay only for what was received/authorised |
| `REJECT` | Policy violation, mismatch, or unresolvable discrepancy |

**Intermediate decisions** (multi-step; episode continues, context revealed):

| Decision | When to use |
|---|---|
| `QUERY_VENDOR` | Request vendor clarification before deciding |
| `ESCALATE` | Escalate to Finance Manager for complex policy questions |
| `HOLD` | Place invoice on hold pending further information |

**Reason codes:**

| Code | When to use |
|---|---|
| `MATCH_CONFIRMED` | All three documents agree, payment is clean |
| `QUANTITY_MISMATCH` | GRN shows fewer units received than invoiced |
| `PRICE_DISCREPANCY` | Invoice unit price differs from agreed PO price beyond tolerance |
| `POLICY_VIOLATION` | A company policy rule is breached (freight cap, unauthorised charge) |
| `NO_PO_FOUND` | No valid OPEN Purchase Order exists for this invoice |
| `DUPLICATE_INVOICE` | This invoice ID has already been paid |
| `VENDOR_MISMATCH` | Invoice vendor name does not match PO vendor name |
| `TAX_DISCREPANCY` | Invoice includes tax not authorised in the PO |
| `PENDING_CLARIFICATION` | Awaiting vendor response (for intermediate steps) |
| `MANAGER_REVIEW` | Escalated to Finance Manager (for intermediate steps) |

---

## Observation Space

**Type:** `APObservation`

| Field | Type | Description |
|---|---|---|
| `task_id` / `task_name` | string | Identifies the active task |
| `task_description` | string | Plain-English description of the scenario |
| `invoice` | `Invoice` | Vendor bill: line items, freight charge, tax amount, invoice total |
| `purchase_orders` | `List[PurchaseOrder]` | All POs in system — includes CLOSED distractors; only OPEN ones authorise payment |
| `goods_receipts` | `List[GoodsReceipt]` | Warehouse receipts — includes GRNs for other POs; match by `po_number` |
| `paid_invoice_ids` | `List[str]` | Invoice IDs already settled — populated in duplicate-detection tasks |
| `company_policy` | string | Free-text policy document with episode-specific thresholds |
| `step_count` / `max_steps` | int | Episode progress. max_steps > 1 for multi-step tasks. |
| `freight_cap` | float | Episode-specific freight cap (also stated in company_policy) |
| `price_tolerance` | float | Episode-specific price tolerance (also stated in company_policy) |
| `action_history` | `List[dict]` | Prior actions taken this episode (populated during multi-step) |
| `context_notes` | `List[str]` | Responses revealed by ESCALATE / QUERY_VENDOR intermediate actions |

**Invoice** also includes `tax_amount: float` (non-zero in `hard_tax_discrepancy`).

---

## Tasks

Ten randomised tasks across three difficulty levels. Each call to `/reset` with a new `seed` generates a different invoice, vendor, product, quantities, prices, and policy thresholds. The agent cannot memorise answers — it must reason from documents on every episode.

### Easy

| Task ID | What the agent must do |
|---|---|
| `easy_perfect_match` | Verify that the invoice, OPEN PO, and GRN all agree (ignoring CLOSED distractors), then approve the full amount. |
| `easy_no_po_found` | Recognise that no OPEN PO exists for the invoice reference and reject immediately. |

### Medium

| Task ID | What the agent must do |
|---|---|
| `medium_quantity_shortfall` | Calculate the correct payable amount based on GRN-confirmed quantities only, then partial-approve. |
| `medium_price_discrepancy` | Detect that the invoice unit price deviates from the agreed PO price beyond the episode-specific tolerance and reject. |
| `medium_split_delivery` | Sum quantities across two GRNs for the same PO (split shipment) and approve the full amount. |
| `medium_vendor_mismatch` | Identify that the invoice vendor name differs subtly from the PO vendor and reject per policy. |

### Hard

| Task ID | What the agent must do | max_steps |
|---|---|---|
| `hard_policy_violation` | Identify that freight exceeds the episode-specific cap. Optionally ESCALATE for Finance Manager confirmation before rejecting. | 3 |
| `hard_duplicate_invoice` | Recognise the invoice ID in the paid ledger. Optionally QUERY_VENDOR before rejecting. | 3 |
| `hard_partial_po_match` | Invoice has two line items; only one is covered by the PO. Partial-approve for the authorised amount only. | 1 |
| `hard_tax_discrepancy` | Vendor adds a tax charge not in the PO. Detect and reject. | 1 |

---

## Reward Design

Rewards are partial and multi-dimensional. Each grader decomposes the score across weighted sub-components so that an agent making the right decision with slightly wrong arithmetic still receives a meaningful learning signal.

| Task | Score formula |
|---|---|
| `easy_perfect_match` | 50% decision + 35% amount accuracy + 15% reason code |
| `easy_no_po_found` | 60% decision + 30% reason code + 10% amount is zero |
| `medium_quantity_shortfall` | 45% decision + 40% amount accuracy + 15% reason code |
| `medium_price_discrepancy` | 55% decision + 30% reason code + 15% explanation quality |
| `medium_split_delivery` | 50% decision + 35% amount (sum of both GRNs) + 15% reason code |
| `medium_vendor_mismatch` | 50% decision + 25% reason code + 15% explanation + 10% amount zero |
| `hard_policy_violation` | 48% decision + 27% reason code + 20% explanation + 5% process bonus |
| `hard_duplicate_invoice` | 48% decision + 27% reason code + 20% explanation + 5% process bonus |
| `hard_partial_po_match` | 45% decision + 38% amount (PO-covered only) + 12% reason code + 5% explanation |
| `hard_tax_discrepancy` | 50% decision + 30% reason code + 20% explanation |

**Amount accuracy tiers** (for tasks with partial-credit on amount):
- Within 1%: full credit (1.0)
- Within 3%: 0.60–0.65
- Within 8%: 0.30–0.40
- Beyond 8%: near zero

**Process bonus** (multi-step tasks only): +0.05 if agent uses the correct intermediate step (ESCALATE / QUERY_VENDOR) before making the right terminal decision.

---

## Project Structure

```
ap-clerk-env/
├── Dockerfile            # Container definition for HF Spaces (port 7860)
├── README.md             # This file
├── openenv.yaml          # OpenEnv spec: tasks, action space, observation space, endpoints
├── inference.py          # Baseline agent — multi-step loop, writes results.json
├── validate.py           # Pre-submission validator — run locally before deploying
├── requirements.txt      # Python dependencies
└── app/
    ├── __init__.py
    ├── main.py           # FastAPI server — /reset /step /state /tasks /health
    ├── models.py         # Pydantic typed models — Observation, Action, Reward
    ├── tasks.py          # Task generators (randomised) and graders (deterministic)
    └── environment.py    # APClerkEnvironment — reset() / step() / state()
```

---

## API Reference

| Endpoint | Method | Body / Params | Returns |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok"}` |
| `/tasks` | GET | — | List of all 10 tasks with difficulty metadata |
| `/reset` | POST | `{"task_id": "...", "seed": 42}` | Observation + session_id |
| `/step` | POST | `{"session_id": "...", "action": {...}}` | Reward (score, breakdown, feedback) + done |
| `/state` | GET | `?session_id=...` | Current session state |
| `/docs` | GET | — | Swagger UI |

`seed` in `/reset` is optional. Omit for a random episode, or pass a fixed integer for reproducible episodes.

---

## Setup

### Requirements

- Python 3.10, 3.11, or 3.12
- Hugging Face account with an API token (free tier is sufficient)

### Install and run locally

```bash
git clone https://huggingface.co/spaces/Pathikreet/ap-clerk-env
cd ap-clerk-env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Open `http://localhost:7860/docs` for the interactive API documentation.

### Run baseline inference

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

Scores are printed per task and written to `results.json`. The inference script handles multi-step episodes automatically — it loops until `done=True`.

### Run with Docker

```bash
docker build -t ap-clerk-env .
docker run -p 7860:7860 ap-clerk-env
```

### Validate before submitting

```bash
python validate.py
```

---

## Environment Variables

All three are required. `inference.py` will exit with a clear error message if any are missing.

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face API token, used as the LLM API key |
| `API_BASE_URL` | OpenAI-compatible endpoint, e.g. `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier, e.g. `Qwen/Qwen2.5-72B-Instruct` |

---

## Design Notes

**Why randomised policy thresholds:** Real AP policy documents change. A new audit year brings different freight caps, updated tax rules, revised price tolerance bands. An agent trained on fixed "$50" rules fails the moment the policy changes. By randomising freight caps ($30–$100) and price tolerances (0.5%–3%) per episode, the environment forces the agent to read and apply the policy as written — not to recall a memorised number.

**Why distractor documents:** Enterprise ERP systems contain thousands of POs — OPEN, CLOSED, cancelled, from dozens of vendors. A real AP clerk must filter to the relevant OPEN PO from the correct vendor. Training on environments with exactly one PO teaches agents to always approve the first document they see. Distractors teach selective attention.

**Why multi-step episodes:** Complex policy cases in real enterprises require consultation. A freight charge question may need Finance Manager sign-off before the clerk can proceed. By allowing ESCALATE/QUERY_VENDOR intermediate steps that reveal additional context, the environment rewards systematic verification — not just lucky one-shot guessing.

**Why partial rewards:** A sparse 0/1 reward destroys gradient signal for near-correct agents. An agent that correctly identifies a quantity shortfall but calculates the payable amount 3% low should learn something from that episode. The weighted breakdown makes the reward surface continuous and learnable.

**Why free-text policy:** Real company policy documents are Word files and PDFs written in natural language. The hard tasks require the agent to read a plain-English policy string and extract a precise numeric rule — the same challenge enterprise AI faces in production.

**Why harder graders (v2):** The original graders were too generous, giving 0.35 credit for REJECT on a shortfall that should be partial-approved. Hard graders with tighter tolerances and higher keyword thresholds ensure only agents that genuinely understand the task score well. Partial credit still exists but requires more precision.
