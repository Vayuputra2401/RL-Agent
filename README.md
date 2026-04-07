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

A reinforcement learning environment that simulates an Accounts Payable clerk performing three-way invoice matching. Given a vendor invoice, a set of Purchase Orders, and warehouse Goods Receipt Notes, an agent must cross-reference the documents, read the company policy, and decide whether to approve, partially approve, or reject the invoice for payment.

Three-way invoice matching is a standard financial control used by companies to prevent overpayments, duplicate payments, and policy violations before money leaves the organisation.

---

## Goal

The agent receives a set of documents and must produce a payment decision:

- **APPROVE_FULL** — all documents agree, pay the full invoice amount
- **APPROVE_PARTIAL** — partial match, pay only for what was received
- **REJECT** — mismatch, policy violation, or unresolvable discrepancy

For complex cases, the agent can take an intermediate step first (ESCALATE, QUERY_VENDOR, HOLD) to reveal additional context before making the final call.

---

## How an Episode Works

### Single-step tasks

```
POST /reset  { "task_id": "easy_perfect_match", "seed": 42 }
→ observation: invoice, purchase_orders, goods_receipts, company_policy

POST /step   { "session_id": "...", "action": { decision, approved_amount, reason_code, explanation } }
→ reward: { score, breakdown, feedback }, done: true
```

### Multi-step tasks

```
POST /reset  → observation (max_steps: 3)
POST /step   { action: { decision: "ESCALATE", ... } }   → done: false, context_notes revealed
POST /step   { action: { decision: "REJECT", ... } }     → done: true, score awarded
```

The agent can skip the intermediate step and go straight to a terminal decision — it just misses the small process bonus for following proper verification procedure.

---

## What the Agent Sees (Observation)

| Field | Description |
|---|---|
| `invoice` | Vendor bill: line items, quantities, unit prices, freight charge, tax amount, total |
| `purchase_orders` | All POs in the system — includes CLOSED historical ones; only OPEN POs authorise payment |
| `goods_receipts` | Warehouse receipts — may include GRNs for other POs; match by `po_number` |
| `paid_invoice_ids` | Invoice IDs already settled — used in duplicate-detection tasks |
| `company_policy` | Plain-text policy document with episode-specific thresholds (freight cap, price tolerance) |
| `step_count` / `max_steps` | Episode progress |
| `action_history` | Prior actions taken this episode |
| `context_notes` | Responses revealed by intermediate actions |

Every episode is randomised from a seed. Freight caps ($30–$100) and price tolerances (0.5%–3%) vary per episode, so the agent must read the policy each time.

---

## What the Agent Does (Action)

| Field | Type | Description |
|---|---|---|
| `decision` | enum | Payment verdict (see below) |
| `approved_amount` | float | Dollar amount to pay. Must be 0.0 if REJECT. |
| `reason_code` | enum | Classification of the decision (see below) |
| `explanation` | string | Plain-English justification (10–600 chars) |

**Terminal decisions:**

| Decision | When |
|---|---|
| `APPROVE_FULL` | Invoice, PO, and GRN all match |
| `APPROVE_PARTIAL` | Partial match — pay only for what was received or authorised |
| `REJECT` | Policy violation, mismatch, no PO, duplicate |

**Intermediate decisions** (multi-step tasks, episode continues):

| Decision | When |
|---|---|
| `ESCALATE` | Escalate to Finance Manager |
| `QUERY_VENDOR` | Request vendor clarification |
| `HOLD` | Place invoice on hold |

**Reason codes:**

`MATCH_CONFIRMED` / `QUANTITY_MISMATCH` / `PRICE_DISCREPANCY` / `POLICY_VIOLATION` / `NO_PO_FOUND` / `DUPLICATE_INVOICE` / `VENDOR_MISMATCH` / `TAX_DISCREPANCY` / `PENDING_CLARIFICATION` / `MANAGER_REVIEW`

---

## Tasks

Ten tasks across three difficulty levels. Each call to `/reset` with a different `seed` produces a different invoice, vendor, product, quantities, prices, and policy thresholds.

### Easy

| Task ID | What the agent must do |
|---|---|
| `easy_perfect_match` | Verify invoice, PO, and GRN all agree, then approve full amount |
| `easy_no_po_found` | Recognise there is no OPEN PO for this invoice and reject |

### Medium

| Task ID | What the agent must do |
|---|---|
| `medium_quantity_shortfall` | Calculate payable amount from GRN-confirmed quantities only and partial-approve |
| `medium_price_discrepancy` | Detect invoice unit price exceeds agreed PO price beyond tolerance and reject |
| `medium_split_delivery` | Sum quantities across two GRNs for the same PO and approve full amount |
| `medium_vendor_mismatch` | Identify subtle vendor name mismatch between invoice and PO and reject |

### Hard

| Task ID | What the agent must do | max_steps |
|---|---|---|
| `hard_policy_violation` | Freight charge exceeds episode-specific cap; optionally escalate before rejecting | 3 |
| `hard_duplicate_invoice` | Invoice ID is already in the paid ledger; optionally query vendor before rejecting | 3 |
| `hard_partial_po_match` | Only one of two invoice line items is covered by the PO; partial-approve for covered lines only | 1 |
| `hard_tax_discrepancy` | Invoice includes a tax charge not in the PO; reject | 1 |

---

## Grading

Scores are partial and broken down across components so near-correct decisions still produce a useful learning signal.

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
| `hard_partial_po_match` | 45% amount (PO-covered lines only) + 38% decision + 12% reason code + 5% explanation |
| `hard_tax_discrepancy` | 50% decision + 30% reason code + 20% explanation |

**Amount accuracy tiers** (for tasks that partially score the approved amount):
- Within 1%: full credit
- Within 3%: 0.60–0.65
- Within 8%: 0.30–0.40
- Beyond 8%: near zero

**Process bonus** (multi-step tasks only): +0.05 added if the agent uses the correct intermediate step before the right terminal decision. An agent that skips straight to REJECT on those tasks scores ~0.80 instead of ~0.85.

All scores are in the open interval (0.01, 0.99).

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks with difficulty metadata |
| `/reset` | POST | Start a new episode, returns observation + session_id |
| `/step` | POST | Submit an action, returns reward + done |
| `/state` | GET | Current session state |
| `/docs` | GET | Swagger UI |

`seed` in `/reset` is optional. Omit for a random episode, or pass a fixed integer for a reproducible one.

---

## Project Structure

```
├── Dockerfile            # Container definition (port 7860)
├── README.md
├── openenv.yaml          # Environment spec: tasks, action space, endpoints
├── inference.py          # Baseline agent — runs all 10 tasks, writes results.json
├── validate.py           # Local pre-submit validator
├── requirements.txt
└── app/
    ├── main.py           # FastAPI server
    ├── models.py         # Pydantic models: Observation, Action, Reward
    ├── tasks.py          # Task generators and graders
    └── environment.py    # APClerkEnvironment: reset() / step() / state()
```

---

## Setup

### Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Open `http://localhost:7860/docs` for the interactive API docs.

### Run inference

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

Scores are printed per task and written to `results.json`.

### Run with Docker

```bash
docker build -t ap-clerk-env .
docker run -p 7860:7860 ap-clerk-env
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes | — | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
