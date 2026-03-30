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

An RL training environment where an agent learns to act as a corporate Accounts Payable clerk. Given a vendor invoice, a Purchase Order, and a Goods Receipt Note, the agent must cross-reference all three documents, apply free-text company policy, and decide whether to approve, partially approve, or reject the invoice for payment.

Three-way invoice matching is a mandatory control in every company that purchases goods. It is the process that prevents overpayments, duplicate payments, and policy violations before money leaves the organisation. This environment models that process faithfully — the agent faces the same reasoning challenges a human AP clerk does.

---

## Environment Overview

### Episode Flow

```
POST /reset  { "task_id": "...", "seed": 42 }
  → { "observation": { invoice, purchase_orders, goods_receipts, company_policy }, "session_id": "uuid" }

POST /step   { "session_id": "uuid", "action": { decision, approved_amount, reason_code, explanation } }
  → { "reward": { "score": 0.0–1.0, "breakdown": {...}, "feedback": "..." }, "done": true }
```

Episodes are single-step. An AP decision is atomic — the clerk approves or rejects in one action, exactly as in real enterprise workflows.

---

## Action Space

**Type:** `APAction`

| Field | Type | Constraint | Description |
|---|---|---|---|
| `decision` | enum | `APPROVE_FULL`, `APPROVE_PARTIAL`, `REJECT` | Payment verdict |
| `approved_amount` | float | ≥ 0.0 | Dollar amount to pay. Must be 0.0 if decision is REJECT. |
| `reason_code` | enum | see below | Diagnostic classification of the decision |
| `explanation` | string | 10–600 chars | Plain-English justification the agent must provide |

**Reason codes:**

| Code | When to use |
|---|---|
| `MATCH_CONFIRMED` | All three documents agree, payment is clean |
| `QUANTITY_MISMATCH` | GRN shows fewer units received than invoiced |
| `PRICE_DISCREPANCY` | Invoice unit price differs from agreed PO price |
| `POLICY_VIOLATION` | A company policy rule is breached (e.g. freight cap) |
| `NO_PO_FOUND` | No valid Purchase Order exists for this invoice |
| `DUPLICATE_INVOICE` | This invoice ID has already been paid |

---

## Observation Space

**Type:** `APObservation`

| Field | Type | Description |
|---|---|---|
| `task_id` / `task_name` | string | Identifies the active task |
| `task_description` | string | Plain-English description of the scenario |
| `invoice` | `Invoice` | Vendor bill: line items, freight charge, invoice total |
| `purchase_orders` | `List[PurchaseOrder]` | Company's pre-authorised orders against which the invoice is matched |
| `goods_receipts` | `List[GoodsReceipt]` | Warehouse confirmations of physical delivery |
| `paid_invoice_ids` | `List[str]` | Invoice IDs already settled — populated in duplicate-detection tasks |
| `company_policy` | string | Free-text policy document the agent must read and apply |
| `step_count` / `max_steps` | int | Episode progress. max_steps is always 1. |

---

## Tasks

Six randomised tasks across three difficulty levels. Each call to `/reset` with a new `seed` generates a different invoice, vendor, product, quantities, and prices. The agent cannot memorise answers — it must reason from the documents on every episode.

### Easy

| Task ID | What the agent must do |
|---|---|
| `easy_perfect_match` | Verify that invoice, PO, and GRN all agree, then approve the full amount. |
| `easy_no_po_found` | Recognise that the referenced PO does not exist and reject immediately. |

### Medium

| Task ID | What the agent must do |
|---|---|
| `medium_quantity_shortfall` | Calculate the correct payable amount based on the quantity actually received per GRN, not the quantity invoiced. |
| `medium_price_discrepancy` | Detect that the invoice unit price deviates from the agreed PO price by more than the 1% policy threshold and reject. |

### Hard

| Task ID | What the agent must do |
|---|---|
| `hard_policy_violation` | Identify that quantities match but a freight charge exceeds the $50 unapproved cap stated in the free-text policy, then reject so the vendor must resubmit with manager approval. |
| `hard_duplicate_invoice` | Recognise that despite valid supporting documents, the invoice ID already appears in the paid ledger and block the duplicate payment. |

---

## Reward Design

Rewards are partial and multi-dimensional. Each grader decomposes the score across weighted sub-components so that an agent making the right decision with slightly wrong arithmetic still receives a meaningful learning signal — not zero.

| Task | Score formula |
|---|---|
| `easy_perfect_match` | 50% decision type + 35% amount accuracy + 15% reason code |
| `easy_no_po_found` | 60% decision type + 30% reason code + 10% amount is zero |
| `medium_quantity_shortfall` | 45% decision type + 40% amount accuracy + 15% reason code |
| `medium_price_discrepancy` | 55% decision type + 30% reason code + 15% price deviation cited in explanation |
| `hard_policy_violation` | 50% decision type + 30% reason code + 20% policy breach cited in explanation |
| `hard_duplicate_invoice` | 50% decision type + 30% reason code + 20% duplicate cited in explanation |

Amount accuracy uses a tolerance curve: within 1% scores full credit, within 5% scores partial credit, beyond 15% scores near zero. This gives the reward function a continuous gradient rather than a sharp cliff.

---

## Project Structure

```
ap-clerk-env/
├── Dockerfile            # Container definition for HF Spaces (port 7860)
├── README.md             # This file
├── openenv.yaml          # OpenEnv spec: tasks, action space, observation space, endpoints
├── inference.py          # Baseline agent — runs all 6 tasks, writes results.json
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
| `/tasks` | GET | — | List of all 6 tasks with difficulty metadata |
| `/reset` | POST | `{"task_id": "...", "seed": 42}` | Observation + session_id |
| `/step` | POST | `{"session_id": "...", "action": {...}}` | Reward (score, breakdown, feedback) + done |
| `/state` | GET | `?session_id=...` | Current session state |
| `/docs` | GET | — | Swagger UI |

`seed` in `/reset` is optional. Omit it for a random episode each call, or pass a fixed integer for a reproducible episode.

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

Scores are printed per task and written to `results.json`.

### Run with Docker

```bash
docker build -t ap-clerk-env .
docker run -p 7860:7860 ap-clerk-env
```

### Validate before submitting

```bash
python validate.py
```

Runs 47 checks across Python version, file structure, OpenEnv spec, all 6 graders, randomisation, FastAPI routes, and Dockerfile compliance. All must pass.

---

## Environment Variables

All three are required. `inference.py` will exit with a clear error message if any are missing.

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face API token, used as the LLM API key |
| `API_BASE_URL` | OpenAI-compatible endpoint, e.g. `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier, e.g. `Qwen/Qwen2.5-72B-Instruct` |

---

## Baseline Scores

Run with `Qwen/Qwen2.5-72B-Instruct` via the HuggingFace inference router, fixed seed:

| Task | Decision | Amount | Score |
|---|---|---|---|
| `easy_perfect_match` | APPROVE_FULL | invoice total | 1.000 |
| `easy_no_po_found` | REJECT | $0.00 | 1.000 |
| `medium_quantity_shortfall` | APPROVE_PARTIAL | received × unit price | 1.000 |
| `medium_price_discrepancy` | REJECT | $0.00 | 1.000 |
| `hard_policy_violation` | REJECT | $0.00 | 1.000 |
| `hard_duplicate_invoice` | REJECT | $0.00 | 1.000 |
| **Mean** | | | **1.000** |

---

## Design Notes

**Why single-step episodes:** An AP payment decision is a one-shot action in real enterprise systems. The invoice either gets approved or returned to the vendor. There is no negotiation mid-episode. The difficulty lies entirely in reading and reasoning across three documents simultaneously, not in sequential decision-making.

**Why partial rewards:** A sparse 0/1 reward destroys gradient signal for near-correct agents. An agent that correctly identifies a quantity shortfall but calculates the payable amount 3% low should learn something from that episode, not receive zero. The weighted breakdown makes the reward surface continuous and learnable.

**Why free-text policy:** Real company policy documents are not structured data. They are Word files and PDFs written in natural language. The hard tasks in this environment require the agent to read a plain-English policy string and extract a precise numeric rule ($50 freight cap) — the same challenge enterprise AI faces in production.

**Why randomised observations:** Hardcoded scenarios allow an agent to memorise the correct answer after one episode. Every `/reset` call with a unique seed produces a different vendor, product, quantity, price, and document set. The agent must reason from first principles every time.
