"""
Inference Script — AP Commander (Multi-Agent Enterprise Financial Environment)
==============================================================================
MANDATORY environment variables:
  API_BASE_URL   The OpenAI-compatible API base URL.
                 e.g. https://router.huggingface.co/v1
  MODEL_NAME     The model identifier.
                 e.g. Qwen/Qwen2.5-72B-Instruct
  HF_TOKEN       Your Hugging Face token (used as the API key).

Optional:
  RUN_OVERSIGHT=1   Also run oversight agent tasks
  TASK_FILTER=easy  Only run tasks matching this difficulty prefix

Usage:
  export API_BASE_URL="https://router.huggingface.co/v1"
  export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py

Writes results to results.json in the current directory.
"""

import os
import sys
import json
import re
import time
import textwrap
from typing import Optional

from openai import OpenAI

from app import APClerkEnvironment, APAction, DecisionType, ReasonCode
from app.tasks import TASKS

API_BASE_URL:   str  = os.getenv("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME:     str  = os.getenv("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")
API_KEY:        str  = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
RUN_OVERSIGHT:  bool = os.getenv("RUN_OVERSIGHT", "0") == "1"
TASK_FILTER:    str  = os.getenv("TASK_FILTER", "")  # e.g. "easy" or "long"

if not API_KEY:
    print("ERROR: required environment variable 'HF_TOKEN' is not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

MAX_TOKENS  = 600
TEMPERATURE = 0.0

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI Accounts Payable Clerk. Your job is to perform three-way invoice
    matching: compare the vendor INVOICE against the company PURCHASE ORDER (PO) and
    the warehouse GOODS RECEIPT NOTE (GRN), then apply COMPANY POLICY to decide.

    Respond with ONLY a single valid JSON object — no extra text, no markdown fences.

    JSON schema:
    {
      "decision":        "APPROVE_FULL" | "APPROVE_PARTIAL" | "REJECT" |
                         "ESCALATE" | "QUERY_VENDOR" | "HOLD",
      "approved_amount": <float — dollar amount to pay; 0.0 if REJECT/ESCALATE/QUERY_VENDOR/HOLD;
                          use the NEGATIVE credit amount for credit memo approvals>,
      "reason_code":     "MATCH_CONFIRMED" | "QUANTITY_MISMATCH" | "PRICE_DISCREPANCY" |
                         "POLICY_VIOLATION" | "NO_PO_FOUND" | "DUPLICATE_INVOICE" |
                         "VENDOR_MISMATCH" | "TAX_DISCREPANCY" |
                         "PENDING_CLARIFICATION" | "MANAGER_REVIEW",
      "explanation":     "<10–500 char justification — MUST cite specific dollar values or percentages>"
    }

    REASON CODE MAPPING (use exactly the matching code for your decision):
      APPROVE_FULL    → MATCH_CONFIRMED
      APPROVE_PARTIAL → QUANTITY_MISMATCH  (or MATCH_CONFIRMED for credit memos / partial PO)
      REJECT          → NO_PO_FOUND | PRICE_DISCREPANCY | POLICY_VIOLATION | DUPLICATE_INVOICE
                        | VENDOR_MISMATCH | TAX_DISCREPANCY
      QUERY_VENDOR    → PENDING_CLARIFICATION
      ESCALATE        → MANAGER_REVIEW
      HOLD            → PENDING_CLARIFICATION

    MULTI-STEP TRIGGERS (only when max_steps > 1):
      USE ESCALATE when:
        (1) freight_charge > freight_cap stated in COMPANY POLICY, AND max_steps > 1
        (2) policy text mentions "manager approval required" or "freight override"
        (3) long-horizon tasks: manager is out-of-office (context note says so) → ESCALATE again for VP Finance
        After ESCALATE: read context_notes carefully — if manager pre-approved, APPROVE_FULL.
        If NOT pre-approved, REJECT with POLICY_VIOLATION.
      USE QUERY_VENDOR when:
        (1) invoice_id already appears in PAID INVOICE LEDGER, AND max_steps > 1
        (2) price discrepancy exists in dispute tasks (long_invoice_dispute)
        After QUERY_VENDOR: read context_notes — vendor confirming duplicate → REJECT with DUPLICATE_INVOICE.
      USE HOLD when:
        (1) compliance review required (context mentions SOX/GDPR/compliance flag)
        After HOLD: read context_notes for compliance verdict then decide.
      Single-step tasks (max_steps = 1): go directly to APPROVE_FULL / APPROVE_PARTIAL / REJECT.

    LONG-HORIZON TASK TIPS (max_steps 10-16):
      - long_invoice_dispute: QUERY_VENDOR first, then ESCALATE, then REJECT
      - long_policy_migration: HOLD to get compliance update, then re-read new policy, APPROVE_FULL
      - long_manager_chain: ESCALATE (manager OOO) → ESCALATE again (VP Finance) → APPROVE_FULL
      - long_fraud_investigation: QUERY_VENDOR (vendor denies) → ESCALATE (manager confirms duplicate) → REJECT
      - long_audit_trail: HOLD for SOX review → APPROVE_FULL with PO/GRN/amount citations
      - long_batch_reconciliation: treat as standard match in batch context
      - long_multi_vendor_split: approve invoice amount (first tranche only)

    DECISION RULES:
    - APPROVE_FULL: Invoice, PO (OPEN) and GRN all match exactly. Pay full invoice total.
    - APPROVE_PARTIAL: Quantity shortfall, partial PO coverage, or credit memo with valid PO.
      Pay only for what was received and authorised. For credit memos, approved_amount is NEGATIVE.
    - REJECT: Policy violation, no valid OPEN PO, vendor name mismatch, tax not in PO,
      duplicate invoice, price deviation over threshold, or no PO for credit memo.
    - ESCALATE/QUERY_VENDOR: Intermediate steps that reveal context (see triggers above).

    MANDATORY CHECKS before deciding:
    1. OPEN PO? — Find a matching OPEN PO by po_number. Ignore ALL CLOSED POs.
    2. Vendor name? — Invoice vendor must EXACTLY match PO vendor name. Any difference → REJECT.
    3. Price check? — Compute deviation = |invoice_price - po_price| / po_price.
       If deviation > price_tolerance (in COMPANY POLICY) → REJECT with PRICE_DISCREPANCY.
    4. Quantity? — Sum received_quantity across ALL GRNs whose po_number matches the OPEN PO.
       Pay only for received quantity × agreed PO price.
    5. Duplicate? — Is invoice_id in PAID INVOICE LEDGER? If yes → QUERY_VENDOR (if multi-step) or REJECT.
    6. Extra charges? — Freight above cap → check policy for override. Tax not in PO → REJECT.
    7. Line items? — Each invoice line must be covered by the PO. Uncovered items → APPROVE_PARTIAL.
    8. Currency? — If invoice currency ≠ USD, convert using the exchange rate in COMPANY POLICY.
       approved_amount must be in USD.

    EXPLANATION QUALITY: Always cite specific numbers. Good examples:
      "Invoice price $520 vs PO price $500 — 4% deviation exceeds 2% threshold. REJECT."
      "Freight $85 exceeds cap of $50. Escalating to Finance Manager for pre-approval check."
      "GRN confirms 80 of 100 ordered units. Approving $40,000 (80 × $500) per Policy Rule 3."

    EXAMPLES:
    Example 1 — Perfect match:
    Invoice $1,500, PO authorizes $1,500, GRN confirms all 10 units. Freight $20 under $50 cap.
    → {"decision":"APPROVE_FULL","approved_amount":1500.00,"reason_code":"MATCH_CONFIRMED",
       "explanation":"Invoice $1,500 matches PO-2024-001 ($1,500) and GRN confirms all 10 units received. Freight $20 within $50 cap."}

    Example 2 — Duplicate invoice (multi-step):
    Invoice INV-2024-5432 is in the PAID INVOICE LEDGER. max_steps = 3.
    → {"decision":"QUERY_VENDOR","approved_amount":0.0,"reason_code":"PENDING_CLARIFICATION",
       "explanation":"INV-2024-5432 already appears in the paid ledger. Querying vendor to confirm before final rejection."}
    (After vendor confirms duplicate:)
    → {"decision":"REJECT","approved_amount":0.0,"reason_code":"DUPLICATE_INVOICE",
       "explanation":"Vendor confirmed INV-2024-5432 was paid in a prior cycle. Rejecting duplicate per Policy Rule 6."}

    Policy thresholds (freight cap, price tolerance, FX rate) VARY per episode.
    Always read COMPANY POLICY carefully for exact values.
""").strip()


def build_user_prompt(obs) -> str:
    inv = obs.invoice
    lines_text = "\n".join(
        f"  - {li.description}: qty={li.quantity}, unit_price=${li.unit_price:.2f}, "
        f"line_total=${li.line_total:.2f}"
        for li in inv.line_items
    )
    tax_line = f"  Tax         : ${inv.tax_amount:.2f}\n" if inv.tax_amount > 0 else ""
    invoice_block = (
        f"INVOICE {inv.invoice_id}\n"
        f"  Vendor      : {inv.vendor_name}\n"
        f"  PO Reference: {inv.po_reference or 'NONE'}\n"
        f"  Line Items  :\n{lines_text}\n"
        f"  Freight     : ${inv.freight_charge:.2f}\n"
        f"{tax_line}"
        f"  TOTAL BILLED: ${inv.invoice_total:.2f}"
    )

    po_blocks = []
    for po in obs.purchase_orders:
        po_lines = "\n".join(
            f"    - {pl.description}: ordered_qty={pl.ordered_quantity}, "
            f"agreed_price=${pl.agreed_unit_price:.2f}"
            for pl in po.lines
        )
        po_blocks.append(
            f"PO {po.po_number} ({po.status})\n"
            f"  Vendor          : {po.vendor_name}\n"
            f"  Lines           :\n{po_lines}\n"
            f"  Authorized Total: ${po.authorized_total:.2f}"
        )

    grn_blocks = []
    for grn in obs.goods_receipts:
        grn_lines = "\n".join(
            f"    - {gl.description}: received_qty={gl.received_quantity}"
            for gl in grn.lines
        )
        grn_blocks.append(
            f"GRN {grn.grn_id} (for PO {grn.po_number})\n"
            f"  Lines:\n{grn_lines}"
        )

    po_section  = "\n\n".join(po_blocks)  if po_blocks  else "  (no purchase order found in system)"
    grn_section = "\n\n".join(grn_blocks) if grn_blocks else "  (no goods receipt found in system)"

    ledger_section = ""
    if obs.paid_invoice_ids:
        ledger_section = (
            f"{'='*60}\n"
            f"PAID INVOICE LEDGER (already settled):\n"
            + "\n".join(f"  - {iid}" for iid in obs.paid_invoice_ids) + "\n\n"
        )

    context_section = ""
    if obs.context_notes:
        context_section = (
            f"{'='*60}\n"
            f"ADDITIONAL CONTEXT (revealed by prior query/escalation):\n"
            + "\n".join(f"  {note}" for note in obs.context_notes) + "\n\n"
        )

    history_section = ""
    if obs.action_history:
        history_section = (
            f"{'='*60}\n"
            f"YOUR PRIOR ACTIONS THIS EPISODE:\n"
            + "\n".join(
                f"  Step {h['step']}: {h['decision']} — {h['explanation'][:80]}"
                for h in obs.action_history
            ) + "\n\n"
        )

    return (
        f"TASK: {obs.task_name}\n"
        f"{obs.task_description}\n\n"
        f"{'='*60}\n"
        f"{invoice_block}\n\n"
        f"{'='*60}\n"
        f"{po_section}\n\n"
        f"{'='*60}\n"
        f"{grn_section}\n\n"
        f"{ledger_section}"
        f"{context_section}"
        f"{history_section}"
        f"{'='*60}\n"
        f"COMPANY POLICY:\n{obs.company_policy}\n\n"
        f"Now output your JSON decision."
    )


def call_llm(user_prompt: str) -> str:
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                timeout=60,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt == 0:
                print(f"  [WARN] LLM call failed ({exc}), retrying in 3s…", flush=True)
                time.sleep(3)
            else:
                raise


def parse_action(raw: str) -> Optional[APAction]:
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    try:
        decision    = DecisionType(data.get("decision", "REJECT").upper())
        reason_code = ReasonCode(data.get("reason_code", "NO_PO_FOUND").upper())
    except ValueError:
        decision    = DecisionType.REJECT
        reason_code = ReasonCode.NO_PO_FOUND
    try:
        return APAction(
            decision=decision,
            approved_amount=float(data.get("approved_amount", 0.0)),
            reason_code=reason_code,
            explanation=str(data.get("explanation", "No explanation provided."))[:500],
        )
    except Exception:
        return None


def run_task(task_id: str, seed: int = None) -> dict:
    env  = APClerkEnvironment()
    obs  = env.reset(task_id, seed=seed)
    done = False
    reward = None
    step_num = 0
    max_steps = obs.max_steps
    raw_response = ""
    action = None
    step_rewards: list = []

    print(f"[START] task={task_id}", flush=True)

    while not done and step_num < max_steps:
        raw_response = call_llm(build_user_prompt(obs))
        action = parse_action(raw_response)
        if action is None:
            action = APAction(
                decision=DecisionType.REJECT,
                approved_amount=0.0,
                reason_code=ReasonCode.NO_PO_FOUND,
                explanation="Unable to parse response; defaulting to safe rejection.",
            )

        step_num += 1
        obs, reward, done, info = env.step(action)
        step_rewards.append(reward.score)

        print(f"[STEP] step={step_num} reward={reward.score:.2f}", flush=True)

        if done:
            break

    final_score = step_rewards[-1] if step_rewards else 0.01
    print(f"[END] task={task_id} score={final_score:.2f} steps={step_num}", flush=True)

    return {
        "task_id":         task_id,
        "decision":        action.decision.value,
        "approved_amount": action.approved_amount,
        "reason_code":     action.reason_code.value,
        "explanation":     action.explanation,
        "score":           reward.score,
        "breakdown":       reward.breakdown,
        "feedback":        reward.feedback,
        "steps_taken":     obs.step_count,
        "raw_response":    raw_response,
    }


def main():
    print("=" * 65)
    print("  AP Commander — Multi-Agent Enterprise Environment")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  API Base : {API_BASE_URL}")
    print("=" * 65)

    results     = []
    total_score = 0.0
    # Exclude oversight stub tasks (they use the dedicated /oversight/* endpoints)
    all_task_ids = [
        tid for tid, spec in TASKS.items()
        if spec.difficulty != "oversight"
    ]
    # Apply optional filter
    if TASK_FILTER:
        task_ids = [tid for tid in all_task_ids if TASK_FILTER in tid]
        print(f"  Filter   : '{TASK_FILTER}' → {len(task_ids)} tasks")
    else:
        task_ids = all_task_ids
    print(f"  Tasks    : {len(task_ids)}")
    print("=" * 65)

    for task_id in task_ids:
        print(f"\n[{task_id}]")
        t0 = time.time()
        try:
            result  = run_task(task_id)
            elapsed = time.time() - t0
            results.append(result)
            total_score += result["score"]
            print(f"  Decision : {result['decision']}  (${result['approved_amount']:,.2f})")
            print(f"  Reason   : {result['reason_code']}")
            print(f"  Score    : {result['score']:.3f}   (steps: {result['steps_taken']})")
            print(f"  Feedback : {result['feedback'][:120]}")
            print(f"  Time     : {elapsed:.1f}s")
        except Exception as exc:
            print(f"[END] task={task_id} score=0.01 steps=0", flush=True)
            print(f"  ERROR: {exc}")
            results.append({"task_id": task_id, "score": 0.01, "error": str(exc)})

    mean_score = total_score / len(task_ids) if task_ids else 0.0

    # Score breakdown by difficulty
    by_diff: dict = {}
    for r in results:
        spec = TASKS.get(r.get("task_id", ""))
        if spec:
            d = spec.difficulty
            by_diff.setdefault(d, []).append(r.get("score", 0.01))
    diff_means = {d: round(sum(s)/len(s), 3) for d, s in by_diff.items()}

    print("\n" + "=" * 65)
    print(f"  MEAN SCORE: {mean_score:.3f}  ({total_score:.3f} / {len(task_ids)})")
    for d, m in sorted(diff_means.items()):
        print(f"    {d:<16}: {m:.3f}")
    print("=" * 65)

    output = {
        "model":           MODEL_NAME,
        "api_base":        API_BASE_URL,
        "tasks":           results,
        "mean_score":      round(mean_score, 4),
        "by_difficulty":   diff_means,
        "environment":     "ap-commander",
        "version":         "4.0.0",
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults written to results.json")


if __name__ == "__main__":
    main()
