"""
Inference Script — AP Clerk Environment
========================================
MANDATORY environment variables:
  API_BASE_URL   The OpenAI-compatible API base URL.
                 e.g. https://router.huggingface.co/v1
  MODEL_NAME     The model identifier.
                 e.g. Qwen/Qwen2.5-72B-Instruct
  HF_TOKEN       Your Hugging Face token (used as the API key).

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

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY:      str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

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
      "decision":        "APPROVE_FULL" | "APPROVE_PARTIAL" | "REJECT",
      "approved_amount": <float — dollar amount to pay, 0.0 if REJECT>,
      "reason_code":     "MATCH_CONFIRMED" | "QUANTITY_MISMATCH" | "PRICE_DISCREPANCY" |
                         "POLICY_VIOLATION" | "NO_PO_FOUND" | "DUPLICATE_INVOICE" |
                         "VENDOR_MISMATCH" | "TAX_DISCREPANCY",
      "explanation":     "<10–500 char plain-English justification>"
    }

    Decision rules:
    - APPROVE_FULL: Invoice, PO and GRN all match; pay the full invoice total.
    - APPROVE_PARTIAL: Partial match (quantity shortfall or partial PO coverage);
      pay only for what was received and authorised.
    - REJECT: Policy violation, no PO, vendor name mismatch, tax discrepancy,
      duplicate invoice, or unresolvable discrepancy; do not pay.

    Mandatory checks before deciding:
    1. Is there a valid OPEN Purchase Order matching the invoice PO reference?
    2. Does the PO vendor name EXACTLY match the invoice vendor name?
    3. Do invoice unit prices match agreed PO prices within the stated policy tolerance?
    4. Do GRN quantities (sum ALL GRNs for this PO) confirm receipt of invoiced goods?
    5. Is this invoice ID already in the paid ledger? (duplicate check)
    6. Does the invoice include any charges not in the PO (freight above cap, tax, fees)?
    7. Are ALL invoice line items covered by the PO?

    Policy thresholds (freight cap, price tolerance) VARY per episode.
    Always read the COMPANY POLICY section carefully for the exact values.

    Ignore CLOSED purchase orders — they are historical records and do not authorise payment.
    Ignore GRNs whose PO reference does not match the invoice's PO reference.
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

        print(
            f"[STEP] step={step_num} action={action.decision.value} "
            f"reward={reward.score:.2f} done={done}",
            flush=True,
        )

        if done:
            break

    print(f"[END] task={task_id} score={reward.score:.2f} steps={step_num}", flush=True)

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
    print("  AP Clerk Environment — Baseline Inference")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  API Base : {API_BASE_URL}")
    print("=" * 65)

    results     = []
    total_score = 0.0
    task_ids    = list(TASKS.keys())

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
            print(f"  ERROR: {exc}")
            results.append({"task_id": task_id, "score": 0.0, "error": str(exc)})

    mean_score = total_score / len(task_ids)
    print("\n" + "=" * 65)
    print(f"  MEAN SCORE: {mean_score:.3f}  ({total_score:.3f} / {len(task_ids)})")
    print("=" * 65)

    output = {
        "model":      MODEL_NAME,
        "api_base":   API_BASE_URL,
        "tasks":      results,
        "mean_score": round(mean_score, 4),
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults written to results.json")


if __name__ == "__main__":
    main()
