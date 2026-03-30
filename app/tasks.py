"""
AP Clerk Environment — Task Definitions, Generators & Graders
Six tasks across easy / medium / hard.
All graders are pure functions: grade_xxx(obs, action) -> APReward.
All generators produce a fresh randomised observation each episode.
"""

from __future__ import annotations
import random as _random
from dataclasses import dataclass
from typing import Callable, Dict

from .models import (
    APObservation, APAction, APReward,
    Invoice, LineItem, PurchaseOrder, POLine, GoodsReceipt, GRNLine,
    DecisionType, ReasonCode,
)

POLICY_STANDARD = """\
ACME Corp Accounts Payable Policy (Effective 2024-01-01)
--------------------------------------------------------
1. Three-Way Match:  Every invoice MUST match a valid Purchase Order (PO) AND a
   Goods Receipt Note (GRN) before payment is authorised.
2. Freight Cap:      Freight / shipping charges exceeding $50.00 per invoice require
   explicit prior approval from the Finance Manager. Do NOT pay unapproved freight.
3. Quantity Rule:    Payment is made only for quantities confirmed received in the GRN.
   If fewer items arrived than were invoiced, approve only the received amount.
4. Price Integrity:  Unit prices must match the agreed PO price. Any deviation > 1%
   must be queried and rejected until a corrected invoice is received.
5. PO Mandatory:     Invoices without a valid PO reference MUST be rejected.
6. Duplicate Block:  Any invoice ID that has already been paid must be rejected
   immediately as a duplicate — regardless of supporting documents.
""".strip()

# ── Shared randomised helpers ─────────────────────────────────────────────────

_OFFICE_ITEMS = [
    ("Ergonomic Office Chair Model-X",   "OfficeWorld Supplies Ltd."),
    ("Height-Adjustable Standing Desk",  "FurniCorp Direct"),
    ("Monitor Arm Dual-Screen Mount",    "TechDesk Co."),
    ("Mesh Back Support Lumbar Cushion", "ErgoSupply Ltd."),
]

_TECH_ITEMS = [
    ("ThinkPad L15 Gen-4 Laptop",        "TechProcure Global"),
    ("Dell XPS 15 Business Laptop",      "CompuSource Inc."),
    ("HP EliteBook 840 G10",             "DataTech Ltd."),
    ("Lenovo IdeaPad 5 Pro",             "DigitalParts Corp."),
]

_CABLE_ITEMS = [
    ("USB-C 2m Braided Cable",           "CableMart Direct"),
    ("HDMI 4K 3m Certified Cable",       "WireZone Ltd."),
    ("Cat6 Ethernet Patch Cable 5m",     "NetCables Co."),
    ("DisplayPort 1.4 2m Cable",         "ConnectPro Ltd."),
]

def _rng(seed):
    return _random.Random(seed)

def _po(rng):  return f"PO-{rng.randint(1000, 9999)}"
def _grn(rng): return f"GRN-{rng.randint(1000, 9999)}"
def _inv(rng): return f"INV-2024-{rng.randint(1000, 9999):04d}"


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1  —  Easy: Perfect Three-Way Match
# ═══════════════════════════════════════════════════════════════════════════════

def generate_easy_perfect(seed=None) -> APObservation:
    rng       = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS)
    qty        = rng.randint(20, 150)
    unit_price = round(rng.choice([60, 80, 100, 120, 150, 200]) * rng.uniform(0.95, 1.05), 2)
    freight    = round(rng.uniform(5.0, 48.0), 2)          # always under $50 cap
    line_total = round(qty * unit_price, 2)
    total      = round(line_total + freight, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    return APObservation(
        task_id="easy_perfect_match",
        task_name="Perfect Three-Way Match",
        task_description=(
            f"A vendor has submitted an invoice for {desc.lower()}s. "
            "Verify the invoice against the PO and the warehouse GRN, "
            "then make the correct payment decision."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=freight, invoice_total=total,
        ),
        purchase_orders=[PurchaseOrder(
            po_number=po_num, vendor_name=vendor,
            lines=[POLine(description=desc, ordered_quantity=qty,
                          agreed_unit_price=unit_price)],
            authorized_total=total, status="OPEN",
        )],
        goods_receipts=[GoodsReceipt(
            grn_id=grn_id, po_number=po_num,
            lines=[GRNLine(description=desc, received_quantity=qty)],
        )],
        company_policy=POLICY_STANDARD,
    )


def grade_easy_perfect(obs: APObservation, action: APAction) -> APReward:
    correct = obs.invoice.invoice_total

    if action.decision == DecisionType.APPROVE_FULL:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 0.2
    else:
        decision_score = 0.0

    amount_score = 0.0
    if action.decision != DecisionType.REJECT:
        diff = abs(action.approved_amount - correct) / correct
        amount_score = 1.0 if diff <= 0.01 else 0.7 if diff <= 0.05 else 0.4 if diff <= 0.15 else 0.1

    if action.reason_code == ReasonCode.MATCH_CONFIRMED:
        reason_score = 1.0
    elif action.reason_code in (ReasonCode.QUANTITY_MISMATCH, ReasonCode.PRICE_DISCREPANCY):
        reason_score = 0.1
    else:
        reason_score = 0.3

    final = round(0.50 * decision_score + 0.35 * amount_score + 0.15 * reason_score, 3)

    parts = []
    if decision_score < 1.0:
        parts.append("Invoice, PO and GRN match exactly — APPROVE_FULL is correct.")
    if amount_score < 1.0:
        parts.append(f"Correct amount is ${correct:,.2f}.")
    if reason_score < 1.0:
        parts.append("MATCH_CONFIRMED is the correct reason code.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "amount": amount_score,
                               "reason": reason_score},
                    feedback=" ".join(parts) or "Perfect decision!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2  —  Easy: No PO Found
# ═══════════════════════════════════════════════════════════════════════════════

def generate_easy_no_po(seed=None) -> APObservation:
    rng        = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS)
    qty        = rng.randint(10, 80)
    unit_price = round(rng.choice([50, 75, 100, 125]) * rng.uniform(0.9, 1.1), 2)
    line_total = round(qty * unit_price, 2)
    inv_id     = _inv(rng)
    fake_po    = f"PO-{rng.randint(8000, 9999)}"       # PO that doesn't exist

    return APObservation(
        task_id="easy_no_po_found",
        task_name="No Purchase Order Found",
        task_description=(
            f"A vendor has submitted an invoice referencing {fake_po}, "
            "but no matching Purchase Order exists in the system. "
            "Apply company policy and make the correct decision."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=fake_po,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=0.0, invoice_total=line_total,
        ),
        purchase_orders=[],          # deliberately empty — PO not in system
        goods_receipts=[],
        company_policy=POLICY_STANDARD,
    )


def grade_easy_no_po(obs: APObservation, action: APAction) -> APReward:
    decision_score = 1.0 if action.decision == DecisionType.REJECT else 0.0
    reason_score   = (1.0 if action.reason_code == ReasonCode.NO_PO_FOUND else
                      0.3 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.1)
    amount_score   = 1.0 if action.approved_amount == 0.0 else 0.0

    final = round(0.60 * decision_score + 0.30 * reason_score + 0.10 * amount_score, 3)

    parts = []
    if decision_score < 1.0:
        parts.append("No PO exists — invoice must be REJECTED per Policy Rule 5.")
    if reason_score < 1.0:
        parts.append("NO_PO_FOUND is the correct reason code.")
    if amount_score < 1.0:
        parts.append("Approved amount must be $0.00 on a rejection.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "amount_zero": amount_score},
                    feedback=" ".join(parts) or "Correct — no PO, no payment!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3  —  Medium: Quantity Shortfall
# ═══════════════════════════════════════════════════════════════════════════════

def generate_medium_shortfall(seed=None) -> APObservation:
    rng        = _rng(seed)
    desc, vendor = rng.choice(_TECH_ITEMS)
    ordered    = rng.randint(50, 200)
    received   = rng.randint(int(ordered * 0.4), int(ordered * 0.9))  # 40–90 % delivered
    unit_price = round(rng.choice([500, 600, 700, 800, 900, 1000]) * rng.uniform(0.95, 1.05), 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    invoice_total = round(ordered * unit_price, 2)
    po_total      = round(ordered * unit_price, 2)

    return APObservation(
        task_id="medium_quantity_shortfall",
        task_name="Quantity Shortfall Reconciliation",
        task_description=(
            f"A vendor has invoiced for {ordered} units of {desc.lower()} "
            f"but the warehouse GRN shows only {received} were delivered. "
            "Calculate the correct payable amount and decide."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=ordered,
                                 unit_price=unit_price,
                                 line_total=invoice_total)],
            freight_charge=0.0, invoice_total=invoice_total,
        ),
        purchase_orders=[PurchaseOrder(
            po_number=po_num, vendor_name=vendor,
            lines=[POLine(description=desc, ordered_quantity=ordered,
                          agreed_unit_price=unit_price)],
            authorized_total=po_total, status="OPEN",
        )],
        goods_receipts=[GoodsReceipt(
            grn_id=grn_id, po_number=po_num,
            lines=[GRNLine(description=desc, received_quantity=received)],
        )],
        company_policy=POLICY_STANDARD,
    )


def grade_medium_shortfall(obs: APObservation, action: APAction) -> APReward:
    received   = obs.goods_receipts[0].lines[0].received_quantity
    unit_price = obs.purchase_orders[0].lines[0].agreed_unit_price
    correct    = round(received * unit_price, 2)

    if action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 1.0
    elif action.decision == DecisionType.REJECT:
        decision_score = 0.35
    else:
        decision_score = 0.0

    amount_score = 0.0
    if action.decision in (DecisionType.APPROVE_PARTIAL, DecisionType.APPROVE_FULL):
        diff = abs(action.approved_amount - correct) / correct
        amount_score = (1.0 if diff <= 0.01 else 0.75 if diff <= 0.05 else
                        0.50 if diff <= 0.10 else 0.25 if diff <= 0.20 else 0.05)

    reason_score = (1.0 if action.reason_code == ReasonCode.QUANTITY_MISMATCH else
                    0.4 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.1)

    final = round(0.45 * decision_score + 0.40 * amount_score + 0.15 * reason_score, 3)

    parts = []
    if action.decision == DecisionType.APPROVE_FULL:
        parts.append("Never pay for goods not received — GRN shows a shortfall.")
    if action.decision == DecisionType.APPROVE_PARTIAL and \
            abs(action.approved_amount - correct) > correct * 0.01:
        parts.append(f"Correct amount = {int(received)} received × ${unit_price:.2f} = ${correct:,.2f}.")
    if reason_score < 1.0:
        parts.append("QUANTITY_MISMATCH is the appropriate reason code.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "amount": amount_score,
                               "reason": reason_score},
                    feedback=" ".join(parts) or "Correct partial approval!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4  —  Medium: Price Discrepancy
# ═══════════════════════════════════════════════════════════════════════════════

def generate_medium_price(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_TECH_ITEMS)
    qty          = rng.randint(20, 100)
    agreed_price = round(rng.choice([400, 500, 600, 800, 1000]) * rng.uniform(0.95, 1.05), 2)
    # invoice charges 5–20 % more than agreed
    markup       = rng.uniform(0.05, 0.20)
    invoice_price = round(agreed_price * (1 + markup), 2)
    line_total    = round(qty * invoice_price, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    return APObservation(
        task_id="medium_price_discrepancy",
        task_name="Unit Price Discrepancy",
        task_description=(
            f"A vendor invoiced {qty} units of {desc.lower()} at ${invoice_price:.2f}/unit. "
            f"However, the agreed PO price is ${agreed_price:.2f}/unit — "
            f"a {markup*100:.1f}% deviation exceeding the 1% policy threshold. "
            "Identify the discrepancy and decide."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=invoice_price, line_total=line_total)],
            freight_charge=0.0, invoice_total=line_total,
        ),
        purchase_orders=[PurchaseOrder(
            po_number=po_num, vendor_name=vendor,
            lines=[POLine(description=desc, ordered_quantity=qty,
                          agreed_unit_price=agreed_price)],
            authorized_total=round(qty * agreed_price, 2), status="OPEN",
        )],
        goods_receipts=[GoodsReceipt(
            grn_id=grn_id, po_number=po_num,
            lines=[GRNLine(description=desc, received_quantity=qty)],
        )],
        company_policy=POLICY_STANDARD,
    )


def grade_medium_price(obs: APObservation, action: APAction) -> APReward:
    inv_price = obs.invoice.line_items[0].unit_price
    po_price  = obs.purchase_orders[0].lines[0].agreed_unit_price
    deviation = (inv_price - po_price) / po_price

    decision_score = 1.0 if action.decision == DecisionType.REJECT else 0.0
    reason_score   = (1.0 if action.reason_code == ReasonCode.PRICE_DISCREPANCY else
                      0.3 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.1)

    expl = action.explanation.lower()
    price_kws  = ["price", "unit price", "agreed", "deviation", "discrepancy",
                  "higher", "markup", "overprice", "%", "po price"]
    hits       = sum(1 for kw in price_kws if kw in expl)
    expl_score = min(1.0, hits / 2)

    final = round(0.55 * decision_score + 0.30 * reason_score + 0.15 * expl_score, 3)

    parts = []
    if decision_score < 1.0:
        parts.append(
            f"Invoice unit price ${inv_price:.2f} deviates {deviation*100:.1f}% "
            f"from agreed PO price ${po_price:.2f} — must REJECT per Policy Rule 4.")
    if reason_score < 1.0:
        parts.append("PRICE_DISCREPANCY is the correct reason code.")
    if expl_score < 0.5:
        parts.append("Mention the price deviation in your explanation.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "price_detection_in_explanation": round(expl_score, 3)},
                    feedback=" ".join(parts) or "Correct — price deviation caught!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5  —  Hard: Policy Violation (Unauthorized Freight)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hard_freight(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_CABLE_ITEMS)
    qty          = rng.randint(100, 500)
    unit_price   = round(rng.choice([10, 12, 15, 18, 20]) * rng.uniform(0.9, 1.1), 2)
    freight      = round(rng.uniform(80.0, 300.0), 2)     # always > $50 cap
    line_total   = round(qty * unit_price, 2)
    total        = round(line_total + freight, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    return APObservation(
        task_id="hard_policy_violation",
        task_name="Policy Violation — Unauthorized Freight Charge",
        task_description=(
            f"A vendor invoice includes a ${freight:.2f} freight charge for {desc.lower()}s. "
            "The PO and GRN quantities match perfectly, but company policy caps "
            "unapproved freight at $50. Identify the violation and decide."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=freight, invoice_total=total,
        ),
        purchase_orders=[PurchaseOrder(
            po_number=po_num, vendor_name=vendor,
            lines=[POLine(description=desc, ordered_quantity=qty,
                          agreed_unit_price=unit_price)],
            authorized_total=line_total, status="OPEN",
        )],
        goods_receipts=[GoodsReceipt(
            grn_id=grn_id, po_number=po_num,
            lines=[GRNLine(description=desc, received_quantity=qty)],
        )],
        company_policy=POLICY_STANDARD,
    )


def grade_hard_freight(obs: APObservation, action: APAction) -> APReward:
    freight = obs.invoice.freight_charge

    if action.decision == DecisionType.REJECT:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        goods_only = obs.invoice.invoice_total - freight
        decision_score = 0.40 if abs(action.approved_amount - goods_only) <= 30 else 0.20
    else:
        decision_score = 0.0

    reason_score = (1.0 if action.reason_code == ReasonCode.POLICY_VIOLATION else
                    0.45 if action.reason_code == ReasonCode.PRICE_DISCREPANCY else 0.1)

    expl = action.explanation.lower()
    freight_kws  = ["freight", "shipping", "policy", "$50", "50",
                    "unauthorized", "unapproved", "cap"]
    hits         = sum(1 for kw in freight_kws if kw in expl)
    expl_score   = min(1.0, hits / 3)

    final = round(0.50 * decision_score + 0.30 * reason_score + 0.20 * expl_score, 3)

    parts = []
    if action.decision != DecisionType.REJECT:
        parts.append(
            f"Freight ${freight:.2f} exceeds the $50 unapproved cap — "
            "Policy Rule 2 requires REJECTION so vendor must resubmit.")
    if reason_score < 1.0:
        parts.append("POLICY_VIOLATION is the correct reason code.")
    if expl_score < 0.6:
        parts.append(f"Mention the ${freight:.2f} freight cap breach in your explanation.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "policy_detection_in_explanation": round(expl_score, 3)},
                    feedback=" ".join(parts) or "Excellent policy enforcement!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6  —  Hard: Duplicate Invoice
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hard_duplicate(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS + _TECH_ITEMS)
    qty          = rng.randint(10, 100)
    unit_price   = round(rng.choice([100, 200, 300, 500, 800]) * rng.uniform(0.95, 1.05), 2)
    line_total   = round(qty * unit_price, 2)
    po_num, grn_id = _po(rng), _grn(rng)
    # same invoice ID as one already paid
    inv_id       = _inv(rng)

    return APObservation(
        task_id="hard_duplicate_invoice",
        task_name="Duplicate Invoice Detection",
        task_description=(
            f"Invoice {inv_id} has been submitted for {qty} units of {desc.lower()}. "
            "All three documents match. However, the payment ledger shows this exact "
            "invoice ID was already paid last month. Identify the duplicate and decide."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=0.0, invoice_total=line_total,
        ),
        purchase_orders=[PurchaseOrder(
            po_number=po_num, vendor_name=vendor,
            lines=[POLine(description=desc, ordered_quantity=qty,
                          agreed_unit_price=unit_price)],
            authorized_total=line_total, status="OPEN",
        )],
        goods_receipts=[GoodsReceipt(
            grn_id=grn_id, po_number=po_num,
            lines=[GRNLine(description=desc, received_quantity=qty)],
        )],
        paid_invoice_ids=[inv_id],      # the smoking gun
        company_policy=POLICY_STANDARD,
    )


def grade_hard_duplicate(obs: APObservation, action: APAction) -> APReward:
    decision_score = 1.0 if action.decision == DecisionType.REJECT else 0.0
    reason_score   = (1.0 if action.reason_code == ReasonCode.DUPLICATE_INVOICE else
                      0.2 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05)

    expl = action.explanation.lower()
    dup_kws    = ["duplicate", "already paid", "previously", "paid", "again",
                  "twice", "ledger", "repeat", "resubmit"]
    hits       = sum(1 for kw in dup_kws if kw in expl)
    expl_score = min(1.0, hits / 2)

    final = round(0.50 * decision_score + 0.30 * reason_score + 0.20 * expl_score, 3)

    parts = []
    if decision_score < 1.0:
        parts.append(
            f"Invoice {obs.invoice.invoice_id} is in the paid ledger — "
            "Policy Rule 6 mandates REJECTION as a duplicate.")
    if reason_score < 1.0:
        parts.append("DUPLICATE_INVOICE is the correct reason code.")
    if expl_score < 0.5:
        parts.append("Mention that this invoice was already paid in your explanation.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "duplicate_detection_in_explanation": round(expl_score, 3)},
                    feedback=" ".join(parts) or "Duplicate correctly identified!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Task registry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskSpec:
    task_id:    str
    name:       str
    difficulty: str
    description: str
    generator:  Callable          # generator(seed=None) -> APObservation
    grader:     Callable          # grader(obs, action)  -> APReward


TASKS: Dict[str, TaskSpec] = {
    "easy_perfect_match": TaskSpec(
        task_id="easy_perfect_match",
        name="Perfect Three-Way Match",
        difficulty="easy",
        description="All three documents agree exactly. Confirm and approve.",
        generator=generate_easy_perfect,
        grader=grade_easy_perfect,
    ),
    "easy_no_po_found": TaskSpec(
        task_id="easy_no_po_found",
        name="No Purchase Order Found",
        difficulty="easy",
        description="Invoice references a PO that does not exist. Reject immediately.",
        generator=generate_easy_no_po,
        grader=grade_easy_no_po,
    ),
    "medium_quantity_shortfall": TaskSpec(
        task_id="medium_quantity_shortfall",
        name="Quantity Shortfall Reconciliation",
        difficulty="medium",
        description="GRN shows fewer items received than invoiced. Recalculate and partial-approve.",
        generator=generate_medium_shortfall,
        grader=grade_medium_shortfall,
    ),
    "medium_price_discrepancy": TaskSpec(
        task_id="medium_price_discrepancy",
        name="Unit Price Discrepancy",
        difficulty="medium",
        description="Invoice unit price deviates from agreed PO price by more than 1%. Reject.",
        generator=generate_medium_price,
        grader=grade_medium_price,
    ),
    "hard_policy_violation": TaskSpec(
        task_id="hard_policy_violation",
        name="Policy Violation — Unauthorized Freight",
        difficulty="hard",
        description="Freight charge exceeds policy cap. Detect violation and reject.",
        generator=generate_hard_freight,
        grader=grade_hard_freight,
    ),
    "hard_duplicate_invoice": TaskSpec(
        task_id="hard_duplicate_invoice",
        name="Duplicate Invoice Detection",
        difficulty="hard",
        description="Invoice ID already appears in the paid ledger. Block the duplicate.",
        generator=generate_hard_duplicate,
        grader=grade_hard_duplicate,
    ),
}


def grade_action(task_id: str, obs: APObservation, action: APAction) -> APReward:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    return TASKS[task_id].grader(obs, action)
