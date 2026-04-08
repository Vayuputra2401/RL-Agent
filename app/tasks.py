"""
AP Clerk Environment — Task Definitions, Generators & Graders
Ten tasks across easy / medium / hard.
All graders are pure functions: grade_xxx(obs, action) -> APReward.
All generators produce a fresh randomised observation each episode.

Improvements v2:
  1. Randomised policy — freight cap and price tolerance vary per episode.
  2. Distractor documents — CLOSED POs and wrong-vendor GRNs included in every episode.
  3. Multi-step support — hard tasks reveal context on ESCALATE / QUERY_VENDOR.
  4. Harder graders — tighter tolerances, more keywords required, stricter partial credit.
  5. Four new task types — split delivery, vendor mismatch, partial PO, tax discrepancy.
"""

from __future__ import annotations
import random as _random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .models import (
    APObservation, APAction, APReward,
    Invoice, LineItem, PurchaseOrder, POLine, GoodsReceipt, GRNLine,
    DecisionType, ReasonCode,
)

# ── Policy configuration ──────────────────────────────────────────────────────

_FREIGHT_CAPS      = [30.0, 50.0, 75.0, 100.0]
_PRICE_TOLERANCES  = [0.005, 0.010, 0.015, 0.020, 0.030]


def _make_policy(freight_cap: float, price_tolerance: float) -> str:
    pct = f"{price_tolerance * 100:.1f}%"
    return (
        f"ACME Corp Accounts Payable Policy (Effective 2024-01-01)\n"
        f"--------------------------------------------------------\n"
        f"1. Three-Way Match:  Every invoice MUST match a valid Purchase Order (PO) AND a\n"
        f"   Goods Receipt Note (GRN) before payment is authorised.\n"
        f"2. Freight Cap:      Freight / shipping charges exceeding ${freight_cap:.2f} per invoice\n"
        f"   require explicit prior approval from the Finance Manager. Do NOT pay unapproved freight.\n"
        f"3. Quantity Rule:    Payment is made only for quantities confirmed received in the GRN.\n"
        f"   If fewer items arrived than were invoiced, approve only the received amount.\n"
        f"4. Price Integrity:  Unit prices must match the agreed PO price. Any deviation > {pct}\n"
        f"   must be queried and rejected until a corrected invoice is received.\n"
        f"5. PO Mandatory:     Invoices without a valid OPEN PO reference MUST be rejected.\n"
        f"6. Duplicate Block:  Any invoice ID that has already been paid must be rejected\n"
        f"   immediately as a duplicate — regardless of supporting documents.\n"
        f"7. Vendor Verification: The vendor name on the invoice must exactly match the\n"
        f"   vendor name on the Purchase Order. Name discrepancies require rejection.\n"
        f"8. Authorised Charges: Only charges explicitly included in the PO are payable.\n"
        f"   Unauthorised additions (tax, surcharges, fees) not in the PO must be rejected."
    ).strip()


# ── Item catalogues ───────────────────────────────────────────────────────────

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

_ALL_ITEMS = _OFFICE_ITEMS + _TECH_ITEMS + _CABLE_ITEMS

# Vendor name mismatches — pairs (correct PO name, invoice typo)
_VENDOR_TYPOS: Dict[str, str] = {
    "OfficeWorld Supplies Ltd.": "OfficeWorld Supplies",
    "FurniCorp Direct":          "Furnicorp Direct",
    "TechDesk Co.":              "TechDesk Company",
    "ErgoSupply Ltd.":           "ErgoSupply Limited",
    "TechProcure Global":        "TechProcure Global Ltd.",
    "CompuSource Inc.":          "CompuSource Inc",
    "DataTech Ltd.":             "DataTech Ltd",
    "DigitalParts Corp.":        "Digital Parts Corp.",
    "CableMart Direct":          "Cable Mart Direct",
    "WireZone Ltd.":             "Wire Zone Ltd.",
    "NetCables Co.":             "NetCables Corp.",
    "ConnectPro Ltd.":           "ConnectPro Limited",
}

# ── RNG and ID helpers ────────────────────────────────────────────────────────

def _rng(seed):  return _random.Random(seed)
def _po(rng):    return f"PO-{rng.randint(1000, 4999)}"
def _grn(rng):   return f"GRN-{rng.randint(1000, 4999)}"
def _inv(rng):   return f"INV-2024-{rng.randint(1000, 9999):04d}"


def _distractor_po(rng: _random.Random, exclude_vendor: str = "") -> PurchaseOrder:
    """CLOSED PO from a different vendor — realistic noise the agent must filter out."""
    candidates = [(d, v) for d, v in _ALL_ITEMS if v != exclude_vendor]
    d_desc, d_vendor = rng.choice(candidates)
    d_qty   = rng.randint(5, 50)
    d_price = round(rng.choice([50, 100, 200, 500]) * rng.uniform(0.9, 1.1), 2)
    return PurchaseOrder(
        po_number=f"PO-{rng.randint(6000, 8999)}",
        vendor_name=d_vendor,
        lines=[POLine(description=d_desc, ordered_quantity=d_qty, agreed_unit_price=d_price)],
        authorized_total=round(d_qty * d_price, 2),
        status="CLOSED",
    )


def _distractor_grn(rng: _random.Random, exclude_po: str = "") -> GoodsReceipt:
    """GRN for a different PO — adds noise the agent must skip over."""
    d_desc, _ = rng.choice(_ALL_ITEMS)
    d_qty     = rng.randint(5, 40)
    d_po      = f"PO-{rng.randint(6000, 8999)}"
    while d_po == exclude_po:
        d_po = f"PO-{rng.randint(6000, 8999)}"
    return GoodsReceipt(
        grn_id=f"GRN-{rng.randint(6000, 8999)}",
        po_number=d_po,
        lines=[GRNLine(description=d_desc, received_quantity=d_qty)],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1  —  Easy: Perfect Three-Way Match
# ═══════════════════════════════════════════════════════════════════════════════

def generate_easy_perfect(seed=None) -> APObservation:
    rng           = _rng(seed)
    desc, vendor  = rng.choice(_OFFICE_ITEMS)
    qty           = rng.randint(20, 150)
    unit_price    = round(rng.choice([60, 80, 100, 120, 150, 200]) * rng.uniform(0.95, 1.05), 2)
    freight_cap   = rng.choice(_FREIGHT_CAPS)
    price_tol     = rng.choice(_PRICE_TOLERANCES)
    freight       = round(rng.uniform(2.0, freight_cap * 0.90), 2)   # always under cap
    line_total    = round(qty * unit_price, 2)
    total         = round(line_total + freight, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    # Distractor documents
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.6:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    return APObservation(
        task_id="easy_perfect_match",
        task_name="Perfect Three-Way Match",
        task_description=(
            f"A vendor has submitted an invoice for {desc.lower()}s. "
            "Verify the invoice against the open Purchase Order and the warehouse GRN. "
            "Closed historical POs in the system can be ignored. "
            "Make the correct payment decision."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=freight, invoice_total=total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_easy_perfect(obs: APObservation, action: APAction) -> APReward:
    correct = obs.invoice.invoice_total

    if action.decision == DecisionType.APPROVE_FULL:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 0.0   # no partial on a perfect match
    else:
        decision_score = 0.0

    amount_score = 0.0
    if action.decision != DecisionType.REJECT:
        diff = abs(action.approved_amount - correct) / correct
        amount_score = (1.0 if diff <= 0.01 else
                        0.60 if diff <= 0.03 else
                        0.30 if diff <= 0.10 else 0.05)

    if action.reason_code == ReasonCode.MATCH_CONFIRMED:
        reason_score = 1.0
    elif action.reason_code in (ReasonCode.QUANTITY_MISMATCH, ReasonCode.PRICE_DISCREPANCY):
        reason_score = 0.05
    else:
        reason_score = 0.2

    final = max(0.01, min(0.99, round(0.50 * decision_score + 0.35 * amount_score + 0.15 * reason_score, 3)))

    parts = []
    if decision_score < 1.0:
        parts.append("Invoice, PO and GRN match exactly — APPROVE_FULL is correct.")
    if amount_score < 1.0 and action.decision != DecisionType.REJECT:
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
    rng          = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS)
    qty          = rng.randint(10, 80)
    unit_price   = round(rng.choice([50, 75, 100, 125]) * rng.uniform(0.9, 1.1), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    line_total   = round(qty * unit_price, 2)
    inv_id       = _inv(rng)
    fake_po      = f"PO-{rng.randint(8000, 9999)}"   # PO that doesn't exist in system

    # CLOSED POs from prior periods — do NOT contain an OPEN PO for this invoice
    pos = []
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    grns: List[GoodsReceipt] = []

    return APObservation(
        task_id="easy_no_po_found",
        task_name="No Purchase Order Found",
        task_description=(
            f"A vendor has submitted an invoice referencing {fake_po}, "
            "but no matching OPEN Purchase Order exists in the system. "
            "Closed historical POs from other vendors are visible but do not apply. "
            "Apply company policy and make the correct decision."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=fake_po,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=0.0, invoice_total=line_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_easy_no_po(obs: APObservation, action: APAction) -> APReward:
    decision_score = 1.0 if action.decision == DecisionType.REJECT else 0.0
    reason_score   = (1.0 if action.reason_code == ReasonCode.NO_PO_FOUND else
                      0.3 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05)
    amount_score   = 1.0 if action.approved_amount == 0.0 else 0.0

    final = max(0.01, min(0.99, round(0.60 * decision_score + 0.30 * reason_score + 0.10 * amount_score, 3)))

    parts = []
    if decision_score < 1.0:
        parts.append("No valid OPEN PO exists — invoice must be REJECTED per Policy Rule 5.")
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
    rng          = _rng(seed)
    desc, vendor = rng.choice(_TECH_ITEMS)
    ordered      = rng.randint(50, 200)
    received     = rng.randint(int(ordered * 0.4), int(ordered * 0.9))
    unit_price   = round(rng.choice([500, 600, 700, 800, 900, 1000]) * rng.uniform(0.95, 1.05), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    invoice_total = round(ordered * unit_price, 2)
    po_total      = round(ordered * unit_price, 2)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=ordered, agreed_unit_price=unit_price)],
        authorized_total=po_total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=received)],
    )]
    # Distractor documents
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.5:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    return APObservation(
        task_id="medium_quantity_shortfall",
        task_name="Quantity Shortfall Reconciliation",
        task_description=(
            f"A vendor has invoiced for {ordered} units of {desc.lower()} "
            f"but the warehouse GRN shows only {received} were delivered. "
            "Calculate the correct payable amount based on received quantity only, "
            "then decide. Ignore closed POs from other vendors."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=ordered,
                                 unit_price=unit_price, line_total=invoice_total)],
            freight_charge=0.0, invoice_total=invoice_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_medium_shortfall(obs: APObservation, action: APAction) -> APReward:
    # Sum received across all GRNs that match the real PO
    real_po    = obs.purchase_orders[0].po_number
    unit_price = obs.purchase_orders[0].lines[0].agreed_unit_price
    received   = sum(
        line.received_quantity
        for grn in obs.goods_receipts
        if grn.po_number == real_po
        for line in grn.lines
    )
    correct = round(received * unit_price, 2)

    if action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 1.0
    elif action.decision == DecisionType.REJECT:
        decision_score = 0.15   # harder: REJECT on shortfall is wrong approach
    else:
        decision_score = 0.0

    amount_score = 0.0
    if action.decision in (DecisionType.APPROVE_PARTIAL, DecisionType.APPROVE_FULL):
        diff = abs(action.approved_amount - correct) / correct if correct > 0 else 1.0
        amount_score = (1.0 if diff <= 0.01 else
                        0.65 if diff <= 0.03 else
                        0.40 if diff <= 0.08 else
                        0.15 if diff <= 0.20 else 0.02)

    reason_score = (1.0 if action.reason_code == ReasonCode.QUANTITY_MISMATCH else
                    0.3 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05)

    final = max(0.01, min(0.99, round(0.45 * decision_score + 0.40 * amount_score + 0.15 * reason_score, 3)))

    parts = []
    if action.decision == DecisionType.APPROVE_FULL:
        parts.append("Never pay for goods not received — GRN shows a shortfall.")
    if action.decision == DecisionType.APPROVE_PARTIAL and \
            abs(action.approved_amount - correct) > correct * 0.01:
        parts.append(
            f"Correct amount = {int(received)} received × ${unit_price:.2f} = ${correct:,.2f}.")
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
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    # Invoice charges at least 3× the tolerance above agreed price
    markup       = rng.uniform(price_tol * 3, 0.20)
    invoice_price = round(agreed_price * (1 + markup), 2)
    line_total    = round(qty * invoice_price, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=agreed_price)],
        authorized_total=round(qty * agreed_price, 2), status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.5:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    return APObservation(
        task_id="medium_price_discrepancy",
        task_name="Unit Price Discrepancy",
        task_description=(
            f"A vendor invoiced {qty} units of {desc.lower()} at ${invoice_price:.2f}/unit. "
            f"However, the agreed PO price is ${agreed_price:.2f}/unit — "
            f"a {markup*100:.1f}% deviation exceeding the policy {price_tol*100:.1f}% threshold. "
            "Identify the discrepancy and decide."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=invoice_price, line_total=line_total)],
            freight_charge=0.0, invoice_total=line_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_medium_price(obs: APObservation, action: APAction) -> APReward:
    inv_price = obs.invoice.line_items[0].unit_price
    po_price  = obs.purchase_orders[0].lines[0].agreed_unit_price
    deviation = (inv_price - po_price) / po_price
    price_tol = obs.price_tolerance

    decision_score = 1.0 if action.decision == DecisionType.REJECT else 0.0
    reason_score   = (1.0 if action.reason_code == ReasonCode.PRICE_DISCREPANCY else
                      0.25 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05)

    expl = action.explanation.lower()
    price_kws = ["price", "unit price", "agreed", "deviation", "discrepancy",
                 "higher", "markup", "overprice", "%", "po price", "mismatch", "exceed"]
    hits       = sum(1 for kw in price_kws if kw in expl)
    expl_score = min(1.0, hits / 3)   # harder: need 3 hits

    final = max(0.01, min(0.99, round(0.55 * decision_score + 0.30 * reason_score + 0.15 * expl_score, 3)))

    parts = []
    if decision_score < 1.0:
        parts.append(
            f"Invoice unit price ${inv_price:.2f} deviates {deviation*100:.1f}% "
            f"from agreed PO price ${po_price:.2f} (threshold {price_tol*100:.1f}%) "
            "— must REJECT per Policy Rule 4.")
    if reason_score < 1.0:
        parts.append("PRICE_DISCREPANCY is the correct reason code.")
    if expl_score < 0.5:
        parts.append("Mention the price deviation with specific values in your explanation.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "price_detection_in_explanation": round(expl_score, 3)},
                    feedback=" ".join(parts) or "Correct — price deviation caught!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5  —  Hard: Policy Violation (Unauthorized Freight)  [multi-step]
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hard_freight(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_CABLE_ITEMS)
    qty          = rng.randint(100, 500)
    unit_price   = round(rng.choice([10, 12, 15, 18, 20]) * rng.uniform(0.9, 1.1), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    # Freight always > cap (at least 20% above)
    freight      = round(rng.uniform(freight_cap * 1.20, freight_cap * 4.0), 2)
    line_total   = round(qty * unit_price, 2)
    total        = round(line_total + freight, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=line_total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.5:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    # Pre-generated context revealed on ESCALATE
    escalation_note = (
        f"ESCALATE — Finance Manager (email response): "
        f"'Confirmed: freight charge of ${freight:.2f} on invoice {inv_id} was NOT "
        f"pre-approved by Finance. Our standard unapproved freight cap is ${freight_cap:.2f}. "
        f"Please reject this invoice and ask the vendor to resubmit with either a corrected "
        f"freight amount (under ${freight_cap:.2f}) or a Finance Manager approval reference.'"
    )

    return APObservation(
        task_id="hard_policy_violation",
        task_name="Policy Violation — Unauthorized Freight Charge",
        task_description=(
            f"A vendor invoice includes a ${freight:.2f} freight charge for {desc.lower()}s. "
            f"The PO and GRN quantities match perfectly, but company policy caps "
            f"unapproved freight at ${freight_cap:.2f}. "
            "For complex policy questions, you may ESCALATE to the Finance Manager "
            "before making your final decision."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=freight, invoice_total=total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
        context_notes=[escalation_note],
    )


def grade_hard_freight(obs: APObservation, action: APAction) -> APReward:
    freight     = obs.invoice.freight_charge
    freight_cap = obs.freight_cap
    cap_str     = str(int(freight_cap))

    # Check if agent escalated during multi-step
    escalated = any(
        h.get("decision") == DecisionType.ESCALATE.value
        for h in obs.action_history
    )

    if action.decision == DecisionType.REJECT:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        goods_only     = obs.invoice.invoice_total - freight
        decision_score = 0.25 if abs(action.approved_amount - goods_only) <= goods_only * 0.02 else 0.10
    else:
        decision_score = 0.0

    reason_score = (1.0 if action.reason_code == ReasonCode.POLICY_VIOLATION else
                    0.35 if action.reason_code == ReasonCode.PRICE_DISCREPANCY else 0.05)

    expl = action.explanation.lower()
    freight_kws = ["freight", "shipping", "policy", f"${cap_str}", cap_str,
                   "unauthorized", "unapproved", "cap", "exceed", "approve"]
    hits         = sum(1 for kw in freight_kws if kw in expl)
    expl_score   = min(1.0, hits / 4)   # harder: need 4 hits

    # Multi-step process bonus: escalating before deciding shows correct AP procedure
    process_score = 0.05 if (escalated and action.decision == DecisionType.REJECT) else 0.0

    final = max(0.01, min(0.99, round(
        0.48 * decision_score + 0.27 * reason_score + 0.20 * expl_score + process_score,
        3
    )))

    parts = []
    if action.decision != DecisionType.REJECT:
        parts.append(
            f"Freight ${freight:.2f} exceeds the ${freight_cap:.2f} unapproved cap — "
            "Policy Rule 2 requires REJECTION so the vendor must resubmit.")
    if reason_score < 1.0:
        parts.append("POLICY_VIOLATION is the correct reason code.")
    if expl_score < 0.5:
        parts.append(
            f"Mention the ${freight:.2f} freight charge and the ${freight_cap:.2f} cap in your explanation.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "policy_detection_in_explanation": round(expl_score, 3),
                               "process_bonus": round(process_score, 3)},
                    feedback=" ".join(parts) or "Excellent policy enforcement!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6  —  Hard: Duplicate Invoice  [multi-step]
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hard_duplicate(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS + _TECH_ITEMS)
    qty          = rng.randint(10, 100)
    unit_price   = round(rng.choice([100, 200, 300, 500, 800]) * rng.uniform(0.95, 1.05), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    line_total   = round(qty * unit_price, 2)
    po_num, grn_id = _po(rng), _grn(rng)
    inv_id       = _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=line_total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.5:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    # Pre-generated context revealed on QUERY_VENDOR
    query_note = (
        f"QUERY_VENDOR response — '{vendor}': "
        f"'We acknowledge that invoice {inv_id} was submitted and processed in a prior "
        f"billing cycle. This appears to be an accidental duplicate submission from our "
        f"accounts team. Please reject it — no further payment is due for this transaction.'"
    )

    return APObservation(
        task_id="hard_duplicate_invoice",
        task_name="Duplicate Invoice Detection",
        task_description=(
            f"Invoice {inv_id} has been submitted for {qty} units of {desc.lower()}. "
            "All three documents match. However, the payment ledger shows this exact "
            "invoice ID was already paid last month. You may QUERY_VENDOR for confirmation "
            "before making your final decision."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=0.0, invoice_total=line_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        paid_invoice_ids=[inv_id],
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
        context_notes=[query_note],
    )


def grade_hard_duplicate(obs: APObservation, action: APAction) -> APReward:
    queried = any(
        h.get("decision") == DecisionType.QUERY_VENDOR.value
        for h in obs.action_history
    )

    decision_score = 1.0 if action.decision == DecisionType.REJECT else 0.0
    reason_score   = (1.0 if action.reason_code == ReasonCode.DUPLICATE_INVOICE else
                      0.15 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.03)

    expl = action.explanation.lower()
    dup_kws    = ["duplicate", "already paid", "previously", "paid", "again",
                  "twice", "ledger", "repeat", "resubmit", "prior"]
    hits       = sum(1 for kw in dup_kws if kw in expl)
    expl_score = min(1.0, hits / 3)   # harder: need 3 hits

    process_score = 0.05 if (queried and action.decision == DecisionType.REJECT) else 0.0

    final = max(0.01, min(0.99, round(
        0.48 * decision_score + 0.27 * reason_score + 0.20 * expl_score + process_score,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append(
            f"Invoice {obs.invoice.invoice_id} is in the paid ledger — "
            "Policy Rule 6 mandates REJECTION as a duplicate.")
    if reason_score < 1.0:
        parts.append("DUPLICATE_INVOICE is the correct reason code.")
    if expl_score < 0.5:
        parts.append("Mention the prior payment in your explanation.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "duplicate_detection_in_explanation": round(expl_score, 3),
                               "process_bonus": round(process_score, 3)},
                    feedback=" ".join(parts) or "Duplicate correctly identified!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 7  —  Medium: Split Delivery
# ═══════════════════════════════════════════════════════════════════════════════

def generate_medium_split(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_TECH_ITEMS)
    total_ordered = rng.randint(50, 200)
    split1        = rng.randint(int(total_ordered * 0.30), int(total_ordered * 0.70))
    split2        = total_ordered - split1
    unit_price    = round(rng.choice([200, 300, 400, 500, 600]) * rng.uniform(0.95, 1.05), 2)
    freight_cap   = rng.choice(_FREIGHT_CAPS)
    price_tol     = rng.choice(_PRICE_TOLERANCES)
    line_total    = round(total_ordered * unit_price, 2)
    po_num        = _po(rng)
    grn_id1, grn_id2 = _grn(rng), _grn(rng)
    inv_id        = _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=total_ordered, agreed_unit_price=unit_price)],
        authorized_total=line_total, status="OPEN",
    )]
    grns = [
        GoodsReceipt(
            grn_id=grn_id1, po_number=po_num,
            lines=[GRNLine(description=desc, received_quantity=split1)],
        ),
        GoodsReceipt(
            grn_id=grn_id2, po_number=po_num,
            lines=[GRNLine(description=desc, received_quantity=split2)],
        ),
    ]
    if rng.random() < 0.65:
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.40:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    return APObservation(
        task_id="medium_split_delivery",
        task_name="Split Delivery Reconciliation",
        task_description=(
            f"A vendor has invoiced for {total_ordered} units of {desc.lower()} "
            f"at ${unit_price:.2f}/unit. The goods arrived in two separate shipments: "
            f"GRN {grn_id1} confirms {split1} units and GRN {grn_id2} confirms "
            f"{split2} units. Verify total received and decide the correct payment."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=total_ordered,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=0.0, invoice_total=line_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_medium_split(obs: APObservation, action: APAction) -> APReward:
    real_po    = obs.purchase_orders[0].po_number
    unit_price = obs.purchase_orders[0].lines[0].agreed_unit_price
    ordered    = obs.purchase_orders[0].lines[0].ordered_quantity
    # Sum all GRNs matching real PO (should equal total_ordered)
    received   = sum(
        line.received_quantity
        for grn in obs.goods_receipts
        if grn.po_number == real_po
        for line in grn.lines
    )
    correct = round(received * unit_price, 2)

    if action.decision == DecisionType.APPROVE_FULL:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 0.25   # probably missed summing both GRNs
    else:
        decision_score = 0.0

    amount_score = 0.0
    if action.decision in (DecisionType.APPROVE_FULL, DecisionType.APPROVE_PARTIAL):
        diff = abs(action.approved_amount - correct) / correct if correct > 0 else 1.0
        amount_score = (1.0 if diff <= 0.01 else
                        0.60 if diff <= 0.05 else
                        0.25 if diff <= 0.15 else 0.03)

    reason_score = (1.0 if action.reason_code == ReasonCode.MATCH_CONFIRMED else
                    0.40 if action.reason_code == ReasonCode.QUANTITY_MISMATCH else 0.05)

    final = max(0.01, min(0.99, round(0.50 * decision_score + 0.35 * amount_score + 0.15 * reason_score, 3)))

    parts = []
    if decision_score < 1.0:
        parts.append(
            f"Both GRNs together confirm all {int(ordered)} units received — APPROVE_FULL.")
    if amount_score < 1.0 and action.decision != DecisionType.REJECT:
        parts.append(
            f"Sum both GRNs: {int(received)} units × ${unit_price:.2f} = ${correct:,.2f}.")
    if reason_score < 1.0:
        parts.append("MATCH_CONFIRMED is correct — all goods received across two shipments.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "amount": amount_score,
                               "reason": reason_score},
                    feedback=" ".join(parts) or "Correct split-delivery reconciliation!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8  —  Medium: Vendor Name Mismatch
# ═══════════════════════════════════════════════════════════════════════════════

def generate_medium_vendor_mismatch(seed=None) -> APObservation:
    rng            = _rng(seed)
    desc, vendor   = rng.choice(_ALL_ITEMS)
    typo_vendor    = _VENDOR_TYPOS.get(vendor, vendor + " Ltd.")
    qty            = rng.randint(20, 100)
    unit_price     = round(rng.choice([100, 200, 300, 400, 500]) * rng.uniform(0.95, 1.05), 2)
    freight_cap    = rng.choice(_FREIGHT_CAPS)
    price_tol      = rng.choice(_PRICE_TOLERANCES)
    line_total     = round(qty * unit_price, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,   # correct PO vendor name
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=line_total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    if rng.random() < 0.60:
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.40:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    return APObservation(
        task_id="medium_vendor_mismatch",
        task_name="Vendor Name Mismatch",
        task_description=(
            f"Invoice {inv_id} for {qty} units of {desc.lower()} "
            f"names the vendor as '{typo_vendor}'. "
            f"However, Purchase Order {po_num} was issued to '{vendor}'. "
            "Policy Rule 7 requires an exact vendor name match. "
            "Identify the discrepancy and decide."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=typo_vendor,   # mismatched
            po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=0.0, invoice_total=line_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_medium_vendor_mismatch(obs: APObservation, action: APAction) -> APReward:
    po_vendor  = obs.purchase_orders[0].vendor_name
    inv_vendor = obs.invoice.vendor_name

    decision_score = 1.0 if action.decision == DecisionType.REJECT else 0.0
    reason_score   = (1.0 if action.reason_code == ReasonCode.VENDOR_MISMATCH else
                      0.4 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05)
    amount_score   = 1.0 if action.approved_amount == 0.0 else 0.0

    expl = action.explanation.lower()
    vendor_kws = ["vendor", "name", "mismatch", "different", "does not match",
                  "supplier", po_vendor.lower().split()[0],
                  inv_vendor.lower().split()[0], "policy", "verification"]
    hits       = sum(1 for kw in vendor_kws if kw in expl)
    expl_score = min(1.0, hits / 3)

    final = max(0.01, min(0.99, round(
        0.50 * decision_score + 0.25 * reason_score +
        0.15 * expl_score    + 0.10 * amount_score,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append(
            f"Invoice vendor '{inv_vendor}' does not match PO vendor '{po_vendor}' "
            "— REJECT per Policy Rule 7.")
    if reason_score < 1.0:
        parts.append("VENDOR_MISMATCH is the correct reason code.")
    if expl_score < 0.5:
        parts.append("Explain the vendor name discrepancy in your explanation.")
    if amount_score < 1.0:
        parts.append("Approved amount must be $0.00 on a rejection.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "vendor_mismatch_cited": round(expl_score, 3),
                               "amount_zero": amount_score},
                    feedback=" ".join(parts) or "Correct — vendor mismatch detected!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 9  —  Hard: Partial PO Coverage
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hard_partial_po(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc1, vendor = rng.choice(_TECH_ITEMS)
    qty1          = rng.randint(10, 50)
    price1        = round(rng.choice([200, 300, 400, 500]) * rng.uniform(0.95, 1.05), 2)
    total1        = round(qty1 * price1, 2)

    # Second line item — no PO coverage (unauthorized purchase)
    desc2, _      = rng.choice(_OFFICE_ITEMS)
    qty2          = rng.randint(5, 30)
    price2        = round(rng.choice([50, 80, 100, 150]) * rng.uniform(0.95, 1.05), 2)
    total2        = round(qty2 * price2, 2)

    freight_cap   = rng.choice(_FREIGHT_CAPS)
    price_tol     = rng.choice(_PRICE_TOLERANCES)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)
    invoice_total = round(total1 + total2, 2)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc1, ordered_quantity=qty1, agreed_unit_price=price1)],
        authorized_total=total1, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[
            GRNLine(description=desc1, received_quantity=qty1),
            GRNLine(description=desc2, received_quantity=qty2),
        ],
    )]
    if rng.random() < 0.70:
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    return APObservation(
        task_id="hard_partial_po_match",
        task_name="Partial PO Coverage",
        task_description=(
            f"Invoice {inv_id} covers two items: "
            f"{qty1} × {desc1.lower()} (${total1:,.2f}) and "
            f"{qty2} × {desc2.lower()} (${total2:,.2f}). "
            f"Purchase Order {po_num} authorises only the {desc1.lower()} line. "
            f"The {desc2.lower()} has no PO authorisation. "
            "Determine the correct payable amount for the covered items only."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[
                LineItem(description=desc1, quantity=qty1, unit_price=price1, line_total=total1),
                LineItem(description=desc2, quantity=qty2, unit_price=price2, line_total=total2),
            ],
            freight_charge=0.0, invoice_total=invoice_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_hard_partial_po(obs: APObservation, action: APAction) -> APReward:
    # Correct amount: only line items covered by an OPEN PO
    po_descs = {
        pl.description.lower()
        for po in obs.purchase_orders
        if po.status == "OPEN"
        for pl in po.lines
    }
    correct = round(sum(
        li.line_total
        for li in obs.invoice.line_items
        if li.description.lower() in po_descs
    ), 2)

    if action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 1.0
    elif action.decision == DecisionType.REJECT:
        decision_score = 0.20   # defensible but policy prefers partial approval
    else:
        decision_score = 0.0

    amount_score = 0.0
    if action.decision in (DecisionType.APPROVE_PARTIAL, DecisionType.APPROVE_FULL):
        diff = abs(action.approved_amount - correct) / correct if correct > 0 else 1.0
        amount_score = (1.0 if diff <= 0.01 else
                        0.60 if diff <= 0.05 else
                        0.30 if diff <= 0.15 else 0.03)

    reason_score = (
        1.0 if action.reason_code in (ReasonCode.NO_PO_FOUND, ReasonCode.POLICY_VIOLATION) else
        0.40 if action.reason_code == ReasonCode.QUANTITY_MISMATCH else 0.05
    )

    expl = action.explanation.lower()
    partial_kws = ["partial", "no po", "unauthorized", "not covered", "not authoris",
                   "only", "line item", "covered", "policy", "uncovered"]
    hits       = sum(1 for kw in partial_kws if kw in expl)
    expl_score = min(1.0, hits / 3)

    final = max(0.01, min(0.99, round(
        0.45 * decision_score + 0.38 * amount_score +
        0.12 * reason_score   + 0.05 * expl_score,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append(
            f"Only PO-covered items are payable — APPROVE_PARTIAL for ${correct:,.2f}.")
    if amount_score < 0.9 and action.decision != DecisionType.REJECT:
        parts.append(f"Pay only the PO-authorised items: ${correct:,.2f}.")
    if reason_score < 1.0:
        parts.append("NO_PO_FOUND or POLICY_VIOLATION are appropriate reason codes.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "amount": amount_score,
                               "reason": reason_score,
                               "explanation_quality": round(expl_score, 3)},
                    feedback=" ".join(parts) or "Correct partial PO reconciliation!", done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 10  —  Hard: Unauthorized Tax Charge
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hard_tax(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_TECH_ITEMS + _OFFICE_ITEMS)
    qty          = rng.randint(10, 80)
    unit_price   = round(rng.choice([100, 200, 300, 500, 800]) * rng.uniform(0.95, 1.05), 2)
    line_total   = round(qty * unit_price, 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)

    tax_rate     = rng.choice([0.05, 0.08, 0.10, 0.13, 0.15, 0.20])
    tax_amount   = round(line_total * tax_rate, 2)
    invoice_total = round(line_total + tax_amount, 2)

    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=line_total,   # PO has NO tax provision
        status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    if rng.random() < 0.70:
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.40:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    return APObservation(
        task_id="hard_tax_discrepancy",
        task_name="Unauthorized Tax Charge",
        task_description=(
            f"Invoice {inv_id} charges ${tax_amount:.2f} in tax ({tax_rate*100:.0f}%) "
            f"on top of the agreed line-item amount of ${line_total:,.2f} for "
            f"{qty} units of {desc.lower()}. "
            f"Purchase Order {po_num} authorises ${line_total:,.2f} with no tax provision. "
            "Determine whether the tax charge is payable per company policy and decide."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=0.0, tax_amount=tax_amount, invoice_total=invoice_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_hard_tax(obs: APObservation, action: APAction) -> APReward:
    tax       = obs.invoice.tax_amount
    line_only = obs.invoice.invoice_total - tax

    if action.decision == DecisionType.REJECT:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        # Giving credit only if agent approves the non-tax portion
        diff = abs(action.approved_amount - line_only) / line_only if line_only > 0 else 1.0
        decision_score = 0.40 if diff <= 0.02 else 0.15
    else:
        decision_score = 0.0

    reason_score = (1.0 if action.reason_code == ReasonCode.TAX_DISCREPANCY else
                    0.50 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05)

    expl = action.explanation.lower()
    tax_str   = str(int(tax))
    tax_kws   = ["tax", "vat", "gst", "levy", "unauthorized", "not authoris",
                 "not in po", "unapproved", "additional charge", "discrepancy",
                 f"${tax_str}", tax_str]
    hits       = sum(1 for kw in tax_kws if kw in expl)
    expl_score = min(1.0, hits / 3)

    final = max(0.01, min(0.99, round(0.50 * decision_score + 0.30 * reason_score + 0.20 * expl_score, 3)))

    parts = []
    if decision_score == 0.0:
        parts.append(
            f"Tax of ${tax:.2f} has no PO authorisation (Policy Rule 8) — "
            "REJECT and request a tax-free corrected invoice.")
    if reason_score < 1.0:
        parts.append("TAX_DISCREPANCY is the correct reason code.")
    if expl_score < 0.5:
        parts.append(f"Mention the unauthorized ${tax:.2f} tax charge in your explanation.")
    return APReward(score=final,
                    breakdown={"decision": decision_score, "reason": reason_score,
                               "tax_detection_in_explanation": round(expl_score, 3)},
                    feedback=" ".join(parts) or "Unauthorized tax charge correctly identified!",
                    done=True)


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
    max_steps:  int = 1           # >1 enables multi-step episodes


TASKS: Dict[str, TaskSpec] = {
    "easy_perfect_match": TaskSpec(
        task_id="easy_perfect_match",
        name="Perfect Three-Way Match",
        difficulty="easy",
        description="All three documents agree exactly. Confirm and approve.",
        generator=generate_easy_perfect,
        grader=grade_easy_perfect,
        max_steps=1,
    ),
    "easy_no_po_found": TaskSpec(
        task_id="easy_no_po_found",
        name="No Purchase Order Found",
        difficulty="easy",
        description="Invoice references a PO that does not exist. Reject immediately.",
        generator=generate_easy_no_po,
        grader=grade_easy_no_po,
        max_steps=1,
    ),
    "medium_quantity_shortfall": TaskSpec(
        task_id="medium_quantity_shortfall",
        name="Quantity Shortfall Reconciliation",
        difficulty="medium",
        description="GRN shows fewer items received than invoiced. Recalculate and partial-approve.",
        generator=generate_medium_shortfall,
        grader=grade_medium_shortfall,
        max_steps=1,
    ),
    "medium_price_discrepancy": TaskSpec(
        task_id="medium_price_discrepancy",
        name="Unit Price Discrepancy",
        difficulty="medium",
        description="Invoice unit price deviates from agreed PO price beyond the policy threshold. Reject.",
        generator=generate_medium_price,
        grader=grade_medium_price,
        max_steps=1,
    ),
    "hard_policy_violation": TaskSpec(
        task_id="hard_policy_violation",
        name="Policy Violation — Unauthorized Freight",
        difficulty="hard",
        description="Freight charge exceeds the episode-specific policy cap. Detect violation and reject.",
        generator=generate_hard_freight,
        grader=grade_hard_freight,
        max_steps=3,
    ),
    "hard_duplicate_invoice": TaskSpec(
        task_id="hard_duplicate_invoice",
        name="Duplicate Invoice Detection",
        difficulty="hard",
        description="Invoice ID already appears in the paid ledger. Block the duplicate.",
        generator=generate_hard_duplicate,
        grader=grade_hard_duplicate,
        max_steps=3,
    ),
    "medium_split_delivery": TaskSpec(
        task_id="medium_split_delivery",
        name="Split Delivery Reconciliation",
        difficulty="medium",
        description="Goods arrived in two shipments across two GRNs. Sum quantities and approve full.",
        generator=generate_medium_split,
        grader=grade_medium_split,
        max_steps=1,
    ),
    "medium_vendor_mismatch": TaskSpec(
        task_id="medium_vendor_mismatch",
        name="Vendor Name Mismatch",
        difficulty="medium",
        description="Invoice vendor name does not exactly match the PO vendor. Reject per policy.",
        generator=generate_medium_vendor_mismatch,
        grader=grade_medium_vendor_mismatch,
        max_steps=1,
    ),
    "hard_partial_po_match": TaskSpec(
        task_id="hard_partial_po_match",
        name="Partial PO Coverage",
        difficulty="hard",
        description="Invoice includes line items not covered by the PO. Partial-approve only authorised items.",
        generator=generate_hard_partial_po,
        grader=grade_hard_partial_po,
        max_steps=1,
    ),
    "hard_tax_discrepancy": TaskSpec(
        task_id="hard_tax_discrepancy",
        name="Unauthorized Tax Charge",
        difficulty="hard",
        description="Vendor adds a tax charge with no PO authorisation. Detect and reject.",
        generator=generate_hard_tax,
        grader=grade_hard_tax,
        max_steps=1,
    ),
}


def grade_action(task_id: str, obs: APObservation, action: APAction) -> APReward:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    return TASKS[task_id].grader(obs, action)
