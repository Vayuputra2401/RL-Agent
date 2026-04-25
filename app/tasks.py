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
import re as _re
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


# ── Explanation quality helpers ───────────────────────────────────────────────

def _has_numeric_citation(expl: str) -> bool:
    """True if explanation contains a specific monetary value ($NNN) or percentage (N.N%)."""
    return bool(_re.search(r'\$\d[\d,.]*|\d+\.?\d*\s*%', expl))


def _explanation_coherence(expl: str, hits: int) -> float:
    """
    Anti-keyword-salad multiplier in [0.5, 1.0].
    Penalises short dumps where >40% of words are exact keyword matches.
    Requires at least 8 words for full credit.
    """
    words = expl.split()
    word_count = max(1, len(words))
    if word_count < 8:
        return 0.5
    density = hits / word_count
    if density > 0.40:
        return 0.6  # keyword salad penalty
    return 1.0


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
                 "higher", "markup", "overprice", "po price", "mismatch", "exceed"]
    hits       = sum(1 for kw in price_kws if kw in expl)
    expl_score = min(1.0, hits / 3) * _explanation_coherence(expl, hits)
    if not _has_numeric_citation(expl):
        expl_score *= 0.5

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
    expl_score   = min(1.0, hits / 4) * _explanation_coherence(expl, hits)

    # Multi-step process bonus: escalating before deciding shows correct AP procedure
    process_score = 0.10 if (escalated and action.decision == DecisionType.REJECT) else 0.0

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
    expl_score = min(1.0, hits / 3) * _explanation_coherence(expl, hits)
    # Bonus for citing the specific invoice ID — shows the agent actually checked the ledger
    if obs.invoice.invoice_id.lower() in expl:
        expl_score = min(1.0, expl_score + 0.30)

    process_score = 0.10 if (queried and action.decision == DecisionType.REJECT) else 0.0

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
    # Best score: both full vendor names cited explicitly
    po_cited  = po_vendor.lower() in expl
    inv_cited = inv_vendor.lower() in expl
    if po_cited and inv_cited:
        expl_score = 1.0
    elif po_cited or inv_cited:
        expl_score = 0.6
    else:
        # Fallback keyword score (max 0.4 without explicit vendor names)
        vendor_kws = ["vendor", "name", "mismatch", "different", "does not match",
                      "supplier", "policy", "verification"]
        hits = sum(1 for kw in vendor_kws if kw in expl)
        expl_score = min(0.4, hits / 4) * _explanation_coherence(expl, hits)

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
    expl_score = min(1.0, hits / 3) * _explanation_coherence(expl, hits)

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
                    0.50 if action.reason_code == ReasonCode.POLICY_VIOLATION else
                    0.30 if action.reason_code == ReasonCode.PRICE_DISCREPANCY else 0.05)

    expl = action.explanation.lower()
    tax_str   = str(int(tax))
    tax_kws   = ["tax", "vat", "gst", "levy", "unauthorized", "not authoris",
                 "not in po", "unapproved", "additional charge", "discrepancy",
                 f"${tax_str}", tax_str]
    hits       = sum(1 for kw in tax_kws if kw in expl)
    expl_score = min(1.0, hits / 3) * _explanation_coherence(expl, hits)

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
# TASK 11  —  Hard: Currency Conversion (EUR invoice vs USD PO)
# ═══════════════════════════════════════════════════════════════════════════════

_EUR_VENDORS = [
    ("Industrial Valve Set (ISO-EN9001)",   "EuroTech Supplies GmbH"),
    ("CNC Precision Bearing Kit",           "Rhein Industrial AG"),
    ("High-Pressure Hydraulic Hose 10m",    "Benelux FluidTech B.V."),
    ("Stainless Steel Fastener Assortment", "Nordics Hardware OY"),
]


def generate_hard_currency(seed=None) -> APObservation:
    rng           = _rng(seed)
    desc, vendor  = rng.choice(_EUR_VENDORS)
    qty           = rng.randint(10, 80)
    eur_price     = round(rng.choice([50, 80, 100, 150, 200]) * rng.uniform(0.95, 1.05), 2)
    freight_cap   = rng.choice(_FREIGHT_CAPS)
    price_tol     = rng.choice(_PRICE_TOLERANCES)
    # Exchange rate EUR → USD (e.g. 1.07)
    fx_rate       = round(rng.choice([1.05, 1.07, 1.09, 1.11, 1.13, 1.15]), 2)
    usd_price     = round(eur_price * fx_rate, 2)
    eur_total     = round(qty * eur_price, 2)
    usd_total     = round(qty * usd_price, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    # Randomise whether the converted total is within or over PO (drives correct decision)
    within_po = rng.random() < 0.55   # majority of cases: approve if conversion is correct
    # PO authorized in USD
    if within_po:
        po_usd_total = round(usd_total * rng.uniform(1.00, 1.03), 2)   # PO slightly above
        correct_decision = "APPROVE_FULL"
    else:
        # Invoice slightly over PO after conversion
        po_usd_total = round(usd_total * rng.uniform(0.94, 0.99), 2)
        correct_decision = "REJECT"

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=usd_price)],
        authorized_total=po_usd_total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.5:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    # Store correct_decision in description for grader use (via task_description field hack-free approach)
    # We'll encode it in context_notes (not shown initially) — grader reads obs directly
    policy = (
        _make_policy(freight_cap, price_tol) +
        f"\n9. Currency Conversion: Invoices in foreign currencies must be converted to USD "
        f"using the rate specified. Current rate: EUR 1.00 = USD {fx_rate:.2f}. "
        f"All approval decisions must be based on the USD equivalent amount."
    )

    return APObservation(
        task_id="hard_currency_conversion",
        task_name="Foreign Currency Invoice Reconciliation",
        task_description=(
            f"A European vendor has submitted invoice {inv_id} for {qty} units of "
            f"{desc.lower()} at EUR {eur_price:.2f}/unit (EUR {eur_total:,.2f} total). "
            f"Company policy requires conversion to USD at the current rate of "
            f"EUR 1.00 = USD {fx_rate:.2f} before comparison against the PO. "
            f"The PO is denominated in USD. Convert the invoice total and decide."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=eur_price, line_total=eur_total)],
            freight_charge=0.0, invoice_total=eur_total,
            currency="EUR",
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=policy,
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_hard_currency(obs: APObservation, action: APAction) -> APReward:
    # Recover fx_rate from policy text
    fx_match = _re.search(r'EUR 1\.00 = USD (\d+\.\d+)', obs.company_policy)
    fx_rate   = float(fx_match.group(1)) if fx_match else 1.0

    inv_eur   = obs.invoice.invoice_total
    inv_usd   = round(inv_eur * fx_rate, 2)
    po_usd    = obs.purchase_orders[0].authorized_total
    deviation = (inv_usd - po_usd) / po_usd if po_usd > 0 else 1.0
    tol       = obs.price_tolerance

    # Correct decision: APPROVE_FULL if converted total ≤ PO (within tolerance), else REJECT
    should_approve = deviation <= tol

    if should_approve:
        if action.decision == DecisionType.APPROVE_FULL:
            decision_score = 1.0
        elif action.decision == DecisionType.APPROVE_PARTIAL:
            decision_score = 0.30
        else:
            decision_score = 0.0
    else:
        if action.decision == DecisionType.REJECT:
            decision_score = 1.0
        elif action.decision == DecisionType.APPROVE_PARTIAL:
            decision_score = 0.15
        else:
            decision_score = 0.0

    # Amount score: expects USD-converted amount if approving
    amount_score = 0.0
    if action.decision in (DecisionType.APPROVE_FULL, DecisionType.APPROVE_PARTIAL):
        expected_amount = inv_usd if should_approve else 0.0
        if expected_amount > 0:
            diff = abs(action.approved_amount - expected_amount) / expected_amount
            amount_score = (1.0 if diff <= 0.01 else
                            0.65 if diff <= 0.03 else
                            0.35 if diff <= 0.10 else 0.05)

    # Explanation: must mention conversion, exchange rate, and USD value
    expl = action.explanation.lower()
    fx_str  = str(fx_rate)
    conv_kws = ["eur", "usd", "convert", "exchange", "rate", fx_str, "foreign", "currency"]
    hits = sum(1 for kw in conv_kws if kw in expl)
    expl_score = min(1.0, hits / 3) * _explanation_coherence(expl, hits)
    # Require at least one numeric citation (the converted value)
    if not _has_numeric_citation(expl):
        expl_score *= 0.5

    reason_score = (
        1.0 if (should_approve and action.reason_code == ReasonCode.MATCH_CONFIRMED) else
        1.0 if (not should_approve and action.reason_code == ReasonCode.PRICE_DISCREPANCY) else
        0.40 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05
    )

    final = max(0.01, min(0.99, round(
        0.45 * decision_score + 0.25 * amount_score +
        0.20 * expl_score + 0.10 * reason_score,
        3
    )))

    parts = []
    if decision_score < 1.0:
        adj = "within" if should_approve else "exceeds"
        parts.append(
            f"EUR {inv_eur:,.2f} × {fx_rate:.2f} = USD {inv_usd:,.2f}. "
            f"This {adj} the PO authorized amount of USD {po_usd:,.2f}.")
    if amount_score < 0.9 and action.decision != DecisionType.REJECT:
        parts.append(f"Approved amount should be the USD equivalent: ${inv_usd:,.2f}.")
    if expl_score < 0.5:
        parts.append("Explain the EUR→USD conversion with specific values in your reasoning.")
    if reason_score < 1.0:
        code = "MATCH_CONFIRMED" if should_approve else "PRICE_DISCREPANCY"
        parts.append(f"{code} is the correct reason code.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "conversion_explained": round(expl_score, 3),
                   "reason": reason_score},
        feedback=" ".join(parts) or "Correct foreign currency reconciliation!",
        done=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 12  —  Hard: Manager Pre-Approved Freight Override  [multi-step]
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hard_manager_preapproval(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_CABLE_ITEMS + _TECH_ITEMS)
    qty          = rng.randint(50, 300)
    unit_price   = round(rng.choice([15, 20, 25, 30, 40]) * rng.uniform(0.9, 1.1), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    # Freight exceeds cap by $20–$50
    freight      = round(freight_cap + rng.uniform(20.0, 50.0), 2)
    line_total   = round(qty * unit_price, 2)
    total        = round(line_total + freight, 2)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    # Generate a realistic approval code
    approval_code = f"MAP-2024-{rng.randint(1000, 9999)}"

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

    # Context revealed ONLY on ESCALATE — contains manager pre-approval
    manager_note = (
        f"[MANAGER] Finance Manager approval on file: "
        f"Freight charge of ${freight:.2f} on invoice {inv_id} for {vendor} "
        f"has been reviewed and pre-authorised by Finance Manager J. Henderson. "
        f"Approval reference: {approval_code}. "
        f"This invoice may be processed in full including the freight charge."
    )

    policy = (
        _make_policy(freight_cap, price_tol) +
        f"\n9. Freight Override: If freight exceeds the ${freight_cap:.2f} cap, "
        f"ESCALATE to Finance for review before making a payment decision. "
        f"If the Finance Manager has pre-approved the freight, full payment is authorised."
    )

    return APObservation(
        task_id="hard_manager_preapproval",
        task_name="Manager Pre-Approved Freight Override",
        task_description=(
            f"Invoice {inv_id} for {qty} units of {desc.lower()} includes a "
            f"${freight:.2f} freight charge, which exceeds the policy cap of "
            f"${freight_cap:.2f}. However, company policy states that freight "
            f"overrides may be authorised by the Finance Manager. "
            f"Use ESCALATE to check for pre-approval before making your final decision."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=freight, invoice_total=total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=policy,
        freight_cap=freight_cap,
        price_tolerance=price_tol,
        context_notes=[manager_note],
    )


def grade_hard_manager_preapproval(obs: APObservation, action: APAction) -> APReward:
    escalated = any(
        h.get("decision") == DecisionType.ESCALATE.value
        for h in obs.action_history
    )
    # Check if the approval code was revealed (context_notes present)
    approval_revealed = any("[MANAGER]" in note for note in obs.context_notes)

    if action.decision == DecisionType.APPROVE_FULL:
        if escalated and approval_revealed:
            decision_score = 1.0   # perfect: escalated, saw approval, approved
        elif escalated:
            decision_score = 0.60  # escalated but context not fully processed
        else:
            decision_score = 0.10  # approved without checking — too permissive
    elif action.decision == DecisionType.REJECT:
        if escalated and approval_revealed:
            decision_score = 0.10  # saw manager approval and still rejected — wrong
        elif escalated:
            decision_score = 0.30  # escalated but misread the context
        else:
            decision_score = 0.50  # reasonable default (freight over cap) but missed process step
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 0.20      # partial is not the right process
    else:
        decision_score = 0.0

    # Amount: full invoice (including freight) is correct once approved
    amount_score = 0.0
    correct_amount = obs.invoice.invoice_total
    if action.decision == DecisionType.APPROVE_FULL:
        diff = abs(action.approved_amount - correct_amount) / correct_amount
        amount_score = (1.0 if diff <= 0.01 else
                        0.60 if diff <= 0.03 else 0.20)

    # Explanation: mention freight, approval code or manager, and decision rationale
    expl = action.explanation.lower()
    ap_kws = ["freight", "manager", "approve", "pre-approv", "authoris", "escalat",
              "override", "cap", "policy", "finance"]
    hits = sum(1 for kw in ap_kws if kw in expl)
    expl_score = min(1.0, hits / 3)
    if not _has_numeric_citation(expl):
        expl_score *= 0.6

    reason_score = (
        1.0 if (action.decision == DecisionType.APPROVE_FULL and
                action.reason_code == ReasonCode.MATCH_CONFIRMED) else
        1.0 if (action.decision == DecisionType.REJECT and
                action.reason_code in (ReasonCode.POLICY_VIOLATION, ReasonCode.MANAGER_REVIEW)) else
        0.40 if action.reason_code == ReasonCode.MANAGER_REVIEW else 0.10
    )

    # Process bonus: escalated → revealed context → approved = perfect procedure
    process_score = 0.15 if (escalated and approval_revealed and
                              action.decision == DecisionType.APPROVE_FULL) else 0.0

    final = max(0.01, min(0.99, round(
        0.40 * decision_score + 0.20 * amount_score +
        0.20 * expl_score + 0.05 * reason_score + process_score,
        3
    )))

    parts = []
    if decision_score < 1.0:
        if not escalated:
            parts.append(
                "Policy Rule 9 requires ESCALATE to check for manager pre-approval "
                "before deciding on freight overages.")
        elif not approval_revealed:
            parts.append("Escalation context not yet revealed — check context_notes after ESCALATE.")
        else:
            parts.append(
                "The Finance Manager pre-approved this freight — APPROVE_FULL is correct.")
    if amount_score < 0.9 and action.decision == DecisionType.APPROVE_FULL:
        parts.append(f"Full invoice amount ${correct_amount:,.2f} is payable with manager approval.")
    if expl_score < 0.5:
        parts.append("Mention the freight amount, cap, and manager pre-approval in your explanation.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "explanation_quality": round(expl_score, 3),
                   "reason": reason_score,
                   "process_bonus": round(process_score, 3)},
        feedback=" ".join(parts) or "Correct — manager approval verified and invoice approved!",
        done=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 13  —  Hard: Credit Memo Processing
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hard_credit_memo(seed=None) -> APObservation:
    rng           = _rng(seed)
    desc, vendor  = rng.choice(_TECH_ITEMS + _OFFICE_ITEMS)
    qty           = rng.randint(5, 40)
    unit_price    = round(rng.choice([100, 200, 300, 400, 500]) * rng.uniform(0.95, 1.05), 2)
    line_total    = round(qty * unit_price, 2)
    freight_cap   = rng.choice(_FREIGHT_CAPS)
    price_tol     = rng.choice(_PRICE_TOLERANCES)
    po_num, grn_id, inv_id = _po(rng), _grn(rng), _inv(rng)

    # ~60% chance the PO exists (agent should APPROVE_PARTIAL with negative amount)
    # ~40% chance no PO (agent should REJECT)
    po_exists = rng.random() < 0.60

    if po_exists:
        pos  = [PurchaseOrder(
            po_number=po_num, vendor_name=vendor,
            lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
            authorized_total=line_total, status="OPEN",
        )]
        grns = [GoodsReceipt(
            grn_id=grn_id, po_number=po_num,
            lines=[GRNLine(description=desc, received_quantity=qty)],
        )]
        correct_decision = "APPROVE_PARTIAL"   # credit memo with valid PO → process as credit
    else:
        pos  = []
        fake_po = f"PO-{rng.randint(8000, 9999)}"
        po_num  = fake_po   # invoice references non-existent PO
        grns    = []
        correct_decision = "REJECT"            # no PO → reject credit memo

    # Distractors
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))
    if rng.random() < 0.40:
        grns.append(_distractor_grn(rng, exclude_po=po_num))

    # Credit memo: negative line total, negative invoice_total
    credit_total = round(-line_total, 2)

    policy = (
        _make_policy(freight_cap, price_tol) +
        "\n9. Credit Memos: When a vendor issues a credit memo (negative invoice), "
        "verify the original PO reference. If a valid OPEN PO exists, apply the credit "
        "(APPROVE_PARTIAL with the negative credit amount). "
        "If no valid PO is found, REJECT the credit memo and request clarification."
    )

    return APObservation(
        task_id="hard_credit_memo",
        task_name="Credit Memo Processing",
        task_description=(
            f"Vendor {vendor} has issued a credit memo {inv_id} for a return of "
            f"{qty} units of {desc.lower()}, crediting ${line_total:,.2f} back to ACME Corp. "
            f"Credit memos must be matched against an OPEN PO before processing. "
            f"{'The PO reference is valid and open in the system.' if po_exists else 'No matching OPEN PO could be located for the credit.'} "
            "Apply company policy and make the correct decision."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=credit_total)],
            freight_charge=0.0, invoice_total=credit_total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=policy,
        freight_cap=freight_cap,
        price_tolerance=price_tol,
    )


def grade_hard_credit_memo(obs: APObservation, action: APAction) -> APReward:
    # Determine if a valid OPEN PO exists for this vendor+PO reference
    po_ref    = obs.invoice.po_reference
    po_exists = any(
        po.po_number == po_ref and po.status == "OPEN"
        for po in obs.purchase_orders
    )
    credit_amount = obs.invoice.invoice_total   # negative value

    if po_exists:
        # Credit memo with valid PO: APPROVE_PARTIAL with negative (credit) amount
        if action.decision == DecisionType.APPROVE_PARTIAL:
            decision_score = 1.0
        elif action.decision == DecisionType.APPROVE_FULL:
            decision_score = 0.10   # full approval on a credit memo makes no sense
        elif action.decision == DecisionType.REJECT:
            decision_score = 0.20   # defensible but incorrect — policy says process the credit
        else:
            decision_score = 0.0
    else:
        # No PO: must REJECT
        if action.decision == DecisionType.REJECT:
            decision_score = 1.0
        elif action.decision == DecisionType.QUERY_VENDOR:
            decision_score = 0.35   # reasonable step but not correct terminal decision
        else:
            decision_score = 0.0

    # Amount score: credit amount (negative) if approving
    amount_score = 0.0
    if action.decision == DecisionType.APPROVE_PARTIAL and po_exists:
        diff = abs(action.approved_amount - credit_amount) / abs(credit_amount) if credit_amount != 0 else 1.0
        amount_score = (1.0 if diff <= 0.01 else
                        0.60 if diff <= 0.05 else
                        0.25 if diff <= 0.15 else 0.05)
    elif action.decision == DecisionType.REJECT and not po_exists:
        amount_score = 1.0 if action.approved_amount == 0.0 else 0.0

    # Reason code
    if po_exists:
        reason_score = (
            1.0 if action.reason_code == ReasonCode.MATCH_CONFIRMED else
            0.40 if action.reason_code == ReasonCode.QUANTITY_MISMATCH else 0.05
        )
    else:
        reason_score = (
            1.0 if action.reason_code == ReasonCode.NO_PO_FOUND else
            0.40 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05
        )

    # Explanation: must mention "credit", "credit memo" or "return", and the amount
    expl = action.explanation.lower()
    credit_kws = ["credit", "credit memo", "return", "refund", "negative", "debit",
                  "offset", "adjustment", "reversal"]
    hits = sum(1 for kw in credit_kws if kw in expl)
    expl_score = min(1.0, hits / 2) * _explanation_coherence(expl, hits)
    if not _has_numeric_citation(expl):
        expl_score *= 0.5

    final = max(0.01, min(0.99, round(
        0.45 * decision_score + 0.35 * amount_score +
        0.10 * reason_score + 0.10 * expl_score,
        3
    )))

    parts = []
    if decision_score < 1.0:
        if po_exists:
            parts.append(
                f"Credit memo with valid PO: APPROVE_PARTIAL with credit amount "
                f"${credit_amount:,.2f} per Policy Rule 9.")
        else:
            parts.append("No valid OPEN PO found for credit memo — REJECT per Policy Rule 5.")
    if amount_score < 0.9 and action.decision == DecisionType.APPROVE_PARTIAL:
        parts.append(
            f"Approved credit amount should equal the credit memo total: ${credit_amount:,.2f}.")
    if expl_score < 0.5:
        parts.append("Mention 'credit memo' or 'return' with the credit value in your explanation.")
    if reason_score < 1.0:
        code = "MATCH_CONFIRMED" if po_exists else "NO_PO_FOUND"
        parts.append(f"{code} is the correct reason code.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "reason": reason_score,
                   "credit_memo_cited": round(expl_score, 3)},
        feedback=" ".join(parts) or "Credit memo correctly processed!",
        done=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LONG-HORIZON TASKS  (Theme #2 — Long-Horizon Planning, Scale AI bonus)
# All have max_steps 10-16, multi-step actor interactions, per-step micro-rewards
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Actor-aware context note generation ─────────────────────────────────────

def _vendor_context_honest(inv_id: str, total: float) -> str:
    return (
        f"[VENDOR] Thank you for your query regarding invoice {inv_id}. "
        f"We confirm this is a valid invoice for ${total:,.2f}. "
        "All line items match our delivery records. Please proceed with payment."
    )

def _vendor_context_dispute(inv_id: str, orig: float, corrected: float) -> str:
    return (
        f"[VENDOR] Regarding invoice {inv_id}: we acknowledge the discrepancy. "
        f"The correct amount is ${corrected:,.2f} (not ${orig:,.2f}). "
        "A corrected invoice will be issued within 2 business days. Please hold payment."
    )

def _vendor_context_duplicate_deny(inv_id: str) -> str:
    return (
        f"[VENDOR] Invoice {inv_id} is NOT a duplicate. "
        "Our records show the previous submission was rejected at your end. "
        "This is a re-submission — please process immediately."
    )

def _manager_context_approve(freight: float, cap: float) -> str:
    return (
        f"[MANAGER] Pre-approval confirmed for freight ${freight:.2f} "
        f"(policy cap: ${cap:.2f}). Approved as one-time exception. "
        "Proceed with full invoice payment."
    )

def _manager_context_deny(freight: float, cap: float) -> str:
    overage = freight - cap
    return (
        f"[MANAGER] No pre-approval on file. Freight of ${freight:.2f} "
        f"exceeds the ${cap:.2f} policy cap by ${overage:.2f}. "
        "Reject and request corrected invoice per Policy Rule 2."
    )

def _manager_unavailable_note() -> str:
    return (
        "[MANAGER] Out of office until Monday. For urgent approvals above $5,000, "
        "please escalate to VP Finance (vp.finance@acme.com). "
        "Standard policy applies in my absence."
    )

def _vp_context_approve(total: float) -> str:
    return (
        f"[VP_FINANCE] Reviewed and approved. Invoice total ${total:,.2f} "
        "has been pre-cleared at the VP level due to strategic vendor relationship. "
        "Proceed with APPROVE_FULL."
    )

def _compliance_cleared(vendor: str) -> str:
    return (
        f"[COMPLIANCE] SOX review complete for vendor {vendor}. "
        "No material weaknesses. Three-way documentation adequate. "
        "Invoice may proceed."
    )


# ── Task: long_invoice_dispute ─────────────────────────────────────────────────

def generate_long_dispute(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS + _TECH_ITEMS)
    qty          = rng.randint(20, 100)
    agreed_price = round(rng.choice([80, 120, 200, 350]) * rng.uniform(0.97, 1.03), 2)
    invoice_price = round(agreed_price * rng.uniform(1.10, 1.30), 2)  # 10-30% over
    freight_cap   = rng.choice(_FREIGHT_CAPS)
    price_tol     = rng.choice(_PRICE_TOLERANCES)
    freight       = round(rng.uniform(5, freight_cap * 0.85), 2)

    line_total_agreed  = round(qty * agreed_price, 2)
    line_total_invoice = round(qty * invoice_price, 2)
    total_invoice      = round(line_total_invoice + freight, 2)
    total_correct      = round(line_total_agreed  + freight, 2)

    po_num = _po(rng)
    grn_id = _grn(rng)
    inv_id = _inv(rng)
    dev    = (invoice_price - agreed_price) / agreed_price

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=agreed_price)],
        authorized_total=total_correct, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    context_notes = [
        f"[VENDOR] Initial response: Invoice {inv_id} is correct per our updated pricing.",
        _vendor_context_dispute(inv_id, total_invoice, total_correct),
        f"[MANAGER] Price deviation of {dev*100:.1f}% exceeds policy tolerance {price_tol*100:.1f}%. Reject original; accept corrected invoice.",
        "[COMPLIANCE] Policy Rule 4 violation confirmed. Corrected invoice required.",
    ]

    return APObservation(
        task_id="long_invoice_dispute",
        task_name="Invoice Dispute Resolution (Long-Horizon)",
        task_description=(
            f"Vendor submitted invoice {inv_id} for {desc.lower()} at ${invoice_price:.2f}/unit "
            f"(agreed PO rate: ${agreed_price:.2f}/unit — deviation {dev*100:.1f}%). "
            f"Engage in a multi-step dispute process: query vendor, escalate to manager, "
            f"review compliance note, then make a final decision. "
            f"Correctly approved amount (if any) uses the PO agreed rate: ${total_correct:,.2f}."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=invoice_price, line_total=line_total_invoice)],
            freight_charge=freight, invoice_total=total_invoice,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
        context_notes=context_notes,
        max_steps=12,
    )


def grade_long_dispute(obs: APObservation, action: APAction) -> APReward:
    history    = obs.action_history
    notes      = obs.context_notes
    # Parse correct total from task description (PO agreed rate * qty + freight)
    # Find correct amount from POs
    po = next((p for p in obs.purchase_orders if p.status == "OPEN"), None)
    if po and po.lines:
        line = po.lines[0]
        qty  = obs.invoice.line_items[0].quantity
        correct_total = round(line.agreed_unit_price * qty + obs.invoice.freight_charge, 2)
    else:
        correct_total = obs.invoice.invoice_total

    used_query    = any(h["decision"] == "QUERY_VENDOR" for h in history)
    used_escalate = any(h["decision"] == "ESCALATE"     for h in history)
    steps_used    = obs.step_count

    # Decision scoring: should REJECT (vendor was wrong, corrected invoice required)
    if action.decision == DecisionType.REJECT:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 0.4
    else:
        decision_score = 0.0

    # Amount
    amount_score = 0.0
    if action.decision == DecisionType.REJECT:
        amount_score = 1.0 if action.approved_amount == 0.0 else 0.3
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        diff = abs(action.approved_amount - correct_total) / correct_total
        amount_score = 1.0 if diff <= 0.01 else 0.50 if diff <= 0.05 else 0.15

    # Reason code
    reason_score = 1.0 if action.reason_code == ReasonCode.PRICE_DISCREPANCY else 0.2

    # Explanation quality
    expl = action.explanation.lower()
    expl_score = 0.0
    if _has_numeric_citation(action.explanation):
        expl_score += 0.5
    if "price" in expl or "deviation" in expl or "discrepan" in expl:
        expl_score += 0.3
    if "corrected" in expl or "reissue" in expl or "reject" in expl:
        expl_score += 0.2
    expl_score = min(1.0, expl_score)

    # Process bonus: reward for using dispute workflow
    process_bonus = 0.0
    if used_query:    process_bonus += 0.08
    if used_escalate: process_bonus += 0.07

    final = max(0.01, min(0.99, round(
        0.38 * decision_score + 0.25 * amount_score +
        0.15 * reason_score   + 0.12 * expl_score  + process_bonus,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append("Vendor admitted error — REJECT and request corrected invoice.")
    if not used_query:
        parts.append("Process tip: QUERY_VENDOR first to document the vendor's admission.")
    if expl_score < 0.5:
        parts.append("Cite the price deviation % and both PO and invoice amounts.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "reason": reason_score, "explanation": round(expl_score,3),
                   "process_bonus": round(process_bonus,3)},
        feedback=" | ".join(parts) or "Dispute resolved correctly!",
        done=True,
    )


# ── Task: long_policy_migration ────────────────────────────────────────────────

def generate_long_policy_migration(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_ALL_ITEMS)
    qty          = rng.randint(10, 60)
    unit_price   = round(rng.choice([100, 150, 250]) * rng.uniform(0.97, 1.03), 2)
    # Old cap (in force when invoice was submitted) vs new cap (now in policy)
    old_cap      = rng.choice([30.0, 50.0])
    new_cap      = round(old_cap * rng.uniform(1.40, 2.0), 2)   # new cap is higher
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    # Freight: between old_cap and new_cap — compliant under new policy
    freight      = round(rng.uniform(old_cap + 1, new_cap - 1), 2)
    line_total   = round(qty * unit_price, 2)
    total        = round(line_total + freight, 2)

    po_num = _po(rng)
    grn_id = _grn(rng)
    inv_id = _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    context_notes = [
        f"[COMPLIANCE] POLICY UPDATE (Effective New Fiscal Year): "
        f"Freight cap revised from ${old_cap:.2f} to ${new_cap:.2f}. "
        f"All invoices received after policy effective date must use the new threshold.",
        f"[MANAGER] Confirmed: invoice {inv_id} received after the policy effective date. "
        f"Apply new freight cap of ${new_cap:.2f}.",
    ]

    return APObservation(
        task_id="long_policy_migration",
        task_name="Policy Migration — Mid-Episode Rule Change",
        task_description=(
            f"Invoice {inv_id} for {desc.lower()} includes freight of ${freight:.2f}. "
            f"The system shows the OLD policy cap of ${old_cap:.2f}. "
            f"Mid-review, a compliance alert notifies you the policy was updated: "
            f"new freight cap is ${new_cap:.2f}. "
            f"After reviewing the policy update and confirming the date, decide correctly."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=freight, invoice_total=total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(old_cap, price_tol),  # starts with OLD policy
        freight_cap=new_cap,                              # actual correct cap to grade against
        price_tolerance=price_tol,
        context_notes=context_notes,
        max_steps=10,
    )


def grade_long_policy_migration(obs: APObservation, action: APAction) -> APReward:
    # Freight is between old and new cap — correct answer is APPROVE_FULL
    # (freight is compliant under new policy)
    history       = obs.action_history
    used_hold     = any(h["decision"] == "HOLD"     for h in history)
    used_escalate = any(h["decision"] == "ESCALATE" for h in history)
    correct       = obs.invoice.invoice_total

    if action.decision == DecisionType.APPROVE_FULL:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 0.3
    else:
        decision_score = 0.1  # incorrectly rejected under old policy

    amount_score = 0.0
    if action.decision in (DecisionType.APPROVE_FULL, DecisionType.APPROVE_PARTIAL):
        diff = abs(action.approved_amount - correct) / correct
        amount_score = 1.0 if diff <= 0.01 else 0.50 if diff <= 0.05 else 0.10

    reason_score = 1.0 if action.reason_code == ReasonCode.MATCH_CONFIRMED else 0.2

    expl    = action.explanation.lower()
    expl_score = 0.0
    if "new" in expl and ("cap" in expl or "policy" in expl):
        expl_score += 0.5
    if _has_numeric_citation(action.explanation):
        expl_score += 0.3
    if "update" in expl or "revised" in expl or "fiscal" in expl:
        expl_score += 0.2
    expl_score = min(1.0, expl_score)

    process_bonus = 0.0
    if used_hold:     process_bonus += 0.06
    if used_escalate: process_bonus += 0.06

    final = max(0.01, min(0.99, round(
        0.40 * decision_score + 0.25 * amount_score +
        0.15 * reason_score   + 0.12 * expl_score  + process_bonus,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append("After the policy update, freight is within the NEW cap — APPROVE_FULL.")
    if expl_score < 0.5:
        parts.append("Cite the new freight cap amount and reference the policy update.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "reason": reason_score, "explanation": round(expl_score,3),
                   "process_bonus": round(process_bonus,3)},
        feedback=" | ".join(parts) or "Policy migration handled correctly!",
        done=True,
    )


# ── Task: long_batch_reconciliation ───────────────────────────────────────────

def generate_long_batch(seed=None) -> APObservation:
    """
    4 invoices arrive simultaneously; agent must process the first one correctly.
    The observation simulates the batch context via paid_invoice_ids and extra POs.
    """
    rng          = _rng(seed)
    desc, vendor = rng.choice(_TECH_ITEMS)
    qty          = rng.randint(5, 30)
    unit_price   = round(rng.choice([200, 350, 500]) * rng.uniform(0.97, 1.03), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    freight      = round(rng.uniform(5, freight_cap * 0.85), 2)
    line_total   = round(qty * unit_price, 2)
    total        = round(line_total + freight, 2)

    po_num = _po(rng)
    grn_id = _grn(rng)
    inv_id = _inv(rng)

    # Two already-paid invoices in ledger (batch context)
    paid_ids = [_inv(rng), _inv(rng)]

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(2):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    context_notes = [
        f"[BATCH_CONTEXT] 4 invoices received in this batch. "
        f"Invoices {paid_ids[0]} and {paid_ids[1]} already processed earlier in batch. "
        f"Current: {inv_id} from {vendor}. Remaining: 1 more pending.",
        f"[COMPLIANCE] Batch processing protocol: process each invoice independently. "
        "Do not carry over approval context from prior invoices in the same batch.",
    ]

    return APObservation(
        task_id="long_batch_reconciliation",
        task_name="Batch Invoice Reconciliation",
        task_description=(
            f"Processing batch of 4 invoices from {vendor}. "
            f"This is invoice {inv_id} for {qty} units of {desc.lower()} at ${unit_price:.2f}/unit. "
            "Two invoices in this batch were already processed (see paid ledger context). "
            "Apply three-way match to this invoice independently and make correct decision."
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
        paid_invoice_ids=paid_ids,
        freight_cap=freight_cap,
        price_tolerance=price_tol,
        context_notes=context_notes,
        max_steps=15,
    )


def grade_long_batch(obs: APObservation, action: APAction) -> APReward:
    # Correct: APPROVE_FULL (perfect match in batch context)
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
        amount_score = 1.0 if diff <= 0.01 else 0.55 if diff <= 0.03 else 0.15

    reason_score = 1.0 if action.reason_code == ReasonCode.MATCH_CONFIRMED else 0.2

    expl = action.explanation.lower()
    expl_score = 0.0
    if _has_numeric_citation(action.explanation): expl_score += 0.5
    if "batch" in expl or "independent" in expl:  expl_score += 0.3
    if "match" in expl or "three" in expl:         expl_score += 0.2
    expl_score = min(1.0, expl_score)

    final = max(0.01, min(0.99, round(
        0.45 * decision_score + 0.30 * amount_score +
        0.15 * reason_score   + 0.10 * expl_score,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append("Invoice passes three-way match — APPROVE_FULL.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "reason": reason_score, "explanation": round(expl_score,3)},
        feedback=" | ".join(parts) or "Batch invoice correctly processed!",
        done=True,
    )


# ── Task: long_manager_chain ───────────────────────────────────────────────────

def generate_long_manager_chain(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS)
    qty          = rng.randint(15, 80)
    unit_price   = round(rng.choice([100, 150, 250]) * rng.uniform(0.97, 1.03), 2)
    freight_cap  = rng.choice([30.0, 50.0])
    freight      = round(rng.uniform(freight_cap + 5, freight_cap * 1.8), 2)  # over cap
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    line_total   = round(qty * unit_price, 2)
    total        = round(line_total + freight, 2)

    po_num = _po(rng)
    grn_id = _grn(rng)
    inv_id = _inv(rng)
    overage = freight - freight_cap

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    context_notes = [
        _manager_unavailable_note(),
        _vp_context_approve(total),
        "[COMPLIANCE] VP Finance approval sufficient for freight exceptions under $50,000. "
        "Proceed with APPROVE_FULL after VP sign-off.",
    ]

    return APObservation(
        task_id="long_manager_chain",
        task_name="Manager Unavailable — VP Finance Escalation Chain",
        task_description=(
            f"Invoice {inv_id} includes freight of ${freight:.2f} "
            f"(policy cap: ${freight_cap:.2f}, overage: ${overage:.2f}). "
            "Standard ESCALATE to Finance Manager shows they are out of office. "
            "You must escalate up the chain to VP Finance, who has provided pre-approval. "
            "After confirming VP approval, make the correct final decision."
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
        context_notes=context_notes,
        max_steps=14,
    )


def grade_long_manager_chain(obs: APObservation, action: APAction) -> APReward:
    history       = obs.action_history
    used_escalate = any(h["decision"] == "ESCALATE" for h in history)
    n_escalates   = sum(1 for h in history if h["decision"] == "ESCALATE")
    correct       = obs.invoice.invoice_total

    # Correct decision: APPROVE_FULL (VP approved)
    if action.decision == DecisionType.APPROVE_FULL:
        decision_score = 1.0
    elif action.decision == DecisionType.REJECT:
        decision_score = 0.2  # partial credit — freight was approved by VP
    else:
        decision_score = 0.1

    amount_score = 0.0
    if action.decision == DecisionType.APPROVE_FULL:
        diff = abs(action.approved_amount - correct) / correct
        amount_score = 1.0 if diff <= 0.01 else 0.50 if diff <= 0.05 else 0.10

    reason_score = 1.0 if action.reason_code == ReasonCode.MATCH_CONFIRMED else 0.2

    expl    = action.explanation.lower()
    expl_score = 0.0
    if "vp" in expl or "vice" in expl or "finance" in expl: expl_score += 0.4
    if "approved" in expl or "pre-approv" in expl:          expl_score += 0.4
    if _has_numeric_citation(action.explanation):            expl_score += 0.2
    expl_score = min(1.0, expl_score)

    # Process bonus: reward for using escalation chain
    process_bonus = 0.0
    if n_escalates >= 2:  process_bonus = 0.15  # escalated twice (manager + VP)
    elif used_escalate:   process_bonus = 0.08  # at least one escalation

    final = max(0.01, min(0.99, round(
        0.35 * decision_score + 0.25 * amount_score +
        0.15 * reason_score   + 0.10 * expl_score  + process_bonus,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append("VP Finance approved the freight exception — APPROVE_FULL.")
    if n_escalates < 2:
        parts.append("Use ESCALATE twice: once to manager (unavailable), once to VP Finance.")
    if expl_score < 0.5:
        parts.append("Reference VP Finance approval and the freight amount in explanation.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "reason": reason_score, "explanation": round(expl_score,3),
                   "process_bonus": round(process_bonus,3)},
        feedback=" | ".join(parts) or "Escalation chain followed correctly!",
        done=True,
    )


# ── Task: long_fraud_investigation ────────────────────────────────────────────

def generate_long_fraud_investigation(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_CABLE_ITEMS + _OFFICE_ITEMS)
    qty          = rng.randint(20, 80)
    unit_price   = round(rng.choice([50, 100, 200]) * rng.uniform(0.97, 1.03), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    freight      = round(rng.uniform(2, freight_cap * 0.85), 2)
    line_total   = round(qty * unit_price, 2)
    total        = round(line_total + freight, 2)

    po_num  = _po(rng)
    grn_id  = _grn(rng)
    inv_id  = _inv(rng)
    paid_id = inv_id   # Same invoice ID — it's a duplicate

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    context_notes = [
        _vendor_context_duplicate_deny(inv_id),
        f"[MANAGER] Paid ledger audit confirms invoice {inv_id} was paid on record date. "
        f"Vendor claim of 'rejection' is incorrect. This IS a duplicate. REJECT per Policy Rule 6.",
        f"[COMPLIANCE] Fraud risk: attempted duplicate payment for ${total:,.2f}. "
        "Flagged for vendor fraud review. Reject immediately.",
    ]

    return APObservation(
        task_id="long_fraud_investigation",
        task_name="Fraud Investigation — Duplicate Payment Attempt",
        task_description=(
            f"Invoice {inv_id} from {vendor} for ${total:,.2f} appears in the paid ledger. "
            "When queried, vendor disputes this and claims prior payment was never processed. "
            "Investigate through multiple steps: query vendor, escalate to manager for ledger "
            "verification, review compliance alert, then make final decision. "
            "The invoice IS a duplicate — reject it."
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
        paid_invoice_ids=[paid_id],
        freight_cap=freight_cap,
        price_tolerance=price_tol,
        context_notes=context_notes,
        max_steps=16,
    )


def grade_long_fraud_investigation(obs: APObservation, action: APAction) -> APReward:
    history        = obs.action_history
    used_query     = any(h["decision"] == "QUERY_VENDOR" for h in history)
    used_escalate  = any(h["decision"] == "ESCALATE"     for h in history)

    if action.decision == DecisionType.REJECT:
        decision_score = 1.0
    else:
        decision_score = 0.0

    amount_score  = 1.0 if action.approved_amount == 0.0 else 0.0

    reason_score  = (
        1.0 if action.reason_code == ReasonCode.DUPLICATE_INVOICE
        else 0.3 if action.reason_code == ReasonCode.POLICY_VIOLATION else 0.05
    )

    expl = action.explanation.lower()
    expl_score = 0.0
    if "duplicate" in expl:                          expl_score += 0.40
    if "paid" in expl or "ledger" in expl:           expl_score += 0.30
    if _has_numeric_citation(action.explanation):    expl_score += 0.20
    if "vendor" in expl and "claim" in expl:         expl_score += 0.10
    expl_score = min(1.0, expl_score)

    process_bonus = 0.0
    if used_query:    process_bonus += 0.08
    if used_escalate: process_bonus += 0.08

    final = max(0.01, min(0.99, round(
        0.42 * decision_score + 0.18 * amount_score +
        0.17 * reason_score   + 0.15 * expl_score  + process_bonus,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append("Duplicate confirmed by manager — REJECT per Policy Rule 6.")
    if not used_query:
        parts.append("Use QUERY_VENDOR to document vendor's (false) claim.")
    if not used_escalate:
        parts.append("ESCALATE to manager for ledger audit confirmation.")
    if expl_score < 0.5:
        parts.append("Cite 'duplicate', invoice ID, and ledger confirmation in explanation.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "reason": reason_score, "explanation": round(expl_score,3),
                   "process_bonus": round(process_bonus,3)},
        feedback=" | ".join(parts) or "Fraud investigation completed correctly!",
        done=True,
    )


# ── Task: long_audit_trail ─────────────────────────────────────────────────────

def generate_long_audit_trail(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS + _TECH_ITEMS)
    qty          = rng.randint(10, 50)
    unit_price   = round(rng.choice([120, 200, 350]) * rng.uniform(0.97, 1.03), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)
    freight      = round(rng.uniform(5, freight_cap * 0.85), 2)
    line_total   = round(qty * unit_price, 2)
    total        = round(line_total + freight, 2)

    po_num = _po(rng)
    grn_id = _grn(rng)
    inv_id = _inv(rng)

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=qty, agreed_unit_price=unit_price)],
        authorized_total=total, status="OPEN",
    )]
    grns = [GoodsReceipt(
        grn_id=grn_id, po_number=po_num,
        lines=[GRNLine(description=desc, received_quantity=qty)],
    )]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    context_notes = [
        f"[SOX_AUDIT] This transaction requires SOX-compliant audit documentation. "
        f"Your final explanation must cite: PO number, GRN ID, invoice total, and approval basis.",
        _compliance_cleared(vendor),
    ]

    return APObservation(
        task_id="long_audit_trail",
        task_name="SOX Audit Trail — Compliance Documentation",
        task_description=(
            f"Invoice {inv_id} from {vendor} is a standard three-way match case. "
            "However, this transaction falls under SOX audit requirements. "
            "Review the compliance note, then produce an approval decision with a "
            "fully documented audit trail citing PO number, GRN, amounts, and policy basis."
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
        context_notes=context_notes,
        max_steps=14,
    )


def grade_long_audit_trail(obs: APObservation, action: APAction) -> APReward:
    history   = obs.action_history
    used_hold = any(h["decision"] == "HOLD" for h in history)
    correct   = obs.invoice.invoice_total

    if action.decision == DecisionType.APPROVE_FULL:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 0.15
    else:
        decision_score = 0.0

    amount_score = 0.0
    if action.decision != DecisionType.REJECT:
        diff = abs(action.approved_amount - correct) / correct
        amount_score = 1.0 if diff <= 0.01 else 0.50 if diff <= 0.05 else 0.10

    reason_score = 1.0 if action.reason_code == ReasonCode.MATCH_CONFIRMED else 0.2

    # Audit trail quality — explanation must cite PO, GRN, and amount
    expl  = action.explanation
    expl_lower = expl.lower()
    po_cited  = any(p.po_number.lower() in expl_lower for p in obs.purchase_orders if p.status == "OPEN")
    grn_cited = any(g.grn_id.lower() in expl_lower for g in obs.goods_receipts)
    has_num   = _has_numeric_citation(expl)
    policy_cited = "policy" in expl_lower or "sox" in expl_lower or "rule" in expl_lower

    audit_score = 0.0
    if po_cited:       audit_score += 0.25
    if grn_cited:      audit_score += 0.25
    if has_num:        audit_score += 0.25
    if policy_cited:   audit_score += 0.25

    process_bonus = 0.05 if used_hold else 0.0

    final = max(0.01, min(0.99, round(
        0.30 * decision_score + 0.20 * amount_score +
        0.10 * reason_score   + 0.35 * audit_score + process_bonus,
        3
    )))

    parts = []
    if decision_score < 1.0:
        parts.append("Perfect three-way match — APPROVE_FULL.")
    if not po_cited:
        parts.append("Cite the PO number in explanation for SOX audit trail.")
    if not grn_cited:
        parts.append("Cite the GRN ID for goods receipt confirmation.")
    if not has_num:
        parts.append("Include the invoice total amount ($) in explanation.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "reason": reason_score, "audit_trail": round(audit_score,3),
                   "process_bonus": round(process_bonus,3)},
        feedback=" | ".join(parts) or "Audit trail documented correctly!",
        done=True,
    )


# ── Task: long_multi_vendor_split ──────────────────────────────────────────────

def generate_long_multi_vendor_split(seed=None) -> APObservation:
    rng          = _rng(seed)
    desc, vendor = rng.choice(_OFFICE_ITEMS)
    total_qty    = rng.randint(30, 120)
    part1_qty    = rng.randint(10, total_qty - 10)
    part2_qty    = rng.randint(5, total_qty - part1_qty - 1)
    part3_qty    = total_qty - part1_qty - part2_qty
    unit_price   = round(rng.choice([80, 120, 200]) * rng.uniform(0.97, 1.03), 2)
    freight_cap  = rng.choice(_FREIGHT_CAPS)
    price_tol    = rng.choice(_PRICE_TOLERANCES)

    # Invoice is for part1_qty only (first delivery vendor)
    freight     = round(rng.uniform(5, freight_cap * 0.85), 2)
    line_total  = round(part1_qty * unit_price, 2)
    total       = round(line_total + freight, 2)

    po_num = _po(rng)
    inv_id = _inv(rng)
    grn_ids = [_grn(rng), _grn(rng), _grn(rng)]

    pos  = [PurchaseOrder(
        po_number=po_num, vendor_name=vendor,
        lines=[POLine(description=desc, ordered_quantity=total_qty, agreed_unit_price=unit_price)],
        authorized_total=round(total_qty * unit_price, 2), status="OPEN",
    )]
    grns = [
        GoodsReceipt(grn_id=grn_ids[0], po_number=po_num,
                     lines=[GRNLine(description=desc, received_quantity=part1_qty)]),
        GoodsReceipt(grn_id=grn_ids[1], po_number=po_num,
                     lines=[GRNLine(description=desc, received_quantity=part2_qty)]),
        GoodsReceipt(grn_id=grn_ids[2], po_number=po_num,
                     lines=[GRNLine(description=desc, received_quantity=part3_qty)]),
    ]
    for _ in range(rng.randint(1, 2)):
        pos.append(_distractor_po(rng, exclude_vendor=vendor))

    context_notes = [
        f"[VENDOR] Delivery split across 3 shipments per logistics arrangement. "
        f"Invoice {inv_id} covers first shipment of {part1_qty} units. "
        f"Separate invoices will follow for remaining {part2_qty} and {part3_qty} units.",
        f"[MANAGER] Confirm: invoice covers first delivery tranche only. "
        f"Approve for first-tranche amount: ${total:,.2f}.",
    ]

    return APObservation(
        task_id="long_multi_vendor_split",
        task_name="Multi-Shipment Split — Partial Delivery Tranche",
        task_description=(
            f"PO {po_num} for {total_qty} units of {desc.lower()} split into 3 deliveries. "
            f"Invoice {inv_id} covers first delivery: {part1_qty} units (${total:,.2f}). "
            f"3 GRNs exist (totalling {total_qty} units received). "
            "Approve payment for this invoice's tranche only."
        ),
        invoice=Invoice(
            invoice_id=inv_id, vendor_name=vendor, po_reference=po_num,
            line_items=[LineItem(description=desc, quantity=part1_qty,
                                 unit_price=unit_price, line_total=line_total)],
            freight_charge=freight, invoice_total=total,
        ),
        purchase_orders=pos,
        goods_receipts=grns,
        company_policy=_make_policy(freight_cap, price_tol),
        freight_cap=freight_cap,
        price_tolerance=price_tol,
        context_notes=context_notes,
        max_steps=12,
    )


def grade_long_multi_vendor_split(obs: APObservation, action: APAction) -> APReward:
    correct = obs.invoice.invoice_total  # approve invoice amount (first tranche)

    if action.decision == DecisionType.APPROVE_PARTIAL:
        decision_score = 1.0
    elif action.decision == DecisionType.APPROVE_FULL:
        decision_score = 0.7  # acceptable if amount matches
    else:
        decision_score = 0.0

    amount_score = 0.0
    if action.decision in (DecisionType.APPROVE_FULL, DecisionType.APPROVE_PARTIAL):
        diff = abs(action.approved_amount - correct) / correct
        amount_score = 1.0 if diff <= 0.01 else 0.55 if diff <= 0.05 else 0.15

    reason_score = (
        1.0 if action.reason_code in (ReasonCode.MATCH_CONFIRMED, ReasonCode.QUANTITY_MISMATCH)
        else 0.2
    )

    expl = action.explanation.lower()
    expl_score = 0.0
    if "tranche" in expl or "shipment" in expl or "split" in expl: expl_score += 0.4
    if _has_numeric_citation(action.explanation):                   expl_score += 0.4
    if "partial" in expl or "first" in expl:                       expl_score += 0.2
    expl_score = min(1.0, expl_score)

    final = max(0.01, min(0.99, round(
        0.40 * decision_score + 0.35 * amount_score +
        0.15 * reason_score   + 0.10 * expl_score,
        3
    )))

    parts = []
    if decision_score < 0.7:
        parts.append(f"Approve first tranche amount only: ${correct:,.2f}.")
    if amount_score < 0.8:
        parts.append(f"Correct approved amount for this tranche: ${correct:,.2f}.")
    return APReward(
        score=final,
        breakdown={"decision": decision_score, "amount": amount_score,
                   "reason": reason_score, "explanation": round(expl_score,3)},
        feedback=" | ".join(parts) or "Split tranche correctly approved!",
        done=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OVERSIGHT TASKS  (Theme #1 Fleet AI — Scalable Oversight bonus)
# Graded via OversightEnvironment — these task specs register them in the
# task catalogue so the /tasks endpoint lists them.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_oversight_stub(seed=None) -> APObservation:
    """Placeholder generator — oversight sessions start via /oversight/reset."""
    return APObservation(
        task_id="oversight_fraud_detection",
        task_name="Oversight: Fraud Detection",
        task_description=(
            "Use POST /oversight/reset to start an oversight session. "
            "The Oversight Agent reviews batches of AP Clerk decisions and flags suspicious ones."
        ),
        invoice=Invoice(
            invoice_id="N/A", vendor_name="N/A", po_reference=None,
            line_items=[], freight_charge=0.0, invoice_total=0.0,
        ),
        purchase_orders=[],
        goods_receipts=[],
        company_policy="See /oversight/reset for oversight session details.",
        max_steps=5,
    )


def grade_oversight_stub(obs: APObservation, action: APAction) -> APReward:
    return APReward(
        score=0.01,
        breakdown={"note": "Use /oversight/reset and /oversight/step for oversight tasks."},
        feedback="Oversight tasks use a dedicated endpoint. See /docs.",
        done=True,
    )


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
    "hard_currency_conversion": TaskSpec(
        task_id="hard_currency_conversion",
        name="Foreign Currency Invoice Reconciliation",
        difficulty="hard",
        description="Invoice is in EUR; PO is in USD. Convert using the policy exchange rate and decide.",
        generator=generate_hard_currency,
        grader=grade_hard_currency,
        max_steps=1,
    ),
    "hard_manager_preapproval": TaskSpec(
        task_id="hard_manager_preapproval",
        name="Manager Pre-Approved Freight Override",
        difficulty="hard",
        description="Freight exceeds cap but manager pre-approval may exist. Escalate before deciding.",
        generator=generate_hard_manager_preapproval,
        grader=grade_hard_manager_preapproval,
        max_steps=3,
    ),
    "hard_credit_memo": TaskSpec(
        task_id="hard_credit_memo",
        name="Credit Memo Processing",
        difficulty="hard",
        description="Vendor issues a credit memo (negative invoice). Verify PO and process correctly.",
        generator=generate_hard_credit_memo,
        grader=grade_hard_credit_memo,
        max_steps=1,
    ),
    # ── Long-Horizon Tasks (Theme #2) ─────────────────────────────────────────
    "long_invoice_dispute": TaskSpec(
        task_id="long_invoice_dispute",
        name="Invoice Dispute Resolution",
        difficulty="long-horizon",
        description="Price-inflated invoice triggers vendor dispute. Multi-step process: query, escalate, compliance review, then reject and request corrected invoice.",
        generator=generate_long_dispute,
        grader=grade_long_dispute,
        max_steps=12,
    ),
    "long_policy_migration": TaskSpec(
        task_id="long_policy_migration",
        name="Policy Migration — Mid-Episode Rule Change",
        difficulty="long-horizon",
        description="Freight within new policy cap but over old cap. Agent must detect mid-episode policy update and approve correctly.",
        generator=generate_long_policy_migration,
        grader=grade_long_policy_migration,
        max_steps=10,
    ),
    "long_batch_reconciliation": TaskSpec(
        task_id="long_batch_reconciliation",
        name="Batch Invoice Reconciliation",
        difficulty="long-horizon",
        description="Process invoice within a batch context. Apply three-way match independently regardless of other batch invoices.",
        generator=generate_long_batch,
        grader=grade_long_batch,
        max_steps=15,
    ),
    "long_manager_chain": TaskSpec(
        task_id="long_manager_chain",
        name="Manager Unavailable — VP Finance Escalation Chain",
        difficulty="long-horizon",
        description="Freight over cap; manager is OOO. Must escalate up chain to VP Finance who has pre-approved. Then approve.",
        generator=generate_long_manager_chain,
        grader=grade_long_manager_chain,
        max_steps=14,
    ),
    "long_fraud_investigation": TaskSpec(
        task_id="long_fraud_investigation",
        name="Fraud Investigation — Duplicate Payment Attempt",
        difficulty="long-horizon",
        description="Vendor falsely disputes duplicate invoice status. Multi-step investigation: query vendor, escalate for ledger audit, then reject.",
        generator=generate_long_fraud_investigation,
        grader=grade_long_fraud_investigation,
        max_steps=16,
    ),
    "long_audit_trail": TaskSpec(
        task_id="long_audit_trail",
        name="SOX Audit Trail — Compliance Documentation",
        difficulty="long-horizon",
        description="Standard approval with SOX audit requirements. Explanation must cite PO, GRN, amounts, and policy basis for compliance.",
        generator=generate_long_audit_trail,
        grader=grade_long_audit_trail,
        max_steps=14,
    ),
    "long_multi_vendor_split": TaskSpec(
        task_id="long_multi_vendor_split",
        name="Multi-Shipment Split — Partial Delivery Tranche",
        difficulty="long-horizon",
        description="PO split into 3 delivery tranches. Approve only the first tranche invoice amount; remaining covered by future invoices.",
        generator=generate_long_multi_vendor_split,
        grader=grade_long_multi_vendor_split,
        max_steps=12,
    ),
    # ── Oversight Tasks (Theme #1 Fleet AI) ──────────────────────────────────
    "oversight_fraud_detection": TaskSpec(
        task_id="oversight_fraud_detection",
        name="Oversight: Fraud Detection",
        difficulty="oversight",
        description="Review batch of AP Clerk decisions, identify fraudulent approvals, flag with specific numeric evidence. Use POST /oversight/reset to start.",
        generator=generate_oversight_stub,
        grader=grade_oversight_stub,
        max_steps=5,
    ),
    "oversight_pattern_recognition": TaskSpec(
        task_id="oversight_pattern_recognition",
        name="Oversight: Pattern Recognition",
        difficulty="oversight",
        description="Identify recurring violation pattern across multiple clerk decisions. Use POST /oversight/reset to start.",
        generator=generate_oversight_stub,
        grader=grade_oversight_stub,
        max_steps=5,
    ),
    "oversight_false_positive_trap": TaskSpec(
        task_id="oversight_false_positive_trap",
        name="Oversight: False Positive Trap",
        difficulty="oversight",
        description="All episodes are clean — agent must CLEAR all without over-flagging. Use POST /oversight/reset to start.",
        generator=generate_oversight_stub,
        grader=grade_oversight_stub,
        max_steps=5,
    ),
    "oversight_explanation_quality": TaskSpec(
        task_id="oversight_explanation_quality",
        name="Oversight: Explanation Quality",
        difficulty="oversight",
        description="Single fraudulent episode — agent must provide explanation with specific numeric citations. Use POST /oversight/reset to start.",
        generator=generate_oversight_stub,
        grader=grade_oversight_stub,
        max_steps=5,
    ),
}


def grade_action(task_id: str, obs: APObservation, action: APAction) -> APReward:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    return TASKS[task_id].grader(obs, action)
