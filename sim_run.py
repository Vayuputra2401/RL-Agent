"""Simulated full pass through all 10 tasks with an optimal agent."""
from app import APClerkEnvironment, APAction, DecisionType, ReasonCode
from app.tasks import TASKS

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 42]
SEP   = "=" * 65


def solve(task_id, obs, env):
    """Simulate an optimal AP clerk — multi-step on hard tasks, single-step elsewhere."""
    steps = []

    # ── Optional intermediate steps for hard multi-step tasks ────────────────
    if task_id == "hard_policy_violation":
        a = APAction(
            decision=DecisionType.ESCALATE,
            approved_amount=0.0,
            reason_code=ReasonCode.MANAGER_REVIEW,
            explanation=(
                f"Freight charge ${obs.invoice.freight_charge:.2f} appears to exceed the "
                f"${obs.freight_cap:.2f} policy cap. Escalating to Finance Manager for "
                "confirmation before making final decision."
            ),
        )
        obs, r, done, _ = env.step(a)
        steps.append(("ESCALATE", r.score, done))

    if task_id == "hard_duplicate_invoice":
        a = APAction(
            decision=DecisionType.QUERY_VENDOR,
            approved_amount=0.0,
            reason_code=ReasonCode.PENDING_CLARIFICATION,
            explanation=(
                f"Invoice {obs.invoice.invoice_id} appears in the paid ledger. "
                "Querying vendor to confirm whether this is a duplicate submission."
            ),
        )
        obs, r, done, _ = env.step(a)
        steps.append(("QUERY_VENDOR", r.score, done))

    # ── Terminal decision ─────────────────────────────────────────────────────
    real_po = obs.purchase_orders[0].po_number if obs.purchase_orders else None

    if task_id == "easy_perfect_match":
        action = APAction(
            decision=DecisionType.APPROVE_FULL,
            approved_amount=obs.invoice.invoice_total,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation="Invoice, PO and GRN match exactly. Freight within policy cap. Approving full amount.",
        )

    elif task_id == "easy_no_po_found":
        action = APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.NO_PO_FOUND,
            explanation=(
                f"Invoice references {obs.invoice.po_reference} but no matching OPEN PO "
                "exists in the system. Rejecting per Policy Rule 5."
            ),
        )

    elif task_id == "medium_quantity_shortfall":
        up   = obs.purchase_orders[0].lines[0].agreed_unit_price
        recv = sum(
            l.received_quantity
            for g in obs.goods_receipts if g.po_number == real_po
            for l in g.lines
        )
        amt = round(recv * up, 2)
        action = APAction(
            decision=DecisionType.APPROVE_PARTIAL,
            approved_amount=amt,
            reason_code=ReasonCode.QUANTITY_MISMATCH,
            explanation=(
                f"GRN confirms only {int(recv)} of {int(obs.invoice.line_items[0].quantity)} "
                f"units received. Approving ${amt:,.2f} per Policy Rule 3."
            ),
        )

    elif task_id == "medium_price_discrepancy":
        inv_p = obs.invoice.line_items[0].unit_price
        po_p  = obs.purchase_orders[0].lines[0].agreed_unit_price
        dev   = (inv_p - po_p) / po_p * 100
        tol   = obs.price_tolerance * 100
        action = APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.PRICE_DISCREPANCY,
            explanation=(
                f"Invoice unit price ${inv_p:.2f} vs agreed PO price ${po_p:.2f} — "
                f"{dev:.1f}% deviation exceeds {tol:.1f}% policy threshold. "
                "Rejecting per Policy Rule 4."
            ),
        )

    elif task_id == "medium_split_delivery":
        recv = sum(
            l.received_quantity
            for g in obs.goods_receipts if g.po_number == real_po
            for l in g.lines
        )
        up  = obs.purchase_orders[0].lines[0].agreed_unit_price
        amt = round(recv * up, 2)
        action = APAction(
            decision=DecisionType.APPROVE_FULL,
            approved_amount=amt,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation=(
                f"Two GRNs confirm all {int(recv)} units received across split shipments. "
                f"Full amount ${amt:,.2f} approved."
            ),
        )

    elif task_id == "medium_vendor_mismatch":
        inv_v = obs.invoice.vendor_name
        po_v  = obs.purchase_orders[0].vendor_name
        action = APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.VENDOR_MISMATCH,
            explanation=(
                f'Invoice vendor "{inv_v}" does not match PO vendor "{po_v}". '
                "Name mismatch violates Policy Rule 7. Rejecting invoice."
            ),
        )

    elif task_id == "hard_policy_violation":
        freight = obs.invoice.freight_charge
        cap     = obs.freight_cap
        action = APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.POLICY_VIOLATION,
            explanation=(
                f"Freight ${freight:.2f} exceeds the ${cap:.2f} unapproved cap. "
                "Finance Manager confirmed NOT pre-approved. "
                "Rejecting per Policy Rule 2."
            ),
        )

    elif task_id == "hard_duplicate_invoice":
        inv_id = obs.invoice.invoice_id
        action = APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.DUPLICATE_INVOICE,
            explanation=(
                f"Invoice {inv_id} already in the paid ledger. "
                "Vendor confirmed duplicate submission. "
                "Rejecting per Policy Rule 6."
            ),
        )

    elif task_id == "hard_partial_po_match":
        po_descs = {
            pl.description.lower()
            for po in obs.purchase_orders if po.status == "OPEN"
            for pl in po.lines
        }
        amt       = round(sum(
            li.line_total for li in obs.invoice.line_items
            if li.description.lower() in po_descs
        ), 2)
        uncovered = [li.description for li in obs.invoice.line_items
                     if li.description.lower() not in po_descs]
        action = APAction(
            decision=DecisionType.APPROVE_PARTIAL,
            approved_amount=amt,
            reason_code=ReasonCode.NO_PO_FOUND,
            explanation=(
                f"Only PO-covered items payable (${amt:,.2f}). "
                f"Items without PO authorisation: {uncovered}. "
                "Partial approval per Policy Rule 8."
            ),
        )

    elif task_id == "hard_tax_discrepancy":
        tax = obs.invoice.tax_amount
        action = APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.TAX_DISCREPANCY,
            explanation=(
                f"Invoice includes ${tax:.2f} tax charge not present in the PO authorisation. "
                "Unapproved additional charge per Policy Rule 8. Rejecting."
            ),
        )

    else:
        action = APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.POLICY_VIOLATION,
            explanation="Unrecognised task — defaulting to safe rejection.",
        )

    obs2, reward, done, _ = env.step(action)
    steps.append((action.decision.value, reward.score, done))
    return steps, reward, obs, obs2


# ── Main simulation ───────────────────────────────────────────────────────────

print()
print(SEP)
print("  AP CLERK ENV — Full Simulated Pass (Optimal Agent)")
print("  10 tasks  |  varied seeds  |  multi-step where applicable")
print(SEP)

total   = 0.0
results = []

for i, (task_id, seed) in enumerate(zip(TASKS.keys(), SEEDS)):
    spec = TASKS[task_id]
    env  = APClerkEnvironment()
    obs  = env.reset(task_id, seed=seed)

    steps, reward, obs_pre, obs_post = solve(task_id, obs, env)

    total += reward.score
    results.append((task_id, reward.score, steps))

    step_str = " -> ".join(
        f"{s[0]}(done={s[2]})" for s in steps
    )
    closed = sum(1 for p in obs_pre.purchase_orders if p.status == "CLOSED")
    dist_grns = sum(
        1 for g in obs_pre.goods_receipts
        if g.po_number not in {p.po_number for p in obs_pre.purchase_orders if p.status == "OPEN"}
    )

    print(f"\n[{i+1:02d}] {task_id}  ({spec.difficulty})")
    print(f"      seed={seed}  freight_cap=${obs_pre.freight_cap:.0f}  price_tol={obs_pre.price_tolerance*100:.1f}%")
    print(f"      Vendor  : {obs_pre.invoice.vendor_name}")
    print(f"      Invoice : ${obs_pre.invoice.invoice_total:,.2f}   "
          f"POs={len(obs_pre.purchase_orders)} ({closed} closed distractors)   "
          f"GRNs={len(obs_pre.goods_receipts)} ({dist_grns} distractor)")
    if obs_post.context_notes:
        print(f"      Context : {obs_post.context_notes[0][:75]}...")
    print(f"      Steps   : {step_str}")
    print(f"      Score   : {reward.score:.3f}   {reward.feedback[:80]}")

print()
print(SEP)
print("  SCOREBOARD")
print(SEP)
for tid, sc, steps in results:
    bar  = "#" * int(sc * 20)
    flag = " <-- multi-step" if len(steps) > 1 else ""
    print(f"  {tid:<32}  {sc:.3f}  |{bar:<20}|{flag}")
print(SEP)
mean = total / len(TASKS)
print(f"  MEAN SCORE : {mean:.3f}   ({total:.3f} / {len(TASKS)}.000)")
print(SEP)
