"""Simulated full pass through all AP Clerk tasks (including long-horizon) with an optimal agent."""
import re as _re
from app import APClerkEnvironment, APAction, DecisionType, ReasonCode
from app.tasks import TASKS

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 42, 17, 31, 57,
         101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
SEP   = "=" * 70


def solve(task_id, obs, env):
    """Simulate an optimal AP clerk — multi-step on hard + long-horizon tasks."""
    steps = []

    # ── Pre-terminal intermediate steps ──────────────────────────────────────

    if task_id == "hard_manager_preapproval":
        _do_escalate(obs, env, steps)

    if task_id == "hard_policy_violation":
        _do_escalate(obs, env, steps)

    if task_id == "hard_duplicate_invoice":
        _do_query_vendor(obs, env, steps)

    # Long-horizon tasks
    if task_id == "long_invoice_dispute":
        # Step 1: Query vendor to document price issue
        obs, r, done, _ = env.step(APAction(
            decision=DecisionType.QUERY_VENDOR,
            approved_amount=0.0,
            reason_code=ReasonCode.PENDING_CLARIFICATION,
            explanation=(
                f"Invoice unit price ${obs.invoice.line_items[0].unit_price:.2f} "
                f"exceeds PO agreed price. Querying vendor to document discrepancy."
            ),
        ))
        steps.append(("QUERY_VENDOR", r.score, done))
        # Step 2: Escalate to manager after vendor response
        obs, r, done, _ = env.step(APAction(
            decision=DecisionType.ESCALATE,
            approved_amount=0.0,
            reason_code=ReasonCode.MANAGER_REVIEW,
            explanation="Escalating to Finance Manager after vendor acknowledged price error.",
        ))
        steps.append(("ESCALATE", r.score, done))

    if task_id == "long_policy_migration":
        # Step 1: HOLD to trigger compliance/policy update context
        obs, r, done, _ = env.step(APAction(
            decision=DecisionType.HOLD,
            approved_amount=0.0,
            reason_code=ReasonCode.PENDING_CLARIFICATION,
            explanation=(
                f"Freight ${obs.invoice.freight_charge:.2f} appears near policy threshold. "
                "Holding pending compliance review of current freight cap policy."
            ),
        ))
        steps.append(("HOLD", r.score, done))

    if task_id == "long_manager_chain":
        # Step 1: ESCALATE to manager (will get OOO response)
        obs, r, done, _ = env.step(APAction(
            decision=DecisionType.ESCALATE,
            approved_amount=0.0,
            reason_code=ReasonCode.MANAGER_REVIEW,
            explanation=(
                f"Freight ${obs.invoice.freight_charge:.2f} exceeds cap "
                f"${obs.freight_cap:.2f}. Escalating to Finance Manager for approval."
            ),
        ))
        steps.append(("ESCALATE-manager", r.score, done))
        # Step 2: ESCALATE again to VP Finance
        obs, r, done, _ = env.step(APAction(
            decision=DecisionType.ESCALATE,
            approved_amount=0.0,
            reason_code=ReasonCode.MANAGER_REVIEW,
            explanation=(
                "Manager is out of office. Escalating to VP Finance for freight override approval."
            ),
        ))
        steps.append(("ESCALATE-VP", r.score, done))

    if task_id == "long_fraud_investigation":
        # Step 1: Query vendor (will deny duplicate)
        obs, r, done, _ = env.step(APAction(
            decision=DecisionType.QUERY_VENDOR,
            approved_amount=0.0,
            reason_code=ReasonCode.PENDING_CLARIFICATION,
            explanation=(
                f"Invoice {obs.invoice.invoice_id} appears in the paid ledger. "
                "Querying vendor to document their claim."
            ),
        ))
        steps.append(("QUERY_VENDOR", r.score, done))
        # Step 2: Escalate to manager for ledger audit confirmation
        obs, r, done, _ = env.step(APAction(
            decision=DecisionType.ESCALATE,
            approved_amount=0.0,
            reason_code=ReasonCode.MANAGER_REVIEW,
            explanation=(
                "Vendor disputes duplicate status. Escalating to manager for paid-ledger audit."
            ),
        ))
        steps.append(("ESCALATE", r.score, done))

    if task_id == "long_audit_trail":
        # Step 1: HOLD for SOX compliance review
        obs, r, done, _ = env.step(APAction(
            decision=DecisionType.HOLD,
            approved_amount=0.0,
            reason_code=ReasonCode.PENDING_CLARIFICATION,
            explanation="SOX audit requirement detected. Holding for compliance documentation review.",
        ))
        steps.append(("HOLD", r.score, done))

    # ── Terminal decision ─────────────────────────────────────────────────────
    action = _terminal_action(task_id, obs)
    obs2, reward, done, _ = env.step(action)
    steps.append((action.decision.value, reward.score, done))
    return steps, reward, obs, obs2


def _do_escalate(obs, env, steps):
    obs_new, r, done, _ = env.step(APAction(
        decision=DecisionType.ESCALATE,
        approved_amount=0.0,
        reason_code=ReasonCode.MANAGER_REVIEW,
        explanation=(
            f"Freight ${obs.invoice.freight_charge:.2f} exceeds policy cap "
            f"${obs.freight_cap:.2f}. Escalating to Finance Manager for pre-approval check."
        ),
    ))
    steps.append(("ESCALATE", r.score, done))
    obs.__dict__.update(obs_new.__dict__)


def _do_query_vendor(obs, env, steps):
    obs_new, r, done, _ = env.step(APAction(
        decision=DecisionType.QUERY_VENDOR,
        approved_amount=0.0,
        reason_code=ReasonCode.PENDING_CLARIFICATION,
        explanation=(
            f"Invoice {obs.invoice.invoice_id} appears in the paid ledger. "
            "Querying vendor to confirm before final rejection."
        ),
    ))
    steps.append(("QUERY_VENDOR", r.score, done))
    obs.__dict__.update(obs_new.__dict__)


def _terminal_action(task_id: str, obs) -> APAction:
    real_po = next(
        (po.po_number for po in obs.purchase_orders if po.status == "OPEN"),
        None
    )

    if task_id == "easy_perfect_match":
        return APAction(
            decision=DecisionType.APPROVE_FULL,
            approved_amount=obs.invoice.invoice_total,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation=(
                f"Invoice ${obs.invoice.invoice_total:,.2f}, PO and GRN match exactly. "
                f"Freight ${obs.invoice.freight_charge:.2f} within cap. Approving full amount."
            ),
        )

    if task_id == "easy_no_po_found":
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.NO_PO_FOUND,
            explanation=(
                f"Invoice references {obs.invoice.po_reference} but no matching OPEN PO "
                "exists in the system. Rejecting per Policy Rule 5."
            ),
        )

    if task_id == "medium_quantity_shortfall":
        up   = obs.purchase_orders[0].lines[0].agreed_unit_price
        recv = sum(
            l.received_quantity
            for g in obs.goods_receipts if g.po_number == real_po
            for l in g.lines
        )
        amt = round(recv * up, 2)
        return APAction(
            decision=DecisionType.APPROVE_PARTIAL,
            approved_amount=amt,
            reason_code=ReasonCode.QUANTITY_MISMATCH,
            explanation=(
                f"GRN confirms only {int(recv)} of "
                f"{int(obs.invoice.line_items[0].quantity)} units received. "
                f"Approving ${amt:,.2f} per Policy Rule 3."
            ),
        )

    if task_id == "medium_price_discrepancy":
        inv_p = obs.invoice.line_items[0].unit_price
        po_p  = obs.purchase_orders[0].lines[0].agreed_unit_price
        dev   = (inv_p - po_p) / po_p * 100
        tol   = obs.price_tolerance * 100
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.PRICE_DISCREPANCY,
            explanation=(
                f"Invoice unit price ${inv_p:.2f} vs agreed PO price ${po_p:.2f} — "
                f"{dev:.1f}% deviation exceeds {tol:.1f}% threshold. Rejecting per Rule 4."
            ),
        )

    if task_id == "medium_split_delivery":
        recv = sum(
            l.received_quantity
            for g in obs.goods_receipts if g.po_number == real_po
            for l in g.lines
        )
        up  = obs.purchase_orders[0].lines[0].agreed_unit_price
        amt = round(recv * up + obs.invoice.freight_charge, 2)
        return APAction(
            decision=DecisionType.APPROVE_FULL,
            approved_amount=amt,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation=(
                f"Two GRNs confirm all {int(recv)} units received across split shipments. "
                f"Full amount ${amt:,.2f} approved."
            ),
        )

    if task_id == "medium_vendor_mismatch":
        inv_v = obs.invoice.vendor_name
        po_v  = obs.purchase_orders[0].vendor_name
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.VENDOR_MISMATCH,
            explanation=(
                f'Invoice vendor "{inv_v}" does not match PO vendor "{po_v}". '
                "Name mismatch violates Policy Rule 7."
            ),
        )

    if task_id == "hard_policy_violation":
        freight = obs.invoice.freight_charge
        cap     = obs.freight_cap
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.POLICY_VIOLATION,
            explanation=(
                f"Freight ${freight:.2f} exceeds ${cap:.2f} cap. "
                "Manager confirmed NOT pre-approved. Rejecting per Policy Rule 2."
            ),
        )

    if task_id == "hard_duplicate_invoice":
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.DUPLICATE_INVOICE,
            explanation=(
                f"Invoice {obs.invoice.invoice_id} already in the paid ledger. "
                "Vendor confirmed duplicate. Rejecting per Policy Rule 6."
            ),
        )

    if task_id == "hard_partial_po_match":
        po_descs = {
            pl.description.lower()
            for po in obs.purchase_orders if po.status == "OPEN"
            for pl in po.lines
        }
        amt = round(sum(
            li.line_total for li in obs.invoice.line_items
            if li.description.lower() in po_descs
        ), 2)
        return APAction(
            decision=DecisionType.APPROVE_PARTIAL,
            approved_amount=amt,
            reason_code=ReasonCode.NO_PO_FOUND,
            explanation=(
                f"Only PO-covered items payable (${amt:,.2f}). "
                "Items without PO authorisation excluded. Partial approval per Rule 8."
            ),
        )

    if task_id == "hard_tax_discrepancy":
        tax = obs.invoice.tax_amount
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.TAX_DISCREPANCY,
            explanation=(
                f"Invoice includes ${tax:.2f} tax not in PO authorisation. "
                "Unapproved charge per Policy Rule 8."
            ),
        )

    if task_id == "hard_currency_conversion":
        fx_match = _re.search(r'EUR 1\.00 = USD (\d+\.\d+)', obs.company_policy)
        fx_rate   = float(fx_match.group(1)) if fx_match else 1.0
        inv_eur   = obs.invoice.invoice_total
        inv_usd   = round(inv_eur * fx_rate, 2)
        po_usd    = obs.purchase_orders[0].authorized_total
        tol       = obs.price_tolerance
        dev       = (inv_usd - po_usd) / po_usd if po_usd > 0 else 1.0
        if dev <= tol:
            return APAction(
                decision=DecisionType.APPROVE_FULL,
                approved_amount=inv_usd,
                reason_code=ReasonCode.MATCH_CONFIRMED,
                explanation=(
                    f"EUR {inv_eur:,.2f} × {fx_rate:.2f} = USD {inv_usd:,.2f}. "
                    f"Within PO ${po_usd:,.2f}. Approving."
                ),
            )
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.PRICE_DISCREPANCY,
            explanation=(
                f"EUR {inv_eur:,.2f} × {fx_rate:.2f} = USD {inv_usd:,.2f} "
                f"exceeds PO ${po_usd:,.2f} by {dev*100:.1f}%."
            ),
        )

    if task_id == "hard_manager_preapproval":
        approval_found = any("[MANAGER]" in n for n in obs.context_notes)
        freight = obs.invoice.freight_charge
        cap     = obs.freight_cap
        if approval_found:
            return APAction(
                decision=DecisionType.APPROVE_FULL,
                approved_amount=obs.invoice.invoice_total,
                reason_code=ReasonCode.MATCH_CONFIRMED,
                explanation=(
                    f"Freight ${freight:.2f} (cap ${cap:.2f}) — Finance Manager "
                    f"pre-approval confirmed. Approving ${obs.invoice.invoice_total:,.2f}."
                ),
            )
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.POLICY_VIOLATION,
            explanation=f"Freight ${freight:.2f} exceeds cap ${cap:.2f}. No pre-approval.",
        )

    if task_id == "hard_credit_memo":
        po_ref    = obs.invoice.po_reference
        po_exists = any(po.po_number == po_ref and po.status == "OPEN"
                        for po in obs.purchase_orders)
        credit_amt = obs.invoice.invoice_total
        if po_exists:
            return APAction(
                decision=DecisionType.APPROVE_PARTIAL,
                approved_amount=credit_amt,
                reason_code=ReasonCode.MATCH_CONFIRMED,
                explanation=(
                    f"Credit memo {obs.invoice.invoice_id} has valid OPEN PO. "
                    f"Credit ${credit_amt:,.2f} processed per Policy Rule 9."
                ),
            )
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.NO_PO_FOUND,
            explanation=f"Credit memo {obs.invoice.invoice_id} — no valid OPEN PO. Rejecting.",
        )

    # ── Long-Horizon Terminal Decisions ──────────────────────────────────────

    if task_id == "long_invoice_dispute":
        po = next((p for p in obs.purchase_orders if p.status == "OPEN"), None)
        qty    = obs.invoice.line_items[0].quantity
        agreed = po.lines[0].agreed_unit_price if po and po.lines else 0.0
        correct_total = round(qty * agreed + obs.invoice.freight_charge, 2)
        inv_p = obs.invoice.line_items[0].unit_price
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.PRICE_DISCREPANCY,
            explanation=(
                f"Invoice price ${inv_p:.2f} vs PO agreed ${agreed:.2f}. "
                f"Vendor acknowledged error — correct amount would be ${correct_total:,.2f}. "
                "Rejecting original; corrected invoice required per Policy Rule 4."
            ),
        )

    if task_id == "long_policy_migration":
        # After HOLD, compliance revealed new policy cap — freight is now compliant
        return APAction(
            decision=DecisionType.APPROVE_FULL,
            approved_amount=obs.invoice.invoice_total,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation=(
                f"Policy update confirmed: new freight cap ${obs.freight_cap:.2f}. "
                f"Freight ${obs.invoice.freight_charge:.2f} is within new cap. "
                f"Three-way match confirmed. Approving ${obs.invoice.invoice_total:,.2f}."
            ),
        )

    if task_id == "long_batch_reconciliation":
        return APAction(
            decision=DecisionType.APPROVE_FULL,
            approved_amount=obs.invoice.invoice_total,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation=(
                f"Independent three-way match confirmed within batch context. "
                f"PO, GRN, and invoice for ${obs.invoice.invoice_total:,.2f} all match. "
                "Approving per standard procedure."
            ),
        )

    if task_id == "long_manager_chain":
        # After two ESCALATEs, VP Finance context should be visible
        return APAction(
            decision=DecisionType.APPROVE_FULL,
            approved_amount=obs.invoice.invoice_total,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation=(
                f"Freight ${obs.invoice.freight_charge:.2f} (cap ${obs.freight_cap:.2f}) — "
                f"VP Finance pre-approval confirmed via escalation chain. "
                f"Approving ${obs.invoice.invoice_total:,.2f}."
            ),
        )

    if task_id == "long_fraud_investigation":
        return APAction(
            decision=DecisionType.REJECT,
            approved_amount=0.0,
            reason_code=ReasonCode.DUPLICATE_INVOICE,
            explanation=(
                f"Invoice {obs.invoice.invoice_id} is in paid ledger. "
                f"Vendor disputed (falsely); manager audit confirmed duplicate. "
                f"Rejecting ${obs.invoice.invoice_total:,.2f} per Policy Rule 6."
            ),
        )

    if task_id == "long_audit_trail":
        po  = next((p for p in obs.purchase_orders if p.status == "OPEN"), None)
        grn = next((g for g in obs.goods_receipts if g.po_number == (po.po_number if po else "")), None)
        po_ref  = po.po_number if po else "N/A"
        grn_ref = grn.grn_id if grn else "N/A"
        return APAction(
            decision=DecisionType.APPROVE_FULL,
            approved_amount=obs.invoice.invoice_total,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation=(
                f"SOX audit trail: PO {po_ref} confirmed OPEN; "
                f"GRN {grn_ref} confirms receipt; "
                f"Invoice total ${obs.invoice.invoice_total:,.2f} matches PO authorisation. "
                "Compliance cleared. Three-way match approved per Policy Rule 1."
            ),
        )

    if task_id == "long_multi_vendor_split":
        return APAction(
            decision=DecisionType.APPROVE_PARTIAL,
            approved_amount=obs.invoice.invoice_total,
            reason_code=ReasonCode.MATCH_CONFIRMED,
            explanation=(
                f"First delivery tranche: invoice ${obs.invoice.invoice_total:,.2f} "
                f"covers {int(obs.invoice.line_items[0].quantity)} units per split delivery arrangement. "
                "Approving first tranche amount only."
            ),
        )

    return APAction(
        decision=DecisionType.REJECT,
        approved_amount=0.0,
        reason_code=ReasonCode.POLICY_VIOLATION,
        explanation="Unrecognised task — defaulting to safe rejection.",
    )


# ── Main simulation ───────────────────────────────────────────────────────────

# Exclude oversight stub tasks (they require the /oversight/* endpoints)
RUNNABLE = {
    tid: spec for tid, spec in TASKS.items()
    if spec.difficulty != "oversight"
}

print()
print(SEP)
print("  AP COMMANDER — Full Simulated Pass (Optimal Agent)")
print(f"  {len(RUNNABLE)} tasks  |  varied seeds  |  multi-step where applicable")
print(SEP)

total   = 0.0
results = []

for i, (task_id, spec) in enumerate(RUNNABLE.items()):
    seed = SEEDS[i % len(SEEDS)]
    env  = APClerkEnvironment()
    obs  = env.reset(task_id, seed=seed)

    steps, reward, obs_pre, obs_post = solve(task_id, obs, env)

    total += reward.score
    results.append((task_id, reward.score, steps, spec.difficulty))

    step_str = " -> ".join(f"{s[0]}(done={s[2]})" for s in steps)
    closed   = sum(1 for p in obs_pre.purchase_orders if p.status == "CLOSED")

    print(f"\n[{i+1:02d}] {task_id}  ({spec.difficulty})")
    print(f"      seed={seed}  freight_cap=${obs_pre.freight_cap:.0f}  "
          f"price_tol={obs_pre.price_tolerance*100:.1f}%")
    print(f"      Vendor  : {obs_pre.invoice.vendor_name}")
    print(f"      Invoice : ${obs_pre.invoice.invoice_total:,.2f}   "
          f"POs={len(obs_pre.purchase_orders)} ({closed} closed)   "
          f"GRNs={len(obs_pre.goods_receipts)}")
    if obs_post.context_notes:
        print(f"      Context : {obs_post.context_notes[0][:80]}...")
    print(f"      Steps   : {step_str}")
    print(f"      Score   : {reward.score:.3f}   {reward.feedback[:80]}")

print()
print(SEP)
print("  SCOREBOARD")
print(SEP)

by_diff: dict = {}
for tid, sc, steps, diff in results:
    bar  = "#" * int(sc * 20)
    flag = f" ({len(steps)} steps)" if len(steps) > 1 else ""
    print(f"  {tid:<40}  {sc:.3f}  |{bar:<20}|{flag}")
    by_diff.setdefault(diff, []).append(sc)

print(SEP)
print("  BY DIFFICULTY:")
for diff in ["easy", "medium", "hard", "long-horizon"]:
    scores = by_diff.get(diff, [])
    if scores:
        m = sum(scores) / len(scores)
        print(f"    {diff:<16}: {m:.3f}  ({len(scores)} tasks)")

mean = total / len(RUNNABLE) if RUNNABLE else 0.0
print(SEP)
print(f"  MEAN SCORE : {mean:.3f}   ({total:.3f} / {len(RUNNABLE)}.000)")
print(SEP)
