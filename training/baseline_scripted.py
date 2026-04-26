"""
baseline.py — Optimal scripted agent baseline against live HF Space.
Runs all 20 runnable tasks via HTTP, records every step, saves JSON + plots.
Usage: python baseline.py
"""
import json, os, re, time, datetime
import requests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ENV = 'https://pathikreet-ap-clerk-env.hf.space'
SEED = 42

# Outputs go into runs/baselines/scripted-agent-DATETIME/ so runs never overwrite each other
_RUN_DIR = os.path.join('runs', 'baselines',
                        f'scripted-agent-{datetime.datetime.now().strftime("%Y-%m-%d_%H%M")}')
os.makedirs(_RUN_DIR, exist_ok=True)
OUT_JSON = os.path.join(_RUN_DIR, 'baseline_results.json')
OUT_PLOT = os.path.join(_RUN_DIR, 'baseline_plot.png')


# ── Helpers ────────────────────────────────────────────────────────────────────

def reset(task_id, seed=SEED):
    r = requests.post(f'{ENV}/reset', json={'task_id': task_id, 'seed': seed}, timeout=20)
    r.raise_for_status()
    return r.json()

def step(session_id, action):
    r = requests.post(f'{ENV}/step', json={'session_id': session_id, 'action': action}, timeout=20)
    r.raise_for_status()
    return r.json()

def make_action(decision, amount, reason, explanation):
    return {'decision': decision, 'approved_amount': amount,
            'reason_code': reason, 'explanation': explanation}


# ── Optimal action sequences per task ─────────────────────────────────────────

def run_task(task_id):
    """Returns list of step dicts: {step, action, score, breakdown, done}"""
    data = reset(task_id)
    sid  = data['session_id']
    obs  = data['observation']
    steps = []

    def do_step(decision, amount, reason, expl):
        r = step(sid, make_action(decision, amount, reason, expl))
        steps.append({
            'step':      len(steps) + 1,
            'action':    decision,
            'score':     r['reward']['score'],
            'breakdown': r['reward'].get('breakdown', {}),
            'feedback':  r['reward'].get('feedback', ''),
            'done':      r['done'],
        })
        return r

    # ── pre-terminal intermediate steps ───────────────────────────────────────
    inv   = obs['invoice']
    total = inv['invoice_total']
    items = inv.get('line_items', [{}])
    pos   = obs.get('purchase_orders', [{}])
    grns  = obs.get('goods_receipts', [{}])
    notes = obs.get('context_notes', [])

    if task_id == 'hard_policy_violation':
        do_step('ESCALATE', 0.0, 'POLICY_VIOLATION',
                f"Freight ${inv.get('freight_charge',0):.2f} may exceed cap. Escalating to Finance Manager.")

    if task_id == 'hard_duplicate_invoice':
        do_step('QUERY_VENDOR', 0.0, 'PENDING_CLARIFICATION',
                f"Invoice {inv['invoice_id']} found in paid ledger. Querying vendor to confirm.")

    if task_id == 'hard_manager_preapproval':
        do_step('ESCALATE', 0.0, 'MANAGER_REVIEW',
                f"Freight ${inv.get('freight_charge',0):.2f} exceeds cap. Checking manager pre-approval.")

    if task_id == 'long_invoice_dispute':
        do_step('QUERY_VENDOR', 0.0, 'PENDING_CLARIFICATION',
                f"Invoice price ${items[0].get('unit_price',0):.2f} exceeds agreed PO price. Querying vendor.")
        do_step('ESCALATE', 0.0, 'MANAGER_REVIEW',
                "Vendor acknowledged error. Escalating to Finance Manager.")

    if task_id == 'long_policy_migration':
        do_step('HOLD', 0.0, 'PENDING_CLARIFICATION',
                f"Freight ${inv.get('freight_charge',0):.2f} near policy threshold. Holding for compliance review.")

    if task_id == 'long_manager_chain':
        do_step('ESCALATE', 0.0, 'MANAGER_REVIEW',
                f"Freight ${inv.get('freight_charge',0):.2f} exceeds cap. Escalating to Finance Manager.")
        do_step('ESCALATE', 0.0, 'MANAGER_REVIEW',
                "Manager out of office. Re-escalating to VP Finance.")

    if task_id == 'long_fraud_investigation':
        do_step('QUERY_VENDOR', 0.0, 'PENDING_CLARIFICATION',
                f"Invoice {inv['invoice_id']} appears in paid ledger. Querying vendor.")
        do_step('ESCALATE', 0.0, 'MANAGER_REVIEW',
                "Vendor disputes duplicate. Escalating to manager for paid-ledger audit.")

    if task_id == 'long_audit_trail':
        do_step('HOLD', 0.0, 'PENDING_CLARIFICATION',
                "SOX audit requirement detected. Holding for compliance documentation review.")

    # ── terminal decisions ─────────────────────────────────────────────────────
    # re-fetch obs from latest step response if available
    po = pos[0] if pos else {}
    po_lines = po.get('lines', [{}])

    if task_id == 'easy_perfect_match':
        do_step('APPROVE_FULL', total, 'MATCH_CONFIRMED',
                f"Invoice ${total:,.2f}, PO and GRN match exactly. Freight within cap. Approving.")

    elif task_id == 'easy_no_po_found':
        do_step('REJECT', 0.0, 'NO_PO_FOUND',
                f"Invoice references {inv.get('po_reference','N/A')} but no matching OPEN PO found. Rejecting per Rule 5.")

    elif task_id == 'medium_quantity_shortfall':
        agreed_up = po_lines[0].get('agreed_unit_price', 0)
        recv_qty  = sum(l.get('received_quantity', 0) for g in grns for l in g.get('lines', []))
        amt = round(recv_qty * agreed_up, 2)
        do_step('APPROVE_PARTIAL', amt, 'QUANTITY_MISMATCH',
                f"GRN confirms {int(recv_qty)} of {int(items[0].get('quantity',0))} units received. Approving ${amt:,.2f} per Rule 3.")

    elif task_id == 'medium_price_discrepancy':
        inv_p = items[0].get('unit_price', 0)
        po_p  = po_lines[0].get('agreed_unit_price', 1)
        dev   = (inv_p - po_p) / po_p * 100
        do_step('REJECT', 0.0, 'PRICE_DISCREPANCY',
                f"Invoice price ${inv_p:.2f} vs PO ${po_p:.2f} — {dev:.1f}% deviation exceeds tolerance. Rejecting per Rule 4.")

    elif task_id == 'medium_split_delivery':
        recv_qty = sum(l.get('received_quantity', 0) for g in grns for l in g.get('lines', []))
        agreed_up = po_lines[0].get('agreed_unit_price', 0)
        amt = round(recv_qty * agreed_up + inv.get('freight_charge', 0), 2)
        do_step('APPROVE_FULL', amt, 'MATCH_CONFIRMED',
                f"Two GRNs confirm all {int(recv_qty)} units received across split shipments. Approving ${amt:,.2f}.")

    elif task_id == 'medium_vendor_mismatch':
        inv_v = inv.get('vendor_name', '')
        po_v  = po.get('vendor_name', '')
        do_step('REJECT', 0.0, 'VENDOR_MISMATCH',
                f'Invoice vendor "{inv_v}" does not match PO vendor "{po_v}". Rejecting per Rule 7.')

    elif task_id == 'hard_policy_violation':
        do_step('REJECT', 0.0, 'POLICY_VIOLATION',
                f"Freight ${inv.get('freight_charge',0):.2f} exceeds cap. Manager: NOT pre-approved. Rejecting per Rule 2.")

    elif task_id == 'hard_duplicate_invoice':
        do_step('REJECT', 0.0, 'DUPLICATE_INVOICE',
                f"Invoice {inv['invoice_id']} already paid per ledger. Vendor confirmed. Rejecting per Rule 6.")

    elif task_id == 'hard_partial_po_match':
        po_descs = {pl.get('description','').lower() for p in pos if p.get('status')=='OPEN' for pl in p.get('lines',[])}
        amt = round(sum(li.get('line_total',0) for li in items if li.get('description','').lower() in po_descs), 2)
        do_step('APPROVE_PARTIAL', amt, 'NO_PO_FOUND',
                f"Only PO-covered items payable (${amt:,.2f}). Unapproved items excluded per Rule 8.")

    elif task_id == 'hard_tax_discrepancy':
        tax = inv.get('tax_amount', 0)
        do_step('REJECT', 0.0, 'TAX_DISCREPANCY',
                f"Invoice includes ${tax:.2f} tax not in PO authorisation. Unapproved per Rule 8.")

    elif task_id == 'hard_currency_conversion':
        policy = obs.get('company_policy', '')
        fx_match = re.search(r'EUR 1\.00 = USD (\d+\.\d+)', policy)
        fx = float(fx_match.group(1)) if fx_match else 1.0
        inv_eur = total
        inv_usd = round(inv_eur * fx, 2)
        po_usd  = po.get('authorized_total', inv_usd)
        tol     = obs.get('price_tolerance', 0.05)
        dev     = (inv_usd - po_usd) / po_usd if po_usd > 0 else 1.0
        if dev <= tol:
            do_step('APPROVE_FULL', inv_usd, 'MATCH_CONFIRMED',
                    f"EUR {inv_eur:,.2f} × {fx:.2f} = USD {inv_usd:,.2f}. Within PO ${po_usd:,.2f}. Approving.")
        else:
            do_step('REJECT', 0.0, 'PRICE_DISCREPANCY',
                    f"EUR {inv_eur:,.2f} × {fx:.2f} = USD {inv_usd:,.2f} exceeds PO ${po_usd:,.2f} by {dev*100:.1f}%.")

    elif task_id == 'hard_manager_preapproval':
        approval = any('[MANAGER]' in n for n in notes)
        if approval:
            do_step('APPROVE_FULL', total, 'MATCH_CONFIRMED',
                    f"Freight override confirmed by Finance Manager. Approving ${total:,.2f}.")
        else:
            do_step('REJECT', 0.0, 'POLICY_VIOLATION',
                    f"Freight ${inv.get('freight_charge',0):.2f} exceeds cap. No pre-approval found.")

    elif task_id == 'hard_credit_memo':
        po_ref    = inv.get('po_reference', '')
        po_exists = any(p.get('po_number') == po_ref and p.get('status') == 'OPEN' for p in pos)
        credit_amt = abs(total)  # credit memos have negative invoice_total — approved_amount must be >= 0
        if po_exists:
            do_step('APPROVE_PARTIAL', credit_amt, 'MATCH_CONFIRMED',
                    f"Credit memo {inv['invoice_id']} has valid OPEN PO. Credit ${credit_amt:,.2f} approved per Rule 9.")
        else:
            do_step('REJECT', 0.0, 'NO_PO_FOUND',
                    f"Credit memo {inv['invoice_id']} — no valid OPEN PO. Rejecting.")

    elif task_id == 'long_invoice_dispute':
        agreed = po_lines[0].get('agreed_unit_price', 0)
        qty    = items[0].get('quantity', 0)
        correct_total = round(qty * agreed + inv.get('freight_charge', 0), 2)
        do_step('REJECT', 0.0, 'PRICE_DISCREPANCY',
                f"Invoice price ${items[0].get('unit_price',0):.2f} vs PO ${agreed:.2f}. Correct amount ${correct_total:,.2f}. Rejecting original per Rule 4.")

    elif task_id == 'long_policy_migration':
        do_step('APPROVE_FULL', total, 'MATCH_CONFIRMED',
                f"Policy update confirmed new freight cap. Freight now compliant. Three-way match confirmed. Approving ${total:,.2f}.")

    elif task_id == 'long_batch_reconciliation':
        do_step('APPROVE_FULL', total, 'MATCH_CONFIRMED',
                f"Independent three-way match confirmed in batch context. PO, GRN, invoice ${total:,.2f} all match. Approving.")

    elif task_id == 'long_manager_chain':
        do_step('APPROVE_FULL', total, 'MATCH_CONFIRMED',
                f"VP Finance pre-approval confirmed via escalation chain. Approving ${total:,.2f}.")

    elif task_id == 'long_fraud_investigation':
        do_step('REJECT', 0.0, 'DUPLICATE_INVOICE',
                f"Invoice {inv['invoice_id']} confirmed duplicate by manager audit. Vendor dispute rejected. Rejecting ${total:,.2f} per Rule 6.")

    elif task_id == 'long_audit_trail':
        po_num  = po.get('po_number', 'N/A')
        grn_id  = grns[0].get('grn_id', 'N/A') if grns else 'N/A'
        do_step('APPROVE_FULL', total, 'MATCH_CONFIRMED',
                f"SOX audit trail: PO {po_num} confirmed OPEN; GRN {grn_id} confirms receipt; Invoice ${total:,.2f} matches. Compliance cleared. Approving per Rule 1.")

    elif task_id == 'long_multi_vendor_split':
        do_step('APPROVE_PARTIAL', total, 'MATCH_CONFIRMED',
                f"First tranche: invoice ${total:,.2f} covers {int(items[0].get('quantity',0))} units per split delivery. Approving first tranche only.")

    return steps


# ── Main ───────────────────────────────────────────────────────────────────────

RUNNABLE_TASKS = [
    # easy
    'easy_perfect_match', 'easy_no_po_found',
    # medium
    'medium_quantity_shortfall', 'medium_price_discrepancy',
    'medium_split_delivery', 'medium_vendor_mismatch',
    # hard
    'hard_policy_violation', 'hard_duplicate_invoice',
    'hard_partial_po_match', 'hard_tax_discrepancy',
    'hard_currency_conversion', 'hard_manager_preapproval', 'hard_credit_memo',
    # long-horizon
    'long_invoice_dispute', 'long_policy_migration', 'long_batch_reconciliation',
    'long_manager_chain', 'long_fraud_investigation', 'long_audit_trail',
    'long_multi_vendor_split',
]

DIFFICULTY = {
    'easy_perfect_match': 'easy', 'easy_no_po_found': 'easy',
    'medium_quantity_shortfall': 'medium', 'medium_price_discrepancy': 'medium',
    'medium_split_delivery': 'medium', 'medium_vendor_mismatch': 'medium',
    'hard_policy_violation': 'hard', 'hard_duplicate_invoice': 'hard',
    'hard_partial_po_match': 'hard', 'hard_tax_discrepancy': 'hard',
    'hard_currency_conversion': 'hard', 'hard_manager_preapproval': 'hard',
    'hard_credit_memo': 'hard',
    'long_invoice_dispute': 'long-horizon', 'long_policy_migration': 'long-horizon',
    'long_batch_reconciliation': 'long-horizon', 'long_manager_chain': 'long-horizon',
    'long_fraud_investigation': 'long-horizon', 'long_audit_trail': 'long-horizon',
    'long_multi_vendor_split': 'long-horizon',
}

COLORS = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c', 'long-horizon': '#9b59b6'}


def main():
    print(f'Baseline run — {ENV}')
    print(f'Started: {datetime.datetime.now().isoformat()}')
    print('=' * 70)

    all_results = {}
    by_diff = {'easy': [], 'medium': [], 'hard': [], 'long-horizon': []}

    for task_id in RUNNABLE_TASKS:
        diff = DIFFICULTY[task_id]
        try:
            steps = run_task(task_id)
            final_score = steps[-1]['score']
            all_results[task_id] = {
                'difficulty': diff,
                'steps': steps,
                'final_score': final_score,
                'num_steps': len(steps),
            }
            by_diff[diff].append(final_score)
            step_str = ' -> '.join(f"{s['action']}({s['score']:.2f})" for s in steps)
            print(f"[{diff[:4]}] {task_id:<38} {final_score:.3f}  |  {step_str}")
        except Exception as e:
            print(f"  ERROR {task_id}: {e}")
            all_results[task_id] = {'difficulty': diff, 'steps': [], 'final_score': 0.01, 'num_steps': 0, 'error': str(e)}
            by_diff[diff].append(0.01)
        time.sleep(0.3)

    # Summary
    print('=' * 70)
    summary = {}
    for diff in ['easy', 'medium', 'hard', 'long-horizon']:
        scores = by_diff[diff]
        if scores:
            mean = sum(scores) / len(scores)
            summary[diff] = {'mean': round(mean, 4), 'scores': [round(s, 4) for s in scores], 'n': len(scores)}
            print(f"  {diff:<14}: mean={mean:.3f}  n={len(scores)}  scores={[round(s,3) for s in scores]}")
    all_scores = [v['final_score'] for v in all_results.values()]
    overall_mean = sum(all_scores) / len(all_scores)
    print(f"  {'OVERALL':<14}: mean={overall_mean:.3f}  n={len(all_scores)}")
    print('=' * 70)

    # Save JSON
    output = {
        'run_type': 'optimal_scripted_agent',
        'env_url': ENV,
        'seed': SEED,
        'timestamp': datetime.datetime.now().isoformat(),
        'summary': summary,
        'overall_mean': round(overall_mean, 4),
        'tasks': all_results,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Saved: {OUT_JSON}')

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Plot 1: Score per task (bar chart)
    ax1 = fig.add_subplot(gs[0, :])
    task_names = list(all_results.keys())
    scores     = [all_results[t]['final_score'] for t in task_names]
    bar_colors = [COLORS[all_results[t]['difficulty']] for t in task_names]
    bars = ax1.bar(range(len(task_names)), scores, color=bar_colors, alpha=0.85, edgecolor='white')
    ax1.set_xticks(range(len(task_names)))
    ax1.set_xticklabels([t.replace('_', '\n') for t in task_names], fontsize=7, rotation=0)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Final Score (0.01–0.99)')
    ax1.set_title('Optimal Agent Baseline — Score per Task', fontsize=13, fontweight='bold')
    ax1.axhline(overall_mean, color='black', linestyle='--', linewidth=1.2, label=f'Overall mean: {overall_mean:.3f}')
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.2f}', ha='center', va='bottom', fontsize=7)
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=d) for d, c in COLORS.items()]
    legend_els.append(plt.Line2D([0],[0], color='black', linestyle='--', label=f'Mean {overall_mean:.3f}'))
    ax1.legend(handles=legend_els, loc='lower right', fontsize=8)

    # Plot 2: Mean score by difficulty
    ax2 = fig.add_subplot(gs[1, 0])
    diffs  = [d for d in ['easy', 'medium', 'hard', 'long-horizon'] if by_diff[d]]
    means  = [sum(by_diff[d])/len(by_diff[d]) for d in diffs]
    colors = [COLORS[d] for d in diffs]
    ax2.bar(diffs, means, color=colors, alpha=0.85, edgecolor='white')
    for i, (d, m) in enumerate(zip(diffs, means)):
        ax2.text(i, m + 0.01, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Mean Score by Difficulty', fontsize=11, fontweight='bold')

    # Plot 3: Step-level rewards for multi-step tasks
    ax3 = fig.add_subplot(gs[1, 1])
    multi_step = {t: v for t, v in all_results.items() if v['num_steps'] > 1}
    for task_id, v in multi_step.items():
        step_scores = [s['score'] for s in v['steps']]
        label = task_id.replace('_', ' ')
        ax3.plot(range(1, len(step_scores)+1), step_scores, marker='o',
                 color=COLORS[v['difficulty']], alpha=0.7, linewidth=1.5,
                 label=label if len(label) < 22 else label[:20]+'…')
    ax3.set_xlabel('Step Number')
    ax3.set_ylabel('Score at Step')
    ax3.set_title('Per-Step Rewards — Multi-Step Tasks', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=6, loc='lower right')

    plt.suptitle(f'AP Commander — Optimal Agent Baseline  |  seed={SEED}  |  {datetime.datetime.now().strftime("%Y-%m-%d")}',
                 fontsize=12, y=1.01)
    plt.savefig(OUT_PLOT, dpi=130, bbox_inches='tight')
    print(f'Saved: {OUT_PLOT}')
    plt.close()


if __name__ == '__main__':
    main()
