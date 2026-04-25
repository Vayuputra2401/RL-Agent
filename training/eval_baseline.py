"""
eval_baseline.py — LLM baseline evaluation (no fine-tuning)
Loads MODEL_NAME in 4-bit, evaluates on all EVAL_TASKS, saves results to
runs/baselines/MODEL-DATETIME/ and uploads to HF Hub.

Usage (HF Spaces / Colab with GPU):
    MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python eval_baseline.py
    HF_TOKEN=hf_... MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct python eval_baseline.py
"""
import os, json, re, datetime, time
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ENV_URL    = 'https://pathikreet-ap-clerk-env.hf.space'
MODEL_NAME = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')
SEEDS      = [42, 99, 7]   # 3 seeds per task → mean score per task
EVAL_TASKS = [
    'easy_perfect_match', 'easy_no_po_found',
    'medium_quantity_shortfall', 'medium_price_discrepancy',
    'medium_split_delivery', 'medium_vendor_mismatch',
    'hard_policy_violation', 'hard_duplicate_invoice',
    'hard_partial_po_match', 'hard_tax_discrepancy',
    'long_invoice_dispute', 'long_policy_migration',
    'long_batch_reconciliation', 'long_manager_chain',
    'long_fraud_investigation', 'long_audit_trail',
    'long_multi_vendor_split',
]
TASK_DIFFICULTY = {
    'easy_perfect_match': 'easy',       'easy_no_po_found': 'easy',
    'medium_quantity_shortfall':'medium','medium_price_discrepancy':'medium',
    'medium_split_delivery':'medium',    'medium_vendor_mismatch':'medium',
    'hard_policy_violation':'hard',      'hard_duplicate_invoice':'hard',
    'hard_partial_po_match':'hard',      'hard_tax_discrepancy':'hard',
    'long_invoice_dispute':'long',       'long_policy_migration':'long',
    'long_batch_reconciliation':'long',  'long_manager_chain':'long',
    'long_fraud_investigation':'long',   'long_audit_trail':'long',
    'long_multi_vendor_split':'long',
}
DIFF_COLORS = {'easy': '#3fb950', 'medium': '#d29922', 'hard': '#f85149', 'long': '#a371f7'}
DIFF_ORDER  = ['easy', 'medium', 'hard', 'long']

SYSTEM_PROMPT = """You are an AI Accounts Payable Clerk. Review the invoice, PO, and GRN, then output ONLY valid JSON:
{"decision": "APPROVE_FULL"|"APPROVE_PARTIAL"|"REJECT"|"ESCALATE"|"QUERY_VENDOR",
 "approved_amount": <float>,
 "reason_code": "MATCH_CONFIRMED"|"QUANTITY_MISMATCH"|"PRICE_DISCREPANCY"|"POLICY_VIOLATION"|"NO_PO_FOUND"|"DUPLICATE_INVOICE"|"VENDOR_MISMATCH"|"TAX_DISCREPANCY"|"PENDING_CLARIFICATION"|"MANAGER_REVIEW",
 "explanation": "<cite specific $ amounts>"}"""

VALID_DECISIONS    = {'APPROVE_FULL','APPROVE_PARTIAL','REJECT','ESCALATE','QUERY_VENDOR','HOLD'}
VALID_REASON_CODES = {'MATCH_CONFIRMED','QUANTITY_MISMATCH','PRICE_DISCREPANCY','POLICY_VIOLATION',
                      'NO_PO_FOUND','DUPLICATE_INVOICE','VENDOR_MISMATCH','TAX_DISCREPANCY',
                      'PENDING_CLARIFICATION','MANAGER_REVIEW'}


def obs_to_prompt(obs):
    inv = obs['invoice']
    lines = '\n'.join(f"  {li['description']}: qty={li['quantity']}, unit_price=${li['unit_price']:.2f}"
                      for li in inv.get('line_items', []))
    pos = '\n'.join(
        f"  PO {p['po_number']} ({p['status']}) {p['vendor_name']}: " +
        ', '.join(f"{l['description']} qty={l['ordered_quantity']} @${l['agreed_unit_price']:.2f}"
                  for l in p.get('lines', []))
        for p in obs.get('purchase_orders', []))
    grns = '\n'.join(
        f"  GRN {g['grn_id']}: " + ', '.join(f"{l['description']} recv={l['received_quantity']}"
                                              for l in g.get('lines', []))
        for g in obs.get('goods_receipts', []))
    context = '\n'.join(f'  {n}' for n in obs.get('context_notes', []))
    paid = ', '.join(obs.get('paid_invoice_ids', []))
    return (f"TASK: {obs['task_name']}\n{obs['task_description']}\n\n"
            f"INVOICE {inv['invoice_id']} | {inv['vendor_name']} | ${inv['invoice_total']:,.2f}\n{lines}\n"
            f"Freight: ${inv.get('freight_charge',0):.2f}\n\n"
            f"PURCHASE ORDERS:\n{pos}\n\nGOODS RECEIPTS:\n{grns}\n"
            + (f"PAID LEDGER: {paid}\n" if paid else "")
            + (f"CONTEXT:\n{context}\n" if context else "")
            + f"\nPOLICY:\n{obs['company_policy']}\n\nOutput JSON decision.")


def parse_action(raw):
    clean = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try:
            a = json.loads(m.group())
            if (a.get('decision') in VALID_DECISIONS and
                a.get('reason_code') in VALID_REASON_CODES and
                isinstance(a.get('approved_amount'), (int, float)) and
                len(a.get('explanation', '')) > 10):
                return a, True
        except Exception:
            pass
    return {'decision': 'REJECT', 'approved_amount': 0.0,
            'reason_code': 'NO_PO_FOUND', 'explanation': 'parse error'}, False


def eval_one(model, tokenizer, task_id, seed):
    import torch
    model.eval()
    try:
        reset = requests.post(f'{ENV_URL}/reset',
                              json={'task_id': task_id, 'seed': seed}, timeout=20).json()
        obs, sid = reset['observation'], reset['session_id']
        msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': obs_to_prompt(obs)}]
        text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=250, temperature=0.1, do_sample=True)
        raw    = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action, fmt_ok = parse_action(raw)
        score = float(requests.post(f'{ENV_URL}/step',
                                    json={'session_id': sid, 'action': action},
                                    timeout=20).json()['reward']['score'])
        return score, raw[:120], action.get('decision', '?'), fmt_ok
    except Exception as e:
        print(f'    error: {e}')
        return 0.01, '', 'ERROR', False


def main():
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print('[AUTH] Logged in.')

    model_slug = MODEL_NAME.split('/')[-1].lower().replace('.', '-')
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
    run_dir = f'/app/runs/baselines/{model_slug}-{ts}'
    os.makedirs(run_dir, exist_ok=True)
    print(f'[RUN] {MODEL_NAME}  →  {run_dir}')

    print(f'[ENV] Checking {ENV_URL}...')
    h = requests.get(f'{ENV_URL}/health', timeout=30).json()
    print(f"[ENV] status={h['status']} tasks={h.get('total_tasks')}")

    print(f'[MODEL] Loading {MODEL_NAME} (4-bit NF4, no LoRA)...')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                  device_map='auto', trust_remote_code=True)
    print('[MODEL] Ready.')

    # Evaluate
    results = {}   # task_id → {scores: [], mean: float, decisions: [], fmt_rate: float}
    parse_failures = 0
    print(f'\n[EVAL] {len(EVAL_TASKS)} tasks × {len(SEEDS)} seeds = {len(EVAL_TASKS)*len(SEEDS)} episodes\n')

    for task_id in EVAL_TASKS:
        diff = TASK_DIFFICULTY[task_id]
        scores, decisions, fmts = [], [], []
        for seed in SEEDS:
            score, raw, dec, fmt_ok = eval_one(model, tokenizer, task_id, seed)
            scores.append(score)
            decisions.append(dec)
            fmts.append(fmt_ok)
            if not fmt_ok:
                parse_failures += 1
            print(f'  [{diff[:4]}] {task_id} seed={seed}: {score:.3f}  {dec}  fmt={fmt_ok}')
            print(f'    {raw[:90]}')
            time.sleep(0.2)
        results[task_id] = {
            'difficulty': diff,
            'scores':     [round(s, 4) for s in scores],
            'mean':       round(sum(scores) / len(scores), 4),
            'decisions':  decisions,
            'fmt_rate':   round(sum(fmts) / len(fmts), 3),
        }

    # Summary
    print('\n' + '='*70)
    by_diff = {}
    for tid, v in results.items():
        by_diff.setdefault(v['difficulty'], []).append(v['mean'])
    for diff in DIFF_ORDER:
        ms = by_diff.get(diff, [])
        if ms:
            print(f"  {diff:<8}: mean={sum(ms)/len(ms):.3f}  tasks={[round(m,3) for m in ms]}")
    all_means = [v['mean'] for v in results.values()]
    overall = sum(all_means) / len(all_means)
    print(f"  overall : mean={overall:.3f}  parse_failures={parse_failures}/{len(EVAL_TASKS)*len(SEEDS)}")
    print('='*70)

    # Save JSON
    output = {
        'run_type':       'llm_baseline_no_finetuning',
        'model':          MODEL_NAME,
        'quantization':   '4-bit NF4 (BitsAndBytes)',
        'lora':           None,
        'timestamp':      datetime.datetime.now().isoformat(),
        'run_dir':        run_dir,
        'env_url':        ENV_URL,
        'seeds':          SEEDS,
        'eval_tasks':     EVAL_TASKS,
        'overall_mean':   round(overall, 4),
        'parse_failures': parse_failures,
        'tasks':          results,
        'by_difficulty':  {d: round(sum(ms)/len(ms), 4) for d, ms in by_diff.items()},
    }
    json_path = os.path.join(run_dir, 'baseline_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'[SAVED] {json_path}')

    # ── Plots ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, max(9, len(results) * 0.5 + 2)))
    fig.patch.set_facecolor('#0d1117')
    gs  = fig.add_gridspec(1, 2, wspace=0.30)

    def _dark(ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#21262d', linewidth=0.7)
        ax.set_axisbelow(True)
        if title:  ax.set_title(title, color='#e6edf3', fontsize=11, fontweight='bold', pad=8)
        if xlabel: ax.set_xlabel(xlabel, color='#8b949e', fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, color='#8b949e', fontsize=8)

    # Panel 1: Per-task mean score (horizontal bar), ordered by difficulty
    ax1 = fig.add_subplot(gs[0, 0])
    tasks  = sorted(results.keys(),
                    key=lambda t: (DIFF_ORDER.index(results[t]['difficulty']), t))
    means  = [results[t]['mean'] for t in tasks]
    colors = [DIFF_COLORS[results[t]['difficulty']] for t in tasks]
    short  = [t.replace('easy_','').replace('medium_','').replace('hard_','').replace('long_','')
               .replace('_',' ').title() for t in tasks]
    yp = range(len(tasks))
    bars = ax1.barh(list(yp), means, color=colors, alpha=0.85, edgecolor='#0d1117')
    ax1.set_yticks(list(yp))
    ax1.set_yticklabels(short, fontsize=8)
    ax1.set_xlim(0, 1.05)
    ax1.axvline(overall, color='#f78166', linestyle='--', linewidth=1.2,
                label=f'Overall mean: {overall:.3f}')
    ax1.axvline(0.5, color='#484f58', linestyle=':', linewidth=1)
    for i, m in enumerate(means):
        ax1.text(m + 0.01, i, f'{m:.3f}', va='center', color='#c9d1d9', fontsize=8)
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=d) for d, c in DIFF_COLORS.items()]
    legend_els.append(plt.Line2D([0],[0], color='#f78166', linestyle='--',
                                  label=f'Mean {overall:.3f}'))
    ax1.legend(handles=legend_els, fontsize=8, facecolor='#161b22',
               edgecolor='#30363d', labelcolor='#c9d1d9', loc='lower right')
    _dark(ax1, f'Untrained Baseline — Per-Task Mean Score ({len(SEEDS)} seeds)',
          xlabel='Mean Score  [0.01 – 0.99]', ylabel='Task')

    # Panel 2: Mean by difficulty
    ax2 = fig.add_subplot(gs[0, 1])
    diffs = [d for d in DIFF_ORDER if d in by_diff]
    d_means = [sum(by_diff.get(d, [0])) / max(1, len(by_diff.get(d, [0]))) for d in diffs]
    d_colors = [DIFF_COLORS[d] for d in diffs]
    bars2 = ax2.bar(diffs, d_means, color=d_colors, alpha=0.85, edgecolor='#0d1117', width=0.5)
    for i, (d, m) in enumerate(zip(diffs, d_means)):
        ax2.text(i, m + 0.02, f'{m:.3f}', ha='center', color='#c9d1d9', fontsize=10,
                 fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.axhline(overall, color='#f78166', linestyle='--', linewidth=1,
                label=f'Overall {overall:.3f}')
    ax2.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    _dark(ax2, 'Mean Score by Difficulty Tier',
          xlabel='Difficulty Tier', ylabel='Mean Score  [0.01 – 0.99]')

    model_short = MODEL_NAME.split('/')[-1]
    fig.suptitle(
        f'{model_short} — Untrained Baseline  |  4-bit NF4  |  {len(SEEDS)} seeds  |  '
        f'{len(EVAL_TASKS)} tasks  |  overall={overall:.3f}  |  '
        f'{datetime.datetime.now().strftime("%Y-%m-%d")}',
        color='#e6edf3', fontsize=10, y=1.01
    )
    fig.text(0.5, 0.0,
             'Baseline = model loaded 4-bit NF4 with no fine-tuning. '
             'Score range [0.01, 0.99]. Tasks: easy (green), medium (yellow), hard (red), long-horizon (purple).',
             ha='center', color='#8b949e', fontsize=7, style='italic')
    plot_path = os.path.join(run_dir, 'baseline_plot.png')
    plt.savefig(plot_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[SAVED] {plot_path}')

    # Upload run folder to HF Space repo
    repo_run_path = run_dir.replace('/app/', '')
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=run_dir,
            path_in_repo=repo_run_path,
            repo_id='Pathikreet/ap-commander-training',
            repo_type='space',
            commit_message=f'Baseline: {model_short} untrained {ts}',
        )
        print(f'[UPLOAD] {repo_run_path} → Pathikreet/ap-commander-training')
    except Exception as e:
        print(f'[UPLOAD] skipped: {e}')

    print(f'\n[DONE] Results in {run_dir}')
    return output


if __name__ == '__main__':
    main()
