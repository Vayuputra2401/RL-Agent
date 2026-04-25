"""
AP Commander — GRPO Training Script
Tracks: overall reward, per-component rewards, decision distribution,
        format compliance, env errors, sample generations, reward curve.
"""
import os, json, re, random, time, datetime, collections
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ENV_URL         = 'https://pathikreet-ap-clerk-env.hf.space'
MODEL_NAME      = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')
NUM_EPOCHS      = int(os.environ.get('NUM_EPOCHS', '3'))
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', '8'))
LOG_SAMPLES_EVERY = 20   # print a sample generation every N reward calls

SYSTEM_PROMPT = """You are an AI Accounts Payable Clerk. Review the invoice, PO, and GRN, then output ONLY valid JSON:
{"decision": "APPROVE_FULL"|"APPROVE_PARTIAL"|"REJECT"|"ESCALATE"|"QUERY_VENDOR",
 "approved_amount": <float>,
 "reason_code": "MATCH_CONFIRMED"|"QUANTITY_MISMATCH"|"PRICE_DISCREPANCY"|"POLICY_VIOLATION"|"NO_PO_FOUND"|"DUPLICATE_INVOICE"|"VENDOR_MISMATCH"|"TAX_DISCREPANCY"|"PENDING_CLARIFICATION"|"MANAGER_REVIEW",
 "explanation": "<cite specific $ amounts>"}"""

TRAIN_TASKS = [
    'easy_perfect_match', 'easy_no_po_found',
    'medium_quantity_shortfall', 'medium_price_discrepancy',
    'medium_split_delivery', 'medium_vendor_mismatch',
    'hard_policy_violation', 'hard_duplicate_invoice',
    'hard_partial_po_match', 'hard_tax_discrepancy',
]
EVAL_TASKS = [
    'easy_perfect_match', 'easy_no_po_found',
    'medium_quantity_shortfall', 'medium_price_discrepancy',
    'medium_split_delivery', 'medium_vendor_mismatch',
    'hard_policy_violation', 'hard_duplicate_invoice',
    'hard_partial_po_match', 'hard_tax_discrepancy',
]

VALID_DECISIONS   = {'APPROVE_FULL','APPROVE_PARTIAL','REJECT','ESCALATE','QUERY_VENDOR','HOLD'}
VALID_REASON_CODES = {'MATCH_CONFIRMED','QUANTITY_MISMATCH','PRICE_DISCREPANCY','POLICY_VIOLATION',
                      'NO_PO_FOUND','DUPLICATE_INVOICE','VENDOR_MISMATCH','TAX_DISCREPANCY',
                      'PENDING_CLARIFICATION','MANAGER_REVIEW'}

# Task difficulty map used by curriculum sampler
_TASK_DIFFICULTY = {
    'easy_perfect_match': 'easy',   'easy_no_po_found': 'easy',
    'medium_quantity_shortfall': 'medium', 'medium_price_discrepancy': 'medium',
    'medium_split_delivery': 'medium',    'medium_vendor_mismatch': 'medium',
    'hard_policy_violation': 'hard',      'hard_duplicate_invoice': 'hard',
    'hard_partial_po_match': 'hard',      'hard_tax_discrepancy': 'hard',
}
_DIFFICULTY_ORDER  = ['easy', 'medium', 'hard']
_UNLOCK_THRESHOLDS = {'easy': 0.70, 'medium': 0.65}


# ── Curriculum sampler ──────────────────────────────────────────────────────────

class CurriculumSampler:
    """
    Tracks per-difficulty running mean and unlocks harder tasks once thresholds
    are met. Used both for building the training dataset and for gating tasks in
    the reward function so early training stays on easier tasks.
    """
    def __init__(self):
        self._rewards:  dict = collections.defaultdict(list)  # task_id → [rewards]
        self.unlocked:  set  = {'easy'}

    def record(self, task_id: str, reward: float):
        self._rewards[task_id].append(reward)
        self._try_unlock()

    def mean_for_difficulty(self, diff: str) -> float:
        vals = []
        for tid, d in _TASK_DIFFICULTY.items():
            if d == diff:
                vals.extend(self._rewards.get(tid, []))
        return sum(vals) / len(vals) if vals else 0.0

    def _try_unlock(self):
        for i, diff in enumerate(_DIFFICULTY_ORDER[:-1]):
            if diff in self.unlocked:
                m = self.mean_for_difficulty(diff)
                if m >= _UNLOCK_THRESHOLDS.get(diff, 0.70):
                    nxt = _DIFFICULTY_ORDER[i + 1]
                    if nxt not in self.unlocked:
                        self.unlocked.add(nxt)
                        print(f'\n[CURRICULUM] Unlocked {nxt}! mean({diff})={m:.3f} '
                              f'>= threshold {_UNLOCK_THRESHOLDS[diff]}')

    def gate_task(self, task_id: str) -> str:
        """If task's difficulty is not yet unlocked, return easiest unlocked task."""
        if _TASK_DIFFICULTY.get(task_id, 'easy') in self.unlocked:
            return task_id
        easiest = [t for t, d in _TASK_DIFFICULTY.items() if d == 'easy']
        return random.choice(easiest)

    def build_dataset_tasks(self) -> list:
        """
        Curriculum-weighted task list:
          easy  → 10 seeds  (always included)
          medium → 5 seeds  (if unlocked)
          hard   → 2 seeds  (if unlocked)
        Returns list of (task_id, seed) pairs.
        """
        rows = []
        seeds_per_diff = {'easy': 10, 'medium': 5, 'hard': 2}
        for task_id, diff in _TASK_DIFFICULTY.items():
            if diff in self.unlocked:
                n = seeds_per_diff[diff]
                rows.extend([(task_id, s) for s in range(1, n + 1)])
        return rows

    def status_line(self) -> str:
        parts = []
        for d in _DIFFICULTY_ORDER:
            m = self.mean_for_difficulty(d)
            unlk = '✓' if d in self.unlocked else '✗'
            parts.append(f'{d}={m:.2f}{unlk}')
        return ' | '.join(parts)


CURRICULUM = CurriculumSampler()


# ── Per-step greedy follow-up policy ───────────────────────────────────────────

def _greedy_followup(obs_dict: dict) -> dict:
    """
    Scripted policy for intermediate follow-up steps (used in multi-step rollouts).
    Reads context_notes added by the environment after ESCALATE/QUERY_VENDOR/HOLD
    and picks the most appropriate next terminal action.
    """
    notes = ' '.join(obs_dict.get('context_notes', [])).lower()
    total = abs(float(obs_dict.get('invoice', {}).get('invoice_total', 0) or 0))

    # Manager / VP approved → APPROVE_FULL
    if any(k in notes for k in ('manager approved', 'vp approved', 'cfo approved',
                                 'pre-approved', 'pre-approv', 'approved by')):
        return {'decision': 'APPROVE_FULL', 'approved_amount': total,
                'reason_code': 'MATCH_CONFIRMED',
                'explanation': f'Approval confirmed via escalation chain. Approving ${total:.2f}.'}

    # Compliance cleared → APPROVE_FULL
    if 'compliance' in notes and any(k in notes for k in ('cleared', 'approved', 'pass')):
        return {'decision': 'APPROVE_FULL', 'approved_amount': total,
                'reason_code': 'MATCH_CONFIRMED',
                'explanation': f'Compliance review cleared. Approving ${total:.2f}.'}

    # Fraudulent / duplicate / deny → REJECT
    if any(k in notes for k in ('fraudulent', 'duplicate', 'already paid', 'deny',
                                 'invalid', 'false claim')):
        return {'decision': 'REJECT', 'approved_amount': 0.0,
                'reason_code': 'DUPLICATE_INVOICE',
                'explanation': 'Vendor response or audit confirms fraud/duplicate. Rejecting.'}

    # Compliance flagged / SOX violation → REJECT
    if any(k in notes for k in ('flagged', 'violation', 'sox', 'gdpr', 'non-compliant')):
        return {'decision': 'REJECT', 'approved_amount': 0.0,
                'reason_code': 'POLICY_VIOLATION',
                'explanation': 'Compliance review flagged a violation. Rejecting.'}

    # Confused vendor / ambiguous → ESCALATE
    if any(k in notes for k in ('confused', 'unclear', 'unable to confirm')):
        return {'decision': 'ESCALATE', 'approved_amount': 0.0,
                'reason_code': 'MANAGER_REVIEW',
                'explanation': 'Vendor response ambiguous. Escalating to manager.'}

    # Default: safe rejection
    return {'decision': 'REJECT', 'approved_amount': 0.0,
            'reason_code': 'PENDING_CLARIFICATION',
            'explanation': 'Could not resolve after investigation. Rejecting for safety.'}


# ── Metrics tracker ────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.step            = 0
        self.reward_history  = []          # (step, mean_reward)
        self.decision_counts = collections.Counter()
        self.parse_failures  = 0
        self.env_errors      = 0
        self.format_scores   = []
        self.reward_by_task  = collections.defaultdict(list)
        self.total_calls     = 0
        self._start_time     = time.time()

    def log_step(self, rewards, decisions, format_ok_list, task_ids, errors):
        self.step += 1
        self.total_calls += len(rewards)
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        self.reward_history.append((self.step, mean_r))
        for d in decisions:
            self.decision_counts[d] += 1
        for ok in format_ok_list:
            self.format_scores.append(1.0 if ok else 0.0)
        for tid, r in zip(task_ids, rewards):
            self.reward_by_task[tid].append(r)
        self.env_errors += errors
        self._flush_live()

    def _flush_live(self):
        recent = self.reward_history[-20:]
        recent_mean = sum(r for _, r in recent) / len(recent) if recent else 0.0
        fmt_rate = sum(self.format_scores) / len(self.format_scores) if self.format_scores else 0.0
        task_means = {t: round(sum(v)/len(v), 3) for t, v in self.reward_by_task.items()}
        elapsed = (time.time() - self._start_time) / 60
        payload = {
            'step':           self.step,
            'total_calls':    self.total_calls,
            'recent_mean':    round(recent_mean, 4),
            'format_rate':    round(fmt_rate, 4),
            'parse_failures': self.parse_failures,
            'env_errors':     self.env_errors,
            'elapsed_min':    round(elapsed, 1),
            'reward_history': [{'step': s, 'reward': r} for s, r in self.reward_history],
            'decision_counts': dict(self.decision_counts),
            'task_means':     task_means,
        }
        try:
            with open('/app/metrics_live.json', 'w') as f:
                json.dump(payload, f)
        except Exception:
            pass

    def print_summary(self):
        recent = self.reward_history[-10:] if self.reward_history else []
        recent_mean = sum(r for _, r in recent) / len(recent) if recent else 0.0
        fmt_rate = sum(self.format_scores) / len(self.format_scores) if self.format_scores else 0.0
        print(f'\n[METRICS] step={self.step} | recent_reward={recent_mean:.3f} | '
              f'format_ok={fmt_rate:.1%} | parse_fails={self.parse_failures} | '
              f'env_errors={self.env_errors} | total_calls={self.total_calls}')
        top_decisions = self.decision_counts.most_common(5)
        print(f'[METRICS] decisions: {dict(top_decisions)}')
        if self.reward_by_task:
            task_means = {t: round(sum(v)/len(v), 3) for t, v in self.reward_by_task.items()}
            print(f'[METRICS] per_task_reward: {task_means}')

    def save_reward_curve(self, path='/app/reward_curve.png'):
        if not self.reward_history:
            return
        steps   = [s for s, _ in self.reward_history]
        rewards = [r for _, r in self.reward_history]

        # Smooth with rolling window
        window = max(1, len(rewards) // 10)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Raw + smoothed reward curve
        axes[0].plot(steps, rewards, alpha=0.3, color='#3498db', label='Per-step reward')
        axes[0].plot(steps[window-1:], smoothed, color='#2980b9', linewidth=2, label=f'Smoothed (w={window})')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Mean Batch Reward')
        axes[0].set_title('Reward Curve During Training', fontweight='bold')
        axes[0].set_ylim(0, 1.0)
        axes[0].legend()
        axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.4)

        # Decision distribution pie
        if self.decision_counts:
            labels = list(self.decision_counts.keys())
            counts = list(self.decision_counts.values())
            colors = ['#2ecc71','#e74c3c','#f39c12','#9b59b6','#3498db','#1abc9c']
            axes[1].pie(counts, labels=labels, autopct='%1.0f%%',
                        colors=colors[:len(labels)], startangle=90)
            axes[1].set_title('Decision Distribution During Training', fontweight='bold')

        plt.suptitle(f'AP Commander GRPO — Training Diagnostics | {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight')
        print(f'[METRICS] Saved reward curve: {path}')
        plt.close()


METRICS = Metrics()

# ── Helpers ────────────────────────────────────────────────────────────────────

def obs_to_prompt(obs: dict) -> str:
    inv = obs['invoice']
    lines = '\n'.join(
        f"  {li['description']}: qty={li['quantity']}, unit_price=${li['unit_price']:.2f}"
        for li in inv.get('line_items', [])
    )
    pos = '\n'.join(
        f"  PO {p['po_number']} ({p['status']}) {p['vendor_name']}: " +
        ', '.join(f"{l['description']} qty={l['ordered_quantity']} @${l['agreed_unit_price']:.2f}"
                  for l in p.get('lines', []))
        for p in obs.get('purchase_orders', [])
    )
    grns = '\n'.join(
        f"  GRN {g['grn_id']} (PO {g['po_number']}): " +
        ', '.join(f"{l['description']} recv={l['received_quantity']}"
                  for l in g.get('lines', []))
        for g in obs.get('goods_receipts', [])
    )
    context = '\n'.join(f'  {n}' for n in obs.get('context_notes', []))
    paid = ', '.join(obs.get('paid_invoice_ids', []))
    return (
        f"TASK: {obs['task_name']}\n{obs['task_description']}\n\n"
        f"INVOICE {inv['invoice_id']} | {inv['vendor_name']} | ${inv['invoice_total']:,.2f}\n{lines}\n"
        f"Freight: ${inv.get('freight_charge',0):.2f}\n\n"
        f"PURCHASE ORDERS:\n{pos}\n\nGOODS RECEIPTS:\n{grns}\n"
        + (f"PAID LEDGER: {paid}\n" if paid else "")
        + (f"CONTEXT:\n{context}\n" if context else "")
        + f"\nPOLICY:\n{obs['company_policy']}\n\nOutput JSON decision."
    )


def parse_action(raw: str) -> tuple[dict, bool]:
    """Returns (action_dict, format_ok). format_ok=False means parse failed."""
    clean = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try:
            action = json.loads(m.group())
            # Validate required fields and enum values
            if (action.get('decision') in VALID_DECISIONS and
                action.get('reason_code') in VALID_REASON_CODES and
                isinstance(action.get('approved_amount'), (int, float)) and
                isinstance(action.get('explanation'), str) and
                len(action.get('explanation', '')) > 10):
                return action, True
        except Exception:
            pass
    METRICS.parse_failures += 1
    return {'decision': 'REJECT', 'approved_amount': 0.0,
            'reason_code': 'NO_PO_FOUND', 'explanation': 'parse error fallback'}, False


def run_episode(task_id: str, action_json: dict, seed=None) -> float:
    try:
        r = requests.post(f'{ENV_URL}/reset',
                          json={'task_id': task_id, 'seed': seed}, timeout=20)
        r.raise_for_status()
        data = r.json()
        step_r = requests.post(f'{ENV_URL}/step',
                               json={'session_id': data['session_id'], 'action': action_json},
                               timeout=20)
        step_r.raise_for_status()
        return float(step_r.json()['reward']['score'])
    except Exception:
        return 0.01


def run_episode_accumulated(task_id: str, first_action: dict, seed=None,
                             discount: float = 0.9, max_steps: int = 5) -> float:
    """
    Run a full multi-step episode accumulating discounted per-step rewards.
    Model's first action starts the episode; _greedy_followup() handles
    subsequent steps so multi-step sequences earn full accumulated credit.
    E.g. QUERY_VENDOR→REJECT = 0.01 + 0.9*0.99 = 0.901 > shortcut REJECT = ~0.4
    """
    try:
        r = requests.post(f'{ENV_URL}/reset',
                          json={'task_id': task_id, 'seed': seed}, timeout=20)
        r.raise_for_status()
        session_id = r.json()['session_id']
        action = first_action
        total  = 0.0
        for step_n in range(max_steps):
            step_r = requests.post(f'{ENV_URL}/step',
                                   json={'session_id': session_id, 'action': action},
                                   timeout=20)
            step_r.raise_for_status()
            result  = step_r.json()
            r_score = float(result['reward']['score'])
            done    = result['done']
            total  += (discount ** step_n) * r_score
            if done:
                break
            action = _greedy_followup(result['observation'])
        return min(0.99, max(0.01, total))
    except Exception:
        return 0.01


# ── Two independent reward functions (guide: use multiple, not one) ─────────────

def env_reward_fn(completions, task_id=None, seed=None, **kwargs):
    """
    Environment reward: accumulated discounted per-step reward from AP Commander.
    Curriculum gating redirects locked tasks to easier ones during early training.
    """
    task_ids = task_id if task_id is not None else ['easy_perfect_match'] * len(completions)
    seeds    = seed    if seed    is not None else [random.randint(1, 999)] * len(completions)

    rewards, decisions, format_ok_list, errors = [], [], [], 0
    for completion, tid, s in zip(completions, task_ids, seeds):
        # Curriculum gate: redirect locked task to easiest unlocked
        gated_tid = CURRICULUM.gate_task(tid)
        if gated_tid != tid:
            print(f'[CURRICULUM] gate {tid} → {gated_tid}')

        action, fmt_ok = parse_action(completion)
        try:
            score = run_episode_accumulated(gated_tid, action, seed=int(s))
        except Exception:
            score = 0.01
            errors += 1
        rewards.append(score)
        decisions.append(action.get('decision', 'UNKNOWN'))
        format_ok_list.append(fmt_ok)
        CURRICULUM.record(gated_tid, score)

        if METRICS.total_calls % LOG_SAMPLES_EVERY == 0:
            gated_note = f'→{gated_tid}' if gated_tid != tid else ''
            print(f'\n[SAMPLE] task={tid}{gated_note} seed={s} fmt={fmt_ok} score={score:.3f}')
            print(f'  {action.get("decision")} ${action.get("approved_amount")} '
                  f'{action.get("reason_code")}')
            print(f'  {str(action.get("explanation",""))[:100]}')
            print(f'  curriculum: {CURRICULUM.status_line()}')

    METRICS.log_step(rewards, decisions, format_ok_list, list(task_ids), errors)
    if METRICS.step % 5 == 0:
        METRICS.print_summary()
        print(f'[CURRICULUM] {CURRICULUM.status_line()}')
    return rewards


def format_reward_fn(completions, **kwargs):
    """Format reward: +0.05 if valid JSON with correct fields, -0.05 otherwise."""
    results = []
    for completion in completions:
        _, ok = parse_action(completion)
        results.append(0.05 if ok else -0.05)
    return results


# ── Eval helper ────────────────────────────────────────────────────────────────

def eval_task(model, tokenizer, task_id: str, seed: int = 99) -> float:
    import torch
    model.eval()
    try:
        reset = requests.post(f'{ENV_URL}/reset', json={'task_id': task_id, 'seed': seed}, timeout=20).json()
        obs, session_id = reset['observation'], reset['session_id']
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': obs_to_prompt(obs)}]
        text    = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs  = tokenizer(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=250, temperature=0.1, do_sample=True)
        raw    = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action, fmt_ok = parse_action(raw)
        score  = float(requests.post(f'{ENV_URL}/step',
                                     json={'session_id': session_id, 'action': action},
                                     timeout=20).json()['reward']['score'])
        print(f'    output: {raw[:120].strip()}')
        return score
    except Exception as e:
        print(f'    eval error: {e}')
        return 0.01


# ── Main ───────────────────────────────────────────────────────────────────────

def _make_run_dir() -> str:
    """Create timestamped run directory under /app/runs/grpo/MODEL-NEpoch-DATETIME."""
    model_slug = MODEL_NAME.split('/')[-1].lower().replace('.', '-')
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
    run_dir = f'/app/runs/grpo/{model_slug}-{NUM_EPOCHS}ep-{ts}'
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    # Authenticate with HF Hub if token provided (needed for gated models like Llama-3)
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print('[AUTH] Logged in to HF Hub.')
    else:
        print('[AUTH] No HF_TOKEN set — using public models only (Qwen recommended).')

    # All run artifacts go into this timestamped dir — never overwrite a previous run
    RUN_DIR = _make_run_dir()
    print(f'[RUN] Artifacts → {RUN_DIR}')

    print(f'[ENV] Checking {ENV_URL}...')
    h = requests.get(f'{ENV_URL}/health', timeout=30).json()
    print(f"[ENV] status={h['status']} tasks={h.get('total_tasks')}")

    print(f'[MODEL] Loading {MODEL_NAME}...')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_dropout=0, bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Baseline eval (before training)
    print('\n[BASELINE] Before training:')
    baseline = {}
    for t in EVAL_TASKS:
        s = eval_task(model, tokenizer, t)
        baseline[t] = s
        print(f'  {t}: {s:.3f}')
    print(f'  Mean: {sum(baseline.values())/len(baseline):.3f}')
    model.train()

    # Dataset contains ALL 10 tasks × 5 seeds = 50 prompts (same as Run 1).
    # gate_task() in env_reward_fn handles curriculum redirection at score time:
    # locked tasks get redirected to easy → model still trains, just on easier logic.
    # As curriculum unlocks, redirection stops and model gets real hard task rewards.
    print('\n[DATASET] Building prompts (all 10 tasks × 5 seeds = 50)...')
    task_seed_pairs = [(tid, s) for tid in TRAIN_TASKS for s in range(1, 6)]
    rows = []
    for task_id, seed in task_seed_pairs:
        try:
            reset = requests.post(f'{ENV_URL}/reset', json={'task_id': task_id, 'seed': seed}, timeout=20).json()
            obs   = reset['observation']
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user',   'content': obs_to_prompt(obs)}]
            rows.append({
                'prompt':   tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                'task_id':  task_id,
                'seed':     seed,
            })
        except Exception as e:
            print(f'  skip {task_id} seed={seed}: {e}')

    dataset = Dataset.from_list(rows)
    print(f'[DATASET] {len(dataset)} samples across {len(TRAIN_TASKS)} tasks | curriculum: {CURRICULUM.status_line()}')

    # Train
    print(f'\n[TRAIN] {NUM_EPOCHS} epochs | {NUM_GENERATIONS} generations/prompt | {len(dataset)} samples')
    model.train()
    # generation_batch_size = per_device_train_batch_size (TRL default).
    # TRL requires: generation_batch_size % num_generations == 0.
    # Simplest fix: set per_device_train_batch_size = num_generations.
    config = GRPOConfig(
        output_dir            = './ap_commander_grpo',
        num_train_epochs      = NUM_EPOCHS,
        per_device_train_batch_size = NUM_GENERATIONS,
        num_generations       = NUM_GENERATIONS,
        gradient_accumulation_steps = 1,
        learning_rate         = 2e-5,
        max_completion_length = 250,
        temperature           = 0.9,
        logging_steps         = 1,
        save_steps            = 999,
        report_to             = 'none',
        remove_unused_columns = False,
    )
    # Two independent reward functions (guide: use multiple, not one combined signal)
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[env_reward_fn, format_reward_fn],
        args=config, train_dataset=dataset,
    )
    result = trainer.train()
    print(f'\n[TRAIN] Done. Loss: {result.training_loss:.4f}')

    METRICS.print_summary()
    METRICS.save_reward_curve(os.path.join(RUN_DIR, 'reward_curve.png'))

    # Save LoRA adapters (guide point 16: save adapters directly, do NOT merge 4-bit naively)
    adapter_dir = os.path.join(RUN_DIR, 'adapter')
    print(f'[SAVE] Saving LoRA adapters to {adapter_dir}...')
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Upload adapter to HF Hub as a model repo
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=adapter_dir,
            repo_id='Pathikreet/ap-commander-adapter',
            repo_type='model',
            commit_message=f'GRPO {datetime.datetime.now().strftime("%Y-%m-%d")} — {MODEL_NAME} {NUM_EPOCHS}ep',
        )
        print('[SAVE] Adapter pushed to HF Hub: Pathikreet/ap-commander-adapter')
    except Exception as e:
        print(f'[SAVE] HF Hub upload skipped: {e}')

    # Post-training eval (all 10 tasks)
    print('\n[POST-EVAL] After training:')
    post = {}
    model.eval()
    for t in EVAL_TASKS:
        s = eval_task(model, tokenizer, t)
        post[t] = s
        print(f'  {t}: {s:.3f}')
    print(f'  Mean: {sum(post.values())/len(post):.3f}')

    print('\n[COMPARE]')
    for t in EVAL_TASKS:
        d = post[t] - baseline[t]
        sym = '+' if d >= 0 else ''
        print(f'  {t:<35} {baseline[t]:.3f} -> {post[t]:.3f}  ({sym}{d:.3f})')

    # ── 4-panel results figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0d1117')
    gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    def _dark(ax, title=''):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#21262d', linewidth=0.7)
        ax.set_axisbelow(True)
        if title: ax.set_title(title, color='#e6edf3', fontsize=10, fontweight='bold', pad=8)

    # Panel 1: Before / After eval bars
    ax1   = fig.add_subplot(gs[0, 0])
    tasks = list(EVAL_TASKS)
    short = [t.replace('easy_','').replace('medium_','').replace('hard_','')
              .replace('_',' ').title() for t in tasks]
    xp    = np.arange(len(tasks))
    ax1.bar(xp - 0.2, [baseline[t] for t in tasks], 0.35,
            label='Before GRPO', color='#f85149', alpha=0.85)
    ax1.bar(xp + 0.2, [post[t]     for t in tasks], 0.35,
            label='After GRPO',  color='#3fb950', alpha=0.85)
    ax1.set_xticks(xp); ax1.set_xticklabels(short, rotation=35, ha='right', fontsize=7)
    ax1.set_ylim(0, 1.05); ax1.axhline(0.5, color='#484f58', linestyle='--', alpha=0.6)
    ax1.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    _dark(ax1, f'Before vs After — {NUM_EPOCHS} Epochs GRPO')

    # Panel 2: Per-task training mean (from live metrics)
    ax2 = fig.add_subplot(gs[0, 1])
    task_means = {t: round(sum(v)/len(v), 3) for t, v in METRICS.reward_by_task.items()}
    if task_means:
        tm_tasks  = list(task_means.keys())
        tm_scores = list(task_means.values())
        tm_short  = [t.replace('easy_','').replace('medium_','').replace('hard_','')
                      .replace('_',' ').title() for t in tm_tasks]
        colors    = ['#3fb950' if s >= 0.7 else '#d29922' if s >= 0.4 else '#f85149'
                     for s in tm_scores]
        yp = range(len(tm_tasks))
        ax2.barh(yp, tm_scores, color=colors, alpha=0.85, edgecolor='#0d1117')
        ax2.set_yticks(list(yp)); ax2.set_yticklabels(tm_short, fontsize=7)
        ax2.set_xlim(0, 1.05)
        ax2.axvline(0.7, color='#3fb950', linestyle='--', linewidth=1, alpha=0.5)
        for i, s in enumerate(tm_scores):
            ax2.text(s + 0.01, i, f'{s:.2f}', va='center', color='#c9d1d9', fontsize=7)
    _dark(ax2, 'Per-Task Training Mean (all seeds)')

    # Panel 3: Decision distribution
    ax3 = fig.add_subplot(gs[1, 0])
    dc  = dict(METRICS.decision_counts)
    if dc:
        colors3 = ['#3fb950','#f85149','#d29922','#a371f7','#58a6ff','#39d353']
        wedges, _, autos = ax3.pie(list(dc.values()), labels=None,
                                   autopct='%1.0f%%', colors=colors3[:len(dc)],
                                   startangle=90, pctdistance=0.75,
                                   wedgeprops=dict(edgecolor='#0d1117', linewidth=1.5))
        for at in autos: at.set_color('#0d1117'); at.set_fontsize(8); at.set_fontweight('bold')
        ax3.legend(list(dc.keys()), loc='lower center', bbox_to_anchor=(0.5, -0.15),
                   ncol=3, fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax3.set_facecolor('#161b22'); fig.patch.set_facecolor('#0d1117')
    ax3.set_title('Decision Distribution During Training', color='#e6edf3',
                  fontsize=10, fontweight='bold', pad=8)

    # Panel 4: Reward curve
    ax4 = fig.add_subplot(gs[1, 1])
    if METRICS.reward_history:
        steps   = [s for s, _ in METRICS.reward_history]
        rewards = [r for _, r in METRICS.reward_history]
        ax4.plot(steps, rewards, color='#58a6ff', alpha=0.30, linewidth=1)
        if len(rewards) >= 5:
            w  = max(3, len(rewards) // 15)
            sm = np.convolve(rewards, np.ones(w)/w, mode='valid')
            ax4.plot(steps[w-1:], sm, color='#79c0ff', linewidth=2, label=f'Smooth (w={w})')
        mean_r = sum(rewards[-20:]) / min(20, len(rewards))
        ax4.axhline(mean_r, color='#f78166', linestyle='--', linewidth=1,
                    label=f'Recent mean: {mean_r:.3f}')
        ax4.set_ylim(0, 1.05)
        ax4.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        ax4.set_xlabel('Training Step', color='#c9d1d9', fontsize=8)
    _dark(ax4, 'Reward Curve')

    _fmt_rate = sum(METRICS.format_scores) / max(1, len(METRICS.format_scores))
    fig.suptitle(
        f'AP Commander GRPO — {MODEL_NAME} | {NUM_EPOCHS} epochs | '
        f'{NUM_GENERATIONS} gen | format={_fmt_rate:.1%} | '
        f'parse_fails={METRICS.parse_failures} | {datetime.datetime.now().strftime("%Y-%m-%d")}',
        color='#e6edf3', fontsize=9, y=0.98
    )
    results_png = os.path.join(RUN_DIR, 'results.png')
    plt.savefig(results_png, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[DONE] Saved {results_png}')

    # Save JSON
    fmt_rate = sum(METRICS.format_scores) / max(1, len(METRICS.format_scores))
    output = {
        'timestamp':       datetime.datetime.now().isoformat(),
        'run_dir':         RUN_DIR,
        'model':           MODEL_NAME,
        'epochs':          NUM_EPOCHS,
        'num_generations': NUM_GENERATIONS,
        'per_device_train_batch_size': NUM_GENERATIONS,
        'train_tasks':     TRAIN_TASKS,
        'eval_tasks':      list(EVAL_TASKS),
        'hardware':        'A10G (HF Spaces)',
        'baseline':        baseline,
        'post_training':   post,
        'delta':           {t: round(post[t] - baseline[t], 4) for t in EVAL_TASKS},
        'metrics': {
            'total_reward_calls': METRICS.total_calls,
            'parse_failures':     METRICS.parse_failures,
            'env_errors':         METRICS.env_errors,
            'format_rate':        round(fmt_rate, 4),
            'decision_counts':    dict(METRICS.decision_counts),
            'per_task_mean':      {t: round(sum(v)/len(v), 4) for t, v in METRICS.reward_by_task.items()},
        },
    }
    results_json = os.path.join(RUN_DIR, 'training_results.json')
    with open(results_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'[DONE] Saved {results_json}')

    # Copy live metrics into run dir as snapshot
    try:
        import shutil
        shutil.copy('/app/metrics_live.json', os.path.join(RUN_DIR, 'metrics_live.json'))
    except Exception:
        pass

    # Persist entire run dir to HF Space repo (runs/grpo/MODEL-NEP-DATETIME/)
    # so artifacts survive container restarts and each run is independently addressable
    repo_run_path = RUN_DIR.replace('/app/', '')  # strip /app/ prefix for repo path
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=RUN_DIR,
            path_in_repo=repo_run_path,
            repo_id='Pathikreet/ap-commander-training',
            repo_type='space',
            commit_message=f'Run artifacts: {os.path.basename(RUN_DIR)}',
            ignore_patterns=['adapter/*'],  # adapter uploaded separately to model repo
        )
        print(f'[UPLOAD] Run folder → {repo_run_path} in Pathikreet/ap-commander-training')
    except Exception as e:
        print(f'[UPLOAD] artifact upload failed: {e}')


if __name__ == '__main__':
    main()
