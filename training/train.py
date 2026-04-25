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
    'long_invoice_dispute', 'long_policy_migration',
    'long_batch_reconciliation', 'long_manager_chain',
    'long_fraud_investigation', 'long_audit_trail',
    'long_multi_vendor_split',
]
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
    'long_invoice_dispute': 'long',       'long_policy_migration': 'long',
    'long_batch_reconciliation': 'long',  'long_manager_chain': 'long',
    'long_fraud_investigation': 'long',   'long_audit_trail': 'long',
    'long_multi_vendor_split': 'long',
}
_DIFFICULTY_ORDER  = ['easy', 'medium', 'hard', 'long']
_UNLOCK_THRESHOLDS = {'easy': 0.70, 'medium': 0.65, 'hard': 0.60}


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
          long   → 2 seeds  (if unlocked)
        Returns list of (task_id, seed) pairs.
        """
        rows = []
        seeds_per_diff = {'easy': 10, 'medium': 5, 'hard': 2, 'long': 2}
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
        self.step               = 0
        self.reward_history     = []   # (step, mean_reward) — overall
        self.diff_reward_hist   = collections.defaultdict(list)  # diff → [(step, mean)]
        self.format_history     = []   # (step, format_rate) — compliance over time
        self.episode_len_hist   = []   # all episode lengths (for histogram)
        self.ep_len_by_task     = collections.defaultdict(list)  # task_id → [lengths]
        self.decision_history   = []   # [(step, Counter)] for stacked-bar over time
        self.decision_counts    = collections.Counter()
        self.parse_failures     = 0
        self.env_errors         = 0
        self.format_scores      = []
        self.reward_by_task     = collections.defaultdict(list)
        self.total_calls        = 0
        self._start_time        = time.time()
        self._step_decisions    = collections.Counter()  # decisions in current step batch

    def log_step(self, rewards, decisions, format_ok_list, task_ids, errors,
                 episode_lengths=None):
        self.step += 1
        self.total_calls += len(rewards)
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        self.reward_history.append((self.step, mean_r))

        # Per-difficulty reward history
        diff_rewards: dict = collections.defaultdict(list)
        for tid, r in zip(task_ids, rewards):
            d = _TASK_DIFFICULTY.get(tid, 'easy')
            diff_rewards[d].append(r)
        for d, rs in diff_rewards.items():
            self.diff_reward_hist[d].append((self.step, sum(rs) / len(rs)))

        for d in decisions:
            self.decision_counts[d] += 1
            self._step_decisions[d] += 1
        # Snapshot decision distribution every step for stacked-bar
        self.decision_history.append((self.step, dict(self._step_decisions)))

        fmt_ok_count = sum(1 for ok in format_ok_list if ok)
        fmt_rate = fmt_ok_count / len(format_ok_list) if format_ok_list else 0.0
        self.format_history.append((self.step, fmt_rate))
        for ok in format_ok_list:
            self.format_scores.append(1.0 if ok else 0.0)

        for tid, r in zip(task_ids, rewards):
            self.reward_by_task[tid].append(r)

        if episode_lengths:
            for tid, ep_len in zip(task_ids, episode_lengths):
                self.episode_len_hist.append(ep_len)
                self.ep_len_by_task[tid].append(ep_len)

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

    def save_all_metrics_figures(self, run_dir: str):
        """
        Save six standard RL research metric figures to run_dir.
        All figures follow conventions used in academic RL papers:
        - Named axes (xlabel, ylabel)
        - Figure caption as fig.text below the plot
        - Dark GitHub-style theme consistent with project
        - Smoothed curves with raw data visible in background
        """
        PALETTE = {'easy': '#3fb950', 'medium': '#d29922', 'hard': '#f85149', 'long': '#a371f7'}
        BG      = '#0d1117'
        PANEL   = '#161b22'
        GRID    = '#21262d'
        TEXT    = '#e6edf3'
        SUBTEXT = '#8b949e'
        ACCENT  = '#58a6ff'

        def _setup(ax, xlabel='', ylabel='', title=''):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color('#30363d')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
            ax.xaxis.grid(True, color=GRID, linewidth=0.4, alpha=0.4)
            ax.set_axisbelow(True)
            if xlabel: ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=9)
            if ylabel: ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=9)
            if title:  ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=8)

        def _smooth(values, window=None):
            if len(values) < 3:
                return values
            w = window or max(3, len(values) // 12)
            return np.convolve(values, np.ones(w)/w, mode='valid'), w

        def _caption(fig, text):
            fig.text(0.5, 0.01, text, ha='center', va='bottom',
                     color=SUBTEXT, fontsize=7, style='italic')

        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

        # ── Figure 1: Mean Episode Return (reward curve) ──────────────────────
        if self.reward_history:
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor(BG)
            steps   = [s for s, _ in self.reward_history]
            rewards = [r for _, r in self.reward_history]
            ax.plot(steps, rewards, color=ACCENT, alpha=0.25, linewidth=1, label='Per-batch mean')
            if len(rewards) >= 5:
                sm, w = _smooth(rewards)
                ax.plot(steps[w-1:], sm, color=ACCENT, linewidth=2, label=f'EMA (w={w})')
            ax.axhline(0.5, color=SUBTEXT, linestyle='--', linewidth=1, alpha=0.5, label='Chance baseline (0.5)')
            recent_mean = sum(rewards[-20:]) / min(20, len(rewards))
            ax.axhline(recent_mean, color='#f78166', linestyle=':', linewidth=1.5,
                       label=f'Recent mean = {recent_mean:.3f}')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT)
            _setup(ax, xlabel='Training Step (reward function call batch)',
                   ylabel='Mean Episode Return  [0.01 – 0.99]',
                   title='Training Reward Curve — AP Commander GRPO')
            _caption(fig, f'Each step = one GRPO batch. Reward = discounted accumulated score from AP Commander environment. | {ts}')
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            p = os.path.join(run_dir, 'fig1_reward_curve.png')
            plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
            plt.close()
            print(f'[METRICS] {p}')

        # ── Figure 2: Per-Difficulty Learning Curves ──────────────────────────
        if self.diff_reward_hist:
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor(BG)
            for diff in _DIFFICULTY_ORDER:
                hist = self.diff_reward_hist.get(diff, [])
                if not hist:
                    continue
                steps_d  = [s for s, _ in hist]
                rewards_d = [r for _, r in hist]
                color = PALETTE.get(diff, ACCENT)
                ax.plot(steps_d, rewards_d, color=color, alpha=0.20, linewidth=1)
                if len(rewards_d) >= 5:
                    sm, w = _smooth(rewards_d)
                    ax.plot(steps_d[w-1:], sm, color=color, linewidth=2.5, label=f'{diff} (n={len(steps_d)})')
                else:
                    ax.plot(steps_d, rewards_d, color=color, linewidth=2.5, label=diff)
            for thr_diff, thr_val in _UNLOCK_THRESHOLDS.items():
                ax.axhline(thr_val, color=PALETTE.get(thr_diff, SUBTEXT),
                           linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9, facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT)
            _setup(ax, xlabel='Training Step',
                   ylabel='Mean Reward per Difficulty Tier  [0.01 – 0.99]',
                   title='Curriculum Learning Curves — Easy / Medium / Hard / Long-Horizon')
            _caption(fig, f'Dashed lines = curriculum unlock thresholds. Each line = rolling mean of all tasks in that difficulty tier. | {ts}')
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            p = os.path.join(run_dir, 'fig2_difficulty_curves.png')
            plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
            plt.close()
            print(f'[METRICS] {p}')

        # ── Figure 3: Episode Length Distribution ─────────────────────────────
        if self.episode_len_hist:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor(BG)
            # Overall histogram
            max_len = max(self.episode_len_hist)
            bins = range(1, max_len + 2)
            axes[0].hist(self.episode_len_hist, bins=bins, color=ACCENT, alpha=0.85,
                         edgecolor=BG, rwidth=0.8)
            _setup(axes[0], xlabel='Episode Length (number of env steps)',
                   ylabel='Count of Episodes',
                   title='Episode Length Distribution (all tasks)')
            axes[0].axvline(np.mean(self.episode_len_hist), color='#f78166',
                            linestyle='--', linewidth=1.5,
                            label=f'Mean = {np.mean(self.episode_len_hist):.1f}')
            axes[0].legend(fontsize=8, facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT)
            # Per-difficulty mean episode length bar
            diff_ep_means = {}
            for diff in _DIFFICULTY_ORDER:
                lens = []
                for tid, d in _TASK_DIFFICULTY.items():
                    if d == diff:
                        lens.extend(self.ep_len_by_task.get(tid, []))
                if lens:
                    diff_ep_means[diff] = np.mean(lens)
            if diff_ep_means:
                diffs  = list(diff_ep_means.keys())
                means  = list(diff_ep_means.values())
                colors = [PALETTE.get(d, ACCENT) for d in diffs]
                axes[1].bar(diffs, means, color=colors, alpha=0.85, edgecolor=BG, width=0.5)
                for i, (d, m) in enumerate(zip(diffs, means)):
                    axes[1].text(i, m + 0.05, f'{m:.1f}', ha='center', color=TEXT, fontsize=9,
                                 fontweight='bold')
                axes[1].set_ylim(0, max(means) * 1.3)
                _setup(axes[1], xlabel='Difficulty Tier',
                       ylabel='Mean Episode Length (steps)',
                       title='Mean Episode Length by Difficulty')
            fig.suptitle('Episode Length Analysis — Multi-Step Decision Behavior', color=TEXT, fontsize=11, y=1.01)
            _caption(fig, f'Long-horizon tasks expected to have higher mean episode lengths as agent learns to use ESCALATE/QUERY_VENDOR. | {ts}')
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            p = os.path.join(run_dir, 'fig3_episode_lengths.png')
            plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
            plt.close()
            print(f'[METRICS] {p}')

        # ── Figure 4: Format Compliance Rate Over Time ────────────────────────
        if self.format_history:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            fig.patch.set_facecolor(BG)
            steps_f  = [s for s, _ in self.format_history]
            fmt_vals = [r for _, r in self.format_history]
            ax.plot(steps_f, fmt_vals, color='#d29922', alpha=0.25, linewidth=1)
            if len(fmt_vals) >= 5:
                sm, w = _smooth(fmt_vals)
                ax.plot(steps_f[w-1:], sm, color='#d29922', linewidth=2.5,
                        label=f'EMA (w={w})')
            final_rate = sum(self.format_scores) / max(1, len(self.format_scores))
            ax.axhline(final_rate, color='#3fb950', linestyle='--', linewidth=1.5,
                       label=f'Overall rate = {final_rate:.1%}')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT)
            _setup(ax, xlabel='Training Step',
                   ylabel='Format Compliance Rate  [0 – 1]',
                   title='JSON Format Compliance Over Training')
            _caption(fig, f'Format compliance = fraction of completions producing valid JSON with correct fields. Parse failures = {self.parse_failures}. | {ts}')
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            p = os.path.join(run_dir, 'fig4_format_compliance.png')
            plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
            plt.close()
            print(f'[METRICS] {p}')

        # ── Figure 5: Decision Distribution Over Time (stacked bar) ──────────
        if self.decision_history and len(self.decision_history) >= 3:
            all_decisions = sorted(set(self.decision_counts.keys()))
            # Sample ~20 evenly-spaced checkpoints for readability
            n_checkpoints = min(20, len(self.decision_history))
            idxs = [int(i * (len(self.decision_history) - 1) / (n_checkpoints - 1))
                    for i in range(n_checkpoints)]
            ckpt_steps  = [self.decision_history[i][0] for i in idxs]
            ckpt_counts = [self.decision_history[i][1] for i in idxs]
            # Convert to fractions
            fracs = []
            for c in ckpt_counts:
                total_c = sum(c.values()) or 1
                fracs.append({d: c.get(d, 0) / total_c for d in all_decisions})
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor(BG)
            dec_colors = ['#3fb950','#f85149','#d29922','#a371f7','#58a6ff','#f0883e']
            bottom = np.zeros(len(ckpt_steps))
            for j, dec in enumerate(all_decisions):
                vals = np.array([f[dec] for f in fracs])
                ax.bar(range(len(ckpt_steps)), vals, bottom=bottom,
                       label=dec, color=dec_colors[j % len(dec_colors)],
                       alpha=0.85, edgecolor=BG)
                bottom += vals
            ax.set_xticks(range(len(ckpt_steps)))
            ax.set_xticklabels([str(s) for s in ckpt_steps], rotation=45, fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT,
                      loc='upper right', bbox_to_anchor=(1.15, 1))
            _setup(ax, xlabel='Training Step (checkpoint)',
                   ylabel='Fraction of Decisions',
                   title='Decision Distribution Over Training (Stacked Bar)')
            _caption(fig, f'Each bar = cumulative decision distribution up to that checkpoint. Ideal: APPROVE_FULL grows for easy tasks, REJECT for fraud/duplicate tasks. | {ts}')
            plt.tight_layout(rect=[0, 0.04, 0.88, 1])
            p = os.path.join(run_dir, 'fig5_decision_distribution.png')
            plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
            plt.close()
            print(f'[METRICS] {p}')

        # ── Figure 6: Per-Task Training Mean (horizontal bar) ─────────────────
        if self.reward_by_task:
            task_means = {t: sum(v)/len(v) for t, v in self.reward_by_task.items()}
            tasks  = sorted(task_means, key=lambda t: (_DIFFICULTY_ORDER.index(_TASK_DIFFICULTY.get(t,'easy')), t))
            means  = [task_means[t] for t in tasks]
            colors = [PALETTE.get(_TASK_DIFFICULTY.get(t,'easy'), ACCENT) for t in tasks]
            short  = [t.replace('easy_','').replace('medium_','').replace('hard_','').replace('long_','').replace('_',' ').title() for t in tasks]

            fig, ax = plt.subplots(figsize=(10, max(4, len(tasks) * 0.45)))
            fig.patch.set_facecolor(BG)
            yp = range(len(tasks))
            ax.barh(list(yp), means, color=colors, alpha=0.85, edgecolor=BG)
            ax.set_yticks(list(yp))
            ax.set_yticklabels(short, fontsize=8)
            ax.set_xlim(0, 1.05)
            overall_mean = sum(means) / len(means)
            ax.axvline(overall_mean, color='#f78166', linestyle='--', linewidth=1.5,
                       label=f'Overall mean = {overall_mean:.3f}')
            ax.axvline(0.5, color=SUBTEXT, linestyle=':', linewidth=1, alpha=0.5)
            for i, m in enumerate(means):
                ax.text(m + 0.01, i, f'{m:.3f}', va='center', color=TEXT, fontsize=7)
            from matplotlib.patches import Patch
            legend_els = [Patch(facecolor=PALETTE[d], label=d.title()) for d in _DIFFICULTY_ORDER if d in PALETTE]
            legend_els.append(plt.Line2D([0],[0], color='#f78166', linestyle='--', label=f'Mean {overall_mean:.3f}'))
            ax.legend(handles=legend_els, fontsize=8, facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT)
            _setup(ax, xlabel='Mean Training Reward  [0.01 – 0.99]',
                   ylabel='Task',
                   title='Per-Task Training Mean Reward (all episodes)')
            _caption(fig, f'Tasks ordered by difficulty. Green ≥ 0.7 = curriculum mastered. Orange = in progress. Red < 0.4 = needs more training. | {ts}')
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            p = os.path.join(run_dir, 'fig6_per_task_means.png')
            plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
            plt.close()
            print(f'[METRICS] {p}')


METRICS = Metrics()
_EPISODE_LOG_PATH: str = ''   # set to run_dir/episodes.jsonl once run_dir is known

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
                             discount: float = 0.9, max_steps: int = 20,
                             episode_log: list | None = None) -> tuple[float, int]:
    """
    Run a full multi-step episode accumulating discounted per-step rewards.
    Returns (score, episode_length) so callers can track step counts.
    Model's first action starts the episode; _greedy_followup() handles
    subsequent steps so multi-step sequences earn full accumulated credit.
    E.g. QUERY_VENDOR→REJECT = 0.01 + 0.9*0.99 = 0.901 > shortcut REJECT = ~0.4

    episode_log: if provided, appended with one dict per env step for JSONL logging.
    """
    try:
        r = requests.post(f'{ENV_URL}/reset',
                          json={'task_id': task_id, 'seed': seed}, timeout=20)
        r.raise_for_status()
        reset_data = r.json()
        session_id = reset_data['session_id']
        action = first_action
        total  = 0.0
        steps_taken = 0
        for step_n in range(max_steps):
            step_r = requests.post(f'{ENV_URL}/step',
                                   json={'session_id': session_id, 'action': action},
                                   timeout=20)
            step_r.raise_for_status()
            result    = step_r.json()
            r_score   = float(result['reward']['score'])
            done      = result['done']
            obs_back  = result.get('observation', {})
            total    += (discount ** step_n) * r_score
            steps_taken = step_n + 1
            if episode_log is not None:
                episode_log.append({
                    'step_n':          step_n,
                    'decision':        action.get('decision'),
                    'approved_amount': action.get('approved_amount'),
                    'reason_code':     action.get('reason_code'),
                    'explanation':     (action.get('explanation') or '')[:120],
                    'step_score':      round(r_score, 4),
                    'done':            done,
                    'context_notes':   obs_back.get('context_notes', []),
                    'action_history':  obs_back.get('action_history', []),
                })
            if done:
                break
            action = _greedy_followup(obs_back)
        return min(0.99, max(0.01, total)), steps_taken
    except Exception as e:
        return 0.01, 1


# ── Two independent reward functions (guide: use multiple, not one) ─────────────

def env_reward_fn(completions, task_id=None, seed=None, **kwargs):
    """
    Environment reward: accumulated discounted per-step reward from AP Commander.
    Curriculum gating redirects locked tasks to easier ones during early training.
    Writes one JSONL record per episode to _EPISODE_LOG_PATH for full verifiability.
    """
    task_ids = task_id if task_id is not None else ['easy_perfect_match'] * len(completions)
    seeds    = seed    if seed    is not None else [random.randint(1, 999)] * len(completions)

    rewards, decisions, format_ok_list, ep_lengths, errors = [], [], [], [], 0
    for completion, tid, s in zip(completions, task_ids, seeds):
        gated_tid = CURRICULUM.gate_task(tid)
        if gated_tid != tid:
            print(f'[CURRICULUM] gate {tid} → {gated_tid}')

        action, fmt_ok = parse_action(completion)
        episode_steps = []
        try:
            score, ep_len = run_episode_accumulated(
                gated_tid, action, seed=int(s), episode_log=episode_steps)
        except Exception as e:
            score, ep_len = 0.01, 1
            errors += 1

        rewards.append(score)
        ep_lengths.append(ep_len)
        decisions.append(action.get('decision', 'UNKNOWN'))
        format_ok_list.append(fmt_ok)
        CURRICULUM.record(gated_tid, score)

        # Write structured episode record to JSONL for full verifiability
        if _EPISODE_LOG_PATH:
            try:
                record = {
                    'reward_step':   METRICS.step + 1,
                    'call_n':        METRICS.total_calls + len(rewards),
                    'task_id':       tid,
                    'gated_task_id': gated_tid,
                    'seed':          int(s),
                    'format_ok':     fmt_ok,
                    'score':         round(score, 4),
                    'episode_len':   ep_len,
                    'final_decision': action.get('decision'),
                    'steps':         episode_steps,
                    'ts':            datetime.datetime.now().isoformat(),
                }
                with open(_EPISODE_LOG_PATH, 'a') as _f:
                    _f.write(json.dumps(record) + '\n')
            except Exception:
                pass

        if METRICS.total_calls % LOG_SAMPLES_EVERY == 0:
            gated_note = f'→{gated_tid}' if gated_tid != tid else ''
            print(f'\n[SAMPLE] task={tid}{gated_note} seed={s} fmt={fmt_ok} '
                  f'score={score:.3f} ep_len={ep_len}')
            print(f'  {action.get("decision")} ${action.get("approved_amount")} '
                  f'{action.get("reason_code")}')
            print(f'  {str(action.get("explanation",""))[:100]}')
            print(f'  curriculum: {CURRICULUM.status_line()}')
            if episode_steps:
                actor_notes = [n for step in episode_steps
                               for n in step.get('context_notes', [])]
                if actor_notes:
                    print(f'  actor_responses: {actor_notes[:2]}')

    METRICS.log_step(rewards, decisions, format_ok_list, list(task_ids), errors,
                     episode_lengths=ep_lengths)
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

    # Point the global episode log path so env_reward_fn can write structured logs
    global _EPISODE_LOG_PATH
    _EPISODE_LOG_PATH = os.path.join(RUN_DIR, 'episodes.jsonl')
    print(f'[RUN] Episode log → {_EPISODE_LOG_PATH}')

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

    # Dataset contains ALL 17 tasks × 5 seeds = 85 prompts.
    # gate_task() in env_reward_fn handles curriculum redirection at reward time:
    # locked tasks (medium/hard/long) redirect to easy during early training.
    # As curriculum unlocks thresholds, redirection stops and full task variety flows.
    print(f'\n[DATASET] Building prompts ({len(TRAIN_TASKS)} tasks × 5 seeds = {len(TRAIN_TASKS)*5})...')
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
    print(f'[DATASET] {len(dataset)} samples across {len(TRAIN_TASKS)} tasks '
          f'({sum(1 for r in rows if _TASK_DIFFICULTY.get(r["task_id"],"easy")=="long")} long-horizon) '
          f'| curriculum: {CURRICULUM.status_line()}')

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
    METRICS.save_all_metrics_figures(RUN_DIR)

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

    # ── Before/After comparison figure (results.png — key result for demo) ────
    BG, TEXT, SUBTEXT = '#0d1117', '#e6edf3', '#8b949e'
    PANEL, GRID = '#161b22', '#21262d'
    _fmt_rate = sum(METRICS.format_scores) / max(1, len(METRICS.format_scores))

    eval_tasks_sorted = sorted(
        EVAL_TASKS,
        key=lambda t: (_DIFFICULTY_ORDER.index(_TASK_DIFFICULTY.get(t,'easy')), t)
    )
    DIFF_COLORS = {'easy': '#3fb950', 'medium': '#d29922', 'hard': '#f85149', 'long': '#a371f7'}

    fig = plt.figure(figsize=(18, max(8, len(eval_tasks_sorted) * 0.45 + 2)))
    fig.patch.set_facecolor(BG)
    gs  = fig.add_gridspec(1, 2, wspace=0.38)

    # Panel left: before/after horizontal bars
    ax_l = fig.add_subplot(gs[0, 0])
    ax_l.set_facecolor(PANEL)
    yp   = np.arange(len(eval_tasks_sorted))
    short = [t.replace('easy_','').replace('medium_','').replace('hard_','').replace('long_','')
              .replace('_',' ').title() for t in eval_tasks_sorted]
    bar_h = 0.35
    bars_b = ax_l.barh(yp - bar_h/2, [baseline.get(t, 0) for t in eval_tasks_sorted],
                       bar_h, label='Before GRPO', color='#f85149', alpha=0.85, edgecolor=BG)
    bars_a = ax_l.barh(yp + bar_h/2, [post.get(t, 0)     for t in eval_tasks_sorted],
                       bar_h, label='After GRPO',  color='#3fb950', alpha=0.85, edgecolor=BG)
    ax_l.set_yticks(yp)
    ax_l.set_yticklabels(short, fontsize=8, color=TEXT)
    ax_l.set_xlim(0, 1.15)
    ax_l.axvline(0.5, color=SUBTEXT, linestyle='--', linewidth=1, alpha=0.5)
    # Color-code y-tick labels by difficulty
    for i, t in enumerate(eval_tasks_sorted):
        ax_l.get_yticklabels()[i].set_color(DIFF_COLORS.get(_TASK_DIFFICULTY.get(t,'easy'), TEXT))
    ax_l.legend(fontsize=9, facecolor=PANEL, edgecolor='#30363d', labelcolor=TEXT)
    ax_l.set_xlabel('Task Score  [0.01 – 0.99]', color=SUBTEXT, fontsize=9)
    ax_l.set_ylabel('Task (color = difficulty tier)', color=SUBTEXT, fontsize=9)
    ax_l.set_title(f'Before vs After GRPO — {NUM_EPOCHS} Epochs', color=TEXT, fontsize=11,
                   fontweight='bold', pad=10)
    ax_l.tick_params(colors=TEXT, labelsize=8)
    for sp in ax_l.spines.values(): sp.set_color('#30363d')
    ax_l.spines['top'].set_visible(False); ax_l.spines['right'].set_visible(False)
    ax_l.xaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax_l.set_axisbelow(True)

    # Panel right: delta (improvement) per task
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.set_facecolor(PANEL)
    deltas = [post.get(t, 0) - baseline.get(t, 0) for t in eval_tasks_sorted]
    d_colors = ['#3fb950' if d >= 0 else '#f85149' for d in deltas]
    ax_r.barh(yp, deltas, color=d_colors, alpha=0.85, edgecolor=BG)
    ax_r.set_yticks(yp)
    ax_r.set_yticklabels(short, fontsize=8, color=TEXT)
    ax_r.axvline(0, color=SUBTEXT, linewidth=1)
    for i, d in enumerate(deltas):
        ax_r.text(d + 0.005 * np.sign(d + 1e-9), i, f'{d:+.3f}',
                  va='center', color=TEXT, fontsize=7)
    ax_r.set_xlabel('Score Delta (After − Before)', color=SUBTEXT, fontsize=9)
    ax_r.set_ylabel('Task', color=SUBTEXT, fontsize=9)
    ax_r.set_title('GRPO Improvement per Task', color=TEXT, fontsize=11,
                   fontweight='bold', pad=10)
    ax_r.tick_params(colors=TEXT, labelsize=8)
    for sp in ax_r.spines.values(): sp.set_color('#30363d')
    ax_r.spines['top'].set_visible(False); ax_r.spines['right'].set_visible(False)
    ax_r.xaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax_r.set_axisbelow(True)

    mean_before = sum(baseline.get(t,0) for t in eval_tasks_sorted) / len(eval_tasks_sorted)
    mean_after  = sum(post.get(t,0)     for t in eval_tasks_sorted) / len(eval_tasks_sorted)
    fig.suptitle(
        f'AP Commander GRPO — {MODEL_NAME.split("/")[-1]}  |  {NUM_EPOCHS} epochs  |  '
        f'{NUM_GENERATIONS} generations  |  {len(TRAIN_TASKS)} tasks\n'
        f'Overall: {mean_before:.3f} → {mean_after:.3f}  (+{mean_after-mean_before:.3f})  '
        f'|  format={_fmt_rate:.1%}  |  parse_fails={METRICS.parse_failures}  '
        f'|  {datetime.datetime.now().strftime("%Y-%m-%d")}',
        color=TEXT, fontsize=10, y=1.01
    )
    fig.text(0.5, -0.01,
             'Task colors: green=easy, yellow=medium, red=hard, purple=long-horizon. '
             'Score range [0.01, 0.99] as per AP Commander environment specification.',
             ha='center', color=SUBTEXT, fontsize=8, style='italic')
    results_png = os.path.join(RUN_DIR, 'results.png')
    plt.savefig(results_png, dpi=130, bbox_inches='tight', facecolor=BG)
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
        'delta':           {t: round(post.get(t,0) - baseline.get(t,0), 4) for t in EVAL_TASKS},
        'overall_baseline': round(mean_before, 4),
        'overall_post':     round(mean_after, 4),
        'overall_delta':    round(mean_after - mean_before, 4),
        'episode_log':      _EPISODE_LOG_PATH,
        'metrics': {
            'total_reward_calls':   METRICS.total_calls,
            'parse_failures':       METRICS.parse_failures,
            'env_errors':           METRICS.env_errors,
            'format_rate':          round(fmt_rate, 4),
            'decision_counts':      dict(METRICS.decision_counts),
            'per_task_mean':        {t: round(sum(v)/len(v), 4) for t, v in METRICS.reward_by_task.items()},
            'mean_episode_length':  round(sum(METRICS.episode_len_hist) / max(1, len(METRICS.episode_len_hist)), 2),
            'by_difficulty_post':   {d: round(sum(post.get(t,0) for t,diff in _TASK_DIFFICULTY.items()
                                                  if diff==d and t in post) /
                                              max(1, sum(1 for t,diff in _TASK_DIFFICULTY.items()
                                                         if diff==d and t in post)), 4)
                                     for d in _DIFFICULTY_ORDER},
        },
        'figures': [
            'fig1_reward_curve.png',
            'fig2_difficulty_curves.png',
            'fig3_episode_lengths.png',
            'fig4_format_compliance.png',
            'fig5_decision_distribution.png',
            'fig6_per_task_means.png',
            'results.png',
        ],
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
