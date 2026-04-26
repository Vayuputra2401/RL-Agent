import gradio as gr
import subprocess, sys, os, threading, json, datetime, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_proc      = None
_eval_proc = None
_lock      = threading.Lock()
METRICS_FILE = '/app/metrics_live.json'
GRPO_DIR     = '/app/runs/grpo'

# Kill any orphaned train.py left from a previous run / soft reload
subprocess.run(['pkill', '-f', 'train.py'], capture_output=True)
# Clear stale stop flag
if os.path.exists('/app/stop_requested'):
    os.remove('/app/stop_requested')


# ── Training process ────────────────────────────────────────────────────────────

_NO_PLOT = [gr.update()] * 8  # 8 plot slots (reward,loss,fmt,diff,eplen,pie,tasks,stats)

def start_training(model_choice, epochs, num_gen, hf_token):
    global _proc
    with _lock:
        if _proc and _proc.poll() is None:
            yield gr.update(value="Already running."), *_NO_PLOT
            return

        env = os.environ.copy()
        env['MODEL_NAME']      = model_choice
        env['NUM_EPOCHS']      = str(int(epochs))
        env['NUM_GENERATIONS'] = str(int(num_gen))
        if hf_token.strip():
            env['HF_TOKEN'] = hf_token.strip()
            env['HUGGING_FACE_HUB_TOKEN'] = hf_token.strip()

        # Wipe stale metrics
        if os.path.exists(METRICS_FILE):
            os.remove(METRICS_FILE)

        _proc = subprocess.Popen(
            [sys.executable, '-u', 'train.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env=env,
        )

    output = ''
    for line in _proc.stdout:
        output += line
        yield gr.update(value=output), *_NO_PLOT

    _proc.wait()
    output += f"\n{'='*60}\nDone (exit {_proc.returncode})"
    yield gr.update(value=output), *build_plots()


# ── Stop training + save plots ──────────────────────────────────────────────────

def _latest_run_dir():
    """Return path to most recently created run folder, or None."""
    import glob
    if not os.path.exists(GRPO_DIR):
        return None
    dirs = sorted(
        [d for d in glob.glob(f'{GRPO_DIR}/*/') if os.path.isdir(d)],
        key=os.path.getmtime, reverse=True
    )
    return dirs[0].rstrip('/') if dirs else None


def stop_training():
    global _proc
    with _lock:
        if not _proc or _proc.poll() is not None:
            return "No training process is currently running.", *build_plots()
        # Write flag file — train.py checks this every step and saves weights before exiting
        with open('/app/stop_requested', 'w') as f:
            f.write('stop')

    # Wait up to 120s for train.py to save weights and exit cleanly
    for _ in range(24):
        time.sleep(5)
        if _proc.poll() is not None:
            break
    else:
        # Fallback: hard terminate if train.py didn't respond
        _proc.terminate()

    m = _load_metrics() or {}
    run_dir = _latest_run_dir()

    if run_dir:
        plot_specs = [
            ('reward_curve_stop.png',    plot_reward_curve(m)),
            ('loss_curve_stop.png',      plot_loss_curve(m)),
            ('format_rate_stop.png',     plot_format_rate(m)),
            ('diff_curves_stop.png',     plot_diff_curves(m)),
            ('ep_lengths_stop.png',      plot_ep_lengths(m)),
            ('decision_dist_stop.png',   plot_decision_dist(m)),
            ('task_scores_stop.png',     plot_task_scores(m)),
            ('stats_stop.png',           plot_stats_panel(m)),
        ]
        for fname, fig in plot_specs:
            fig.savefig(os.path.join(run_dir, fname), dpi=130,
                        bbox_inches='tight', facecolor='#0d1117')
            plt.close(fig)
        with open(os.path.join(run_dir, 'metrics_snapshot_stop.json'), 'w') as f:
            json.dump(m, f, indent=2)
        status = (f"🛑 Stopped at step {m.get('step', '?')}. "
                  f"8 plots + metrics snapshot saved to {run_dir}")
    else:
        status = "🛑 Training stopped. No run directory found to save plots."

    return status, *build_plots()


# ── Baseline eval process ────────────────────────────────────────────────────────

def start_eval_baseline(model_choice, hf_token):
    global _eval_proc
    with _lock:
        if _eval_proc and _eval_proc.poll() is None:
            yield "Eval already running."
            return
        if _proc and _proc.poll() is None:
            yield "Training is running — wait for it to finish first."
            return

        env = os.environ.copy()
        env['MODEL_NAME'] = model_choice
        if hf_token.strip():
            env['HF_TOKEN'] = hf_token.strip()
            env['HUGGING_FACE_HUB_TOKEN'] = hf_token.strip()

        _eval_proc = subprocess.Popen(
            [sys.executable, '-u', 'eval_baseline.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env=env,
        )

    output = ''
    for line in _eval_proc.stdout:
        output += line
        yield gr.update(value=output)

    _eval_proc.wait()
    output += f"\n{'='*60}\nEval done (exit {_eval_proc.returncode})"
    yield gr.update(value=output)


# ── Plot builders ────────────────────────────────────────────────────────────────

def _load_metrics():
    if not os.path.exists(METRICS_FILE):
        return None
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _dark_fig(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor('#0d1117')
    return fig


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#c9d1d9', labelsize=8)
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#e6edf3')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#21262d', linewidth=0.8)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


def plot_reward_curve(m):
    fig = _dark_fig(figsize=(7, 3))
    ax  = fig.add_subplot(111)
    _style_ax(ax, 'Reward Curve', 'Step', 'Reward')

    history = m.get('reward_history', [])
    if history:
        steps   = [h['step'] for h in history]
        rewards = [h['reward'] for h in history]
        ax.plot(steps, rewards, color='#58a6ff', alpha=0.35, linewidth=1, label='Per-step')

        if len(rewards) >= 5:
            w = max(3, len(rewards) // 15)
            sm = np.convolve(rewards, np.ones(w)/w, mode='valid')
            ax.plot(steps[w-1:], sm, color='#79c0ff', linewidth=2, label=f'Smooth (w={w})')

        # Running mean annotation
        mean_r = sum(rewards[-20:]) / min(20, len(rewards))
        ax.axhline(mean_r, color='#f78166', linestyle='--', linewidth=1,
                   label=f'Recent mean: {mean_r:.3f}')

        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#c9d1d9', loc='lower right')

        # Annotate current step
        ax.annotate(f'step {steps[-1]}  r={rewards[-1]:.3f}',
                    xy=(steps[-1], rewards[-1]),
                    xytext=(-30, 12), textcoords='offset points',
                    color='#f0f6fc', fontsize=7,
                    arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1))
    else:
        ax.text(0.5, 0.5, 'Waiting for training data…',
                ha='center', va='center', color='#8b949e',
                transform=ax.transAxes, fontsize=10)

    fig.tight_layout()
    return fig


def plot_decision_dist(m):
    fig = _dark_fig(figsize=(4, 3))
    ax  = fig.add_subplot(111)
    dc  = m.get('decision_counts', {})

    if dc:
        labels = list(dc.keys())
        counts = list(dc.values())
        colors = ['#3fb950','#f85149','#d29922','#a371f7','#58a6ff','#39d353']
        wedges, texts, autotexts = ax.pie(
            counts, labels=None, autopct='%1.0f%%',
            colors=colors[:len(labels)], startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(linewidth=1.5, edgecolor='#0d1117')
        )
        for at in autotexts:
            at.set_color('#0d1117')
            at.set_fontsize(8)
            at.set_fontweight('bold')
        ax.legend(labels, loc='lower center', bbox_to_anchor=(0.5, -0.18),
                  ncol=3, fontsize=7, facecolor='#161b22',
                  edgecolor='#30363d', labelcolor='#c9d1d9')
        ax.set_title('Decision Distribution', fontsize=10, fontweight='bold',
                     color='#e6edf3', pad=6)
        fig.patch.set_facecolor('#0d1117')
    else:
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.text(0.5, 0.5, 'No decisions yet…',
                ha='center', va='center', color='#8b949e',
                transform=ax.transAxes, fontsize=10)
        ax.axis('off')

    fig.tight_layout()
    return fig


def plot_task_scores(m):
    task_means = m.get('task_means', {})
    fig = _dark_fig(figsize=(7, 3))
    ax  = fig.add_subplot(111)
    _style_ax(ax, 'Per-Task Mean Reward', '', 'Mean Reward')

    if task_means:
        tasks  = list(task_means.keys())
        scores = [task_means[t] for t in tasks]
        short  = [t.replace('easy_','').replace('medium_','').replace('hard_','')
                   .replace('long_','').replace('_',' ').title() for t in tasks]
        colors = ['#3fb950' if s >= 0.7 else '#d29922' if s >= 0.4 else '#f85149'
                  for s in scores]
        y = range(len(tasks))
        ax.barh(y, scores, color=colors, alpha=0.85,
                edgecolor='#0d1117', linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.axvline(0.7, color='#3fb950', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(0.4, color='#d29922', linestyle='--', linewidth=1, alpha=0.6)
        for i, s in enumerate(scores):
            ax.text(s + 0.01, i, f'{s:.2f}', va='center',
                    color='#c9d1d9', fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No per-task data yet…',
                ha='center', va='center', color='#8b949e',
                transform=ax.transAxes, fontsize=10)

    fig.tight_layout()
    return fig


def plot_stats_panel(m):
    """Small stats: format rate, parse fails, env errors, total calls."""
    fig = _dark_fig(figsize=(4, 3))
    ax  = fig.add_subplot(111)
    ax.set_facecolor('#161b22')
    ax.axis('off')

    step        = m.get('step', 0)
    total_calls = m.get('total_calls', 0)
    fmt_rate    = m.get('format_rate', 0.0)
    parse_fails = m.get('parse_failures', 0)
    env_errors  = m.get('env_errors', 0)
    recent_mean = m.get('recent_mean', 0.0)
    elapsed     = m.get('elapsed_min', 0.0)

    rows = [
        ('Training Step',   str(step)),
        ('Reward Calls',    str(total_calls)),
        ('Recent Mean',     f'{recent_mean:.3f}'),
        ('Format Rate',     f'{fmt_rate:.1%}'),
        ('Parse Failures',  str(parse_fails)),
        ('Env Errors',      str(env_errors)),
        ('Elapsed',         f'{elapsed:.1f} min'),
    ]

    for i, (label, val) in enumerate(rows):
        y = 0.92 - i * 0.13
        ax.text(0.05, y, label, color='#8b949e', fontsize=9, transform=ax.transAxes)
        color = '#f85149' if (label in ('Parse Failures','Env Errors') and int(val or 0) > 0) \
                else '#3fb950' if label == 'Recent Mean' and float(val or 0) >= 0.7 \
                else '#58a6ff'
        ax.text(0.65, y, val, color=color, fontsize=9, fontweight='bold',
                transform=ax.transAxes)

    ax.set_title('Live Stats', fontsize=10, fontweight='bold',
                 color='#e6edf3', pad=6)
    fig.patch.set_facecolor('#0d1117')
    fig.tight_layout()
    return fig



def plot_loss_curve(m):
    fig = _dark_fig(figsize=(7, 3))
    ax  = fig.add_subplot(111)
    _style_ax(ax, 'Loss & Grad Norm (Live)', 'Step', 'Loss')
    history = m.get('loss_history', [])
    if history:
        steps  = [h['step'] for h in history]
        losses = [h['loss'] for h in history]
        grads  = [h.get('grad_norm', 0) for h in history]
        ax.plot(steps, losses, color='#58a6ff', alpha=0.4, linewidth=1, label='loss')
        if len(losses) >= 5:
            w  = max(3, len(losses) // 12)
            sm = np.convolve(losses, np.ones(w)/w, mode='valid')
            ax.plot(steps[w-1:], sm, color='#79c0ff', linewidth=2, label=f'smooth (w={w})')
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
        ax2 = ax.twinx()
        ax2.set_facecolor('#161b22')
        ax2.plot(steps, grads, color='#f0883e', alpha=0.5, linewidth=1, label='grad norm')
        ax2.set_ylabel('Grad Norm', color='#f0883e', fontsize=8)
        ax2.tick_params(colors='#f0883e', labelsize=7)
        ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#c9d1d9', loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Waiting for loss data…',
                ha='center', va='center', color='#8b949e',
                transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    return fig


def plot_format_rate(m):
    fig = _dark_fig(figsize=(5, 3))
    ax  = fig.add_subplot(111)
    _style_ax(ax, 'Format Compliance Rate', 'Step', 'Rate')
    history = m.get('format_history', [])
    if history:
        steps = [h['step'] for h in history]
        rates = [h['rate'] for h in history]
        ax.plot(steps, rates, color='#d29922', alpha=0.3, linewidth=1)
        if len(rates) >= 5:
            w  = max(3, len(rates) // 12)
            sm = np.convolve(rates, np.ones(w)/w, mode='valid')
            ax.plot(steps[w-1:], sm, color='#d29922', linewidth=2, label=f'Smooth (w={w})')
        overall = m.get('format_rate', 0)
        ax.axhline(overall, color='#3fb950', linestyle='--', linewidth=1.2,
                   label=f'Overall: {overall:.1%}')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    else:
        ax.text(0.5, 0.5, 'Waiting...', ha='center', va='center',
                color='#8b949e', transform=ax.transAxes)
    fig.tight_layout()
    return fig


def plot_diff_curves(m):
    fig = _dark_fig(figsize=(5, 3))
    ax  = fig.add_subplot(111)
    _style_ax(ax, 'Per-Difficulty Rewards', 'Step', 'Reward')
    diff_hist = m.get('diff_reward_hist', {})
    colors = {'easy': '#3fb950', 'medium': '#d29922', 'hard': '#f85149', 'long': '#a371f7'}
    plotted = False
    for diff, color in colors.items():
        hist = diff_hist.get(diff, [])
        if not hist:
            continue
        steps   = [h['step'] for h in hist]
        rewards = [h['reward'] for h in hist]
        ax.plot(steps, rewards, color=color, alpha=0.2, linewidth=1)
        if len(rewards) >= 5:
            w  = max(3, len(rewards) // 12)
            sm = np.convolve(rewards, np.ones(w)/w, mode='valid')
            ax.plot(steps[w-1:], sm, color=color, linewidth=2, label=diff)
        else:
            ax.plot(steps, rewards, color=color, linewidth=2, label=diff)
        plotted = True
    if plotted:
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    else:
        ax.text(0.5, 0.5, 'Waiting...', ha='center', va='center',
                color='#8b949e', transform=ax.transAxes)
    fig.tight_layout()
    return fig


def plot_ep_lengths(m):
    fig = _dark_fig(figsize=(5, 3))
    ax  = fig.add_subplot(111)
    _style_ax(ax, 'Mean Episode Length', 'Step', 'Steps')
    history = m.get('ep_len_history', [])
    if history:
        steps = [h['step'] for h in history]
        lens  = [h['mean_len'] for h in history]
        ax.plot(steps, lens, color='#58a6ff', alpha=0.3, linewidth=1)
        if len(lens) >= 5:
            w  = max(3, len(lens) // 12)
            sm = np.convolve(lens, np.ones(w)/w, mode='valid')
            ax.plot(steps[w-1:], sm, color='#58a6ff', linewidth=2)
        ax.axhline(1.0, color='#8b949e', linestyle='--', linewidth=0.8, alpha=0.5, label='1-step baseline')
        ax.set_ylim(0, max(lens) * 1.2 + 0.5)
        ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    else:
        ax.text(0.5, 0.5, 'Waiting...', ha='center', va='center',
                color='#8b949e', transform=ax.transAxes)
    fig.tight_layout()
    return fig

def build_plots():
    plt.close('all')  # close all open figures before creating new ones
    m = _load_metrics() or {}
    return (plot_reward_curve(m), plot_loss_curve(m), plot_format_rate(m),
            plot_diff_curves(m), plot_ep_lengths(m),
            plot_decision_dist(m), plot_task_scores(m), plot_stats_panel(m))


# ── UI ───────────────────────────────────────────────────────────────────────────

MODELS = [
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
]

with gr.Blocks(title='AP Commander Training', theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div style="background:#0d1117;padding:18px 24px;border-radius:8px;
                border:1px solid #30363d;margin-bottom:4px">
      <h1 style="margin:0;color:#e6edf3;font-size:1.4em">
        🏛️ AP Commander — GRPO Training
      </h1>
      <p style="margin:4px 0 0;color:#8b949e;font-size:0.85em">
        Trains against
        <a href="https://pathikreet-ap-clerk-env.hf.space/docs"
           style="color:#58a6ff">pathikreet-ap-clerk-env.hf.space</a>
        · 24 tasks · Fleet AI oversight · adaptive curriculum
      </p>
    </div>
    """)

    # ── Config row ──────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        model_dd = gr.Dropdown(choices=MODELS, value=MODELS[0], label='Model',
                               scale=2)
        hf_token = gr.Textbox(label='HF Token (for gated models like Llama)',
                              placeholder='hf_…', type='password', scale=2)
        epochs_sl = gr.Slider(1, 10, value=6, step=1, label='Epochs', scale=1)
        numgen_sl = gr.Slider(4, 64, value=32, step=4,
                              label='Generations / prompt', scale=1)

    with gr.Row():
        start_btn = gr.Button('🚀  Start Training', variant='primary', size='lg', scale=4)
        stop_btn  = gr.Button('🛑  Stop & Save Plots', variant='secondary', size='lg', scale=1)
        reset_btn = gr.Button('⚡ Force Reset', variant='secondary', size='lg', scale=1)
    stop_status = gr.Textbox(label='', visible=True, interactive=False, max_lines=2,
                             placeholder='Stop status will appear here…')

    # ── Live metrics rows ───────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=5):
            plot_reward  = gr.Plot(label='Reward Curve',      show_label=False)
        with gr.Column(scale=5):
            plot_loss    = gr.Plot(label='Loss & Grad Norm',  show_label=False)

    with gr.Row():
        with gr.Column(scale=4):
            plot_fmt     = gr.Plot(label='Format Compliance', show_label=False)
        with gr.Column(scale=4):
            plot_diff    = gr.Plot(label='Per-Difficulty',    show_label=False)
        with gr.Column(scale=4):
            plot_eplen   = gr.Plot(label='Episode Lengths',   show_label=False)

    with gr.Row():
        with gr.Column(scale=3):
            plot_stats   = gr.Plot(label='Live Stats',        show_label=False)
        with gr.Column(scale=3):
            plot_pie     = gr.Plot(label='Decision Dist.',    show_label=False)
        with gr.Column(scale=6):
            plot_tasks   = gr.Plot(label='Per-Task Rewards',  show_label=False)

    # ── Log ─────────────────────────────────────────────────────────────────────
    with gr.Accordion('📋  Training Log', open=False):
        logs = gr.Textbox(label='', lines=25, max_lines=50, autoscroll=True,
                          show_label=False)

    refresh_btn = gr.Button('🔄  Refresh Charts', size='sm')

    # ── Final results image (saved by train.py on completion) ───────────────────
    with gr.Accordion('📊  Before / After Comparison', open=False):
        result_img = gr.Image(label='Before vs After', height=380)

    gr.HTML('<hr style="border-color:#30363d;margin:12px 0">')

    # ── Baseline Eval section ────────────────────────────────────────────────────
    gr.HTML("""
    <div style="background:#161b22;padding:12px 18px;border-radius:6px;
                border:1px solid #30363d;margin-bottom:6px">
      <h2 style="margin:0 0 4px;color:#e6edf3;font-size:1.1em">
        📊  Untrained Baseline Evaluation
      </h2>
      <p style="margin:0;color:#8b949e;font-size:0.82em">
        Evaluates the selected model <b>without any fine-tuning</b> on all 10 tasks × 3 seeds.
        Results saved to <code>runs/baselines/MODEL-DATETIME/</code> and uploaded to the repo.
      </p>
    </div>
    """)
    eval_btn = gr.Button('📈  Run Baseline Eval (no training)', variant='secondary', size='lg')

    with gr.Accordion('📋  Eval Log', open=False):
        eval_logs = gr.Textbox(label='', lines=20, max_lines=40, autoscroll=True,
                               show_label=False)

    with gr.Accordion('📊  Baseline Result Plot', open=False):
        baseline_img = gr.Image(label='Baseline Result', height=380)

    # ── Wiring ───────────────────────────────────────────────────────────────────
    start_btn.click(
        fn=start_training,
        inputs=[model_dd, epochs_sl, numgen_sl, hf_token],
        outputs=[logs, plot_reward, plot_loss, plot_fmt, plot_diff, plot_eplen, plot_pie, plot_tasks, plot_stats],
    )

    def force_reset():
        global _proc, _eval_proc
        subprocess.run(['pkill', '-f', 'train.py'], capture_output=True)
        if os.path.exists('/app/stop_requested'):
            os.remove('/app/stop_requested')
        _proc = None
        _eval_proc = None
        return gr.update(value='Reset done — ready to start.')

    reset_btn.click(fn=force_reset, inputs=[], outputs=[stop_status])

    eval_btn.click(
        fn=start_eval_baseline,
        inputs=[model_dd, hf_token],
        outputs=[eval_logs],
    )

    def _refresh():
        plots = build_plots()  # 8 figures
        img = None
        if os.path.exists(GRPO_DIR):
            candidates = sorted([
                os.path.join(dp, 'results.png')
                for dp, _, files in os.walk(GRPO_DIR)
                if 'results.png' in files
            ])
            if candidates:
                img = candidates[-1]
        bimg = None
        bdir = '/app/runs/baselines'
        if os.path.exists(bdir):
            bc = sorted([
                os.path.join(dp, 'baseline_plot.png')
                for dp, _, files in os.walk(bdir)
                if 'baseline_plot.png' in files
            ])
            if bc:
                bimg = bc[-1]
        return (*plots, img, bimg)

    _plot_outputs = [plot_reward, plot_loss, plot_fmt, plot_diff,
                     plot_eplen, plot_pie, plot_tasks, plot_stats,
                     result_img, baseline_img]

    refresh_btn.click(fn=_refresh, outputs=_plot_outputs)

    stop_btn.click(
        fn=stop_training,
        outputs=[stop_status, plot_reward, plot_loss, plot_fmt, plot_diff,
                 plot_eplen, plot_pie, plot_tasks, plot_stats],
    )

    timer = gr.Timer(15)
    timer.tick(fn=_refresh, outputs=_plot_outputs)

demo.launch(server_name='0.0.0.0', server_port=7860)
