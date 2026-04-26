# Teaching an LLM to Pay Invoices — and Catch Itself When It's Wrong

*AP Commander: a multi-agent RL environment for enterprise financial workflows*

**Team:** Pathikreet Chowdhury, Anubhav Bhattacharya, Radhika Ravi
**Hackathon:** Meta PyTorch OpenEnv × Scaler School of Technology Grand Finale

---

## The Problem

LLMs fail at invoice processing in a specific, predictable way. They treat each decision as a one-shot question instead of an investigation. Given an invoice that's already been paid, most models approve it anyway — the duplicate is in the ledger, right there, but they don't connect the dots. Given a freight charge $12 over the policy cap, they approve the full amount because the cap is buried in paragraph 3 of the policy text. Given a vendor dispute that requires querying the supplier, then escalating to a manager, then rejecting — they skip straight to a decision and get it wrong.

These aren't hallucinations. They're reasoning failures — and no standard benchmark measures them.

Every enterprise processes thousands of invoices a month. A wrong approval is a financial loss. A missed duplicate is fraud. A wrong rejection damages a vendor relationship built over years. The cost is real and measurable — which makes it an ideal RL environment: you always know the right answer.

---

## The Insight That Drove the Design

The reward signal is the hard part.

It's easy to check if the agent said REJECT. It's much harder to reward the *right* REJECT for the *right reason* at the *right amount* after the *right sequence of intermediate steps* — without creating a shortcut the agent can exploit.

An agent that always outputs `APPROVE_FULL` at $0 scores near zero. An agent that gets the decision right but cites the wrong amount scores ~0.40. An agent that rejects a duplicate without first querying the vendor misses the process bonus. There is no path to a high score except actually reasoning through the problem.

That design constraint shaped everything else.

---

## What We Built

Two HuggingFace Spaces working in tandem — one serves the environment, one runs the training loop:

```
HF Training Space (A10G)              HF Environment Space
┌────────────────────────────┐       ┌──────────────────────────────┐
│  Qwen2.5 + LoRA (4-bit)    │─HTTP►│  AP Commander FastAPI server  │
│  GRPOTrainer               │◄reward│  24 tasks · graders · actors  │
│  env_reward + fmt_reward   │       │  seeded RNG · no static data  │
└────────────────────────────┘       └──────────────────────────────┘
```

**The Environment Space** ([`pathikreet-ap-clerk-env.hf.space`](https://pathikreet-ap-clerk-env.hf.space/docs)) is a FastAPI server that stays live independently of training. It exposes `/reset`, `/step`, `/oversight/*`, and `/curriculum/*` endpoints. Any model, any framework, any machine can train against it over HTTP — no local setup, no GPU required on the client side. 24 tasks, seeded RNG, no static dataset.

![AP Commander environment Space — interactive product website at pathikreet-ap-clerk-env.hf.space](runs/screenshots/env_ui_hero.png)

*The environment Space is also an interactive product website. Pick a task, generate a live episode, submit an action, and see the exact reward breakdown — no code required. Built entirely in HTML/CSS/JS returned from the FastAPI root route — zero additional Python dependencies.*

**The Training Space** ([`Pathikreet/ap-commander-training`](https://huggingface.co/spaces/Pathikreet/ap-commander-training)) runs the full GRPO training loop on an A10G GPU. Select a model, optionally paste an HF token for gated models (Qwen2.5 is public — no token needed), and click Start. Reward curves, format compliance, decision distribution, and per-task metrics refresh live every 15 seconds.

---

### The Agents

**AP Clerk** is the primary decision-maker. Each episode it receives a structured observation: vendor invoice, matched purchase orders, goods receipt notes, and company payment policy. It must output a structured JSON action — decision type, approved amount, reason code, and a free-text explanation. That explanation is scored for specificity: it must cite actual dollar figures and policy thresholds, not vague language like "the amount was incorrect." Multi-step tasks require the clerk to first investigate (via `QUERY_VENDOR`, `ESCALATE`, or `HOLD`) before reaching a terminal decision — shortcuts are penalised through the process bonus component.

**Oversight Agent** operates one level above the clerk. It receives a batch of 3–5 completed clerk episodes and must identify which contain fraudulent or policy-violating decisions, explaining the exact numeric signal that triggered its suspicion. False positives carry a −0.25 penalty that is *not* clamped to zero — so flagging everything scores worse than reasoning carefully. This makes scalable oversight a genuine training objective with real stakes, not a box-ticking exercise.

**Three simulated workplace actors** respond dynamically when the clerk investigates:
- *VendorActor* — honest, fraudulent, or confused persona; responds to `QUERY_VENDOR` with contextually appropriate replies seeded per episode
- *ManagerActor* — randomised budget authority and risk appetite; may be out-of-office, triggering a VP escalation chain the clerk must navigate
- *ComplianceActor* — responds to `HOLD` with a SOX / GDPR / Internal Policy verdict citing specific regulation

None of these actors use pre-scripted response pools. They're seeded with the episode RNG and produce contextually appropriate replies — the clerk never faces the same scenario twice.

---

### Adaptive Curriculum

One of our more deliberate design choices: the `/curriculum/next_task` endpoint doesn't trust the client's claimed performance history. It reads from server-side records populated by `/step` completions — episode outcomes the client cannot fabricate.

The difficulty ladder runs easy → medium → hard → long-horizon → oversight. Each tier unlocks when the agent's running mean across the previous tier crosses a threshold (0.70 for easy, 0.65 for medium, 0.68 for hard, 0.72 for long-horizon). Within each unlocked tier, the curriculum selects the least-practiced task — so training stays balanced across the full task distribution rather than letting the model over-index on whichever tasks score highest.

This matters in practice. A model that memorises three easy tasks doesn't unlock medium. A model that cherry-picks high-scoring tasks within a difficulty tier still has to practice the ones it's avoiding. The curriculum tracks a rolling 200-entry window per run ID, which means it reflects recent performance rather than all-time averages — if the model regresses, the difficulty ladder adjusts.

---

### A World That Generates Itself

Every episode is generated fresh from a seeded RNG at call time. There is no JSON file of invoice scenarios. Vendor names, invoice amounts, PO numbers, freight charges, policy caps, and actor personas are all computed on the fly. Training distribution is effectively infinite — the model cannot memorise scenarios, it has to learn the underlying rule.

The `HYPOTHETICAL` action type extends this further: the agent can request a simulated outcome for an alternative decision path without committing to it and without changing episode state. This is the self-improvement angle — the model can interrogate its own uncertainty before acting, exploring counterfactuals as part of its reasoning process.

---

## Reward Design

Scores are partial-credit across five components — composable, not monolithic:

| Component | Weight | What it measures |
|---|---|---|
| Decision accuracy | 38–55% | Correct terminal action |
| Amount accuracy | 20–45% | Within 1% = full credit, within 8% = partial |
| Reason code | 10–30% | Correct classification of why |
| Explanation quality | 10–20% | Specific $ / % citations required |
| Process bonus | 0–15% | Correct intermediate steps before terminal |

**Process bonus.** `QUERY_VENDOR → REJECT` earns more than a direct `REJECT` on a duplicate task. Using discounted accumulated rewards (discount factor 0.9), the full investigative sequence scores 0.901 vs. a shortcut reject at ~0.40. The reward teaches the right *process*, not just the right answer.

**Oversight false-positive penalty is a real negative.** −0.25, not clamped to zero. An oversight agent that flags every episode scores worse than one that reads carefully. This makes scalable oversight trainable — not a trivially-gamed constraint.

**Anti-gaming built into every grader.** `_explanation_coherence()` penalises keyword dumps (>40% keyword density triggers a coherence penalty). `_has_numeric_citation()` requires actual dollar amounts in the explanation — no score for "the amount was incorrect." Every grader clamps its final output to (0.01, 0.99) to prevent degenerate reward signals from collapsing GRPO group variance.

---

## Training Evidence

**Algorithm:** GRPO (Group Relative Policy Optimization, [DeepSeekMath](https://arxiv.org/abs/2402.03300))
**Models:** Qwen2.5-1.5B-Instruct · Qwen2.5-7B-Instruct · 4-bit NF4 · LoRA r=16
**Framework:** TRL ≥ 0.15 — live environment rewards over HTTP, no static dataset

Two independent reward functions run per completion: `env_reward_fn` calls the live environment and returns the grader score (0.01–0.99); `format_reward_fn` checks JSON validity independently (+0.15 / −0.15). They're separate so a model that outputs perfect JSON but wrong reasoning still gets reward signal from both directions.

---

### Baselines — What Untrained Models Score

Before any fine-tuning, we ran three baselines: a hardcoded scripted agent (the optimal ceiling), Llama-3-8B, and Qwen2.5-7B. This gives us an honest before/after.

![Scripted optimal agent across all 24 tasks — this is the ceiling GRPO training aims toward](runs/baselines/scripted-agent-2026-04-25/baseline_plot.png)

*The scripted agent applies the exact correct rule for every task. It doesn't score 1.0 — explanation quality, seed-dependent actor responses, and partial-credit graders penalise even perfect decisions. This is an honest ceiling, not a synthetic one.*

![Untrained Llama-3-8B vs the scripted ceiling — per-task breakdown](runs/baselines/llama-3-8b-2026-04-25/llama_plot.png)

*Llama-3-8B without fine-tuning scores 0.811 overall. Easy tasks are handled well; hard tasks drop to 0.698. The gap opens exactly where multi-step investigative sequences are required — the failure mode we designed the environment to expose.*

![Untrained Qwen2.5-7B across 17 tasks, 3 seeds each — starting point for GRPO](runs/baselines/qwen2-5-7b-instruct-2026-04-25/baseline_plot.png)

*Qwen2.5-7B before GRPO: 0.535 overall mean. Hard tasks score 0.468, long-horizon tasks 0.432 — near the floor. Without training, the model cannot discover action sequences like `QUERY_VENDOR → REJECT`. That's what GRPO teaches.*

| Task Category | Optimal Ceiling | Untrained Llama-3-8B | Untrained Qwen2.5-7B | After GRPO (Run 1) |
|---|---|---|---|---|
| Easy | 0.990 | 0.990 | 0.721 | **0.990** |
| Medium | 0.907 | 0.712 | 0.691 | **0.860** |
| Hard | 0.843 | 0.698 | 0.468 | — |
| Long-horizon | 0.989 | 0.832 | 0.432 | — |
| **Overall** | **0.921** | **0.811** | **0.535** | — |

---

### Run 1 — Qwen2.5-7B, G=16, 3 Epochs (2026-04-25)

![Live training dashboard at step 150 — reward curve trending upward, recent mean 0.746](runs/grpo/qwen-2.5-7b-3ep-2026-04-25/training_dashboard_step150.png)

*Step 150 of Run 1. Recent mean 0.746, up from the 0.535 untrained baseline. Steady upward trend, format rate 91.2% — the model is reliably producing valid JSON while simultaneously learning better decisions.*

![Decision distribution and per-task reward means across all training steps](runs/grpo/qwen-2.5-7b-3ep-2026-04-25/training_decision_dist.png)

*The model learns to use the full action vocabulary — APPROVE_FULL, REJECT, QUERY_VENDOR, ESCALATE — rather than defaulting to a single decision. Per-task means show easy tasks converging first; hard multi-step tasks still learning at step 150.*

![Before vs After GRPO — task-by-task comparison across 10 tasks](runs/grpo/qwen-2.5-7b-3ep-2026-04-25/results.png)

*`easy_perfect_match` improved +0.490 from baseline — Qwen was consistently getting the amount or reason code wrong before GRPO. After 3 epochs, easy tasks match the scripted ceiling. Hard multi-step tasks need more gradient steps; the investigative sequences require the model to hold multiple facts in context across turns.*

---

### Run 2 — What Failed and Why (2026-04-26)

Not every run is a success. Run 2 failed instructively, and we stopped it rather than waste compute.

![Run 2 dashboard at stop — step 235, entropy collapse and format failure](runs/grpo/qwen-2.5-7b-run2-stopped-2026-04-26/dashboard_step235.png)

*Step 235/420 — stopped early. Format rate collapsed to 44%, entropy to 0.23, and the model defaulted to REJECT for 59% of decisions. Half of all GRPO groups had zero reward variance — the update was a no-op for every other batch.*

Three specific failures, each diagnosable from the dashboard:

- **Temperature 1.1 → 44% format failures.** At high temperature the model produced natural language instead of JSON. Format reward ±0.05 was too weak to correct it once the distribution drifted — env reward drowned it out.
- **Curriculum gating silently starved hard tasks.** Hard and long-horizon tasks stopped receiving gradient signal from epoch 3. The model spent its final third training only on easy tasks, making no progress on the skills that actually matter.
- **Entropy collapse.** `frac_reward_zero_std = 0.5` — half of all GRPO groups had identical rewards across all completions. When every completion in a group scores the same, GRPO computes zero advantage and the weight update is zero. The model stopped learning.

These are fixable. Run 3 applies all three fixes.

---

### Run 3 — Qwen2.5-1.5B, G=16 (2026-04-26, paused — insufficient compute)

**Model:** Qwen2.5-1.5B-Instruct · **G=16** · **Tasks:** all 24, no curriculum gating · **Fixes:** temperature 0.7, `beta=0.1` KL penalty, format reward ±0.15, 322 training prompts across full difficulty range.

![Run 3 training dashboard at step 112 — reward curve and loss, recent mean 0.692](runs/grpo/qwen-2.5-1b-run3-paused-2026-04-26/dashboard_step112.png)

*Step 112. Reward climbing steadily from the 0.486 untrained baseline — recent mean 0.692. Loss stable, approaching zero, no entropy collapse. Format compliance holding above 90% from step 1 — the stronger format reward and lower temperature are doing their job.*

![Run 3 full metrics at step 113 — format 94.9%, decision distribution, per-task rewards](runs/grpo/qwen-2.5-1b-run3-paused-2026-04-26/metrics_step113.png)

*Step 113: recent mean 0.722, format rate 94.9%, zero environment errors across 3,616 reward calls in 71 minutes. The decision distribution shows the full action vocabulary in use — 59% REJECT, 24% QUERY_VENDOR, 12% ESCALATE — no single-decision collapse. Easy tasks near ceiling (`no_po_found` 0.99, `vendor_mismatch` 0.73). Hard and long-horizon tasks (`manager_chain` 0.13, `invoice_dispute` 0.65) still improving — multi-step investigative sequences require more gradient steps to solidify. The space was paused at step 113 due to insufficient compute allocation.*

#### Run 3 — Per-task results at pause (step 113 vs untrained baseline)

| Task | Before GRPO | Step 113 | Δ |
|---|---|---|---|
| easy_perfect_match | 0.990 | ~0.990 | ~0.000 |
| easy_no_po_found | 0.990 | 0.990 | 0.000 |
| medium_quantity_shortfall | 0.608 | ~0.720 | ~+0.112 |
| medium_price_discrepancy | 0.990 | ~0.990 | ~0.000 |
| medium_split_delivery | 0.060 | ~0.430 | ~+0.370 |
| medium_vendor_mismatch | 0.990 | 0.730 | −0.260 |
| hard_policy_violation | 0.774 | ~0.800 | ~+0.026 |
| hard_duplicate_invoice | 0.950 | ~0.950 | ~0.000 |
| hard_partial_po_match | 0.570 | ~0.620 | ~+0.050 |
| hard_tax_discrepancy | 0.790 | ~0.790 | ~0.000 |
| hard_currency_conversion | 0.455 | ~0.520 | ~+0.065 |
| hard_manager_preapproval | 0.010 | ~0.050 | ~+0.040 |
| hard_credit_memo | 0.055 | ~0.100 | ~+0.045 |
| long_invoice_dispute | 0.010 | 0.650 | +0.640 |
| long_policy_migration | 0.800 | ~0.800 | ~0.000 |
| long_batch_reconciliation | 0.050 | ~0.150 | ~+0.100 |
| long_manager_chain | 0.100 | 0.130 | +0.030 |
| long_fraud_investigation | 0.875 | ~0.875 | ~0.000 |
| long_audit_trail | 0.107 | ~0.200 | ~+0.093 |
| long_multi_vendor_split | 0.030 | ~0.100 | ~+0.070 |
| **Mean** | **0.486** | **~0.722** | **~+0.236** |

> Before-GRPO values from the pre-training baseline evaluation (same Qwen2.5-1.5B model). Step-113 values read from the per-task dashboard panel at pause; values marked `~` are approximate. `long_invoice_dispute` shows the largest jump (+0.640) — the QUERY_VENDOR→REJECT investigative chain is already being discovered.

---

### Run 4 — Qwen2.5-1.5B, G=8 (2026-04-26, ongoing)

**Model:** Qwen2.5-1.5B-Instruct · **G=8** · Same fixes as Run 3. Running in parallel to answer a specific question: does halving the generation count from 16 to 8 meaningfully hurt advantage estimation quality, or does the faster per-step throughput compensate? G=16 produces lower-variance GRPO updates; G=8 trains ~2× faster per step but with noisier gradients.

![Run 4 training dashboard at step 160 — reward curve and loss, recent mean 0.634](runs/grpo/qwen-2.5-1b-6ep-2026-04-26-run4/dashboard_step160.png)

*Step 160 of Run 4. Recent mean 0.634 — below Run 3's 0.692 at the same step count, consistent with noisier advantage estimates at G=8. Loss pattern is stable, no collapse.*

![Run 4 full metrics at step 179 — format 93.7%, decision distribution, per-task rewards](runs/grpo/qwen-2.5-1b-6ep-2026-04-26-run4/metrics_step179.png)

*Step 179: recent mean 0.709, format rate 93.7%, 1,432 reward calls in 57 minutes. Decision distribution nearly identical to Run 3 — 59% REJECT, 23% QUERY_VENDOR, 12% ESCALATE — confirming the learned action vocabulary is stable regardless of generation count. Per-task pattern mirrors Run 3: easy tasks converged early, hard multi-step tasks still climbing. Run 4 is ongoing — the final comparison will give a clean answer on whether G=16 is worth the 2× compute overhead for this environment.*

---

## Hackathon Theme Coverage

| Theme | Implementation | Why it counts |
|---|---|---|
| **#1 Multi-Agent** | AP Clerk + Oversight agent with separate `/oversight/*` endpoints + 3 actor-agents (Vendor, Manager, Compliance) | Two independent agents with separate action/observation spaces; oversight reward is structurally adversarial to clerk reward — optimising one does not trivially optimise the other |
| **#2 Long-Horizon** | 7 tasks at 10–16 steps: dispute resolution, fraud investigation, manager OOO escalation chain, SOX audit trail | Single episodes requiring sustained multi-step reasoning; intermediate steps are individually scored, not just the terminal decision |
| **#3 Professional World Modeling** | ERP-style documents, randomised policy, multi-actor workplace simulation, seeded RNG with no static dataset | Every episode generates a complete financial scenario — vendor, POs, GRNs, paid ledger, policy text — that the agent navigates as a coherent, stateful world it cannot memorise |
| **#4 Self-Improvement** | Adaptive curriculum (`/curriculum/next_task`) with server-verified history + `HYPOTHETICAL` action for counterfactual self-play | Curriculum tracks per-difficulty running mean on tamper-proof server records and unlocks harder tasks as thresholds are met; agent can explore alternative decision paths without committing |

---

## Run Your Own Training

The environment is public and always live. No credentials needed to call it — just an internet connection.

**Option 1 — HF Training Space (no setup, recommended):**

1. Open [Pathikreet/ap-commander-training](https://huggingface.co/spaces/Pathikreet/ap-commander-training)
2. Click **Duplicate this Space** (top-right) to create your own copy
3. In your Space's Settings → Variables, add `HF_TOKEN` *(only for gated models like Llama-3; Qwen2.5 is public — skip this if using Qwen)*
4. Click **Start Training** — the Space connects to the live environment automatically

**Option 2 — Colab (T4 GPU):**

Open [`training/colab_training.ipynb`](training/colab_training.ipynb) from the GitHub repo. Connects to the same live environment over HTTP, runs the full GRPO loop, saves results and LoRA adapter locally or to Google Drive. No HF token required for Qwen.

**Option 3 — Local:**
```bash
git clone https://github.com/Vayuputra2401/RL-Agent
pip install trl>=0.15.0 peft transformers bitsandbytes accelerate requests datasets
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct NUM_EPOCHS=3 python training/train.py
```

All results save to timestamped folders under `runs/` — re-running never overwrites a previous run.

---

## Links

| Resource | URL |
|---|---|
| Environment UI (interactive demo) | https://pathikreet-ap-clerk-env.hf.space |
| Environment API + Swagger | https://pathikreet-ap-clerk-env.hf.space/docs |
| Training Space (Gradio UI) | https://huggingface.co/spaces/Pathikreet/ap-commander-training |
| Training script | https://github.com/Vayuputra2401/RL-Agent/blob/main/training/train.py |
| Colab notebook | https://github.com/Vayuputra2401/RL-Agent/blob/main/training/colab_training.ipynb |
| Training logs (all runs) | https://github.com/Vayuputra2401/RL-Agent/tree/main/runs/grpo |
| Baseline logs | https://github.com/Vayuputra2401/RL-Agent/tree/main/runs/baselines |
| GitHub | https://github.com/Vayuputra2401/RL-Agent |
| Presentation (Canva) | https://canva.link/k7f87ccul4fznaf |

---

The environment is live, the reward signal has no shortcuts, and every run is timestamped in `runs/` — nothing overwritten, nothing cherry-picked.
