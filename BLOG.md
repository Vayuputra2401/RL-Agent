# Teaching an LLM to Pay Invoices — and Catch Itself When It's Wrong

*AP Commander: a multi-agent RL environment for enterprise financial workflows*

**Team:** Pathikreet Chowdhury, Anubhav Bhattacharya, Radhika Ravi
**Hackathon:** Meta PyTorch OpenEnv × Scaler School of Technology Grand Finale

---

## The Problem

LLMs fail at invoice processing in a specific, predictable way. They treat each decision as a one-shot question instead of an investigation. Given an invoice that's already been paid, most models approve it anyway — the duplicate is in the ledger, right there, but they don't connect the dots. Given a freight charge $12 over the policy cap, they approve the full amount because the cap is buried in paragraph 3 of the policy text. Given a vendor dispute that requires querying the supplier, then escalating to a manager, then rejecting — they skip straight to a decision and get it wrong.

These aren't hallucinations. They're reasoning failures — and no standard benchmark measures them.

Every enterprise processes thousands of invoices a month. A wrong approval is a financial loss. A missed duplicate is fraud. A wrong rejection damages a vendor relationship built over years. The cost of these failures is real and measurable — which makes it an ideal RL environment: you always know the right answer.

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
HF Training Space (A10G)            HF Environment Space
┌──────────────────────────┐        ┌──────────────────────────────┐
│  Qwen2.5-7B + LoRA       │─(HTTP)►│  AP Commander FastAPI server │
│  GRPOTrainer             │◄reward─│  24 tasks · graders · actors │
│  [env_reward, fmt_reward]│        │  seeded RNG · no static data │
└──────────────────────────┘        └──────────────────────────────┘
```

**The Environment Space** ([`pathikreet-ap-clerk-env.hf.space`](https://pathikreet-ap-clerk-env.hf.space/docs)) is a FastAPI server that stays live independently of training. It exposes `/reset`, `/step`, `/oversight/*`, and `/curriculum/*` endpoints. Any model, any framework, any machine can train against it by making HTTP calls — no local setup needed. 24 tasks, seeded RNG, no static dataset.

![AP Commander environment Space — interactive product website at pathikreet-ap-clerk-env.hf.space](runs/screenshots/env_ui_hero.png)

*The environment Space serves as both an API backend and an interactive product website. Pick a task, generate a live episode, submit an action, and see the exact reward breakdown — no code required. The "Live" indicator in the nav confirms the FastAPI server is healthy. Built entirely in HTML/CSS/JS returned from the FastAPI root route — zero additional dependencies.*

**The Training Space** ([`Pathikreet/ap-commander-training`](https://huggingface.co/spaces/Pathikreet/ap-commander-training)) is a Gradio UI that runs the full GRPO training loop on an A10G GPU. Open it, select a model, paste your HF token if using a gated model (Qwen is public — token not required), and click Start. Reward curves, decision distribution, and per-task metrics update live every 15 seconds. Want to run it yourself? See [Run Your Own Training](#run-your-own-training) below.

**AP Clerk agent** — receives a structured observation: invoice, purchase orders, goods receipt notes, and company policy. Outputs a structured JSON decision: action, approved amount, reason code, and an explanation that must cite specific dollar figures to score well.

**Oversight agent (Fleet AI)** — receives a batch of completed clerk episodes, identifies which contain fraudulent or policy-violating decisions, and explains its reasoning with numeric evidence. Penalised for false positives — oversight is a genuine training objective, not a trivially-satisfied check.

**Three simulated workplace actors** respond dynamically as the clerk investigates:
- *VendorActor* — honest, fraudulent, or confused persona; responds to `QUERY_VENDOR`
- *ManagerActor* — budget authority and risk appetite vary per episode; may be out-of-office, triggering a VP escalation chain
- *ComplianceActor* — responds to `HOLD` with a SOX / GDPR / Internal Policy verdict

Every episode is generated fresh at runtime from a seeded RNG. There is no static dataset. The agent must reason, not memorise.

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

Two design choices worth calling out:

**Process bonus.** `QUERY_VENDOR → REJECT` earns more than a direct `REJECT` on a duplicate task. Using discounted accumulated rewards (discount=0.9), the full investigative sequence scores 0.901 vs. a shortcut reject at ~0.40. The reward signal teaches the right *process*, not just the right answer.

**Oversight false-positive penalty is a real negative.** −0.25, not clamped to zero. An oversight agent that flags everything scores worse than one that reasons carefully. This makes scalable oversight a trainable skill with real consequences, not a box-ticking exercise.

Anti-hacking measures are built into every grader: `_explanation_coherence()` penalises keyword dumps (>40% keyword density), and `_has_numeric_citation()` requires actual dollar amounts — not vague language like "the amount was incorrect."

---

## Training Evidence

**Algorithm:** GRPO (Group Relative Policy Optimization, [DeepSeekMath](https://arxiv.org/abs/2402.03300))
**Models:** Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct, 4-bit NF4, LoRA r=16
**Framework:** TRL ≥ 0.15 — live environment rewards over HTTP, no static dataset

Two independent reward functions: `env_reward_fn` calls the live environment and returns the grader score (0.01–0.99); `format_reward_fn` checks JSON validity independently (+0.15 / −0.15).

### The Agents

**AP Clerk agent** is the primary decision-maker. Each episode it receives a structured observation — vendor invoice, matched purchase orders, goods receipt notes, and company policy — and must output a structured JSON action: decision type, approved amount, reason code, and a free-text explanation. The explanation is scored for specificity: it must cite actual dollar figures and percentages, not vague language. Multi-step tasks require the clerk to first investigate (QUERY_VENDOR, ESCALATE, HOLD) before reaching a terminal decision — shortcuts are penalised.

**Oversight agent** operates at a higher level of abstraction. It receives a batch of 3–5 completed clerk episodes and must identify which ones contain fraudulent or policy-violating decisions, explaining the specific numeric signal that triggered its suspicion. False positives carry a −0.25 penalty — not clamped to zero — so an oversight agent that flags everything scores worse than one that reasons carefully. This makes scalable oversight a trainable skill with real stakes, not a checkbox.

**Three workplace actors** respond dynamically during clerk investigations: a VendorActor (honest, fraudulent, or confused persona), a ManagerActor (randomised budget authority and risk appetite — may be out-of-office, triggering a VP escalation chain), and a ComplianceActor (responds to HOLD with a SOX / GDPR / Internal Policy verdict). None of them are pre-scripted responses — they're seeded with the episode RNG and generate contextually appropriate replies.

### Baselines — What the Untrained Models Score

Before any fine-tuning, we established three baselines: a hardcoded scripted agent (the optimal ceiling), Llama-3-8B, and Qwen2.5-7B.

![Scripted optimal agent across all 20 tasks — this is the ceiling GRPO training aims toward](runs/baselines/scripted-agent-2026-04-25/baseline_plot.png)

*The scripted agent applies the exact correct rule for every task. It doesn't score 1.0 — explanation quality, seed-dependent actor responses, and partial-credit graders penalise even a perfect decision. This is an honest ceiling, not a synthetic one.*

![Untrained Llama-3-8B vs the scripted ceiling — per-task breakdown](runs/baselines/llama-3-8b-2026-04-25/llama_plot.png)

*Llama-3-8B without fine-tuning scores 0.811 overall. It handles easy tasks well but drops to 0.698 on hard tasks — the multi-step investigative sequences are where untrained models fail.*

![Untrained Qwen2.5-7B across 17 tasks, 3 seeds each — this is the starting point for GRPO](runs/baselines/qwen2-5-7b-instruct-2026-04-25/baseline_plot.png)

*Qwen2.5-7B before GRPO: 0.535 overall mean. Hard tasks score 0.468, long-horizon tasks 0.432 — near the floor. The model cannot discover action sequences like QUERY\_VENDOR → REJECT without training.*

| Task Category | Optimal Ceiling | Untrained Llama-3-8B | Untrained Qwen2.5-7B | After GRPO 3ep |
|---|---|---|---|---|
| Easy (2 tasks) | 0.990 | 0.990 | 0.721 | **0.990** |
| Medium (4 tasks) | 0.907 | 0.712 | 0.691 | **0.860** |
| Hard (4 tasks) | 0.843 | 0.698 | 0.468 | — |
| Long-horizon (7 tasks) | 0.989 | 0.832 | 0.432 | — |
| **Overall** | **0.921** | **0.811** | **0.535** | — |

---

### Run 1 — Qwen2.5-7B, 3 Epochs (2026-04-25)

![Live training dashboard at step 150 — reward curve trending upward, recent mean 0.746](runs/grpo/qwen-2.5-7b-3ep-2026-04-25/training_dashboard_step150.png)

*Step 150 of Run 1. Recent mean reward 0.746, up from the 0.535 untrained baseline. The reward curve shows consistent upward trend across the first 150 steps. Format rate 91.2% — the model is reliably outputting valid JSON.*

![Decision distribution and per-task reward means across all training steps](runs/grpo/qwen-2.5-7b-3ep-2026-04-25/training_decision_dist.png)

*Left: the model learns to use the full action vocabulary — APPROVE\_FULL, REJECT, QUERY\_VENDOR, ESCALATE — rather than defaulting to one decision. Right: per-task training means show easy tasks converging first, hard multi-step tasks still learning.*

![Before vs After GRPO — task-by-task comparison across 10 tasks](runs/grpo/qwen-2.5-7b-3ep-2026-04-25/results.png)

*`easy_perfect_match` improved +0.490 from baseline — Qwen was getting the amount or reason code wrong before GRPO. Easy tasks match the ceiling after 3 epochs. Hard multi-step tasks need more epochs; the correct investigative sequences require more gradient steps to surface.*

---

### Run 2 — What Failed and Why (2026-04-26)

![Run 2 dashboard at stop — step 235, showing entropy collapse and format failure](runs/grpo/qwen-2.5-7b-run2-stopped-2026-04-26/dashboard_step235.png)

*Run 2 stopped at step 235/420. The dashboard tells the story: format rate dropped to 44%, entropy collapsed to 0.23, and the model defaulted to REJECT for 59% of decisions. Half of all GRPO groups had zero reward variance — no learning signal.*

Run 2 was extended to 17 tasks with 160 prompts and stopped early. What went wrong, specifically:

- **Temperature 1.1 → 44% format failures.** At high temperature the model generated natural language instead of JSON. Format reward ±0.05 was too weak to correct it — the env reward signal drowned it out.
- **Curriculum gating silently locked hard tasks from epoch 3.** Hard and long-horizon tasks stopped receiving any gradient signal. The model spent the final third of the run training only on easy tasks.
- **Entropy collapse.** `frac_reward_zero_std = 0.5` — half of all GRPO groups had identical rewards across all 16 generations. When every completion in a group scores the same, GRPO computes zero advantage and the update is a no-op.

---

### Run 3 — Qwen2.5-1.5B, G=16 (2026-04-26, ongoing)

**Model:** Qwen2.5-1.5B-Instruct · **Generations per prompt:** 16 · **Tasks:** 24 (all difficulties, no curriculum gating) · **Fixes from Run 2:** temperature → 0.7, `beta=0.1`, format reward → ±0.15, 322 training prompts.

![Run 3 training dashboard at step 112 — reward curve and loss curve, recent mean 0.692](runs/screenshots/training_dashboard_overview_step112.png)

*Step 112 of Run 3. Reward curve climbing steadily from the 0.535 untrained Qwen baseline — recent mean 0.692. Loss is stable and approaching zero with no entropy collapse, a direct contrast to Run 2's temperature-1.1 implosion. Format compliance holding above 90% throughout.*

![Run 3 full metrics at step 113 — format compliance 94.9%, decision distribution, per-task mean rewards](runs/screenshots/training_dashboard_metrics_step113.png)

*Step 113 breakdown: recent mean 0.722, format rate 94.9%, zero environment errors across 3,616 reward calls in 71 minutes. Decision distribution shows the clerk using the full action vocabulary — 59% REJECT, 24% QUERY\_VENDOR, 12% ESCALATE — rather than collapsing to a single output as in Run 2. Easy tasks have converged near ceiling (`no_po_found` 0.99, `vendor_mismatch` 0.73). Hard and long-horizon tasks (`manager_chain` 0.13, `invoice_dispute` 0.65) are still learning — expected, since multi-step investigative sequences require more gradient steps to surface reliably.*

---

### Run 4 — Qwen2.5-1.5B, G=8 (2026-04-26, ongoing)

**Model:** Qwen2.5-1.5B-Instruct · **Generations per prompt:** 8 · Same fixes as Run 3. Running in parallel to compare advantage estimation quality: G=16 produces lower-variance GRPO updates at the cost of 2× more environment calls per step; G=8 trains faster per step but with noisier gradients.

![Run 4 training dashboard at step 160 — reward curve and loss curve, recent mean 0.634](runs/screenshots/run4_dashboard_step160.png)

*Step 160 of Run 4. Recent mean 0.634 — lower than Run 3's 0.692 at the same model size, consistent with the noisier advantage estimates from halving the generation count. Loss curve shows the same stable pattern: no collapse, no runaway gradient norm.*

![Run 4 full metrics at step 179 — format compliance 93.7%, decision distribution, per-task mean rewards](runs/screenshots/run4_metrics_step179.png)

*Step 179: recent mean 0.709, format rate 93.7%, 1,432 reward calls in 57 minutes. Decision distribution is near-identical to Run 3 — 59% REJECT, 23% QUERY\_VENDOR, 12% ESCALATE — confirming the learned action vocabulary is stable across generation counts. Per-task pattern mirrors Run 3: easy tasks converged first, hard multi-step tasks still climbing. The G=8 vs G=16 comparison will give us a clean read on whether more generations per prompt are worth the extra compute for this environment.*

**[PLACEHOLDER: Run 3 and Run 4 final before/after comparison table — replace when runs complete]**

---

## Hackathon Theme Coverage

| Theme | Implementation | Why it counts |
|---|---|---|
| **#1 Multi-Agent** | AP Clerk + Oversight agent with separate `/oversight/*` endpoints + 3 actor-agents | Two independent agents with separate action/observation spaces; oversight reward is structurally adversarial to clerk reward |
| **#2 Long-Horizon** | 7 tasks at 10–16 steps: dispute resolution, fraud investigation, manager OOO chain, SOX audit trail | Single episodes requiring sustained multi-step reasoning; intermediate steps are individually scored, not just the terminal decision |
| **#3 Professional World Modeling** | ERP-style documents, randomised policy, multi-actor workplace simulation | Every episode generates a complete financial scenario — vendor, POs, GRNs, paid ledger, policy text — that the agent navigates as a coherent, stateful world |
| **#4 Self-Improvement** | Adaptive curriculum (`/curriculum/next_task`) + `HYPOTHETICAL` action for counterfactual self-play | Curriculum tracks per-difficulty running mean and unlocks harder tasks as thresholds are met; agent can explore alternative decision paths without committing |

---

## Run Your Own Training

The environment is public and always live. You don't need any credentials to call it — just an internet connection.

**Option 1 — HF Training Space (recommended, no setup):**

1. Open [Pathikreet/ap-commander-training](https://huggingface.co/spaces/Pathikreet/ap-commander-training)
2. Duplicate the Space to your own HF account (top-right → Duplicate this Space)
3. In your duplicated Space, go to Settings → Variables and add `HF_TOKEN` with your HF token *(only needed for gated models like Llama-3; Qwen2.5 is public — you can skip this)*
4. Click Start Training — the Space connects to the live environment automatically

**Option 2 — Colab (T4 GPU):**

Open [`training/colab_training.ipynb`](training/colab_training.ipynb) from the GitHub repo. The notebook connects to the same live environment over HTTP, runs the full GRPO loop, and saves results + LoRA adapter locally (or to Google Drive if you mount it). No HF token required for Qwen.

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
| Environment API + Swagger | https://pathikreet-ap-clerk-env.hf.space/docs |
| Training Space (Gradio UI) | https://huggingface.co/spaces/Pathikreet/ap-commander-training |
| GitHub | https://github.com/Vayuputra2401/RL-Agent |

---

The environment is live, the reward signal has no shortcuts, and the training evidence is in the `runs/` folder — every run timestamped, nothing overwritten.
