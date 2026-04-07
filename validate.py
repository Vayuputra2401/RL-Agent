"""
Pre-Submission Validation Script — AP Clerk Environment
========================================================
Runs all checklist items locally WITHOUT needing a running server or HF token.
All checks are self-contained.

Usage:
    python validate.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import sys
import json
import importlib

PASS  = "[PASS]"
FAIL  = "[FAIL]"
INFO  = "[INFO]"
errors = []


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)


# ── 1. Python version ─────────────────────────────────────────────────────────
print("\n[1] Python version")
major, minor = sys.version_info.major, sys.version_info.minor
check("Python >= 3.10", (major, minor) >= (3, 10),
      f"found {major}.{minor}")

# ── 2. Required files present ─────────────────────────────────────────────────
print("\n[2] Required files present")
import os
required_files = [
    "inference.py",
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt",
    "app/__init__.py",
    "app/main.py",
    "app/models.py",
    "app/tasks.py",
    "app/environment.py",
]
for f in required_files:
    check(f"File exists: {f}", os.path.isfile(f))

# ── 3. openenv.yaml valid ─────────────────────────────────────────────────────
print("\n[3] openenv.yaml compliance")
try:
    import yaml
    with open("openenv.yaml") as fh:
        spec = yaml.safe_load(fh)
    check("openenv.yaml is valid YAML", True)
    check("has 'name' field",      "name"      in spec)
    check("has 'version' field",   "version"   in spec)
    check("has 'tasks' field",     "tasks"     in spec)
    check("has 'endpoints' field", "endpoints" in spec)
    check("has 'reward_range'",    "reward_range" in spec)
    task_ids = [t["id"] for t in spec.get("tasks", [])]
    check("6+ tasks declared in openenv.yaml", len(task_ids) >= 6,
          f"found {len(task_ids)}")
    endpoints = spec.get("endpoints", {})
    for ep in ["reset", "step", "state", "tasks", "health"]:
        check(f"endpoint '{ep}' declared", ep in endpoints)
except ImportError:
    print(f"  {INFO}  pyyaml not installed — skipping YAML parse (pip install pyyaml)")
except Exception as e:
    check("openenv.yaml readable", False, str(e))

# ── 4. Pydantic models importable ─────────────────────────────────────────────
print("\n[4] Pydantic models")
try:
    from app.models import (
        APAction, APObservation, APReward,
        DecisionType, ReasonCode,
        Invoice, LineItem, PurchaseOrder, POLine, GoodsReceipt, GRNLine,
        ResetRequest, StepRequest, ResetResponse, StepResponse,
        StateResponse, TaskInfo,
    )
    check("All Pydantic models import cleanly", True)
    check("DecisionType has 3+ values (incl. intermediate actions)",
          len(DecisionType) >= 3)
    check("ReasonCode has 6+ values (incl. new codes)",
          len(ReasonCode)   >= 6)
except Exception as e:
    check("Pydantic models importable", False, str(e))

# ── 5. Tasks & graders ────────────────────────────────────────────────────────
print("\n[5] Tasks and graders")
try:
    from app.tasks import TASKS, grade_action
    from app.models import APAction, DecisionType, ReasonCode

    check("6+ tasks registered", len(TASKS) >= 6, f"found {len(TASKS)}")

    difficulty_counts: dict = {}
    for spec in TASKS.values():
        difficulty_counts[spec.difficulty] = difficulty_counts.get(spec.difficulty, 0) + 1
    for diff in ["easy", "medium", "hard"]:
        check(f"At least 2 tasks at difficulty='{diff}'",
              difficulty_counts.get(diff, 0) >= 2,
              f"found {difficulty_counts.get(diff, 0)}")

    # Run each grader with a fixed seed and dummy action — verify score in [0,1]
    dummy_action = APAction(
        decision=DecisionType.REJECT,
        approved_amount=0.0,
        reason_code=ReasonCode.NO_PO_FOUND,
        explanation="Validation dummy action for pre-submission check.",
    )
    for task_id, spec in TASKS.items():
        obs = spec.generator(seed=0)
        reward = grade_action(task_id, obs, dummy_action)
        in_range = 0.0 <= reward.score <= 1.0
        check(f"Grader '{task_id}' returns score in [0,1]", in_range,
              f"got {reward.score}")

    # Verify perfect action scores 1.0 on easy_perfect_match
    perfect_action = APAction(
        decision=DecisionType.APPROVE_FULL,
        approved_amount=TASKS["easy_perfect_match"].generator(seed=1).invoice.invoice_total,
        reason_code=ReasonCode.MATCH_CONFIRMED,
        explanation="All three documents match exactly. Full invoice approved.",
    )
    obs1  = TASKS["easy_perfect_match"].generator(seed=1)
    score = grade_action("easy_perfect_match", obs1, perfect_action).score
    check("Perfect action on easy_perfect_match scores >= 0.99", score >= 0.99, f"got {score}")

except Exception as e:
    check("Tasks and graders", False, str(e))

# ── 6. Environment class ──────────────────────────────────────────────────────
print("\n[6] APClerkEnvironment reset/step/state")
try:
    from app.environment import APClerkEnvironment
    from app.models import APAction, DecisionType, ReasonCode

    env = APClerkEnvironment()
    obs = env.reset("medium_quantity_shortfall", seed=42)
    check("reset() returns APObservation", obs is not None)
    check("reset() obs has invoice",       hasattr(obs, "invoice"))
    check("reset() step_count == 0",       obs.step_count == 0)

    action = APAction(
        decision=DecisionType.APPROVE_PARTIAL,
        approved_amount=obs.goods_receipts[0].lines[0].received_quantity
                        * obs.purchase_orders[0].lines[0].agreed_unit_price,
        reason_code=ReasonCode.QUANTITY_MISMATCH,
        explanation="Paying only for quantities confirmed received in the GRN.",
    )
    obs2, reward, done, info = env.step(action)
    check("step() returns reward",        reward is not None)
    check("step() done=True",             done is True)
    check("step() score in [0,1]",        0.0 <= reward.score <= 1.0)
    check("step() breakdown is dict",     isinstance(reward.breakdown, dict))

    state = env.state()
    check("state() returns dict",         isinstance(state, dict))
    check("state() done=True",            state["done"] is True)

    # Confirm episode_score matches reward.score
    check("state() episode_score matches reward",
          abs(state["episode_score"] - reward.score) < 1e-9)

except Exception as e:
    check("APClerkEnvironment", False, str(e))

# ── 7. Randomisation ──────────────────────────────────────────────────────────
print("\n[7] Randomisation — different seeds produce different episodes")
try:
    from app.tasks import TASKS
    gen   = TASKS["easy_perfect_match"].generator
    obs_a = gen(seed=42)
    obs_b = gen(seed=99)
    check("seed=42 vs seed=99 give different invoice totals",
          obs_a.invoice.invoice_total != obs_b.invoice.invoice_total,
          f"{obs_a.invoice.invoice_total} vs {obs_b.invoice.invoice_total}")
    obs_c = gen(seed=42)
    check("same seed is reproducible",
          obs_a.invoice.invoice_total == obs_c.invoice.invoice_total)
except Exception as e:
    check("Randomisation", False, str(e))

# ── 8. FastAPI app importable ─────────────────────────────────────────────────
print("\n[8] FastAPI app")
try:
    from app.main import app as fastapi_app
    check("FastAPI app imports cleanly", True)
    routes = [r.path for r in fastapi_app.routes]
    for path in ["/health", "/tasks", "/reset", "/step", "/state"]:
        check(f"Route '{path}' registered", path in routes)
except Exception as e:
    check("FastAPI app", False, str(e))

# ── 9. inference.py env-var guard ─────────────────────────────────────────────
print("\n[9] inference.py mandatory env-var checks")
import subprocess, os as _os
env_clean = {k: v for k, v in _os.environ.items()
             if k not in ("HF_TOKEN", "API_KEY", "API_BASE_URL", "MODEL_NAME")}
result = subprocess.run(
    [sys.executable, "inference.py"],
    capture_output=True, text=True, env=env_clean
)
check("inference.py exits non-zero when env vars missing",
      result.returncode != 0,
      f"exit code was {result.returncode}")
check("inference.py prints ERROR for missing vars",
      "ERROR" in result.stderr or "ERROR" in result.stdout)

# ── 10. Dockerfile sanity ─────────────────────────────────────────────────────
print("\n[10] Dockerfile")
with open("Dockerfile") as fh:
    dockerfile = fh.read()
check("Dockerfile uses python:3.11",      "python:3.11" in dockerfile)
check("Dockerfile EXPOSEs port 7860",     "7860" in dockerfile)
check("Dockerfile copies app/",           "COPY app/" in dockerfile)
check("Dockerfile copies inference.py",   "COPY inference.py" in dockerfile)
check("Dockerfile copies openenv.yaml",   "COPY openenv.yaml" in dockerfile)
check("Dockerfile runs uvicorn",          "uvicorn" in dockerfile)
check("Dockerfile has non-root USER",     "USER appuser" in dockerfile)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
if errors:
    print(f"  FAILED — {len(errors)} check(s) did not pass:")
    for e in errors:
        print(f"    ✗ {e}")
    print("="*55)
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED — ready to submit!")
    print("="*55)
    sys.exit(0)
