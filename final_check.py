"""
Final judge-criteria check against the live HF Space.
"""
import urllib.request, json, sys

base = "https://pathikreet-ap-clerk-env.hf.space"
PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" -- {detail}" if detail else ""))
        errors.append(label)

def get(path):
    req = urllib.request.Request(f"{base}{path}")
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, json.loads(r.read())

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{base}{path}", data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, json.loads(r.read())

# ── Phase 1: Automated Validation Gate ───────────────────────────────────────
print("\n[PHASE 1] Automated Validation Gate")

status, health = get("/health")
check("HF Space deploys and responds (200)", status == 200)
check("/health returns status=ok", health.get("status") == "ok")

status, tasks = get("/tasks")
check("GET /tasks returns 200", status == 200)
check("3+ tasks registered",  len(tasks) >= 3,  f"found {len(tasks)}")
check("6+ tasks registered",  len(tasks) >= 6,  f"found {len(tasks)}")
diffs = [t["difficulty"] for t in tasks]
check("easy tasks present",   "easy"   in diffs)
check("medium tasks present", "medium" in diffs)
check("hard tasks present",   "hard"   in diffs)

status, reset1 = post("/reset", {"task_id": "easy_perfect_match", "seed": 42})
check("POST /reset returns 200", status == 200)
check("reset() returns session_id", "session_id" in reset1)
check("reset() returns observation", "observation" in reset1)
sid = reset1["session_id"]
obs = reset1["observation"]
check("observation has invoice",         "invoice"         in obs)
check("observation has purchase_orders", "purchase_orders" in obs)
check("observation has goods_receipts",  "goods_receipts"  in obs)
check("observation has company_policy",  "company_policy"  in obs)
check("observation step_count=0",        obs.get("step_count") == 0)

action = {
    "session_id": sid,
    "action": {
        "decision": "APPROVE_FULL",
        "approved_amount": obs["invoice"]["invoice_total"],
        "reason_code": "MATCH_CONFIRMED",
        "explanation": "Invoice PO and GRN match exactly. Freight under cap. Approving full amount."
    }
}
status, step1 = post("/step", action)
check("POST /step returns 200", status == 200)
check("step() returns reward",  "reward" in step1)
check("step() done=True",       step1.get("done") is True)
score = step1["reward"]["score"]
check("score in [0.0, 1.0]",   0.0 <= score <= 1.0, f"got {score}")
check("perfect action scores >= 0.99", score >= 0.99, f"got {score}")
check("reward has breakdown",  isinstance(step1["reward"].get("breakdown"), dict))
check("reward has feedback",   isinstance(step1["reward"].get("feedback"), str))

status, state1 = get(f"/state?session_id={sid}")
check("GET /state returns 200",       status == 200)
check("state() has task_id",          "task_id"       in state1)
check("state() has step_count",       "step_count"    in state1)
check("state() has episode_score",    "episode_score" in state1)
check("state() done=True after step", state1.get("done") is True)
check("state() score matches step",   abs(state1["episode_score"] - score) < 1e-9)

# ── Phase 2: Grader Quality ───────────────────────────────────────────────────
print("\n[PHASE 2] Grader Quality -- all 6 tasks, varied scores")

task_scores = {}
for task in tasks:
    tid = task["task_id"]
    _, r = post("/reset", {"task_id": tid, "seed": 1})
    o = r["observation"]
    wrong = {
        "session_id": r["session_id"],
        "action": {
            "decision": "APPROVE_FULL",
            "approved_amount": 99999.0,
            "reason_code": "MATCH_CONFIRMED",
            "explanation": "Approving everything always regardless of documents."
        }
    }
    _, sr = post("/step", wrong)
    task_scores[tid] = sr["reward"]["score"]
    check(f"Grader {tid} score in [0,1]", 0.0 <= sr["reward"]["score"] <= 1.0)

score_vals = list(task_scores.values())
check(
    "Graders do NOT all return same score (disqualification check)",
    len(set(round(s, 3) for s in score_vals)) > 1,
    f"all returned: {score_vals}"
)

# ── Phase 3: Randomisation ───────────────────────────────────────────────────
print("\n[PHASE 3] Randomisation -- agent cannot memorise")

_, ra = post("/reset", {"task_id": "medium_quantity_shortfall", "seed": 1})
_, rb = post("/reset", {"task_id": "medium_quantity_shortfall", "seed": 2})
total_a = ra["observation"]["invoice"]["invoice_total"]
total_b = rb["observation"]["invoice"]["invoice_total"]
check("Different seeds produce different invoices", total_a != total_b,
      f"{total_a} vs {total_b}")
_, rc = post("/reset", {"task_id": "medium_quantity_shortfall", "seed": 1})
total_c = rc["observation"]["invoice"]["invoice_total"]
check("Same seed is reproducible", total_a == total_c, f"{total_a} vs {total_c}")

# ── Phase 4: Partial Reward Signal ───────────────────────────────────────────
print("\n[PHASE 4] Partial reward signal (not sparse binary)")

_, r2 = post("/reset", {"task_id": "medium_quantity_shortfall", "seed": 42})
sid2 = r2["session_id"]
obs2 = r2["observation"]
received   = obs2["goods_receipts"][0]["lines"][0]["received_quantity"]
unit_price = obs2["purchase_orders"][0]["lines"][0]["agreed_unit_price"]
correct_amt = received * unit_price

partial_action = {
    "session_id": sid2,
    "action": {
        "decision": "APPROVE_PARTIAL",
        "approved_amount": round(correct_amt * 1.04, 2),   # 4% off
        "reason_code": "QUANTITY_MISMATCH",
        "explanation": "GRN shows fewer units received than invoiced. Approving received quantity only."
    }
}
_, sr2 = post("/step", partial_action)
partial_score = sr2["reward"]["score"]
check("Partial credit: 4% wrong amount scores > 0",    partial_score > 0.0)
check("Partial credit: 4% wrong amount scores < 1.0",  partial_score < 1.0,
      f"got {partial_score}")
print(f"         Partial score (4% off correct amount): {partial_score}  <-- gradient signal works")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
if errors:
    print(f"  FAILED -- {len(errors)} check(s):")
    for e in errors:
        print(f"    x {e}")
    sys.exit(1)
else:
    print("  ALL LIVE CHECKS PASSED -- ready for judge evaluation!")
print("="*55)
