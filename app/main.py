"""
AP Commander — FastAPI Server
Exposes AP Clerk, Oversight, and Curriculum endpoints.

AP Clerk:    POST /reset  POST /step  GET /state
Oversight:   POST /oversight/reset  POST /oversight/step  GET /oversight/state
Curriculum:  GET  /curriculum/next_task
Meta:        GET /tasks  GET /health  GET /stats
"""

from __future__ import annotations
import uuid
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateResponse, TaskInfo,
    OversightResetRequest, OversightResetResponse,
    OversightStepRequest, OversightStepResponse,
    CurriculumRequest, CurriculumResponse,
)
from .environment import APClerkEnvironment
from .tasks import TASKS
from oversight_environment import OversightEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ap-clerk-env")

app = FastAPI(
    title="AP Commander — Multi-Agent Enterprise Financial Environment",
    description=(
        "OpenEnv-compatible multi-agent environment for enterprise AP workflows. "
        "AP Clerk agent (27 tasks) + Fleet AI Oversight Agent + Adaptive Curriculum. "
        "Themes: Multi-Agent (#1), Long-Horizon Planning (#2), World Modeling (#3), "
        "Self-Improvement (#4)."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions:           Dict[str, APClerkEnvironment]  = {}
_oversight_sessions: Dict[str, OversightEnvironment] = {}
_session_times:      Dict[str, datetime]             = {}
_session_run_ids:    Dict[str, str]                  = {}  # session_id → run_id
_SESSION_TTL = timedelta(hours=1)

# Server-side curriculum history — populated by /step when done=True.
# Keyed by run_id (X-Run-Id request header, or "default").
# Rolling window of 200 entries per run_id to prevent unbounded growth.
_curriculum_history: Dict[str, List[dict]] = {}
_CURRICULUM_WINDOW = 200

# In-memory stats counters
_stats: Dict[str, Any] = {
    "total_episodes":     0,
    "completed_episodes": 0,
    "total_score":        0.0,
    "task_counts":        {},   # task_id → {"count": int, "score_sum": float}
}


def _prune_sessions() -> None:
    """Remove sessions older than _SESSION_TTL."""
    cutoff = datetime.utcnow() - _SESSION_TTL
    stale   = [sid for sid, t in _session_times.items() if t < cutoff]
    for sid in stale:
        _sessions.pop(sid, None)
        _oversight_sessions.pop(sid, None)
        _session_times.pop(sid, None)
        _session_run_ids.pop(sid, None)


def _get_session(session_id: str) -> APClerkEnvironment:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found. Call /reset first."
        )
    return _sessions[session_id]


def _get_oversight_session(session_id: str) -> OversightEnvironment:
    if session_id not in _oversight_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Oversight session {session_id!r} not found. Call /oversight/reset first."
        )
    return _oversight_sessions[session_id]


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body style="font-family:monospace;padding:2rem;background:#0f0f0f;color:#e0e0e0">
    <h1>AP Commander — Multi-Agent Enterprise Financial Environment</h1>
    <p>OpenEnv multi-agent environment: AP Clerk + Fleet AI Oversight + Adaptive Curriculum.</p>
    <p style="color:#aaa">
      27 tasks &nbsp;|&nbsp; easy / medium / hard / long-horizon / oversight &nbsp;|&nbsp;
      multi-step (up to 16 steps) &nbsp;|&nbsp; actor-agents (vendor, manager, compliance) &nbsp;|&nbsp;
      adaptive curriculum
    </p>
    <h3 style="color:#7fdbff">AP Clerk (Theme #3 + #2)</h3>
    <p style="color:#888">POST /reset → POST /step → GET /state</p>
    <h3 style="color:#7fdbff">Oversight Agent (Theme #1 Fleet AI)</h3>
    <p style="color:#888">POST /oversight/reset → POST /oversight/step → GET /oversight/state</p>
    <h3 style="color:#7fdbff">Adaptive Curriculum (Theme #4)</h3>
    <p style="color:#888">GET /curriculum/next_task?session_history=[...]</p>
    <ul>
      <li><a href="/docs" style="color:#7fdbff">/docs</a> — Swagger UI</li>
      <li><a href="/tasks" style="color:#7fdbff">/tasks</a> — List all 27 tasks</li>
      <li><a href="/health" style="color:#7fdbff">/health</a> — Health check</li>
      <li><a href="/stats" style="color:#7fdbff">/stats</a> — Live episode statistics</li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "environment": "ap-commander",
        "version": "4.0.0",
        "agents": ["ap-clerk", "oversight"],
        "total_tasks": len(TASKS),
        "themes": ["multi-agent", "long-horizon", "world-modeling", "self-improvement"],
    }


@app.get("/tasks", response_model=list)
async def list_tasks():
    return [
        TaskInfo(
            task_id=tid,
            name=spec.name,
            difficulty=spec.difficulty,
            description=spec.description,
        )
        for tid, spec in TASKS.items()
    ]


@app.post("/reset", response_model=ResetResponse)
async def reset(body: Optional[ResetRequest] = None, request: Request = None):
    if body is None:
        body = ResetRequest()
    _prune_sessions()
    session_id = body.session_id or str(uuid.uuid4())
    run_id = (request.headers.get("X-Run-Id", "default") if request else "default")
    env = APClerkEnvironment()
    try:
        obs = env.reset(body.task_id, seed=body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _sessions[session_id]          = env
    _session_times[session_id]     = datetime.utcnow()
    _session_run_ids[session_id]   = run_id
    # Track episode start in stats
    _stats["total_episodes"] += 1
    tid = body.task_id
    if tid not in _stats["task_counts"]:
        _stats["task_counts"][tid] = {"count": 0, "score_sum": 0.0}
    logger.info("reset  session=%s  task=%s  seed=%s", session_id, body.task_id, body.seed)
    return ResetResponse(
        observation=obs,
        session_id=session_id,
        info={"message": f"Episode started for task '{body.task_id}'",
              "seed": body.seed},
    )


@app.post("/step", response_model=StepResponse)
async def step(body: StepRequest):
    env = _get_session(body.session_id)
    try:
        obs, reward, done, info = env.step(body.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Update stats on episode completion
    if done:
        _stats["completed_episodes"] += 1
        _stats["total_score"]        += reward.score
        tid = obs.task_id
        if tid not in _stats["task_counts"]:
            _stats["task_counts"][tid] = {"count": 0, "score_sum": 0.0}
        _stats["task_counts"][tid]["count"]     += 1
        _stats["task_counts"][tid]["score_sum"] += reward.score
        # Record in server-side curriculum history (keyed by run_id from /reset)
        run_id = _session_run_ids.get(body.session_id, "default")
        bucket = _curriculum_history.setdefault(run_id, [])
        bucket.append({"task_id": tid, "score": reward.score})
        if len(bucket) > _CURRICULUM_WINDOW:
            del bucket[:-_CURRICULUM_WINDOW]
    logger.info(
        "step   session=%s  decision=%s  score=%.3f",
        body.session_id, body.action.decision, reward.score,
    )
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
async def state(session_id: str):
    env = _get_session(session_id)
    s = env.state()
    return StateResponse(
        session_id=session_id,
        task_id=s["task_id"],
        step_count=s["step_count"],
        episode_score=s["episode_score"],
        done=s["done"],
        current_observation=s["current_observation"],
    )


@app.get("/stats")
async def stats():
    completed = _stats["completed_episodes"]
    mean_score = round(_stats["total_score"] / completed, 4) if completed > 0 else 0.0
    task_breakdown = {
        tid: {
            "count":      v["count"],
            "mean_score": round(v["score_sum"] / v["count"], 4) if v["count"] > 0 else 0.0,
        }
        for tid, v in _stats["task_counts"].items()
    }
    return {
        "total_episodes":     _stats["total_episodes"],
        "completed_episodes": completed,
        "mean_score":         mean_score,
        "task_breakdown":     task_breakdown,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Oversight Agent endpoints  (Theme #1 Fleet AI)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/oversight/reset", response_model=OversightResetResponse)
async def oversight_reset(body: Optional[OversightResetRequest] = None):
    """
    Start a new Oversight Agent session.
    The agent receives a batch of completed AP Clerk episodes and must
    identify which ones contain suspicious/fraudulent decisions.
    """
    if body is None:
        body = OversightResetRequest()
    _prune_sessions()
    session_id = str(uuid.uuid4())
    env = OversightEnvironment()
    obs = env.reset(seed=body.seed, num_episodes=body.num_episodes)
    obs.session_id = session_id
    _oversight_sessions[session_id] = env
    _session_times[session_id] = datetime.utcnow()
    logger.info("oversight_reset  session=%s  num_episodes=%d", session_id, body.num_episodes)
    return OversightResetResponse(
        observation=obs,
        session_id=session_id,
        info={
            "message": f"Oversight session started with {body.num_episodes} episodes to review.",
            "seed": body.seed,
            "instructions": (
                "Review each episode_summary and submit OversightAction for each. "
                "Verdicts: CLEAR | FLAG_FOR_REVIEW | ESCALATE_TO_AUDIT. "
                "Include specific numeric signal in your reasoning."
            ),
        },
    )


@app.post("/oversight/step", response_model=OversightStepResponse)
async def oversight_step(body: OversightStepRequest):
    """
    Submit an Oversight verdict for one episode in the batch.
    Repeat until all episodes are reviewed (done=True).
    """
    env = _get_oversight_session(body.session_id)
    try:
        obs, reward, done, info = env.step(body.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info(
        "oversight_step  session=%s  episode=%s  verdict=%s  score=%.3f",
        body.session_id, body.action.episode_id, body.action.verdict, reward.score,
    )
    return OversightStepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/oversight/state")
async def oversight_state(session_id: str):
    """Get current state of an Oversight session."""
    env = _get_oversight_session(session_id)
    return {"session_id": session_id, **env.state()}


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive Curriculum endpoint  (Theme #4 Self-Improvement)
# ══════════════════════════════════════════════════════════════════════════════

# Task difficulty ordering for curriculum
_DIFFICULTY_ORDER = ["easy", "medium", "hard", "long-horizon", "oversight"]

_CURRICULUM_THRESHOLDS = {
    "easy":          0.70,   # unlock medium
    "medium":        0.65,   # unlock hard
    "hard":          0.68,   # unlock long-horizon
    "long-horizon":  0.72,   # unlock oversight
}

_TASKS_BY_DIFFICULTY: Dict[str, List[str]] = {}
for _tid, _spec in TASKS.items():
    _d = _spec.difficulty
    _TASKS_BY_DIFFICULTY.setdefault(_d, []).append(_tid)


@app.post("/curriculum/next_task", response_model=CurriculumResponse)
async def curriculum_next_task(body: Optional[CurriculumRequest] = None, request: Request = None):
    """
    Recommend the next task based on server-verified episode history.

    The curriculum uses server-side records populated by /step completions.
    Client-provided session_history is accepted only as a cold-start fallback
    when no server-side history exists for this run_id.

    Difficulty ladder: easy → medium → hard → long-horizon → oversight
    """
    if body is None:
        body = CurriculumRequest()

    run_id = (request.headers.get("X-Run-Id", "default") if request else "default")
    server_history = _curriculum_history.get(run_id, [])

    # Use server-side history; fall back to client only when server has no data
    if server_history:
        raw_history = server_history
    else:
        raw_history = [{"task_id": e.task_id, "score": e.score}
                       for e in body.session_history]

    history = raw_history  # list of dicts with task_id + score

    # Compute mean score per difficulty level from recent history
    # history entries are dicts: {"task_id": str, "score": float}
    scores_by_diff: Dict[str, List[float]] = {}
    for entry in history:
        tid  = entry["task_id"] if isinstance(entry, dict) else entry.task_id
        sc   = entry["score"]   if isinstance(entry, dict) else entry.score
        spec = TASKS.get(tid)
        if spec:
            diff = spec.difficulty
            scores_by_diff.setdefault(diff, []).append(sc)

    mean_by_diff = {
        d: sum(s) / len(s) for d, s in scores_by_diff.items() if s
    }

    # Determine highest unlocked difficulty
    current_diff = "easy"
    unlocked = ["easy"]
    for diff in _DIFFICULTY_ORDER[1:]:
        prev_diff = _DIFFICULTY_ORDER[_DIFFICULTY_ORDER.index(diff) - 1]
        thresh    = _CURRICULUM_THRESHOLDS.get(prev_diff, 0.70)
        if mean_by_diff.get(prev_diff, 0.0) >= thresh:
            current_diff = diff
            unlocked.append(diff)
        else:
            break

    # Choose least-practiced task at current difficulty
    available = _TASKS_BY_DIFFICULTY.get(current_diff, [])
    if not available:
        available = _TASKS_BY_DIFFICULTY.get("easy", [])
        current_diff = "easy"

    practiced_counts: Dict[str, int] = {}
    for entry in history:
        tid = entry["task_id"] if isinstance(entry, dict) else entry.task_id
        practiced_counts[tid] = practiced_counts.get(tid, 0) + 1

    recommended = min(available, key=lambda tid: practiced_counts.get(tid, 0))

    # Build reason string
    if current_diff == "easy":
        reason = "Starting with easy tasks to build foundational skills."
    else:
        prev = _DIFFICULTY_ORDER[_DIFFICULTY_ORDER.index(current_diff) - 1]
        mean = mean_by_diff.get(prev, 0.0)
        thresh = _CURRICULUM_THRESHOLDS.get(prev, 0.70)
        reason = (
            f"Mean score on {prev} tasks is {mean:.2f} "
            f"(threshold: {thresh:.2f}) — unlocked {current_diff}."
        )

    return CurriculumResponse(
        recommended_task_id=recommended,
        difficulty=current_diff,
        reason=reason,
        unlocked_tasks=[t for d in unlocked for t in _TASKS_BY_DIFFICULTY.get(d, [])],
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def start():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, workers=1)
