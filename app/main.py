"""
AP Clerk Environment — FastAPI Server
Exposes: POST /reset  POST /step  GET /state  GET /tasks  GET /health
"""

from __future__ import annotations
import uuid
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateResponse, TaskInfo,
)
from .environment import APClerkEnvironment
from .tasks import TASKS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ap-clerk-env")

app = FastAPI(
    title="AP Clerk Environment",
    description=(
        "OpenEnv-compatible environment simulating an AI Accounts Payable Clerk "
        "performing Three-Way Invoice Matching. "
        "Six randomised tasks across easy / medium / hard difficulty."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, APClerkEnvironment] = {}


def _get_session(session_id: str) -> APClerkEnvironment:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found. Call /reset first."
        )
    return _sessions[session_id]


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body style="font-family:monospace;padding:2rem;background:#0f0f0f;color:#e0e0e0">
    <h1>🧾 AP Clerk Environment</h1>
    <p>OpenEnv-compatible AI Accounts Payable environment for Three-Way Invoice Matching.</p>
    <p style="color:#aaa">6 randomised tasks &nbsp;|&nbsp; easy / medium / hard &nbsp;|&nbsp; partial-credit rewards</p>
    <ul>
      <li><a href="/docs" style="color:#7fdbff">/docs</a> — Swagger UI</li>
      <li><a href="/tasks" style="color:#7fdbff">/tasks</a> — List all tasks</li>
      <li><a href="/health" style="color:#7fdbff">/health</a> — Health check</li>
    </ul>
    <p style="color:#888">POST /reset → POST /step → GET /state</p>
    </body></html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "ap-clerk-env", "version": "2.0.0"}


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
async def reset(body: ResetRequest):
    session_id = body.session_id or str(uuid.uuid4())
    env = APClerkEnvironment()
    try:
        obs = env.reset(body.task_id, seed=body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _sessions[session_id] = env
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


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
