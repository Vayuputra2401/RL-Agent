"""
AP Commander — Typed Models
OpenEnv-required models: Observation, Action, Reward for AP Clerk and Oversight agents.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


# ── Document primitives ───────────────────────────────────────────────────────

class LineItem(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(gt=0)
    line_total: float


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    po_reference: Optional[str] = None
    line_items: List[LineItem]
    freight_charge: float = 0.0
    tax_amount: float = 0.0
    invoice_total: float
    currency: str = "USD"


class POLine(BaseModel):
    description: str
    ordered_quantity: float = Field(gt=0)
    agreed_unit_price: float = Field(gt=0)


class PurchaseOrder(BaseModel):
    po_number: str
    vendor_name: str
    lines: List[POLine]
    authorized_total: float
    status: str = "OPEN"


class GRNLine(BaseModel):
    description: str
    received_quantity: float = Field(ge=0)


class GoodsReceipt(BaseModel):
    grn_id: str
    po_number: str
    lines: List[GRNLine]


# ── Action space ──────────────────────────────────────────────────────────────

class DecisionType(str, Enum):
    APPROVE_FULL    = "APPROVE_FULL"
    APPROVE_PARTIAL = "APPROVE_PARTIAL"
    REJECT          = "REJECT"
    QUERY_VENDOR    = "QUERY_VENDOR"
    ESCALATE        = "ESCALATE"
    HOLD            = "HOLD"
    HYPOTHETICAL    = "HYPOTHETICAL"   # Training-only: explore counterfactual path


class ReasonCode(str, Enum):
    MATCH_CONFIRMED       = "MATCH_CONFIRMED"
    QUANTITY_MISMATCH     = "QUANTITY_MISMATCH"
    PRICE_DISCREPANCY     = "PRICE_DISCREPANCY"
    POLICY_VIOLATION      = "POLICY_VIOLATION"
    NO_PO_FOUND           = "NO_PO_FOUND"
    DUPLICATE_INVOICE     = "DUPLICATE_INVOICE"
    VENDOR_MISMATCH       = "VENDOR_MISMATCH"
    TAX_DISCREPANCY       = "TAX_DISCREPANCY"
    PENDING_CLARIFICATION = "PENDING_CLARIFICATION"
    MANAGER_REVIEW        = "MANAGER_REVIEW"


class APAction(BaseModel):
    decision: DecisionType
    approved_amount: float = Field(ge=0.0)
    reason_code: ReasonCode
    explanation: str = Field(min_length=10, max_length=600)


# ── Observation space ─────────────────────────────────────────────────────────

class APObservation(BaseModel):
    task_id: str
    task_name: str
    task_description: str
    invoice: Invoice
    purchase_orders: List[PurchaseOrder]
    goods_receipts: List[GoodsReceipt]
    company_policy: str
    paid_invoice_ids: List[str] = []
    step_count: int = 0
    max_steps: int = 1
    freight_cap: float = 50.0
    price_tolerance: float = 0.01
    action_history: List[Dict[str, Any]] = []
    context_notes: List[str] = []


# ── Reward model ──────────────────────────────────────────────────────────────

class APReward(BaseModel):
    score: float = Field(ge=0.01, le=0.99)
    breakdown: Dict[str, Any]
    feedback: str
    done: bool = True

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.01, min(0.99, float(v)))


# ── API wrappers ──────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy_perfect_match"
    session_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: APAction
    session_id: str


class StepResponse(BaseModel):
    observation: APObservation
    reward: APReward
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: APObservation
    session_id: str
    info: Dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    task_id: Optional[str]
    step_count: int
    episode_score: float
    done: bool
    current_observation: Optional[APObservation]


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str


# ── Actor models (multi-agent, Theme #1) ─────────────────────────────────────

class ActorPersona(str, Enum):
    honest      = "honest"
    fraudulent  = "fraudulent"
    confused    = "confused"


class ActorType(str, Enum):
    vendor     = "vendor"
    manager    = "manager"
    compliance = "compliance"


class ActorResponse(BaseModel):
    actor_type:  ActorType
    persona:     ActorPersona
    message:     str
    verdict:     str                 # e.g. "confirmed", "denied", "approved", "flagged"
    context_tag: str                 # prefix like "[VENDOR]", "[MANAGER]", "[COMPLIANCE]"


# ── Oversight models (Fleet AI, Theme #1 bonus) ───────────────────────────────

class EpisodeSummary(BaseModel):
    episode_id:        str
    task_id:           str
    invoice_id:        str
    vendor_name:       str
    invoice_total:     float
    currency:          str
    final_decision:    str
    approved_amount:   float
    reason_code:       str
    explanation:       str
    action_history:    List[Dict[str, Any]] = []
    is_fraudulent:     bool = False     # ground truth, hidden from agent in prompt
    fraud_type:        Optional[str] = None   # "duplicate", "price_inflation", "fake_vendor"


class OversightObservation(BaseModel):
    session_id:         str
    episode_summaries:  List[EpisodeSummary]
    known_fraud_patterns: List[str] = []    # hints from training corpus
    audit_budget:       int = 2             # max flags allowed this round
    step_count:         int = 0
    max_steps:          int = 5            # one verdict per episode
    action_history:     List[Dict[str, Any]] = []


class OversightAction(BaseModel):
    episode_id: str
    verdict:    Literal["CLEAR", "FLAG_FOR_REVIEW", "ESCALATE_TO_AUDIT"]
    signal:     str = Field(min_length=10, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)


class OversightReward(BaseModel):
    score:     float = Field(ge=0.01, le=0.99)
    breakdown: Dict[str, Any]
    feedback:  str
    done:      bool = False

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.01, min(0.99, float(v)))


class OversightResetRequest(BaseModel):
    seed: Optional[int] = None
    num_episodes: int = Field(default=5, ge=3, le=8)


class OversightResetResponse(BaseModel):
    observation: OversightObservation
    session_id:  str
    info:        Dict[str, Any]


class OversightStepRequest(BaseModel):
    action:     OversightAction
    session_id: str


class OversightStepResponse(BaseModel):
    observation: OversightObservation
    reward:      OversightReward
    done:        bool
    info:        Dict[str, Any]


# ── Curriculum models (Theme #4) ──────────────────────────────────────────────

class CurriculumEntry(BaseModel):
    task_id: str
    score:   float


class CurriculumRequest(BaseModel):
    session_history: List[CurriculumEntry] = []


class CurriculumResponse(BaseModel):
    recommended_task_id: str
    difficulty:          str
    reason:              str
    unlocked_tasks:      List[str] = []
