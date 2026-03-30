"""
AP Clerk Environment — Typed Models
All OpenEnv-required models: Observation, Action, Reward.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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


class ReasonCode(str, Enum):
    MATCH_CONFIRMED    = "MATCH_CONFIRMED"
    QUANTITY_MISMATCH  = "QUANTITY_MISMATCH"
    PRICE_DISCREPANCY  = "PRICE_DISCREPANCY"
    POLICY_VIOLATION   = "POLICY_VIOLATION"
    NO_PO_FOUND        = "NO_PO_FOUND"
    DUPLICATE_INVOICE  = "DUPLICATE_INVOICE"


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


# ── Reward model ──────────────────────────────────────────────────────────────

class APReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, Any]
    feedback: str
    done: bool = True


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
