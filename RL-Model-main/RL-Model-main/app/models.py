from pydantic import BaseModel
from typing import List, Optional, Dict


class CrisisAction(BaseModel):
    action_type: str
    zone_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_amount: Optional[int] = None
    priority: Optional[str] = None


class ZoneState(BaseModel):
    id: str
    flood_level: float
    population: int
    survivors: int
    risk_level: str
    rescue_teams_deployed: int
    alert_sent: bool
    resources_allocated: Dict[str, int]
    is_evacuated: bool
    neglected_steps: int


class CrisisObservation(BaseModel):
    zones: List[ZoneState]
    available_resources: Dict[str, int]
    time_step: int
    max_steps: int
    people_saved: int
    total_population: int
    task_id: str
    task_name: str
    task_description: str
    alerts_sent: List[str]
    episode_seed: int
    step_budget_remaining: int


class ResetResult(BaseModel):
    observation: CrisisObservation
    session_id: str


class CrisisReward(BaseModel):
    score: float
    people_saved_ratio: float
    correct_decisions: float
    resource_efficiency: float
    time_penalty: float
    step_reward: float
    penalties: float
    feedback: str
    breakdown: Dict


class StepResult(BaseModel):
    observation: CrisisObservation
    reward: CrisisReward
    done: bool
    info: Dict