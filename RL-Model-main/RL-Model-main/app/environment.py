import random
import uuid
import copy
import math
from typing import Dict, Tuple

from app.models import CrisisAction, ZoneState, CrisisObservation, ResetResult, StepResult
from app.reward import compute_step_reward
from app.tasks import TASKS

NEIGHBORS = {
    "NORTH": ["CENTER", "WEST"],
    "SOUTH": ["CENTER", "EAST"],
    "EAST": ["CENTER", "SOUTH"],
    "WEST": ["CENTER", "NORTH"],
    "CENTER": ["NORTH", "SOUTH", "EAST", "WEST"]
}

class CrisisFlowEnvironment:
    def __init__(self, task_id: str, seed: int):
        self.task_id = task_id
        self.seed = seed if seed is not None else random.randint(0, 999999)
        self.rng = random.Random(self.seed)

        task = TASKS[self.task_id]
        self.task_name = task["name"]
        self.task_description = task["description"]
        self.max_steps = task["max_steps"]
        self.initial_config = copy.deepcopy(task["initial_config"])

    def _reset_internal(self):
        self.zones = []
        for z in self.initial_config["zones"]:
            f_level = z["flood_level"]
            self.zones.append({
                "id": z["id"],
                "flood_level": f_level,
                "population": z["population"],
                "survivors": z["population"],
                "risk_level": self._compute_risk_level(f_level),
                "rescue_teams_deployed": 0,
                "alert_sent": False,
                "resources_allocated": {"medical": 0, "food": 0, "shelter": 0},
                "is_evacuated": False,
                "neglected_steps": 0
            })

        self.available_resources = copy.deepcopy(self.initial_config["resources"])
        self.time_step = 0
        self.alerts_sent = []
        self.people_saved = 0
        self.total_population = sum(z["population"] for z in self.zones)

        self.total_actions_taken = 0
        self.valid_actions_taken = 0
        self.total_resources_used = 0
        self.effective_resources_used = 0
        self.total_rescues_dispatched = 0
        self.high_zone_rescues = 0
        self.alert_history = {}
        self.first_alert_step = 999

    def reset(self) -> ResetResult:
        self._reset_internal()
        self.session_id = str(uuid.uuid4())
        return ResetResult(observation=self._get_observation(), session_id=self.session_id)

    def _get_observation(self) -> CrisisObservation:
        zones = [ZoneState(**z) for z in self.zones]
        return CrisisObservation(
            zones=zones,
            available_resources=self.available_resources,
            time_step=self.time_step,
            max_steps=self.max_steps,
            people_saved=self.people_saved,
            total_population=self.total_population,
            task_id=self.task_id,
            task_name=self.task_name,
            task_description=self.task_description,
            alerts_sent=self.alerts_sent,
            episode_seed=self.seed,
            step_budget_remaining=self.max_steps - self.time_step
        )

    def state(self) -> dict:
        return {
            "zones": copy.deepcopy(self.zones),
            "available_resources": copy.deepcopy(self.available_resources),
            "time_step": self.time_step,
            "max_steps": self.max_steps,
            "people_saved": self.people_saved,
            "total_population": self.total_population,
            "total_actions_taken": self.total_actions_taken,
            "valid_actions_taken": self.valid_actions_taken,
            "total_resources_used": self.total_resources_used,
            "effective_resources_used": self.effective_resources_used,
            "total_rescues_dispatched": self.total_rescues_dispatched,
            "high_zone_rescues": self.high_zone_rescues,
            "alert_history": copy.deepcopy(self.alert_history),
            "first_alert_step": self.first_alert_step
        }

    def _compute_risk_level(self, f):
        if f < 0.3: return "LOW"
        elif f < 0.6: return "MEDIUM"
        elif f < 0.75: return "HIGH"
        else: return "CRITICAL"

    def _validate_action(self, action: CrisisAction) -> Tuple[bool, str]:
        if action.action_type not in ["dispatch_rescue", "send_alert", "allocate_resource", "wait"]:
            return False, "Invalid action"

        if action.action_type != "wait" and not action.zone_id:
            return False, "Missing zone"

        return True, "Valid"

    def step(self, action: CrisisAction) -> StepResult:
        pre_state = self.state()
        self.total_actions_taken += 1

        is_valid, reason = self._validate_action(action)
        if is_valid:
            self.valid_actions_taken += 1

            if action.action_type == "dispatch_rescue":
                zone = next(z for z in self.zones if z["id"] == action.zone_id)

                if zone["flood_level"] > 0.85:
                    saved = math.floor(0.2 * zone["survivors"])
                elif zone["flood_level"] > 0.6:
                    saved = math.floor(0.3 * zone["survivors"])
                else:
                    saved = math.floor(0.4 * zone["survivors"])

                zone["survivors"] -= saved
                self.people_saved += saved

        self._apply_dynamics()
        self.time_step += 1

        reward = compute_step_reward(action, pre_state, self.state(), is_valid, False)

        done = self.time_step >= self.max_steps
        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=done,
            info={"reason": reason}
        )

    def _apply_dynamics(self):
        for z in self.zones:
            if z["flood_level"] < 1.0:
                inc = self.rng.uniform(0.05, 0.15)
                if z["flood_level"] > 0.6:
                    inc *= 1.5
                z["flood_level"] += inc

            if z["rescue_teams_deployed"] == 0:
                z["neglected_steps"] += 1
            else:
                z["neglected_steps"] = 0

            if z["neglected_steps"] > 2:
                loss = math.floor(z["survivors"] * 0.03)
                z["survivors"] -= loss

        for r in self.available_resources:
            self.available_resources[r] -= math.floor(self.available_resources[r] * 0.05)
