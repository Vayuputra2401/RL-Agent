from typing import Dict
from app.models import CrisisAction, CrisisReward

def compute_step_reward(action, pre_state, post_state, is_valid, is_done):
    time_step = post_state["time_step"]
    max_steps = post_state["max_steps"]

    total_population = post_state["total_population"]
    people_saved = post_state["people_saved"]

    people_saved_ratio = people_saved / total_population if total_population else 0

    correct_decisions = post_state["valid_actions_taken"] / max(1, post_state["total_actions_taken"])

    resource_efficiency = post_state.get("effective_resources_used", 0) / max(1, post_state.get("total_resources_used", 1))

    early_bonus = max(0, 1 - time_step / max_steps)

    step_reward = (
        people_saved_ratio * 0.5 +
        correct_decisions * 0.2 +
        resource_efficiency * 0.2 +
        early_bonus * 0.1
    )

    penalties = 0

    if not is_valid:
        penalties += 0.1

    if action.action_type == "wait":
        penalties += 0.02 * time_step

    if time_step > max_steps * 0.7:
        penalties += 0.05

    return CrisisReward(
        score=max(0, min(1, step_reward - penalties)),
        people_saved_ratio=people_saved_ratio,
        correct_decisions=correct_decisions,
        resource_efficiency=resource_efficiency,
        time_penalty=0,
        step_reward=step_reward,
        penalties=penalties,
        feedback="",
        breakdown={}
    )