from typing import Dict

def grade_alert_identification(initial_state: Dict, final_state: Dict) -> float:
    total_high_critical = 3
    correct_alerts = 0
    false_alerts = 0
    
    alert_history = final_state.get("alert_history", {})
    first_alert_step = final_state.get("first_alert_step", final_state["max_steps"])
    
    for zone in initial_state["zones"]:
        z_id = zone["id"]
        f_lvl = zone["flood_level"]
        
        expected_priority = "LOW"
        if f_lvl >= 0.75: expected_priority = "CRITICAL"
        elif f_lvl >= 0.5: expected_priority = "HIGH"
        elif f_lvl >= 0.3: expected_priority = "MEDIUM"
        
        if z_id in alert_history:
            priority = alert_history[z_id]["priority"]
            if f_lvl >= 0.5 and priority == expected_priority:
                correct_alerts += 1
            if f_lvl < 0.3 and priority != "LOW":
                false_alerts += 1
                
    alert_score = correct_alerts / total_high_critical
    no_false_alerts = 1.0 if false_alerts == 0 else 0.0
    timeliness = 1.0 - (first_alert_step / final_state["max_steps"])
    
    score = alert_score * 0.60 + no_false_alerts * 0.20 + timeliness * 0.20
    return max(0.0, min(1.0, score))


def grade_resource_prioritization(initial_state: Dict, final_state: Dict) -> float:
    survival_ratio = final_state["people_saved"] / final_state["total_population"]
    
    initial_avail = 44 # 4+15+15+10 from initial config
    total_used = final_state.get("total_resources_used", 0)
    resource_use = total_used / initial_avail
    
    total_rescues = final_state.get("total_rescues_dispatched", 0)
    high_rescues = final_state.get("high_zone_rescues", 0)
    priority_score = high_rescues / total_rescues if total_rescues > 0 else 0.0
    
    time_score = 1.0 if final_state["time_step"] < final_state["max_steps"] else 0.5
    
    # Heavier on survival ratio allowing good RL agents to score .6 to .8
    score = survival_ratio * 0.45 + resource_use * 0.25 + priority_score * 0.20 + time_score * 0.10
    return max(0.0, min(1.0, score))


def grade_full_crisis_management(initial_state: Dict, final_state: Dict) -> float:
    survival_ratio = final_state["people_saved"] / final_state["total_population"]
    
    zones_alerted = sum(1 for z in final_state["zones"] if z["alert_sent"])
    alert_coverage = zones_alerted / 5.0
    if zones_alerted < 3:
        alert_coverage *= 0.5
    
    rescues = max(1, final_state.get("total_rescues_dispatched", 1))
    rescue_efficiency = min(1.0, final_state["people_saved"] / (rescues * 1000))
    
    total_actions = max(1, final_state.get("total_actions_taken", 1))
    effective_actions = final_state.get("effective_resources_used", 0)
    resource_efficiency = effective_actions / total_actions
    
    time_penalty = final_state["time_step"] / final_state["max_steps"]
    
    score = (
        survival_ratio * 0.45 +
        alert_coverage * 0.10 +
        rescue_efficiency * 0.20 +
        resource_efficiency * 0.15 +
        (1.0 - time_penalty) * 0.10
    )
    return max(0.0, min(1.0, score))

TASKS = {
    "alert_identification": {
        "id": "alert_identification",
        "name": "Alert Identification",
        "description": "A flood is forming. Identify all HIGH or CRITICAL risk zones and send correctly-prioritized alerts to each before time runs out. Relies strictly on diagnostic evaluation.",
        "difficulty": "easy",
        "max_steps": 5,
        "grader_fn": grade_alert_identification,
        "initial_config": {
            "zones": [
                {"id": "NORTH", "flood_level": 0.6, "population": 2000},
                {"id": "SOUTH", "flood_level": 0.6, "population": 1500},
                {"id": "EAST", "flood_level": 0.8, "population": 3000},
                {"id": "WEST", "flood_level": 0.1, "population": 1000},
                {"id": "CENTER", "flood_level": 0.1, "population": 1200}
            ],
            "resources": {"rescue_teams": 3, "medical": 10, "food": 10, "shelter": 10}
        }
    },
    "resource_prioritization": {
        "id": "resource_prioritization",
        "name": "Resource Prioritization",
        "description": "Allocate severely limited supplies across randomized zones. Resources undergo active decay if stalled! Prioritize deployment properly to succeed.",
        "difficulty": "medium",
        "max_steps": 8,
        "grader_fn": grade_resource_prioritization,
        "initial_config": {
            "zones": [
                {"id": "NORTH", "flood_level": 0.4, "population": 500},
                {"id": "SOUTH", "flood_level": 0.5, "population": 1200},
                {"id": "EAST", "flood_level": 0.6, "population": 3000},
                {"id": "WEST", "flood_level": 0.7, "population": 800},
                {"id": "CENTER", "flood_level": 0.8, "population": 2200}
            ],
            "resources": {"rescue_teams": 4, "medical": 15, "food": 15, "shelter": 10}
        }
    },
    "full_crisis_management": {
        "id": "full_crisis_management",
        "name": "Full Crisis Management",
        "description": "Devastating, cascading flood approaches. Manage alerts, dynamic unannounced resource caches, and rescue timings. CRITICAL zones heavily cascade floodwaters into connected neighbors accelerating destruction. Planning ahead is the only way an agent succeeds.",
        "difficulty": "hard",
        "max_steps": 12,
        "grader_fn": grade_full_crisis_management,
        "initial_config": {
            "zones": [
                {"id": "NORTH", "flood_level": 0.3, "population": 1000},
                {"id": "SOUTH", "flood_level": 0.4, "population": 4000},
                {"id": "EAST", "flood_level": 0.5, "population": 2500},
                {"id": "WEST", "flood_level": 0.6, "population": 800},
                {"id": "CENTER", "flood_level": 0.65, "population": 3200}
            ],
            "resources": {"rescue_teams": 5, "medical": 20, "food": 20, "shelter": 15}
        }
    }
}
