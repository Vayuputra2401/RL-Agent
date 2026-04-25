import requests
import time
import os
import json

BASE_URL = "http://localhost:7860"


def wait_for_server():
    print("[INFO] Waiting for server...")
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print("[INFO] Server is ready")
                return
        except:
            pass
        time.sleep(1)
    raise Exception("Server failed to start")


def safe_post(url, payload):
    for _ in range(3):
        try:
            r = requests.post(url, json=payload, timeout=5)
            if r.status_code != 200:
                raise Exception(f"Bad status: {r.status_code}")
            return r.json()
        except Exception as e:
            print(f"[WARN] Retry request: {e}")
            time.sleep(1)
    return None


def call_llm(prompt):
    try:
        response = requests.post(
            f"{os.environ.get('API_BASE_URL')}/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=10
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[WARN] LLM bad response: {response.status_code}")
            return None

    except Exception as e:
        print(f"[WARN] LLM call failed: {e}")
        return None


def run_task(task_id, max_steps):
    print(f"[START] task={task_id}", flush=True)

    reset = safe_post(f"{BASE_URL}/reset", {"task_id": task_id})
    if not reset:
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return

    session_id = reset.get("session_id")
    if not session_id:
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return

    rewards = []
    done = False
    step = 0

    while not done and step < max_steps:

        prompt = f"""
You are managing a crisis response system.
Decide the best action for task: {task_id}.

Respond ONLY in JSON:
{{
  "action_type": "wait",
  "zone_id": null,
  "resource_type": null,
  "resource_amount": null,
  "priority": null
}}
"""

        llm_output = call_llm(prompt)

        try:
            action = json.loads(llm_output) if llm_output else {}
        except:
            action = {}

        if not action:
            action = {
                "action_type": "wait",
                "zone_id": None,
                "resource_type": None,
                "resource_amount": None,
                "priority": None
            }

        result = safe_post(
            f"{BASE_URL}/step",
            {
                "session_id": session_id,
                "action": action
            }
        )

        if not result:
            print(f"[STEP] step={step+1} reward=0.0", flush=True)
            break

        reward = result.get("reward", {}).get("score", 0.0)
        done = result.get("done", True)

        rewards.append(reward)
        step += 1

        print(f"[STEP] step={step} reward={reward}", flush=True)

        time.sleep(0.2)

    score = sum(rewards)/len(rewards) if rewards else 0.0

    print(f"[END] task={task_id} score={score} steps={step}", flush=True)


def main():
    try:
        wait_for_server()

        tasks = {
            "alert_identification": 5,
            "resource_prioritization": 8,
            "full_crisis_management": 12
        }

        for task, steps in tasks.items():
            run_task(task, steps)

        print("[INFO] Inference completed")

    except Exception as e:
        print(f"[FATAL] {e}")


if __name__ == "__main__":
    main()