# 🌊 CrisisFlow: Intelligent Flood Response Environment

CrisisFlow is a reinforcement learning-style simulation environment designed to model real-world flood disaster management. It enables AI agents to make strategic decisions such as rescue dispatch, alert systems, and resource allocation under dynamic and uncertain conditions.

---

## 🚀 Key Features

* 🌊 **Dynamic Flood Simulation**

  * Non-linear flood growth
  * Cascading effects across neighboring zones
  * Time-based escalation

* 🧠 **Agent Decision Framework**

  * Dispatch rescue teams
  * Send alerts to vulnerable zones
  * Allocate critical resources (medical, food, shelter)

* 🎯 **Reward System**

  * Multi-factor scoring:

    * People saved
    * Decision correctness
    * Resource efficiency
    * Early action bonus
  * Penalties for delays, invalid actions, and poor prioritization

* 🔥 **Realistic Constraints**

  * Resource decay over time
  * Neglect penalties for ignored zones
  * Trade-offs between urgency and efficiency

---

## 🏗️ Architecture

```
CrisisFlow/
│
├── app/
│   ├── main.py           # FastAPI server
│   ├── environment.py    # Core simulation logic
│   ├── reward.py         # Reward shaping
│   ├── tasks.py          # Task definitions
│   ├── models.py         # Data schemas
│
├── inference.py          # Agent loop (LLM-based)
├── Dockerfile            # Deployment
├── openenv.yaml          # OpenEnv compliance
```

---

## ⚙️ API Endpoints

* `POST /reset` → Initialize environment
* `POST /step` → Apply action
* `GET /state` → Get current state
* `GET /health` → Health check

---

## 🧪 Running Locally

### 1. Build Docker Image

```
docker build -t crisisflow .
```

### 2. Run Server

```
docker run -p 7860:7860 crisisflow
```

### 3. Test Health

```
http://localhost:7860/health
```

---

## 🤖 Running the Agent

Set environment variables:

```
API_BASE_URL=http://localhost:7860
MODEL_NAME=gpt-4o-mini
HF_TOKEN=your_token_here
```

Run:

```
python inference.py
```

---

## 🧠 Design Philosophy

CrisisFlow focuses on:

* Real-world realism over toy simulations
* Multi-objective optimization
* Strategic planning under constraints

Unlike static environments, it introduces:

* cascading disasters
* delayed consequences
* resource trade-offs

---

## 🏆 Hackathon Highlights

* OpenEnv-compliant environment
* Deterministic yet complex dynamics
* Strong reward shaping for agent learning
* Designed for real-world disaster scenarios

---

## 🚀 Future Improvements

* Multi-agent coordination
* Reinforcement learning training loop
* Real-world data integration

---

## 👨‍💻 Author

Built with focus on intelligent systems and real-world impact.
