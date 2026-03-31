# Reward Hacking Detector in Toy RL Environments

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A personal project to learn reinforcement learning (RL). With the help of scaffolding TODOs from Claude (see [prompt](reward_hacking_project_spec.md) and [this post]() recounting my experiences), this project implements some RL algorithms from scratch in custom GridWorld environments using [Gymnasium](gymnasium.farama.org). We use FastAPI for our backend and an interactive Streamlit dashboard for our frontend. We also implement a custom renderer in Matplotlib.

In particular, this project studies _reward hacking_ (see [section 4 of this paper](https://arxiv.org/pdf/1606.06565)). Reward hacking is an incarnation of Goodhart's law whereby an agent learns to maximise the measurable proxy at the expense of the actual goal. A detection pipeline then flags hacking behaviour using policy divergence (KL divergence vs. a BFS-optimal reference), trajectory clustering, and reward decomposition.

> [GIF placeholder — add side-by-side navigation GIF here]
> Left: agent with low proxy reward navigates directly to goal. Right: agent with high proxy reward loops to collect coins, ignoring the goal.

<!-- ---

## Key Results

*(Fill in after running the parameter sweep.)*

| Coin Reward | Q-Learning Hacking Rate | DQN Hacking Rate |
|-------------|------------------------|------------------|
| 0.1 | — | — |
| 0.5 | — | — |
| 1.0 | — | — |
| 5.0 | — | — |

At coin reward >= X, Y% of agents exhibit reward hacking as measured by the Proxy Reliance Score (PRS > 0.7 threshold). -->

---

## Installation

```bash
git clone https://github.com/your-username/reward-hacking-detector.git
cd reward-hacking-detector
pip install -r requirements.txt
```

For development dependencies (linting, type checking, tests):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

**1. Train a Q-learning agent:**
```bash
python -m src.cli train --config training_default --agent q_learning --episodes 1000
```

**2. Evaluate against test environments:**
```bash
python -m src.cli evaluate --experiment-id <id> --test-env test_coin_moved
```

**3. Launch the interactive dashboard:**
```bash
streamlit run dashboard/streamlit_app.py
```

---

## Project Structure

```
reward_hacking/
├── src/
│   ├── environment/        # GridWorld env, configs, renderer
│   ├── agents/             # Optimal (BFS), Q-learning, DQN
│   ├── detection/          # Proxy Reliance Score, KL divergence, clustering
│   ├── experiments/        # Training loop, evaluation, parameter sweep
│   └── api/                # FastAPI REST backend
├── dashboard/              # Streamlit interactive dashboard
├── notebooks/              # Exploratory analysis notebook
├── tests/                  # pytest suite
├── docs/
│   └── METHODOLOGY.md      # Experimental design and detection metrics
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

<!-- ---

## Background: What is Reward Hacking?

Goodhart's Law states: *"When a measure becomes a target, it ceases to be a good measure."* In reinforcement learning, the reward function is that measure. Designing a reward function that perfectly captures human intent is extremely difficult in practice, so RL practitioners often specify a *proxy* reward — a measurable quantity that is hoped to correlate with the true objective. Reward hacking occurs when an agent discovers a strategy that scores highly on the proxy without satisfying the underlying intent.

The phenomenon is not limited to toy examples. Real RL systems have learned to exploit game physics for points rather than play as intended, manipulate simulated muscles to appear to walk without actually locomoting, and exploit simulator bugs to achieve astronomical scores that correspond to no coherent behaviour. In each case, the agent did exactly what it was rewarded to do — the reward function was wrong, not the agent. This framing is sometimes called *proxy misalignment*: the proxy and the true objective diverge precisely in the situations where the agent is most capable.

For AI safety, reward hacking is not a curiosity — it is a core alignment problem. As AI systems become more capable, the gap between a flawed proxy and the true objective becomes exploitable in increasingly subtle and consequential ways. Understanding when and why reward hacking occurs, and building reliable detectors for it, is a necessary step toward scalable oversight of advanced AI systems. This project studies the phenomenon in a controlled, reproducible setting where the ground truth is known, building intuitions and tools that transfer to harder problems. -->

<!-- ---

## Links

- [Methodology Document](docs/METHODOLOGY.md) — experimental design, metric formulas, results tables
- [Analysis Notebook](notebooks/analysis.ipynb) — plots and interpretation
- [Dashboard](http://localhost:8501) — live after `streamlit run dashboard/streamlit_app.py`
- [REST API Docs](http://localhost:8000/docs) — live after `uvicorn src.api.app:app --reload` -->

---

## License

MIT License. See [LICENSE](LICENSE) for details.
