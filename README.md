---
title: Markov Agent Orchestrator
description: A pixel-art arena where multi-agent orchestration is treated as a stochastic decision problem, evolving from contextual bandits through cooperative Markov games to multi-agent reinforcement learning.
---

# Markov Agent Orchestrator

Most multi-agent frameworks hard-code the workflow: planner, then researcher, then critic, then done. This one treats "which agent should act next" as a **sequential decision problem under uncertainty** and learns the answer.

The orchestrator maintains an explicit Dirichlet belief over competing solution hypotheses, samples an action from a policy, observes a genuinely stochastic transition, and accumulates a decomposed reward. Then it renders the whole thing as a playable pixel-art game.

```text
      ┌─ Planner ─┐                    entropy fog clears as the belief sharpens
Research      Critic          ◄──►     agents ring a central core
      └─ Verifier ┘                    each invocation is a sampled outcome
```

## Why this exists

Three properties drive every design choice:

* The successor state is **sampled, never computed**. Replaying the same action from the same state lands somewhere else.
* Uncertainty is a **probability distribution**, so information gain is measured in bits rather than asserted.
* Every number on screen comes from a **persisted trace**, so any claim in the UI can be traced to the step that produced it.

## The measured result

The policy stack spans five implementations. The interesting question is whether the sophisticated ones earn their complexity, so [backend/tools/campaign.py](backend/tools/campaign.py) runs each policy twice over identical task instances — once carrying learned parameters between episodes, once starting fresh — and reports the paired difference.

| Policy | Stage | Δ carried − fresh | Reward trend across thirds |
| --- | --- | --- | --- |
| `contextual_bandit` | 1 | **−0.80 ± 0.26** ✱ | +2.17 → +2.20 → +1.93 |
| `mdp` | 2 | **+1.07 ± 0.44** ✱ | +0.51 → +2.56 → +2.48 |
| `markov_game` | 3 | +0.60 ± 0.38 | +0.40 → +1.10 → +1.97 |
| `marl` | 4 | **+1.13 ± 0.36** ✱ | +0.62 → +2.08 → +1.86 |
| `random` | control | 0.00 ± 0.00 | flat |

✱ exceeds two standard errors. 40 episodes per arm.

Two things worth reading twice. The contextual bandit gets **significantly worse** when it carries learning: LinUCB's ridge matrices accumulate, its exploration bonus shrinks, and it locks onto arm values fitted to task instances that no longer apply. That is stage-1 myopia showing up as a measured regression rather than a footnote.

And `random` returns *exactly* zero with zero variance, which is the control working: a policy with no state to carry must produce identical arms, and it does.

## The policy stack

| Stage | Policy | Mechanism | The limitation it exposes |
| --- | --- | --- | --- |
| 0 | `random`, `heuristic` | Uniform / hand-tuned scoring | No learning at all |
| 1 | `contextual_bandit` | Disjoint LinUCB over a 12-dimensional state context | Optimizes immediate reward; no credit across time |
| 2 | `mdp` | Tabular Q-learning plus a linear approximator, Boltzmann exploration | Agents are flat action labels |
| 3 | `markov_game` | Per-player values plus a learned pairwise synergy matrix | Cooperative case only |
| 4 | `marl` | VDN additive mixing, abstention baselines, difference rewards | Linear function approximation |

Stage 3 is where the framing changes: agents stop being action labels and become players. Coalition value is the sum of member values plus learned synergy, so the policy chooses **how many agents to fan out to**, rather than being told.

## What you get

* **Arena** — pixel sprites ringed around the core, entropy rendered as fog that clears as the belief sharpens, damage-number reward popups, combo counter, HP-style meters.
* **Graph** — React Flow interaction graph where edge width is message volume and labels carry the mean probability weight.
* **Rewards** — cumulative and per-step reward, entropy against information gain, the full reward decomposition, per-agent contribution with cost efficiency.
* **Traces** — every step, expandable into the policy's action distribution, the reward breakdown and the state transition.
* **Research Library** — 43 curated papers, 81 citation edges, a nine-category taxonomy, and live search across arXiv, Semantic Scholar, Papers With Code and an MCP tool provider.

## Quick start

The Python environment must live **outside** any cloud-synced folder. A venv inside OneDrive or Dropbox syncs tens of thousands of files and trips data-loss scanners on pip's vendored CA bundle.

```powershell
py -3 -m venv C:\venvs\markov-agent-orchestrator
C:\venvs\markov-agent-orchestrator\Scripts\Activate.ps1
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000> and press start. Or run both with `.\scripts\dev.ps1`.

> [!TIP]
> Do not run `npm run build` while `npm run dev` is live. They share `.next`, and the production build wipes the dev server's chunks.

## The agents are simulated

The six agents are stochastic processes, not language model calls. Each is a parameter set covering cost, latency, token draw, a Beta competence prior, evidence strength and noise; the transition kernel samples from those distributions.

No model SDK is used and no API key is required, so an episode costs nothing and is reproducible from its seed. **Token and dollar figures in the UI are sampled quantities, not billed ones.**

That is deliberate. Policy learning needs thousands of episodes, and paying for inference on each would make the research loop unaffordable and non-reproducible. The simulation sits behind a single `AgentReport` contract so it can be swapped for live execution without touching the decision layer — see stage 5 in [docs/ResearchRoadmap.md](docs/ResearchRoadmap.md), including the two fields that have no ground truth outside simulation.

The only outbound calls belong to the research providers, and all of them fall back to a local corpus when offline.

## Research Intelligence Layer

| Provider | Contribution |
| --- | --- |
| arXiv | Preprint coverage via the public Atom API |
| Semantic Scholar | Real citation counts and reference edges |
| Papers With Code | Which papers have reproducible implementations |
| `HITSMCPResearchProvider` | External MCP tool server over JSON-RPC |
| Curated corpus | Offline foundation, always available |

Providers **degrade rather than fail**: on a network error the response is served from the curated corpus and flagged `degraded`, which the UI surfaces instead of passing stale data off as live.

The MCP provider stays useful with no endpoint configured — it ranks the local corpus using Kleinberg HITS over the citation graph, separating authorities from hubs.

## Layout

```text
backend/
  app/orchestration/   State, actions, stochastic transitions, rewards, policies, engine
  app/research/        Provider abstraction + arXiv / Semantic Scholar / PwC / HITS MCP
  app/api/             REST + WebSocket routers
  tools/               balance.py (single-episode probe), campaign.py (cross-episode)
  tests/               39 tests
frontend/
  app/                 Title screen, orchestrator arena, research library
  components/game/     Arena, HUD, Controls, GraphView, RewardDashboard, TraceExplorer
  components/pixel/    Sprite renderer with animation playback
docs/                  Architecture, ResearchRoadmap, DesignDecisionLog
research/              Curated offline corpus
scripts/               dev.ps1, fetch-sprites.ps1
```

## Testing and measurement

```powershell
cd backend
python -m pytest -q                                  # 39 tests
python -m tools.balance --episodes 40                # single-episode balance
python -m tools.campaign --episodes 40               # cross-episode learning
```

The balance harness exists because playing the game revealed what the tests could not: the win condition was originally unreachable in 100% of episodes. Agents emit correct evidence roughly 75% of the time, so `p(truth)` asymptotes near 0.65 and the original target of 0.88 was impossible by construction. That story is recorded in [docs/DesignDecisionLog.md](docs/DesignDecisionLog.md).

## Art

Sprites were generated with [PixelLab](https://www.pixellab.ai) via its MCP server and are re-fetchable with `.\scripts\fetch-sprites.ps1`, which pulls each character archive and regenerates the manifest. The renderer falls back to hand-authored procedural SVG sprites when no generated art is present, so the app never depends on the asset pipeline having run.

The roster is styled after a corporate-hacker-noir aesthetic using original archetypes rather than any protected character designs.

## Documentation

* [docs/Architecture.md](docs/Architecture.md) — decision formalism, transition kernel, reward decomposition, persistence
* [docs/ResearchRoadmap.md](docs/ResearchRoadmap.md) — the staged path and what each stage removes
* [docs/DesignDecisionLog.md](docs/DesignDecisionLog.md) — 13 decisions with rejected alternatives and accepted costs
