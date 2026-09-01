---
title: Research Roadmap
description: The cooperative stochastic game the arena implements today, the escalation gate that makes it a decision under uncertainty, and the assumptions each open research direction would relax.
author: Markov Agent Orchestrator
ms.date: 2026-09-01
ms.topic: concept
keywords:
  - markov game
  - stochastic game
  - extensive form game
  - agent orchestration
  - research roadmap
estimated_reading_time: 14
---

## How to read this roadmap

This document used to describe a ladder: a contextual bandit, then an MDP, then a Markov game, then multi-agent RL, each stage removing a limitation of the one below it. Three of those rungs have been removed from the registry, so the ladder framing has been removed with them. DDL-025 records why.

What replaces it is flatter and easier to argue with. The first half describes the game the arena implements now, which is a cooperative stochastic game with an escalation gate. The second half lists open directions, each named by the assumption it would relax rather than by its position in a sequence, because nothing here establishes that those assumptions have to fall in a particular order.

The corresponding literature lives in the Research Library and is seeded from [research/seed_corpus.json](../research/seed_corpus.json). The library still carries the bandit, MDP and MARL categories, and should: the papers remain relevant to the problem even where the implementations are gone.

## The game

An episode is a finite-horizon stochastic game with one strategic player. Nature draws a latent hypothesis `h*` before anything else happens; the orchestrator never observes it and acts throughout on a Dirichlet belief over the `K` candidates. Each move it makes selects a coalition, nature resolves what that coalition produced, and the belief updates on the evidence returned.

Two features distinguish it from a plain MDP over agent labels. The action set is state-dependent, because escalation gates it. And the payoff of a move is not additive over the agents in it, which is what makes the coalition rather than the agent the unit of choice.

### The escalation gate is the first move

Before escalation, the legal set is `INVOKE_GENERALIST`, `ESCALATE` and `TERMINATE`. The specialists are unreachable. Escalating costs budget and cannot be undone, and `min_solo_steps` requires a solo attempt first, so the gate is a decision informed by an observation rather than a switch thrown at step zero.

That single restriction is what makes the whole thing an experiment rather than a workflow engine. "Should this be orchestrated at all" is the question the arena exists to answer, so it has to be a move in the game rather than a configuration value. [docs/Architecture.md](Architecture.md) draws the resulting tree in extensive form, and the arena renders any finished run in the same shape from its persisted traces.

### The fixed rules that bracket it

Four policies do not learn, and they are not placeholders. Any claim that a learned policy helps is a comparative claim, and these are what it is compared against.

`single_agent` never escalates and is the control every experiment is paired on. `fixed_sequence` escalates immediately and walks a hardcoded rotation, which is the upper bookend on orchestration cost and, in simulation, a surprisingly strong one. `heuristic` escalates on stall and encodes the conventional wisdom worth beating: research when uncertain, criticize when quality lags, verify before terminating. `random` is the floor, and a learned policy that cannot clear it has not learned anything worth carrying.

Implemented in [backend/app/orchestration/policies/baselines.py](../backend/app/orchestration/policies/baselines.py), [single_agent.py](../backend/app/orchestration/policies/single_agent.py) and [fixed_sequence.py](../backend/app/orchestration/policies/fixed_sequence.py).

### The learned player

Agents are players rather than action labels. Each has a value function over the shared state, and a learned pairwise synergy matrix captures what an additive decomposition cannot.

$$
V(s, C) = \sum_{i \in C} Q_i(s) + \sum_{i < j \in C} W_{ij} - \lambda (|C| - 1) \cdot \text{cost pressure}(s)
$$

The score for `RUN_PARALLEL` is the best multi-player coalition value, so coalition size is an output of the policy rather than an input. Synergy absorbs the residual between observed reward and the additive prediction, which is where a pairing like Research plus Verification becomes measurable rather than assumed. Coalitions are enumerated up to size three, which is 41 subsets over six agents.

The scope limit is real and worth stating plainly. This is the cooperative branch only: one shared reward, no adversarial or mixed-motive dynamics, and a coordination structure fixed at pairwise. DDL-017 goes further and argues that even the cooperative Markov game framing is generous, because the specialists move when instructed rather than simultaneously. Read this policy as a coalition-valuation mechanism that the game framing motivated, not as evidence that the framing is correct.

Key references: Shapley 1953 on stochastic games, Littman 1994 on the Markov game framing, Hu and Wellman 2003 on Nash Q-learning, Littman 2001 on the friend branch of general-sum games, Oliehoek and Amato 2016 on Dec-POMDPs.

Implemented in [backend/app/orchestration/policies/markov_game.py](../backend/app/orchestration/policies/markov_game.py).

### What was removed, and what the literature still says

Three learned policies were deleted: disjoint LinUCB over the state context, tabular Q-learning blended with a linear approximator, and VDN-style multi-agent RL with abstention baselines and difference rewards. Over 40 episodes all three escalated in every episode while expressing no coalition, which is `always_orchestrate` arrived at by a more expensive route. DDL-025 records the decision and its cost.

Their motivating problems did not go away with the code, and neither did the papers. Choosing which worker to invoke is still a contextual bandit problem, and the reason the bandit undervalued the Planner and the Memory agent, whose payoff arrives several steps later, is still the reason a purely myopic router will misprice delayed work. What is gone is the claim that this repository has measured any of that.

* Bandit framing: Li et al. 2010 on LinUCB, Auer et al. 2002 on finite-time analysis, Chu et al. 2011 on linear payoff regret, Agrawal and Goyal 2013 on Thompson sampling.
* Sequential credit: Watkins and Dayan 1992 on Q-learning convergence, Kaelbling et al. 1998 on belief-state planning, Ng et al. 1999 on potential-based shaping, Sutton et al. 1999 on temporal abstraction.
* Per-agent credit: Sunehag et al. 2018 on value decomposition networks, Foerster et al. 2018 on counterfactual policy gradients, Rashid et al. 2018 on monotonic mixing, Son et al. 2019 on relaxing factorization constraints, Yu et al. 2022 on on-policy alternatives, Zhang et al. 2021 for the theoretical map.

## Live agent execution

Everything above can be learned against a generative model of agent behavior. The agents are stochastic processes parameterized in [backend/app/orchestration/agents.py](../backend/app/orchestration/agents.py), not model calls. That is deliberate: policy learning needs thousands of episodes, and paying for inference on every one would make the research loop unaffordable and non-reproducible.

Live mode replaces the generative model with real execution while leaving the decision layer untouched. It is implemented, not planned: a step splits into `live/open`, which asks the policy who should act, and `live/report`, which folds in what a real agent actually produced.

### The seam it went through

The entire simulation was already confined behind one dataclass. `TransitionModel._invoke` produces an `AgentReport`, and nothing downstream knows or cares how that report was produced:

```python
@dataclass
class AgentReport:
    agent_id: str
    outcome: str                 # success | partial | failure
    outcome_probability: float
    competence_sample: float
    cost_usd: float
    latency_ms: float
    tokens: int
    evidence_mass: float
    correct_evidence: bool
    summary: str
    source: str = "simulated"    # added for live mode
    metered: bool = False        # added for live mode
    claimed_hypothesis: int | None = None
```

This roadmap previously predicted an `AgentExecutor` protocol with a `SimulatedExecutor` and a `ModelExecutor` behind it. That is not what shipped, and the difference is instructive. An in-process executor assumes the backend can call a model, and it cannot: the thing doing the work is a coding agent driving the arena over HTTP, so the arena is called by the model rather than calling it.

The step therefore splits in half instead. `live/open` runs the policy and returns briefs without advancing the run; `live/report` takes what the agent actually produced and folds it in through `reports_override`. The prediction that the reward model, entropy accounting, trace schema, WebSocket protocol and frontend would need no changes held. The prediction about where the seam would sit did not.

Three fields were added rather than reused. `source` and `metered` mark a report as measured rather than sampled, because a number whose provenance is unrecorded cannot be defended later, and `claimed_hypothesis` is what replaced the oracle.

### The problem it exposed, and the answer taken

Two fields had no ground truth outside simulation.

`correct_evidence` compares against a latent hypothesis the simulator knows, and production has no such oracle. Three replacements were considered: verifier-scored evidence, which is cheap but circular when the verifier is wrong; self-consistency across independent samples, well studied but multiplying cost per invocation; and a held-out judge scoring against retrieved sources, the strongest signal and the most expensive.

The second was taken, in a form the two-phase step made cheap. A live report carries `claimed_hypothesis`, the index the agent's own work supports, and belief mass follows the claim. Confidence therefore rises only when independent agents converge on the same hypothesis, and disagreement raises entropy rather than lowering it. Truth is not asserted anywhere; it emerges from agreement or fails to. `test_disagreement_raises_entropy_more_than_agreement` holds that property.

Two consequences are worth stating. An agent that confidently asserts nonsense moves the belief until something disputes it, which is the principal-agent problem of DDL-017 arriving as a measurable outcome. And an agent whose work supports none of the hypotheses should omit the field rather than guess, because a guessed index defeats the mechanism entirely.

`evidence_mass` remains open. In live mode it is still derived from a per-agent constant scaled by outcome and confidence rather than estimated from the response. Estimating it properly means measuring how much retrieved content shifts a belief encoder, which is the practical form of the mutual-information acquisition criterion in Houlsby et al. 2011. Until that exists, live evidence weight is a modelled quantity sitting inside an otherwise measured pipeline, and that is the honest description of it.

### What live mode does hold

* Simulated and live execution are selected per run and never pooled in a comparison.
* A live episode produces the same trace schema as a simulated one.
* Cost, latency and tokens come from measured values, and a report that omits them is refused rather than filled in from the agent spec. DDL-015 records the fabricated experiment that forced this.

## Open directions

Each of these relaxes one assumption the implementation currently makes. They are listed in no particular order, because nothing measured here says which assumption binds hardest.

### Relaxing full observability

The policy sees the whole state. Real agents return partial, noisy views of a task, which makes this a Dec-POMDP rather than a fully observed Markov game.

The work is to add per-agent observation functions, replace the shared state with per-agent belief states, and move to recurrent or history-conditioned value estimation. Oliehoek and Amato 2016 is the reference treatment, and Kaelbling et al. 1998 provides the single-agent foundation already reflected in the Dirichlet belief.

### Relaxing model-free control

The policies are model-free, yet the transition kernel is known and cheap to sample, which makes it an unusually good candidate for search.

Adding Monte Carlo tree search over sampled transitions, with the coalition value function as the leaf evaluator, follows the pattern Silver et al. 2016 demonstrated. Kocsis and Szepesvari 2006 supplies UCT, and Browne et al. 2012 catalogues the variants worth trying. Sutton et al. 1999 points at the further step of treating each agent invocation as an option with its own termination condition.

### Relaxing the cooperative assumption

The cooperative assumption holds when all agents serve one objective. It breaks under a shared budget with competing priorities, or when agents come from different vendors with different incentives, which DDL-018 makes reachable by assigning models per agent.

Moving to general-sum equilibrium learning would follow Hu and Wellman 2003 and Littman 2001, with Bowling and Veloso 2002 addressing the non-stationarity that appears once several policies adapt simultaneously.

### Relaxing one run at a time

Comparing policies fairly needs more than watching individual runs. The experiment harness supplies paired deltas and blind pairwise judging; what it does not supply is offline replay.

The work is offline evaluation of the kind introduced alongside LinUCB in Li et al. 2010, which would let a new policy be scored against logged episodes without re-running them. Potential-based shaping guarantees from Ng et al. 1999 and their multi-agent extension in Devlin and Kudenko 2011 say which reward modifications preserve the optimal policy and therefore keep such comparisons valid. [backend/tools/balance.py](../backend/tools/balance.py) is the batch half of this, already built.

## Standing discovery

The Research Library runs two standing queries per taxonomy category through the provider registry, so the corpus tracks new work in Markov games, stochastic games, MARL, agent orchestration, reinforcement learning, planning, multi-agent systems, tool use, and agent routing. The category definitions and their queries live in [backend/app/research/taxonomy.py](../backend/app/research/taxonomy.py).
