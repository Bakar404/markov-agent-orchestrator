---
title: Research Roadmap
description: The staged path from contextual bandit orchestration through cooperative Markov games to multi-agent reinforcement learning, plus the extension point for replacing simulated agents with live model execution.
author: Markov Agent Orchestrator
ms.date: 2026-08-20
ms.topic: concept
keywords:
  - contextual bandit
  - markov decision process
  - markov game
  - multi-agent reinforcement learning
  - agent orchestration
  - research roadmap
estimated_reading_time: 16
---

## How to read this roadmap

Each stage removes a specific limitation of the one before it. Stages 0 through 4 are implemented and switchable at run creation time. Stages 5 and beyond are designed but not built, and every one of them names the extension point it would attach to.

The corresponding literature lives in the Research Library and is seeded from [research/seed_corpus.json](../research/seed_corpus.json).

## Stage 0: baselines

Uniform random selection and a hand-tuned need-based heuristic.

Neither policy learns. They exist because any claim that a learned policy helps needs a control condition, and because the heuristic encodes the conventional wisdom that learned policies should be measured against: research when uncertain, criticize when quality lags, verify before terminating.

Implemented in [backend/app/orchestration/policies/baselines.py](../backend/app/orchestration/policies/baselines.py).

## Stage 1: contextual bandit orchestration

Disjoint LinUCB with one ridge regression model per action over the shared 12-dimensional state context.

$$
\text{UCB}_a(s) = \theta_a^\top \varphi(s) + \alpha \sqrt{\varphi(s)^\top A_a^{-1} \varphi(s)}
$$

The bandit learns which agent suits which state, and the exploration bonus shrinks as an arm accumulates observations. What it cannot do is reason about consequences. It optimizes immediate reward, so it systematically undervalues the Planner and the Memory agent, whose payoff arrives several steps later.

That failure mode is the reason stage 2 exists, and it is visible in the reward dashboard: bandit runs show strong early reward and weaker terminal bonuses.

Key references: Li et al. 2010 on LinUCB, Auer et al. 2002 on finite-time bandit analysis, Chu et al. 2011 on linear payoff regret, Agrawal and Goyal 2013 on Thompson sampling as the posterior-sampling alternative.

Implemented in [backend/app/orchestration/policies/bandit.py](../backend/app/orchestration/policies/bandit.py).

## Stage 2: Markov decision process orchestration

Tabular Q-learning over a discretized state, blended with a per-action linear approximator so the policy still produces sensible scores in states it has never visited.

$$
Q(s,a) \leftarrow Q(s,a) + \eta \Big( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big)
$$

The bootstrap term is what changes the behavior. Value propagates backward across steps, so the policy will pay for a Planner call now to earn a better terminal bonus later.

The limitation it exposes is representational. Agents remain flat action labels, so the policy cannot express that two agents are worth more together than separately, and `RUN_PARALLEL` stays an opaque action whose composition is decided outside the policy.

Key references: Watkins and Dayan 1992 on Q-learning convergence, Kaelbling et al. 1998 on belief-state planning, Ng et al. 1999 on potential-based shaping, Sutton et al. 1999 on temporal abstraction.

Implemented in [backend/app/orchestration/policies/mdp.py](../backend/app/orchestration/policies/mdp.py).

## Stage 3: cooperative Markov game orchestration

Agents become players. Each has its own value function over the shared state, and a learned pairwise synergy matrix captures what additive decomposition cannot.

$$
V(s, C) = \sum_{i \in C} Q_i(s) + \sum_{i < j \in C} W_{ij} - \lambda (|C| - 1) \cdot \text{cost pressure}(s)
$$

The score for `RUN_PARALLEL` is the best multi-player coalition value, so coalition size becomes an output of the policy rather than an input. Synergy absorbs the residual between observed reward and the additive prediction, which is where pairings like Research plus Verification become measurable rather than assumed.

The limitation is scope. This is the cooperative branch only, with a shared reward and no adversarial or mixed-motive dynamics, and the coordination structure is fixed at pairwise.

Key references: Shapley 1953 on stochastic games, Littman 1994 on the Markov game framing, Hu and Wellman 2003 on Nash Q-learning, Littman 2001 on the friend branch of general-sum games, Oliehoek and Amato 2016 on Dec-POMDPs.

Implemented in [backend/app/orchestration/policies/markov_game.py](../backend/app/orchestration/policies/markov_game.py).

## Stage 4: multi-agent reinforcement learning orchestration

Independent learners with VDN-style additive value factorization and difference-reward credit assignment.

$$
Q_{\text{tot}}(s, C) = \sum_{i \in C} Q_i(s) + \sum_{i \notin C} B_i(s)
$$

Two additions matter. The abstention baseline `B_i` gives *not* invoking an agent an explicit learned value, turning omission into a decision. The difference-reward head estimates each agent's marginal contribution, which is the counterfactual signal that lets a shared reward train individual policies.

The limitation is function approximation. Everything is linear, so the policy cannot represent interactions beyond what the feature vector already encodes.

Key references: Sunehag et al. 2018 on value decomposition networks, Foerster et al. 2018 on counterfactual multi-agent policy gradients, Rashid et al. 2018 on monotonic mixing, Son et al. 2019 on relaxing factorization constraints, Yu et al. 2022 on on-policy alternatives, Zhang et al. 2021 for the theoretical map.

Implemented in [backend/app/orchestration/policies/marl.py](../backend/app/orchestration/policies/marl.py).

## Stage 5: live agent execution

Everything above learns against a generative model of agent behavior. The agents are stochastic processes parameterized in [backend/app/orchestration/agents.py](../backend/app/orchestration/agents.py), not model calls. That is deliberate for stages 0 through 4: policy learning needs thousands of episodes, and paying for inference on every one of them would make the research loop unaffordable and slow.

Stage 5 replaces the generative model with real execution while leaving the decision layer untouched.

### The extension point

The entire simulation is confined behind one dataclass. `TransitionModel._invoke` produces an `AgentReport`, and nothing downstream knows or cares how that report was produced:

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
```

Introducing an executor protocol alongside the existing sampler is enough to swap the implementation:

```python
class AgentExecutor(Protocol):
    async def invoke(
        self,
        spec: AgentSpec,
        state: OrchestratorState,
        task: str,
    ) -> AgentReport: ...
```

`SimulatedExecutor` wraps the current sampling logic. A `ModelExecutor` would issue the real call, measure wall-clock latency, read token counts and cost from the provider response, and map the result onto the same fields. The policy stack, reward model, entropy accounting, trace schema, WebSocket protocol, and every frontend component consume `AgentReport` and would need no changes.

This mirrors the pattern already proven in the research layer, where five very different sources sit behind one `ResearchProvider` contract.

### The problem this exposes

Two fields have no ground truth outside simulation.

`correct_evidence` currently compares against a latent hypothesis the simulator knows. Production has no such oracle. Candidate replacements, in rough order of cost:

* Verifier-scored evidence, where the Verification agent's judgement stands in for correctness. Cheap, but circular when the verifier itself is wrong.
* Self-consistency across repeated sampling, treating agreement as a correctness proxy. Well studied, but multiplies cost per invocation.
* A held-out judge model scoring evidence against retrieved sources. Strongest signal, highest cost, and introduces its own bias.

`evidence_mass` currently comes from a per-agent constant scaled by outcome. In production it would need to be estimated from the response, for example by measuring how much the retrieved content shifts a belief encoder, which is the practical form of the mutual-information acquisition criterion in Houlsby et al. 2011.

Neither problem is a blocker for the decision layer. Both are the reason stage 5 is a research stage rather than an integration task, and pretending otherwise would misrepresent the difficulty.

### Acceptance criteria

* A single configuration switch selects simulated or live execution.
* A live episode produces the same trace schema as a simulated one.
* Policies trained in simulation can be loaded and evaluated against live execution without retraining.
* Cost and latency in the reward come from measured values, and the interface distinguishes measured from sampled.

## Stage 6: partial observability

Stages 0 through 5 give the policy full visibility of the state. Real agents return partial, noisy views of a task, which makes this a Dec-POMDP rather than a fully observed Markov game.

The work is to add per-agent observation functions, replace the shared state with per-agent belief states, and move to recurrent or history-conditioned value estimation. Oliehoek and Amato 2016 is the reference treatment, and Kaelbling et al. 1998 provides the single-agent foundation already reflected in the Dirichlet belief.

## Stage 7: planning on top of learned values

The current policies are model-free. The transition kernel is known and cheap to sample, which makes it an ideal candidate for search.

Adding Monte Carlo tree search over sampled transitions, with the stage 3 or stage 4 value function as the leaf evaluator, follows the pattern that Silver et al. 2016 demonstrated. Kocsis and Szepesvari 2006 supplies UCT, and Browne et al. 2012 catalogues the variants worth trying. Sutton et al. 1999 points at the further step of treating each agent invocation as an option with its own termination condition.

## Stage 8: equilibrium learning beyond the cooperative case

The cooperative assumption holds when all agents serve one objective. It breaks under a shared budget with competing priorities, or when agents come from different vendors with different incentives.

Moving to general-sum equilibrium learning would follow Hu and Wellman 2003 and Littman 2001, with Bowling and Veloso 2002 addressing the non-stationarity that appears once several policies adapt simultaneously.

## Stage 9: offline evaluation and policy comparison

Comparing orchestration policies fairly needs more than watching individual runs.

The work is batch episode execution across seeds, confidence intervals on cumulative reward and cost efficiency, and offline replay evaluation of the kind introduced alongside LinUCB in Li et al. 2010. Potential-based shaping guarantees from Ng et al. 1999 and their multi-agent extension in Devlin and Kudenko 2011 tell us which reward modifications preserve the optimal policy and therefore keep comparisons valid.

## Standing discovery

The Research Library runs two standing queries per taxonomy category through the provider registry, so the corpus tracks new work in Markov games, stochastic games, MARL, agent orchestration, reinforcement learning, planning, multi-agent systems, tool use, and agent routing. The category definitions and their queries live in [backend/app/research/taxonomy.py](../backend/app/research/taxonomy.py).
