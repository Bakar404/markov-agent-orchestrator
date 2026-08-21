---
title: Design Decision Log
description: Recorded architectural decisions for the Markov Agent Orchestrator, each with the context that forced it, the alternatives considered, and the consequences accepted.
author: Markov Agent Orchestrator
ms.date: 2026-08-20
ms.topic: reference
keywords:
  - architecture decision record
  - design rationale
  - markov game
  - reward shaping
  - research library
estimated_reading_time: 14
---

## Conventions

Each entry records the decision, the situation that forced it, what was rejected, and what the decision costs. Entries are append-only. Superseding an entry means adding a new one that references it, not editing the original.

## DDL-001: represent uncertainty as a Dirichlet belief rather than a scalar

A scalar confidence field would have been simpler, and it is what most agent frameworks expose.

It was rejected because information gain is central to the reward function, and a scalar cannot support it. Entropy requires a distribution. Without one, information gain becomes an invented number, and the claim that the orchestrator maximizes information gain becomes marketing rather than mathematics.

The orchestrator maintains a Dirichlet concentration vector over `K` competing solution hypotheses. Entropy is Shannon entropy of the posterior mean in bits, and information gain is the difference between consecutive entropies.

The cost is a larger state, a `belief_dim` parameter to justify, and the need to explain hypotheses to anyone reading the interface. In exchange, information gain, uncertainty, and confidence all derive from one object, and the metrics panel can show the arithmetic rather than asserting the result.

## DDL-002: sample the successor state, and record the probability of the branch taken

The brief called for stochastic transitions where the same action does not always produce the same next state. A weaker reading would have been to add noise to a deterministic result.

Instead the outcome is genuinely sampled: a Beta competence draw, then a categorical branch across success, partial, and failure, then log-normal draws for cost, latency, and tokens.

The realized branch probability is stored on the trace as `transition_probability`, separate from the policy's `action_probability`. Conflating the two would erase the difference between a confident choice that failed improbably and a gamble that paid off, which is precisely the behavior a stochastic orchestrator exists to expose.

## DDL-003: return the reward as a decomposition, never as a scalar

`RewardModel.compute` returns a `RewardBreakdown` with each term intact plus per-agent credit shares.

A scalar would have been sufficient for the learning algorithms and nothing else. The decomposition serves three consumers that a scalar cannot: the reward dashboard renders the terms directly, the Markov game and MARL policies need per-agent shares for difference rewards, and anyone tuning weights can see which term dominates instead of guessing.

The cost is a wider trace row and a reward object that must stay in sync with the weight configuration. A test asserts that the terms sum to the reported reward on every step of every episode, which catches drift immediately.

## DDL-004: sample from the policy distribution instead of taking the argmax

Every policy exposes a distribution over legal actions, and the engine samples from it.

Taking the argmax would produce better average reward and a duller, less informative system. Sampling makes exploration visible on the canvas, gives `action_probability` a real meaning as the probability of the branch actually taken, and prevents the demo from collapsing into a fixed sequence after a few episodes of learning.

Temperature is a per-policy parameter, so the exploration and exploitation balance is tunable without changing the mechanism.

## DDL-005: simulate agent execution rather than calling live models

The six agents are stochastic processes parameterized by cost, latency, token draw, Beta competence prior, evidence strength, and noise strength. No language model is called anywhere in the project, and there is no model SDK in the dependency list.

Policy learning needs thousands of episodes. Paying for inference on each one would make the research loop unaffordable, slow, and non-reproducible, and it would put the interesting part of the system behind an API key.

The consequence is that token consumption and dollar cost in the interface are sampled quantities, not billed ones. The documentation states this plainly rather than letting the numbers imply otherwise.

The simulation is confined behind the `AgentReport` dataclass so it can be replaced without touching the decision layer. That path is specified as stage 5 in [ResearchRoadmap.md](ResearchRoadmap.md), including the two fields that have no ground truth outside simulation.

## DDL-006: gate TERMINATE behind a minimum step count

`TERMINATE` was originally legal from step 0. With LinUCB starting at zero weights, the near-uniform softmax gave it roughly a one-in-eight chance on the first step, so episodes routinely ended before doing anything. This surfaced as an intermittently failing round-trip test, but the test was reporting a genuine behavioral defect rather than a flaw in itself.

Raising the terminal penalty for premature termination was considered and rejected, because it treats a structural constraint as a reward-shaping problem and leaves the policy free to learn the lesson slowly and expensively.

`RunConfig.min_steps_before_terminate` now removes `TERMINATE` from the legal action set until the orchestrator has produced something. Three is the default, and the value is exposed on the run creation API.

This is a statement about the problem, not a workaround: stopping before any work exists is not a meaningful decision. The cost is one more configuration knob and a constraint the policy no longer gets to learn on its own.

## DDL-007: SQLite now, PostgreSQL-compatible from the start

The ORM uses only column types available in both engines, sessions are handled identically, and the connection string is the sole difference. SQLite-specific behavior is limited to two pragmas applied conditionally at connect time.

Choosing PostgreSQL immediately would have added a service dependency to a project whose main value is that it runs locally with two commands. Choosing SQLite without discipline would have created a migration project later.

The cost is forgoing PostgreSQL-specific features such as native JSONB indexing.

## DDL-008: persist an engine snapshot rather than replay history

Each step writes a snapshot containing the state, the policy parameters, and the serialized random number generator state.

Replaying an episode from its seed was the alternative, and it is attractive because it stores less. It was rejected because restore time grows with episode length, and because any change to the engine silently invalidates every stored run.

Snapshot restore is constant time and preserves the exact random stream, which a test verifies by comparing the next step from a restored engine against the original.

The cost is a larger `runs` row and the requirement that every policy implements `state_dict` and `load_state_dict`.

## DDL-009: research providers degrade, they do not fail

Every provider catches its own network, parse, and HTTP errors, answers from the curated corpus, and sets a `degraded` flag with the underlying error attached.

Propagating provider failures would make the Research Library unusable offline, on a rate limit, or during any upstream incident. Silently substituting local results would be worse: it would present stale data as live.

The flag travels through the service response to the interface, so the user sees that a result set came from the fallback path and why. The cost is that every provider needs its own fallback, which is enforced by the shared contract.

## DDL-010: write original summaries and label the offline citation edges honestly

The seed corpus contains original one-line summaries written for this project rather than publisher abstracts, which avoids reproducing copyrighted text.

Citation edges are a curated approximation of intellectual lineage, and the corpus file says so in its own `notice` field. They exist so the citation graph is populated on a cold start rather than showing an empty canvas. When a live provider returns real reference data it supersedes them.

`citation_count` is zero for every seed record on purpose. Inventing plausible counts would have looked better and been false. The API reports in-corpus in-degree instead, which is a real and well-defined quantity, and real counts arrive from Semantic Scholar when the network is available.

## DDL-011: give the MCP provider a working offline mode based on HITS

`HITSMCPResearchProvider` calls a JSON-RPC tool server when `HITS_MCP_ENDPOINT` is configured. With no endpoint it would ordinarily be inert, which would leave a named integration point that does nothing in the default configuration.

Instead it ranks the curated corpus using Kleinberg HITS over the citation graph, separating authorities from hubs. That is the same hub and authority signal a remote research tool would be expected to supply, computed locally.

The provider therefore demonstrates its own contract out of the box, and the hub and authority report is exposed on the API so a ranking can be explained rather than asserted.

## DDL-012: make coalition size an output of the policy

`RUN_PARALLEL` could have taken a fixed coalition size, or delegated composition to a heuristic.

From stage 3 onward the policy scores every coalition up to size three and treats the best multi-player value as the score for `RUN_PARALLEL`. Choosing to fan out and choosing who to fan out to become the same decision, which is the point of moving to a game-theoretic framing.

Earlier policies that cannot express coalitions fall back to need-based sampling weighted by their own action scores, so the action stays available across the whole stack.

The cost is enumerating coalitions on every scoring call. With six agents and a size cap of three that is 41 subsets, which is negligible.

## DDL-013: let information gain go negative

A Critic that surfaces an unconsidered failure mode legitimately reopens hypotheses and raises entropy. Clamping information gain at zero would hide that, and would reward the orchestrator for avoiding criticism.

The Critic carries the highest noise strength in the agent registry, so negative information gain appears regularly, and the interface displays it as a negative number against the entropy endpoints that produced it.

The consequence is that a step can be worth taking while scoring negatively on the information gain term, which is the correct incentive and occasionally a surprising one.
