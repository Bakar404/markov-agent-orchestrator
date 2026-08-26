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

## DDL-014: narrow to the experiment harness, and archive the rest to a tag

Four products had accumulated in one repository: the comparison harness, a cross-episode learning lab, policy profiles that outlive an episode, and a Copilot CLI extension. Three of them diluted the claim the repository makes.

Removed at `9f37f1c`, recoverable with `git checkout pre-narrowing`:

| Archived | Footprint | Why |
| --- | --- | --- |
| Cross-episode learning lab | `api/campaign.py`, `services/campaign_service.py`, `tools/campaign.py`, `frontend/app/campaign/` | A second product with its own page, CLI tool and result format. It answers whether carrying parameters between episodes helps, which is a different question from whether orchestration helps |
| Policy profiles | `api/profiles.py`, `services/policy_profile_service.py`, `models.PolicyProfile`, `test_profiles.py` | Only meaningful for carrying learning between episodes, which is the campaign story |
| Copilot CLI extension | `.github/extensions/markov-arena/` | Never functional on the shipped CLI version; extension discovery does not work there |

`RunConfig.from_dict` filters unknown keys, so runs recorded with a `policy_profile` still restore. Nothing in the archived set is referenced by what remains.

The cost is that the carried-versus-fresh finding — that a contextual bandit gets measurably worse when it carries learning, because LinUCB's ridge matrix only accumulates and its exploration shrinks monotonically — now lives only in git history. The mechanism is preserved as a warning beside the policy stack; the measurement is not.

## DDL-015: refuse fabricated reports rather than filling them in

Supersedes the second half of DDL-005, which said cost and token figures are sampled. In live mode they are now measured or the report is refused.

A live experiment was driven end to end and produced a significant result — one arm winning five of five blind comparisons. It was void. The control arm had submitted the same placeholder text on four of five seeds, and cost had come from the agent spec's base rates rather than from a model, so the cost delta was arithmetic over the roster. Every existing guard passed. The fabrication was only found by hashing message content by hand.

The engine was the larger offender. A report that omitted `tokens` and `cost_usd` silently received `spec.base_tokens` and `spec.base_cost_usd`, which is the engine inventing the number that became the headline.

Live reports now require a non-empty `response` and measured `tokens`, `latency_ms` and `cost_usd`. A run refuses a summary it has already recorded, and two agents submitting identical text within one step is refused. If a call cannot be measured it is not live, and sim mode remains honest about being sampled.

Two detectors cover history and anything that slips past. Reports are hashed per arm and identical runs counted, because replaying one answer across seeds is one comparison rather than five. And cost is treated as unmeasured when the paired difference has zero variance across seeds — real per-call costs differ, so a stderr of exactly zero means the figures were looked up. That second check works on runs recorded before it existed and does not depend on the caller admitting anything.

The cost is a stricter contract that will reject drivers written against the old one, which is intended.

## DDL-016: separate the driver, the arms, and the judge into different sessions

Driving an experiment by hand puts one agent into three conflicting roles: it produces the control answer, produces the orchestrated answer, and then judges its own output. Those separations were being held up by good intentions, and DDL-015 records what happened when they were not.

`scripts/run-experiment.ps1` spawns a fresh session per brief, so no arm inherits another's context, and a third session judges with the labels stripped and the presentation order shuffled per seed. Which position the control occupied is recorded in the verdict notes, so any judgment can be re-checked. Control runs first across every seed.

The driver never reasons about the task. A driver that contributes content is an unmeasured third arm.

Two consequences worth stating. Cost is reported in AIU because that is the unit the CLI bills; no dollar rate is exposed and inventing one would defeat the purpose of measuring. And because every child is a fresh session, each pays roughly twenty-three thousand tokens of cached context, so per-call totals barely vary and the cost axis currently counts sessions rather than work.

## DDL-017: reject Markov games as the framing, and treat the comparison as a make-or-buy decision

Shapley's stochastic games require simultaneous moves and per-player rewards that differ. The specialists here move when instructed and share the orchestrator's objective, so two of the three defining features are absent. The framing was aspirational.

What the experiment actually compares is whether to do work with one capable generalist or to delegate it to cheaper specialists and pay the coordination overhead. That is a make-or-buy decision, and the literature that fits it is transaction cost economics: the escalation gate is the boundary of the firm, and orchestration is worth buying exactly when coordination costs less than the capability it purchases.

Two adjacent frames stay useful and are kept. The orchestrator cannot observe whether a cheap worker did the work or produced something shaped like work, which is a principal-agent problem with hidden information — and it is why reports carry a claimed hypothesis and a calibrated confidence rather than a bare answer. And choosing which worker to invoke remains a contextual bandit problem, which is implemented.

The cost is that the repository's name no longer describes it, and that the learned Markov game policy is now presented as one arm among several rather than as the thesis.

## DDL-018: assign models per agent, and hold both arms to the same budget

Follows DDL-017. If a make-or-buy comparison is the framing, the arms have to differ in how they spend rather than in how much.

The comparison this repository ran until now was one agent against several agents on the same model, which conflates two variables and leaves the cost axis nearly meaningless — DDL-016 records that per-call totals barely varied because fixed session context dominated them. Buying a capable orchestrator and cheap specialists changes that: different models have genuinely different rates, so the cost column becomes a bill rather than a count of spawns.

`RunConfig` carries `default_model` and a per-agent `agent_models` override, and the brief returned by `live/open` names the model that should answer it. The driver spawns each child on the model its brief specifies. Nothing in the engine knows what a model costs; that arrives through the measured report, which DDL-015 now requires.

The budget constraint is not new code. Both arms already take `budget_usd`, so a matched-budget experiment is a matter of setting it equally and letting the arms exhaust it differently. What matters is that the framing is stated: an arm that outspends the control has not demonstrated anything except that more money buys more.

The cost is a real failure mode that did not exist before. An orchestrator delegating to cheap workers can spend its budget on coordination and receive confident, wrong answers it cannot verify — which is the principal-agent problem from DDL-017 arriving as a measurable outcome rather than as theory. That is the result worth finding either way.
