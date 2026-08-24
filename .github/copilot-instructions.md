---
title: Copilot Instructions
description: Repository guidance for GitHub Copilot, including the protocol for running a control-versus-orchestration experiment in the Markov Agent Orchestrator arena.
---

## Repository overview

Markov Agent Orchestrator answers one question: **is multi-agent orchestration actually better
than a single agent on this task?** A FastAPI backend owns the episodes; a Next.js pixel-art
frontend renders them.

* Backend: `backend/` — FastAPI, SQLAlchemy, SQLite. Runs on `http://localhost:8000`.
* Frontend: `frontend/` — Next.js 14. Runs on `http://localhost:3000`.
* Tests: `cd backend; python -m pytest tests/ -q`
* Start both: `.\scripts\dev.ps1`

## When the user asks you to run the arena

**Run an experiment, not a single run.** One run in isolation proves nothing: without a
single-agent control on the same task and seed, any result could be the task, the model, or the
day. If the user asks for one run anyway, do it, but tell them it is not a result.

Default experiment: two arms, `control` and `cascade`, on the same seed.

### The rules that matter

1. **The policy decides who acts. You decide what that agent produces.** Never choose the agent
   yourself, and never report a step you did not actually perform.
2. **Write the rubric before you run anything.** You will be scoring your own work, which is a
   conflict of interest. Committing to what "good" means in advance is the only thing that stops
   the score becoming a post-hoc justification. Show the rubric to the user first.
3. **Same seed for every arm.** Differences are paired on shared seeds; unpaired arms report
   `paired_seeds: 0` and tell you nothing.

### Step 0 — agree the plan

Tell the user, before spending anything:

* the task, and 4 genuinely competing hypotheses (four rephrasings of one answer will collapse
  the belief trivially and produce a fake result)
* the rubric you will score answers against
* the arms, the seed, and the rough cost — each live step is a real model call

### Step 1 — create both arms

```powershell
$api  = 'http://localhost:8000'
$exp  = '<experiment-name>'
$seed = 101
$task = '<task>'
$hyp  = '["<h0>","<h1>","<h2>","<h3>"]'

$ids = @{}
foreach ($arm in 'control', 'cascade') {
    $body = "{`"task`":`"$task`",`"strategy`":`"$arm`",`"arm`":`"$arm`",`"seed`":$seed,
              `"experiment`":`"$exp`",`"mode`":`"live`",`"belief_dim`":4,
              `"hypotheses`":$hyp,`"max_steps`":8,`"budget_usd`":1.5}"
    $r = Invoke-RestMethod "$api/api/runs" -Method Post -Body $body -ContentType 'application/json'
    $ids[$arm] = $r.id
    Write-Host "$arm -> $($r.id)  watch=http://localhost:3000/?run=$($r.id)"
}
```

Give the user both watch URLs. `control` will stay solo; `cascade` will escalate when it stalls.

### Step 2 — drive each arm to completion

For each arm, loop these two calls. **Do the control arm first**, so your solo answer is not
contaminated by having already done the work with specialists.

```powershell
$id = $ids['control']   # then repeat for 'cascade'

# open: the policy picks who acts. Does not advance the run.
$p = Invoke-RestMethod "$api/api/runs/$id/live/open" -Method Post
$p.agents; $p.briefs[0].instruction; $p.briefs[0].context.belief_ranked
```

Read the instruction, then **actually do that work** — research it, critique it, verify it,
whatever the brief asks. Then report it:

```powershell
$reports = @($p.agents | ForEach-Object {
    @{ agent_id=$_; outcome='success'; confidence=0.8; claimed_hypothesis=0
       response='<what you found>'; summary='<one line>' } })
$rb = @{ token=$p.token; reports=$reports } | ConvertTo-Json -Depth 6
$r = Invoke-RestMethod "$api/api/runs/$id/live/report" -Method Post -Body $rb -ContentType 'application/json'
"step=$($r.step.step) reward=$($r.step.reward) entropy=$($r.step.entropy_after) done=$($r.step.done)"
```

Repeat until `done` is `True`, then move to the next arm.

### Step 3 — score both arms against the rubric you wrote

```powershell
foreach ($arm in 'control','cascade') {
    $v = @{ score=<0-1>; judge='copilot'; rubric='<the rubric from step 0>'
            notes='<why this score>' } | ConvertTo-Json
    Invoke-RestMethod "$api/api/runs/$($ids[$arm])/verdict" -Method Post -Body $v -ContentType 'application/json'
}
```

Score the *answers*, not the effort. An arm that spent more and arrived at the same place scores
the same, not higher.

### Step 4 — report the comparison

```powershell
$c = Invoke-RestMethod "$api/api/experiments/$exp"
$c.verdict
$c.arms | ForEach-Object { "$($_.arm): quality=$($_.mean_quality) cost=$($_.mean_cost_usd) tokens=$($_.mean_tokens)" }
$c.caveats
```

Give the user the verdict sentence, the per-arm numbers, **and the caveats verbatim**. Two arms
on one seed is thin evidence and the caveats will say so — do not paper over that. If the
difference is not significant, say the arms were indistinguishable rather than picking a winner.

### Reporting honestly

* `outcome` — `success` only if the brief was fulfilled, `partial` for progress, `failure` when
  you could not do it or got it wrong. A failure is useful data, not something to avoid.
* `confidence` — 0 to 1, calibrated. Inflating this corrupts the belief update.
* `claimed_hypothesis` — the index your work actually supports. **Omit it if your work supported
  none of them.** There is no hidden ground truth in live mode, so belief mass follows claims and
  confidence only rises when independent agents agree. Guessing an index defeats the mechanism.
* `cost_usd`, `latency_ms`, `tokens` — include when known; they are estimated otherwise.

### Strategies

`GET /api/meta` returns the catalog. `control` never escalates and is mandatory in every
experiment. `cascade` escalates when the solo attempt stalls. `always_orchestrate` escalates
immediately. The four learned strategies currently escalate 100% of the time, so they add cost
without adding a distinct behaviour — prefer `cascade` unless the user asks for them.

### Errors

* `409` on report — no step is open, or the token is stale. Call `live/open` again.
* `422` on report — the agent set does not match the open step. Report exactly the agents listed.
* `409` on open — the run terminated, or it is a sim run.
* `paired_seeds: 0` in the comparison — the arms ran different seeds. Re-run them on one seed.
