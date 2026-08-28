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
* CLI: `cli/` — `arena run`, a paired experiment driven from the terminal.
* Bridge: `bridge/` — Agent Framework workflows, for the arms the arena does not route itself.
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
* the rubric you will judge answers against, written now so it cannot be bent later
* the arms, the seeds, and how many steps each gets

**Use at least 5 seeds per arm.** One seed gives one comparison, and a single comparison cannot
distinguish a real difference from a coin flip. If the user wants a quick look, say that a
one-seed run is a smoke test rather than a result.

**Give the arms enough steps to differ.** `cascade` only escalates once the solo attempt stalls,
so a short run may never open the gate — and an arm that never escalates ran the same solo agent
as the control. The comparison will refuse to give a verdict in that case, correctly. Eight steps
is a reasonable floor; `always_orchestrate` escalates at step 2 regardless.

Base the rubric on what the answer has to do. Microsoft Foundry's evaluators use a 1-5 Likert
scale with 3 as the pass mark, and split writing quality into *coherence* (logical flow) and
*fluency* (readability); borrow that split and add the dimensions the task actually needs, such
as whether a stated constraint was satisfied or whether the steps are specific enough to follow.

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
       response='<what you found>'; summary='<one line>'
       tokens=<measured>; latency_ms=<measured>; cost_usd=<measured> } })
$rb = @{ token=$p.token; reports=$reports } | ConvertTo-Json -Depth 6
$r = Invoke-RestMethod "$api/api/runs/$id/live/report" -Method Post -Body $rb -ContentType 'application/json'
"step=$($r.step.step) reward=$($r.step.reward) entropy=$($r.step.entropy_after) done=$($r.step.done)"
```

Repeat until `done` is `True`, then move to the next arm.

### Step 3 — judge the arms blind, pairwise

Absolute scores compress. A judge rating one answer at a time drifts to the top of the range,
and two arms come back 0.96 against 0.98 — noise between two scorings rather than a difference.
A forced choice between two answers does not have that problem, which is why MT-Bench and
Chatbot Arena compare rather than score.

So: put the two final answers side by side **without their arm labels**, decide which is better
against the rubric you wrote in step 0, and record it.

```powershell
$pw = @{ run_a=$ids['control']; run_b=$ids['cascade']; winner='<a|b|tie>'
         judge='copilot'; rubric='<the rubric from step 0>'
         notes='<what decided it>' } | ConvertTo-Json
Invoke-RestMethod "$api/api/experiments/$exp/pairwise" -Method Post -Body $pw -ContentType 'application/json'
```

Record `tie` honestly when neither is better. A tie is evidence the arms are indistinguishable,
which is a real finding; forcing a preference manufactures a result that is not there.

One comparison per seed. Five seeds gives five comparisons, which is the minimum for a win rate
to clear two standard errors against a coin flip.

Optionally also record an absolute score per run for the record:

```powershell
$v = @{ score=<0-1>; judge='copilot'; rubric='<rubric>'; notes='<why>' } | ConvertTo-Json
Invoke-RestMethod "$api/api/runs/$($ids['control'])/verdict" -Method Post -Body $v -ContentType 'application/json'
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
* `response` — required and non-empty. What you actually produced.
* `cost_usd`, `latency_ms`, `tokens` — **required**. Report what the call actually spent. These
  used to be optional and were filled in from the agent spec, which produced a cost column that
  was arithmetic over the roster rather than a measurement. If you cannot measure them, you are
  not running live — use sim mode, which is honest about being sampled.

The API refuses a report that repeats work the run already recorded. That is not a bug to work
around: if a step has nothing new, report `partial` or `failure` with what you actually tried.
Never replay an earlier answer to fill a step, and never invent one to finish an arm faster — a
short honest run is evidence, and a padded one is not.

### Strategies

`GET /api/meta` returns the catalog. `control` never escalates and is mandatory in every
experiment. `cascade` escalates when the solo attempt stalls. `always_orchestrate` escalates
immediately. The four learned strategies currently escalate 100% of the time, so they add cost
without adding a distinct behaviour — prefer `cascade` unless the user asks for them.

**You cannot drive an arm whose `external_driver` is set.** `maf_sequential`, `maf_concurrent`
and `maf_handoff` are routed by a Microsoft Agent Framework workflow that decides who acts, so
`live/open` refuses to pick an agent and tells the caller to declare one. Creating such a run
from here produces a run nothing can step. If the user asks for one, do not create it — point
them at the driver that can:

```powershell
C:\venvs\arena-maf\Scripts\python bridge/run.py --experiment <name> --arm control --arm maf_sequential `
    --task "<task>" --hypothesis "<h0>" --hypothesis "<h1>" --hypothesis "<h2>" --hypothesis "<h3>" `
    --rubric-file <rubric.md> --seeds 101 102 103 104 105
```

Check `external_driver` in the catalog rather than matching on the `maf_` prefix, so a strategy
added later is handled correctly without this file being updated.

### The other two entry points

The protocol above is for driving the arena yourself, turn by turn, which is the right shape
when the user wants to watch it happen. Two alternatives exist and are better for a full
experiment:

* `cli/bin/arena.js` — `arena run` drives both arms and judges them blind, spawning a fresh
  Copilot session per agent. It refuses externally driven arms up front rather than mid-run.
* `bridge/run.py` — every arm on every seed through Agent Framework, then blind pairwise
  judging. The only way to run the `maf_*` arms.

### Errors

* `409` on report — no step is open, or the token is stale. Call `live/open` again.
* `422` on report — the agent set does not match the open step. Report exactly the agents listed.
* `409` on open — the run terminated, or it is a sim run.
* `409` on open saying the run is externally driven — the arm belongs to `bridge/run.py`, not to
  this protocol. Do not try to satisfy it by declaring an action yourself; that would make you
  the orchestrator being measured.
* `paired_seeds: 0` in the comparison — the arms ran different seeds. Re-run them on one seed.
