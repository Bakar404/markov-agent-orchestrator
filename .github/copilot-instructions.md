---
title: Copilot Instructions
description: Repository guidance for GitHub Copilot, including the protocol for driving a live orchestration run in the Markov Agent Orchestrator arena.
---

## Repository overview

Markov Agent Orchestrator treats "which agent should act next" as a sequential decision problem
under uncertainty. A FastAPI backend owns the episode; a Next.js pixel-art frontend renders it.

* Backend: `backend/` — FastAPI, SQLAlchemy, SQLite. Runs on `http://localhost:8000`.
* Frontend: `frontend/` — Next.js 14. Runs on `http://localhost:3000`.
* Tests: `cd backend; python -m pytest tests/ -q`
* Start both: `.\scripts\dev.ps1`

## Driving a live arena run

A run has one of two modes. **Sim** mode samples outcomes from probability distributions and is
driven by the browser's play button. **Live** mode has no simulated outcomes: you are the agent,
and each step is a real invocation you perform.

Use live mode when the user asks you to run, drive, or play the arena.

### The rule that matters

The policy decides *who acts*. You decide *what that agent produces*. Never choose the agent
yourself, and never report a step you did not actually perform. The entire measurement depends
on those two things being honest.

### Protocol

Three calls, in a loop. Commands are PowerShell, which is the shell on this machine.

1. **Create the run.** Pick 4 genuinely competing hypotheses for the task.

    ```powershell
    $body = '{"task":"<task>","policy":"marl","mode":"live","belief_dim":4,"hypotheses":["<h0>","<h1>","<h2>","<h3>"],"max_steps":20,"budget_usd":1.5}'
    $run = Invoke-RestMethod -Uri http://localhost:8000/api/runs -Method Post -Body $body -ContentType 'application/json'
    $id = $run.id; Write-Host "run=$id watch=http://localhost:3000/?run=$id"
    ```

    Tell the user they can watch at the printed `watch` URL.

2. **Open a step.** Returns the agent the policy chose plus a brief. Does not advance the run.

    ```powershell
    $p = Invoke-RestMethod -Uri "http://localhost:8000/api/runs/$id/live/open" -Method Post
    $p.agents; $p.briefs[0].instruction; $p.briefs[0].context.belief_ranked
    ```

    Read the instruction and the ranked belief, then **actually do that work** — research it,
    critique it, verify it, whatever the brief asks.

3. **Report what you produced.** One entry per agent listed in `$p.agents`.

    ```powershell
    $reports = @($p.agents | ForEach-Object { @{ agent_id=$_; outcome='success'; confidence=0.8; claimed_hypothesis=0; response='<what you found>'; summary='<one line>' } })
    $rb = @{ token=$p.token; reports=$reports } | ConvertTo-Json -Depth 6
    $r = Invoke-RestMethod -Uri "http://localhost:8000/api/runs/$id/live/report" -Method Post -Body $rb -ContentType 'application/json'
    "step=$($r.step.step) reward=$($r.step.reward) entropy=$($r.step.entropy_after) done=$($r.step.done)"
    ```

Repeat 2 and 3 until `done` is `True`. Then summarize the episode for the user: final confidence,
total cost, cumulative reward, and which hypothesis the belief settled on.

### Reporting honestly

* `outcome` — `success` only if the brief was fulfilled, `partial` for progress, `failure` when
  you could not do it or got it wrong. A failure is useful data, not something to avoid.
* `confidence` — 0 to 1, calibrated. Inflating this corrupts the belief update.
* `claimed_hypothesis` — the index your work actually supports. **Omit it if your work supported
  none of them.** There is no hidden ground truth in live mode, so belief mass follows claims and
  confidence only rises when independent agents agree. Guessing an index defeats the mechanism.
* `cost_usd`, `latency_ms`, `tokens` — include when known; they are estimated otherwise.

### Errors

* `409` on report — no step is open, or the token is stale. Call `live/open` again.
* `422` on report — the agent set does not match the open step. Report exactly the agents listed.
* `409` on open — the run terminated, or it is a sim run.
