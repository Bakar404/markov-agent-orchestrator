<#
.SYNOPSIS
    Drives a paired control-vs-orchestration experiment using separate Copilot CLI sessions.

.DESCRIPTION
    The driver never reasons about the task. It creates the runs, hands each brief to a fresh
    child session, records what that child actually spent, and asks a third session to pick a
    winner without telling it which arm is which. Those separations are the point: a driver that
    contributes content is an unmeasured third arm, and a judge that knows the labels is not blind.

    Control runs first for every seed, so the solo answers cannot be contaminated by having
    already done the work with specialists.

    Cost is reported in AIU, not dollars. AIU is the unit the CLI actually bills; no USD rate is
    exposed, and inventing one would defeat the purpose of measuring at all.

.EXAMPLE
    .\scripts\run-experiment.ps1 -Task "our CI suite takes 45 minutes..." `
        -Hypotheses 'shard it','split pre/post merge','fix test design','merge queue' `
        -Experiment ci-debt -Seeds 101,102,103,104,105
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Task,
    [Parameter(Mandatory)][string[]]$Hypotheses,
    [string]$Experiment = 'experiment',
    [int[]]$Seeds = @(101, 102, 103, 104, 105),
    [string]$ControlArm = 'control',
    [string]$TreatmentArm = 'always_orchestrate',
    [int]$MaxSteps = 5,
    [double]$BudgetUsd = 20.0,
    [string]$Api = 'http://localhost:8000',
    [string]$OrchestratorModel = '',
    [string]$WorkerModel = '',
    [string]$JudgeModel = '',
    [switch]$AllowTools,
    [int]$JudgeExcerptChars = 6000
)

$ErrorActionPreference = 'Stop'
$script:ChildCalls = 0

function Assert-Backend {
    try { Invoke-RestMethod "$Api/api/meta" -TimeoutSec 10 | Out-Null }
    catch { throw "Backend not reachable at $Api. Start it with .\scripts\dev.ps1" }
}

function Invoke-Child {
    <#
        One fresh Copilot session. No memory of any other arm, which is what keeps the control
        a control. Returns the response plus what the call actually consumed.
    #>
    param([Parameter(Mandatory)][string]$Prompt, [string]$Model = '')

    $usageFile = Join-Path $env:TEMP "copilot-usage-$([guid]::NewGuid().ToString('N')).json"
    $cliArgs = @('-p', $Prompt, '-s', '--no-color', '--usage-output-file', $usageFile)
    if ($Model) { $cliArgs += @('--model', $Model) }
    if ($AllowTools) { $cliArgs += @('--allow-tool', 'shell') }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & copilot @cliArgs 2>&1 | Out-String
    $sw.Stop()
    $script:ChildCalls++

    if (-not (Test-Path $usageFile)) {
        throw "Copilot wrote no usage file. Without measured tokens the run cannot be reported honestly."
    }
    $usage = Get-Content $usageFile -Raw | ConvertFrom-Json
    Remove-Item $usageFile -ErrorAction SilentlyContinue

    $td = $usage.tokenDetails
    $tokens = [int]($td.input.tokenCount + $td.output.tokenCount + $td.cache_write.tokenCount + $td.cache_read.tokenCount)

    [pscustomobject]@{
        Response  = $output.Trim()
        Tokens    = [Math]::Max($tokens, 1)
        LatencyMs = [double]$usage.totalApiDurationMs
        Aiu       = [double]$usage.totalNanoAiu / 1e9
        WallMs    = $sw.Elapsed.TotalMilliseconds
    }
}

function New-Run {
    param([string]$Arm, [int]$Seed, [hashtable]$AgentModels, [string]$DefaultModel)
    $body = @{
        task          = $Task
        strategy      = $Arm
        arm           = $Arm
        seed          = $Seed
        experiment    = $Experiment
        mode          = 'live'
        belief_dim    = $Hypotheses.Count
        hypotheses    = $Hypotheses
        max_steps     = $MaxSteps
        budget_usd    = $BudgetUsd
        default_model = $DefaultModel
        agent_models  = $AgentModels
    } | ConvertTo-Json -Depth 5
    (Invoke-RestMethod "$Api/api/runs" -Method Post -Body $body -ContentType 'application/json').id
}

function Build-Prompt {
    param($Brief, [string]$Arm, [int]$Step)
    $ranked = ($Brief.context.belief_ranked | ForEach-Object {
        "  [$($_.index)] $($_.label)  (p=$([math]::Round($_.probability, 3)))"
    }) -join "`n"

    @"
You are one agent inside an experiment. Do the work described, then stop.

TASK
$Task

COMPETING HYPOTHESES
$ranked

YOUR ROLE: $($Brief.agent_id)
$($Brief.instruction)

RULES
- Produce the work itself. Do not describe what you would do.
- If you cannot do something, say so plainly rather than inventing it.
- No preamble, no sign-off. Output only the work.
- End with a final line exactly: HYPOTHESIS: <index>   (or HYPOTHESIS: none)
"@
}

function Get-ClaimedHypothesis {
    param([string]$Text)
    $m = [regex]::Match($Text, '(?im)^HYPOTHESIS:\s*(\d+|none)\s*$')
    if (-not $m.Success) { return $null }
    if ($m.Groups[1].Value -ieq 'none') { return $null }
    [int]$m.Groups[1].Value
}

function Invoke-Arm {
    param([string]$RunId, [string]$Arm)

    while ($true) {
        $run = Invoke-RestMethod "$Api/api/runs/$RunId"
        if ($run.terminated -or $run.step_count -ge $MaxSteps) { break }

        $opened = Invoke-RestMethod "$Api/api/runs/$RunId/live/open" -Method Post

        # No agents means the policy escalated. Nothing was produced, so nothing is reported.
        if (-not $opened.agents -or $opened.agents.Count -eq 0) {
            Invoke-RestMethod "$Api/api/runs/$RunId/live/report" -Method Post `
                -Body (@{ token = $opened.token; reports = @() } | ConvertTo-Json -Depth 4) `
                -ContentType 'application/json' | Out-Null
            Write-Host "    step $($run.step_count + 1): escalated" -ForegroundColor DarkYellow
            continue
        }

        $reports = @()
        foreach ($brief in $opened.briefs) {
            $result = Invoke-Child -Prompt (Build-Prompt -Brief $brief -Arm $Arm -Step $run.step_count) -Model $brief.model
            $body = ($result.Response -replace '(?im)^HYPOTHESIS:.*$', '').Trim()

            # An empty child response is a failed step, not a step to skip or to fill in.
            $outcome = if ($body) { 'success' } else { 'failure' }
            if (-not $body) { $body = "$($brief.agent_id) produced no output." }

            $firstLine = ($body -split "`n" | Where-Object { $_.Trim() } | Select-Object -First 1).Trim()
            $report = @{
                agent_id   = $brief.agent_id
                outcome    = $outcome
                confidence = 0.7
                response   = $body
                summary    = $firstLine.Substring(0, [Math]::Min(160, $firstLine.Length))
                tokens     = $result.Tokens
                latency_ms = $result.LatencyMs
                cost_usd   = $result.Aiu
            }
            $claimed = Get-ClaimedHypothesis $result.Response
            if ($null -ne $claimed -and $claimed -lt $Hypotheses.Count) { $report.claimed_hypothesis = $claimed }
            $reports += $report
        }

        $payload = @{ token = $opened.token; reports = $reports } | ConvertTo-Json -Depth 6
        try {
            $stepped = Invoke-RestMethod "$Api/api/runs/$RunId/live/report" -Method Post `
                -Body $payload -ContentType 'application/json'
        } catch {
            # The API refuses replayed or unmeasured work. That is a real failure of this run,
            # not something to retry until it is accepted.
            Write-Host "    REFUSED: $($_.ErrorDetails.Message)" -ForegroundColor Red
            Invoke-RestMethod "$Api/api/runs/$RunId/live/abandon" -Method Post -ErrorAction SilentlyContinue | Out-Null
            break
        }

        $s = $stepped.step
        Write-Host ("    step {0}: {1} -> {2}  ({3} tok, {4:N2} AIU{5})" -f `
            $s.step, ($opened.agents -join '+'), $s.outcome, ($reports | Measure-Object tokens -Sum).Sum, ($reports | Measure-Object cost_usd -Sum).Sum, `
            $(if ($opened.briefs[0].model) { ", $($opened.briefs[0].model)" } else { '' }))
        if ($s.done) { break }
    }
}

function Get-FinalAnswer {
    param([string]$RunId)
    $messages = Invoke-RestMethod "$Api/api/runs/$RunId/messages"
    ($messages | Where-Object { $_.kind -like 'report:*' } | ForEach-Object { $_.content }) -join "`n`n"
}

function Invoke-Judge {
    param([string]$ControlId, [string]$TreatmentId, [int]$Seed, [string]$Rubric)

    $answerControl = Get-FinalAnswer $ControlId
    $answerTreatment = Get-FinalAnswer $TreatmentId
    if (-not $answerControl -or -not $answerTreatment) {
        Write-Host "  seed ${Seed}: one arm produced nothing; not judged" -ForegroundColor DarkYellow
        return
    }

    # Judges favour whichever answer they read first, so the order is shuffled per seed.
    $controlIsA = (Get-Random -Minimum 0 -Maximum 2) -eq 0
    $a = if ($controlIsA) { $answerControl } else { $answerTreatment }
    $b = if ($controlIsA) { $answerTreatment } else { $answerControl }

    $trim = { param($t) if ($t.Length -gt $JudgeExcerptChars) { $t.Substring(0, $JudgeExcerptChars) + "`n[truncated]" } else { $t } }

    $prompt = @"
You are judging two answers to the same question. You do not know how either was produced,
and you must not speculate about it.

QUESTION
$Task

RUBRIC
$Rubric

ANSWER A
$(& $trim $a)

ANSWER B
$(& $trim $b)

Decide which better satisfies the rubric. A tie is a legitimate verdict and you should use it
when neither is clearly better - do not invent a preference.

Reply with exactly two lines:
VERDICT: A
REASON: <one sentence>
"@

    $judged = Invoke-Child -Prompt $prompt -Model $JudgeModel
    $m = [regex]::Match($judged.Response, '(?im)^VERDICT:\s*(A|B|TIE)\s*$')
    if (-not $m.Success) {
        Write-Host "  seed ${Seed}: judge gave no parseable verdict; not recorded" -ForegroundColor Red
        return
    }

    $choice = $m.Groups[1].Value.ToUpper()
    $winner = switch ($choice) {
        'TIE' { 'tie' }
        'A'   { if ($controlIsA) { 'a' } else { 'b' } }
        'B'   { if ($controlIsA) { 'b' } else { 'a' } }
    }
    $reasonMatch = [regex]::Match($judged.Response, '(?im)^REASON:\s*(.+)$')
    $reason = if ($reasonMatch.Success) { $reasonMatch.Groups[1].Value.Trim() } else { '' }

    $body = @{
        run_a  = $ControlId
        run_b  = $TreatmentId
        winner = $winner
        judge  = 'copilot-cli (separate session)'
        rubric = $Rubric
        notes  = "seed $Seed; control shown as $(if ($controlIsA) { 'A' } else { 'B' }); $reason"
    } | ConvertTo-Json
    Invoke-RestMethod "$Api/api/experiments/$([uri]::EscapeDataString($Experiment))/pairwise" `
        -Method Post -Body $body -ContentType 'application/json' | Out-Null

    Write-Host "  seed ${Seed}: winner=$winner  ($reason)" -ForegroundColor Cyan
}

# ------------------------------------------------------------------ execute

Assert-Backend
if ($Hypotheses.Count -lt 2) { throw "Need at least two competing hypotheses." }

$rubric = 'Diagnosis before prescription; discriminating evidence named; specific enough to act on this week; tradeoffs stated; coherent and readable. Tiebreak: what a staff engineer could hand their team on Monday.'

Write-Host "experiment '$Experiment': $($Seeds.Count) seeds x 2 arms x up to $MaxSteps steps" -ForegroundColor Green
if (-not $AllowTools) { Write-Host "children run without tools; a researcher brief should say it consulted nothing" -ForegroundColor DarkGray }

$runs = @{}
foreach ($seed in $Seeds) {
    # The control buys one capable generalist. The treatment buys an orchestrator of the same
    # grade plus cheaper specialists, against the same budget, so the comparison is how to spend
    # rather than how much.
    $workers = @{}
    if ($WorkerModel) {
        foreach ($id in 'planner', 'researcher', 'critic', 'verifier', 'memory', 'executor') {
            $workers[$id] = $WorkerModel
        }
    }
    if ($OrchestratorModel) { $workers['generalist'] = $OrchestratorModel }

    $runs[$seed] = @{
        Control   = New-Run -Arm $ControlArm -Seed $seed -AgentModels @{} -DefaultModel $OrchestratorModel
        Treatment = New-Run -Arm $TreatmentArm -Seed $seed -AgentModels $workers -DefaultModel $OrchestratorModel
    }
}

# Control first, across every seed, so no solo answer follows a specialist one.
foreach ($phase in @(@{ Key = 'Control'; Arm = $ControlArm }, @{ Key = 'Treatment'; Arm = $TreatmentArm })) {
    Write-Host "`n== $($phase.Arm)" -ForegroundColor Green
    foreach ($seed in $Seeds) {
        Write-Host "  seed $seed  watch=http://localhost:3000/?run=$($runs[$seed][$phase.Key])"
        Invoke-Arm -RunId $runs[$seed][$phase.Key] -Arm $phase.Arm
    }
}

Write-Host "`n== blind judging" -ForegroundColor Green
foreach ($seed in $Seeds) {
    Invoke-Judge -ControlId $runs[$seed].Control -TreatmentId $runs[$seed].Treatment -Seed $seed -Rubric $rubric
}

Write-Host "`n== comparison" -ForegroundColor Green
$c = Invoke-RestMethod "$Api/api/experiments/$([uri]::EscapeDataString($Experiment))"
Write-Host $c.verdict -ForegroundColor White
$c.arms | ForEach-Object {
    "  {0}: runs={1} esc={2} tokens={3:N0} cost={4:N2} AIU quality={5}" -f `
        $_.arm, $_.runs, $_.escalated, $_.mean_tokens, $_.mean_cost_usd, $_.mean_quality
}
Write-Host ""
$c.caveats | ForEach-Object { Write-Host "  ! $_" -ForegroundColor Yellow }
Write-Host "`n$script:ChildCalls child sessions. Open http://localhost:3000/compare?experiment=$Experiment"
