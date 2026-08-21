<#
.SYNOPSIS
    Downloads the PixelLab character archives and regenerates the sprite manifest.

.DESCRIPTION
    Each character ZIP contains rotations plus one folder per animation, so nothing here has to
    guess per-animation UUIDs. Re-runnable: change the IDs below, run it, and both the PNGs and
    frontend/public/sprites/manifest.json are rebuilt.

    Downloads are public (the character UUID is the access key), so no API key is needed. The
    endpoint returns HTTP 423 while animation jobs are still rendering, hence the retry loop.
#>
[CmdletBinding()]
param(
    [int]$RetryCount = 12,
    [int]$RetryDelaySeconds = 20
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$repoRoot = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $repoRoot 'frontend\public\sprites'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Mr. Robot themed roster: archetypes evoking the show's world, not character likenesses.
$characters = [ordered]@{
    planner      = '3bf751bf-6ab8-4f0b-aa33-73599b61189b'
    researcher   = '2bdca27b-382f-4e7b-8613-48f694111ad1'
    critic       = '39ea6a3d-4616-4d85-94c9-1be2eb19e343'
    verifier     = '6293db5d-3d18-4dac-84c6-0de86718d6df'
    memory       = '6b2be1f0-92d3-4dc6-a58b-51c34b4b1327'
    executor     = 'f5134b78-8967-48de-83dc-46baadddbd8e'
    orchestrator = '61cc6445-54ed-43a6-b79d-ce8fb7e79a60'
}

$descriptions = @{
    planner      = 'Hooded hacker, hood up, face in shadow'
    researcher   = 'Short dark hair, black leather jacket, headphones'
    critic       = 'Corporate executive, navy suit, slicked blond hair'
    verifier     = 'Security analyst, white shirt and tie, ID lanyard'
    memory       = 'Server technician, grey jumpsuit, coiled cable'
    executor     = 'Hacker in a black hoodie carrying a laptop'
    orchestrator = 'Weathered figure in a worn dark jacket and cap'
}

# Idle loops slowly; attack is a one-shot that hands control back to idle.
$fps = @{ idle = 6; attack = 14 }

$tmpRoot = Join-Path $env:TEMP "pixellab-sprites-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

$manifestSprites = [ordered]@{}

foreach ($name in $characters.Keys) {
    $id = $characters[$name]
    $zipPath = Join-Path $tmpRoot "$name.zip"
    $url = "https://api.pixellab.ai/mcp/characters/$id/download"

    $downloaded = $false
    for ($attempt = 1; $attempt -le $RetryCount -and -not $downloaded; $attempt++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
            $downloaded = $true
        } catch {
            $status = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.value__ } else { 0 }
            if ($status -eq 423 -and $attempt -lt $RetryCount) {
                Write-Host "  $name still rendering (423), retry $attempt/$RetryCount" -ForegroundColor DarkYellow
                Start-Sleep -Seconds $RetryDelaySeconds
            } else {
                throw
            }
        }
    }

    $extractDir = Join-Path $tmpRoot $name
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $extractDir)

    # The top folder is named after the character state ("Idle"), so discover it rather than
    # hard-coding it.
    $stateDir = Get-ChildItem $extractDir -Directory | Select-Object -First 1

    $entry = [ordered]@{
        src         = "/sprites/$name.png"
        size        = 68
        description = $descriptions[$name]
    }

    foreach ($direction in @('south', 'east', 'north', 'west')) {
        $source = Join-Path $stateDir.FullName "rotations\$direction.png"
        if (-not (Test-Path $source)) { continue }
        $suffix = if ($direction -eq 'south') { '' } else { "-$direction" }
        Copy-Item $source (Join-Path $outDir "$name$suffix.png") -Force
    }

    $animRoot = Join-Path $stateDir.FullName 'animations'
    $animationCount = 0
    if (Test-Path $animRoot) {
        $animations = [ordered]@{}
        foreach ($animDir in Get-ChildItem $animRoot -Directory) {
            $animName = $animDir.Name
            $southDir = Join-Path $animDir.FullName 'south'
            if (-not (Test-Path $southDir)) { continue }

            $frameFiles = @(Get-ChildItem $southDir -Filter 'frame_*.png' | Sort-Object Name)
            if ($frameFiles.Count -eq 0) { continue }

            $frames = @()
            for ($i = 0; $i -lt $frameFiles.Count; $i++) {
                $target = "$name-$animName-$i.png"
                Copy-Item $frameFiles[$i].FullName (Join-Path $outDir $target) -Force
                $frames += "/sprites/$target"
            }

            $clip = [ordered]@{
                frames = $frames
                fps    = if ($fps.ContainsKey($animName)) { $fps[$animName] } else { 8 }
            }
            if ($animName -eq 'attack') { $clip['once'] = $true }
            $animations[$animName] = $clip
        }
        if ($animations.Count -gt 0) {
            $entry['animations'] = $animations
            $animationCount = $animations.Count
        }
    }

    $manifestSprites[$name] = $entry
    Write-Host "$name  ($animationCount animations)" -ForegroundColor Cyan
}

$manifest = [ordered]@{
    generator   = 'pixellab-mcp'
    generatedAt = (Get-Date -Format 'yyyy-MM-dd')
    note        = 'Mr. Robot themed roster. create_character (standard, 4 directions, chibi, low top-down) at 1 generation each, plus template animations at 1 generation per direction. Regenerate with scripts/fetch-sprites.ps1.'
    sprites     = $manifestSprites
}

$manifest | ConvertTo-Json -Depth 8 |
    Set-Content -Path (Join-Path $outDir 'manifest.json') -Encoding UTF8

Remove-Item $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`nSprites and manifest written to $outDir" -ForegroundColor Green
