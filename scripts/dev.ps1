<#
.SYNOPSIS
    Starts the Markov Agent Orchestrator backend and, when present, the frontend.

.DESCRIPTION
    Launches uvicorn in its own window and the Next.js dev server in the current window.
    The Python environment is expected to live outside any cloud-synced folder.

.PARAMETER VenvPath
    Root of the virtual environment. Defaults to C:\venvs\markov-agent-orchestrator.

.PARAMETER BackendPort
    Port for the FastAPI server. Defaults to 8000.

.EXAMPLE
    .\scripts\dev.ps1

.EXAMPLE
    .\scripts\dev.ps1 -VenvPath D:\envs\markov -BackendPort 8080
#>
[CmdletBinding()]
param(
    [string]$VenvPath = 'C:\venvs\markov-agent-orchestrator',
    [int]$BackendPort = 8000
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $repoRoot 'backend'
$frontendDir = Join-Path $repoRoot 'frontend'
$python = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Error @"
Python environment not found at $python

Create it first:
    py -3 -m venv $VenvPath
    $VenvPath\Scripts\Activate.ps1
    pip install -r "$backendDir\requirements.txt"
"@
}

Write-Host "Starting backend on http://localhost:$BackendPort" -ForegroundColor Cyan
Start-Process -FilePath $python `
    -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--reload', '--port', "$BackendPort") `
    -WorkingDirectory $backendDir

if (-not (Test-Path (Join-Path $frontendDir 'package.json'))) {
    Write-Host 'Frontend not present yet. Backend is running on its own.' -ForegroundColor Yellow
    Write-Host "API docs: http://localhost:$BackendPort/docs" -ForegroundColor Green
    return
}

if (-not (Test-Path (Join-Path $frontendDir 'node_modules'))) {
    Write-Host 'Installing frontend dependencies' -ForegroundColor Cyan
    Push-Location $frontendDir
    try { npm install } finally { Pop-Location }
}

Write-Host 'Starting frontend on http://localhost:3000' -ForegroundColor Cyan
Push-Location $frontendDir
try { npm run dev } finally { Pop-Location }
