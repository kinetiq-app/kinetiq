<#
.SYNOPSIS
  Kinetiq v5 Instant Live Development Pipeline
  Starts the unified FastAPI + Static PWA server and exposes it via a secure Cloudflare HTTPS Tunnel.
  Changes made to frontend/ or backend/ code are visible instantly without any deployment wait.

.USAGE
  .\dev_tunnel.ps1
#>

param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Gate0Dir = Join-Path $ScriptDir 'evals\gate0'
$Cloudflared = Join-Path $ScriptDir 'cloudflared.exe'

if (-not (Test-Path $Cloudflared)) {
    Write-Host "Downloading portable cloudflared.exe..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $Cloudflared
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Kinetiq v5 - Instant Live Development Pipeline" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Start or check prototype_api (which serves both PWA and API)
$apiProcess = $null
$healthCheck = try { (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue).status } catch { $null }

if ($healthCheck -eq 'ok') {
    Write-Host "[OK] Unified server is already running on http://127.0.0.1:$Port" -ForegroundColor Green
} else {
    Write-Host "[*] Starting unified server on http://127.0.0.1:$Port..." -ForegroundColor Yellow
    $apiProcess = Start-Process -FilePath 'python' `
        -ArgumentList @('-m', 'uvicorn', 'prototype_api.main:app', '--host', '127.0.0.1', '--port', "$Port") `
        -WorkingDirectory $Gate0Dir -PassThru -NoNewWindow
    
    # Wait for server to be responsive
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 1 -ErrorAction SilentlyContinue
            if ($h.status -eq 'ok') { $ready = $true; break }
        } catch {}
    }
    if (-not $ready) {
        Write-Host "[ERROR] Unified server failed to respond on http://127.0.0.1:$Port" -ForegroundColor Red
        if ($apiProcess) { Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue }
        exit 1
    }
    Write-Host "[OK] Unified server is active." -ForegroundColor Green
}

# 2. Start Cloudflare Tunnel
Write-Host "[*] Establishing secure HTTPS Cloudflare Tunnel..." -ForegroundColor Yellow
$logFile = [System.IO.Path]::GetTempFileName()
$tunnelProcess = Start-Process -FilePath $Cloudflared `
    -ArgumentList @('tunnel', '--url', "http://127.0.0.1:$Port") `
    -RedirectStandardError $logFile -PassThru -NoNewWindow

$tunnelUrl = $null
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    if (Test-Path $logFile) {
        $content = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
        if ($content -match 'https://[a-zA-Z0-9-]+\.trycloudflare\.com') {
            $tunnelUrl = $matches[0]
            break
        }
    }
}

if (-not $tunnelUrl) {
    Write-Host "[ERROR] Could not extract tunnel URL. Check log at: $logFile" -ForegroundColor Red
    if ($tunnelProcess) { Stop-Process -Id $tunnelProcess.Id -Force -ErrorAction SilentlyContinue }
    if ($apiProcess) { Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue }
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  LIVE PUBLIC HTTPS URL (Share with users or open on phone):" -ForegroundColor Green
Write-Host "  $tunnelUrl" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Features of this live pipeline:" -ForegroundColor Cyan
Write-Host "  - 100% same-origin: ZERO CORS errors and ZERO mixed-content issues"
Write-Host "  - Instant updates: edit frontend/ or backend/ and refresh to see changes"
Write-Host "  - Fully independent of ManuP001 or external repositories"
Write-Host ""
Write-Host "Press Ctrl+C to terminate the live tunnel and server." -ForegroundColor Yellow

try {
    while ($true) {
        Start-Sleep -Seconds 1
        if ($tunnelProcess.HasExited) { break }
    }
}
finally {
    Write-Host "`nStopping tunnel and local services..." -ForegroundColor Yellow
    if ($tunnelProcess -and -not $tunnelProcess.HasExited) {
        Stop-Process -Id $tunnelProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $logFile -Force -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Yellow
}
