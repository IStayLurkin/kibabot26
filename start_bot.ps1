Set-Location $PSScriptRoot

# 1. Docker Desktop
$dockerReady = docker info 2>$null
if (-not $dockerReady) {
    Write-Host "Starting Docker Desktop..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host "Waiting for Docker to be ready..."
    $timeout = 60
    $elapsed = 0
    do {
        Start-Sleep -Seconds 3
        $elapsed += 3
        $dockerReady = docker info 2>$null
    } while (-not $dockerReady -and $elapsed -lt $timeout)

    if (-not $dockerReady) {
        Write-Host "WARNING: Docker did not start in time — SearXNG will not be available."
    }
}

# 3. Ensure SearXNG is running
if ($dockerReady) {
    $searxng = docker ps --filter "name=kiba-searxng" --filter "status=running" --format "{{.Names}}" 2>$null
    if (-not $searxng) {
        Write-Host "Starting SearXNG..."
        docker compose up -d searxng
        Start-Sleep -Seconds 4
    } else {
        Write-Host "SearXNG already running."
    }
}

# 4. Launch the bot
Write-Host "Launching Kiba Bot..."
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\bot.py"
