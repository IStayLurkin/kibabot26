Set-Location $PSScriptRoot

# 1. Ensure Ollama is running
if (-not (Get-Process "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "Starting Ollama server..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# 2. Ensure SearXNG is running
$searxng = docker ps --filter "name=kiba-searxng" --filter "status=running" --format "{{.Names}}" 2>$null
if (-not $searxng) {
    Write-Host "Starting SearXNG..."
    docker compose up -d searxng
    # Give it a few seconds to bind the port
    Start-Sleep -Seconds 4
} else {
    Write-Host "SearXNG already running."
}

# 3. Launch the bot
Write-Host "Launching Kiba Bot..."
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\bot.py"
