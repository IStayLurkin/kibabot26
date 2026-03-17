Set-Location $PSScriptRoot

# 1. Ensure Ollama is running
if (-not (Get-Process "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "Starting Ollama server..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# 2. Launch the bot
Write-Host "Launching Kiba Bot..."
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\bot.py"
