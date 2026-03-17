Set-Location $PSScriptRoot

# 1. Ensure Ollama is running
if (-not (Get-Process "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "Starting Ollama server..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# 2. Ensure kiba model is built
Write-Host "Verifying kiba model..."
ollama list | Select-String "kiba" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Building kiba model from modelfile..."
    ollama create kiba -f "$PSScriptRoot\kiba.modelfile"
}

# 3. Launch the bot
Write-Host "Launching Kiba Bot..."
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\bot.py"
