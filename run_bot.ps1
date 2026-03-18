Set-Location $PSScriptRoot

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Missing virtual environment interpreter at $pythonExe. Recreate the venv before starting the bot."
}

try {
    & $pythonExe --version | Out-Null
} catch {
    throw "The virtual environment interpreter exists but could not start. Recreate the venv or reinstall Python before starting the bot."
}

# Check Ollama is running before launching
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
} catch {
    throw "Ollama is not running. Start Ollama before launching the bot."
}

& $pythonExe bot.py 2>&1 | Tee-Object -FilePath ".\bot.log" -Append
