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

& $pythonExe bot.py
