Set-Location $PSScriptRoot

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$watchfilesExe = Join-Path $PSScriptRoot ".venv\Scripts\watchfiles.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Missing virtual environment interpreter at $pythonExe. Recreate the venv before starting the bot watcher."
}

if (-not (Test-Path $watchfilesExe)) {
    throw "Missing watchfiles executable at $watchfilesExe. Install dependencies before starting the bot watcher."
}

try {
    & $pythonExe --version | Out-Null
} catch {
    throw "The virtual environment interpreter exists but could not start. Recreate the venv or reinstall Python before starting the bot watcher."
}

$command = '"' + $pythonExe + '" bot.py'

& $watchfilesExe $command cogs services core tasks database bot.py
