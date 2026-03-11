Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python -m watchfiles "python bot.py" cogs services core tasks bot.py