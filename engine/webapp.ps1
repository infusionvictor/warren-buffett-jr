# Lanzador de la app web de Warren Buffett Jr (Windows / PowerShell)
# Uso:  .\webapp.ps1        -> abre http://localhost:8765
# Requiere el venv ya creado en engine\.venv (ver RUN_WINDOWS.md).

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Error "No existe el entorno virtual en engine\.venv. Corre primero los pasos de RUN_WINDOWS.md."
    exit 1
}

# Forzar UTF-8 para que los acentos y símbolos (≥, █) no rompan en Windows.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Warren Buffett Jr -> http://localhost:8765  (Ctrl+C para detener)" -ForegroundColor Green
& $py (Join-Path $here "scripts\webapp.py")
