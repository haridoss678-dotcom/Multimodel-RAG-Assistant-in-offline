<#
  Starts the RAG Assistant (API + UI, one process, one port).

      .\scripts\run_server.ps1            # port 8000
      .\scripts\run_server.ps1 -Port 8080

  It exists because three things have to be true before uvicorn starts, and a
  bare `python -m uvicorn` guarantees none of them (CLAUDE.md §7 item 59).
#>
param([int]$Port = 8000, [switch]$Stop)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# uvicorn 0.53 runs a SUPERVISOR plus a worker, so the server is always two
# python.exe processes, not one. That matters operationally: stop only the
# worker and the supervisor may respawn it; stop only the supervisor and the
# worker lingers holding storage/qdrant, so the next start dies with the
# opaque "Storage folder is already accessed by another instance" (§7 item 10).
# Parent-first, and never a blanket kill of every python.exe on the machine -
# this box also runs unrelated Python (§7 item 1's uv-managed interpreter).
function Get-RagServers {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'rag_assistant\.api\.main' } |
        Sort-Object { $_.ParentProcessId -eq $PID } -Descending)
}

function Stop-RagServers {
    $found = Get-RagServers
    if ($found.Count -eq 0) { Write-Host "No RAG Assistant server running."; return }
    # Parents before children, so a supervisor cannot respawn what we just stopped.
    $ordered = $found | Sort-Object { ($found | Where-Object ProcessId -eq $_.ParentProcessId).Count }
    foreach ($proc in $ordered) {
        Write-Host "Stopping PID $($proc.ProcessId)"
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

if ($Stop) { Stop-RagServers; exit 0 }

# 1. PATH, rebuilt from the registry.
#    §7 items 8 and 11: a freshly opened terminal - especially one spawned from
#    Explorer or a stale cached environment - does not reliably carry the
#    persistent PATH, and a real user hit `WinError 2` (ffmpeg not found) this
#    exact way while the tool sat on the User PATH the whole time.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

# Use this checkout's Windows virtual environment. The old `venv` directory
# may be a copied environment from another machine and is not relocatable.
$pythonCandidates = @(
    (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot "..\venv\Scripts\python.exe")
)
$python = $pythonCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $python) {
    Write-Error "No project Python environment found. Create one with 'py -3.11 -m venv .venv' and install requirements.txt."
    exit 1
}
Write-Host "Using Python: $python" -ForegroundColor DarkGray

# 2. Refuse to start a SECOND server.
#    §7 item 10: embedded Qdrant allows exactly one process to hold
#    storage/qdrant. A leftover server turns every request from the new one
#    into an opaque 500, and the cause is invisible unless you go looking for
#    stray processes. Better to refuse here and say which PID to stop.
$running = Get-RagServers
if ($running.Count -gt 0) {
    Write-Host "A RAG Assistant server is already running:" -ForegroundColor Yellow
    $running | ForEach-Object { Write-Host "  PID $($_.ProcessId) (parent $($_.ParentProcessId))" }
    Write-Host "Stop it first:  .\scripts\run_server.ps1 -Stop" -ForegroundColor Yellow
    exit 1
}

# 3. Say what the machine looks like BEFORE the slow part starts.
#    §7 item 15 measured chat latency degrading sharply below ~1GB free, so
#    "close some apps" is advice backed by a number, not a platitude.
$os = Get-CimInstance Win32_OperatingSystem
$freeGb = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
Write-Host "Free RAM: $freeGb GB" -ForegroundColor Cyan
if ($freeGb -lt 1.0) {
    Write-Host "  Under 1 GB free - answers will be markedly slower (measured)." -ForegroundColor Yellow
}
Write-Host "Open http://127.0.0.1:$Port   (Ctrl+C here to stop - do not just close the window)" -ForegroundColor Cyan
Write-Host "Log file: logs\server.log" -ForegroundColor Cyan

# -u so the log file is written as things happen rather than at exit: a
# buffered log is empty in exactly the case you need it (§7 item 59).
& $python -u -m uvicorn rag_assistant.api.main:app --port $Port
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
