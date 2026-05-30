[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8050,
    [switch]$Live,
    [switch]$WithMonitoring,
    [switch]$ForceInstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function New-LocalVenv {
    $candidates = @(
        @("py", "-3.11"),
        @("py", "-3"),
        @("python")
    )

    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
            continue
        }

        if ($candidate.Count -eq 1) {
            & $candidate[0] -m venv $VenvDir 2>$null
        } else {
            & $candidate[0] $candidate[1] -m venv $VenvDir 2>$null
        }

        if ($LASTEXITCODE -eq 0 -and (Test-Path $VenvPython)) {
            return
        }
    }

    throw "Could not create .venv. Install Python 3.11+ and try again."
}

function Test-Imports {
    param(
        [string[]]$Modules
    )

    $checks = $Modules | ForEach-Object { "import $_" }
    $previousEap = $ErrorActionPreference
    $hasNativePref = Test-Path Variable:\PSNativeCommandUseErrorActionPreference

    if ($hasNativePref) {
        $previousNativePref = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }

    try {
        $ErrorActionPreference = "Continue"
        & $VenvPython -c ($checks -join "; ") 1>$null 2>$null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previousEap
        if ($hasNativePref) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePref
        }
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating local virtual environment in .venv..."
    New-LocalVenv
}

$extras = if ($WithMonitoring) {
    ".[torch,dashboard,monitoring]"
} else {
    ".[torch,dashboard]"
}

$requiredModules = @("neuroforge", "torch", "aiohttp")
if ($ForceInstall -or -not (Test-Imports -Modules $requiredModules)) {
    Write-Host "Installing NeuroForge UI dependencies into .venv..."
    & $VenvPython -m pip install -U pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed."
    }

    & $VenvPython -m pip install -e $extras
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency install failed."
    }
}

$launchArgs = @(
    "-m", "neuroforge.runners.cli",
    "ui",
    "--host", $BindHost,
    "--port", $Port
)

if ($Live) {
    $launchArgs += "--live"
}

Write-Host "Starting NeuroForge dashboard on http://$BindHost`:$Port"
& $VenvPython @launchArgs
exit $LASTEXITCODE
