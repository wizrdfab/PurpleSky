param(
    [switch]$Installer,
    [switch]$InstallInno,
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

if (-not $Installer -or -not $InstallInno -or -not $InstallDeps) {
    $rawArgs = $args | ForEach-Object { $_.ToString().ToLowerInvariant() }
    if (-not $Installer -and ($rawArgs -contains "--installer")) { $Installer = $true }
    if (-not $InstallInno -and ($rawArgs -contains "--installinno")) { $InstallInno = $true }
    if (-not $InstallDeps -and ($rawArgs -contains "--installdeps")) { $InstallDeps = $true }
}

function Get-IsccPath {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    if ($env:INNO_SETUP) {
        $candidate = Join-Path $env:INNO_SETUP "ISCC.exe"
        if (Test-Path $candidate) { return $candidate }
    }
    $paths = @()
    if ($env:ProgramFiles -and (Test-Path $env:ProgramFiles)) {
        $paths += Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"
    }
    if (${env:ProgramFiles(x86)} -and (Test-Path ${env:ProgramFiles(x86)})) {
        $paths += Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"
    }
    if ($env:LOCALAPPDATA -and (Test-Path $env:LOCALAPPDATA)) {
        $paths += Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    }
    foreach ($path in $paths) {
        if (Test-Path $path) { return $path }
    }
    $regKeys = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
        "HKCU:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1"
    )
    foreach ($key in $regKeys) {
        if (Test-Path $key) {
            $installDir = (Get-ItemProperty $key).InstallLocation
            if ($installDir) {
                $candidate = Join-Path $installDir "ISCC.exe"
                if (Test-Path $candidate) { return $candidate }
            }
        }
    }
    $classKeys = @(
        "HKLM:\SOFTWARE\Classes\InnoSetupScriptFile\shell\Compile\command",
        "HKLM:\SOFTWARE\WOW6432Node\Classes\InnoSetupScriptFile\shell\Compile\command"
    )
    foreach ($key in $classKeys) {
        if (Test-Path $key) {
            $command = (Get-ItemProperty $key).'(default)'
            if ($command -and $command -match '"([^"]*ISCC\.exe)"') {
                $candidate = $Matches[1]
                if (Test-Path $candidate) { return $candidate }
            }
        }
    }
    return $null
}

& $python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pip') else 1)" | Out-Null
if ($LASTEXITCODE -ne 0) {
    & $python -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip is not available. Install it or use a Python build with pip."
        exit 1
    }
}

& $python -c "import importlib.util, sys; sys.exit(0 if (importlib.util.find_spec('PyInstaller') or importlib.util.find_spec('pyinstaller')) else 1)" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller not found. Installing..."
    & $python -m pip install pyinstaller --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyInstaller install failed. Try: $python -m pip install pyinstaller"
        exit 1
    }
}

function Test-ModuleImport([string]$ModuleName) {
    & $python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}

$modulesToCheck = @(
    "numpy",
    "pandas",
    "joblib",
    "sklearn",
    "scipy",
    "threadpoolctl",
    "lightgbm",
    "pybit",
    "customtkinter"
)

$missingModules = @()
foreach ($module in $modulesToCheck) {
    if (Test-ModuleImport $module) {
        Write-Host "OK: $module"
    } else {
        Write-Host "Missing: $module"
        $missingModules += $module
    }
}

if ($missingModules.Count -gt 0) {
    if (-not $InstallDeps) {
        Write-Host ("Dependencies missing: " + ($missingModules -join ", "))
        Write-Host "Run: .\\build.ps1 --InstallDeps"
        exit 1
    }
    Write-Host "Installing requirements..."
    & $python -m pip install -r requirements.txt --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Requirements install failed. Try: $python -m pip install -r requirements.txt"
        exit 1
    }
    $missingModules = @()
    foreach ($module in $modulesToCheck) {
        if (Test-ModuleImport $module) {
            Write-Host "OK: $module"
        } else {
            Write-Host "Missing: $module"
            $missingModules += $module
        }
    }
    if ($missingModules.Count -gt 0) {
        Write-Host ("Dependencies still missing after install: " + ($missingModules -join ", "))
        exit 1
    }
}

& $python -m PyInstaller --noconfirm --onedir --windowed --name "PurpleSky" `
  --icon "purplesky.ico" `
  --add-data "ui_theme.json;." `
  --add-data "live_dashboard.html;." `
  --add-data "static;static" `
  --collect-data customtkinter `
  --collect-all lightgbm `
  --collect-all sklearn `
  --collect-all scipy `
  --collect-all joblib `
  --collect-all threadpoolctl `
  launcher.py

$distDir = Join-Path $root "dist\\PurpleSky"
$modelsDir = Join-Path $root "models"
if (Test-Path $modelsDir) {
    $targetModels = Join-Path $distDir "models"
    Copy-Item -Path $modelsDir -Destination $targetModels -Recurse -Force
}

$licenseFile = Join-Path $root "LICENSE"
if (Test-Path $licenseFile) {
    Copy-Item -Path $licenseFile -Destination $distDir -Force
}
$sourceInfo = Join-Path $root "SOURCE.txt"
if (Test-Path $sourceInfo) {
    Copy-Item -Path $sourceInfo -Destination $distDir -Force
}

$sourceDir = Join-Path $root "dist\\source"
New-Item -ItemType Directory -Force -Path $sourceDir | Out-Null
$sourceZip = Join-Path $sourceDir "PurpleSky-source.zip"
$pySource = @'
import pathlib
import zipfile
import sys

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])

exclude_dirs = {
    ".git",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "logs",
    "data",
    "models_v9_test",
    "models_v9_tests_binance",
}
exclude_ext = {".pyc", ".pyo", ".log", ".csv", ".jsonl", ".db", ".sqlite", ".tmp"}
exclude_names = {
    ".DS_Store",
    "Thumbs.db",
    ".env",
    "key_profiles_local.json",
    "closed_pnl_stats.json",
}
exclude_prefixes = (
    "bot_state_",
    "key_profiles_",
    "closed_pnl",
    "live_metrics_",
    "signals_",
    "open_orders_",
    "positions_raw_",
    "data_history_",
    "balance_history_",
)

def should_skip(path: pathlib.Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    if any(part in exclude_dirs for part in rel.parts):
        return True
    if path.name in exclude_names:
        return True
    if path.name.startswith(exclude_prefixes):
        return True
    if path.suffix.lower() in exclude_ext:
        return True
    return False

with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        zf.write(path, path.relative_to(root).as_posix())
'@
$scriptPath = Join-Path $sourceDir "make_source_zip.py"
Set-Content -Path $scriptPath -Value $pySource -Encoding UTF8
& $python $scriptPath $root $sourceZip
if ($LASTEXITCODE -ne 0) {
    Write-Host "Source archive build failed."
    exit 1
}
Remove-Item -Path $scriptPath -Force

$distSourceDir = Join-Path $distDir "source"
if (Test-Path $sourceZip) {
    New-Item -ItemType Directory -Force -Path $distSourceDir | Out-Null
    Copy-Item -Path $sourceZip -Destination $distSourceDir -Force
}

if ($Installer) {
    $iscc = Get-IsccPath
    if (-not $iscc -and $InstallInno) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            & winget install -e --id JRSoftware.InnoSetup --accept-package-agreements --accept-source-agreements
        } else {
            $choco = Get-Command choco -ErrorAction SilentlyContinue
            if ($choco) {
                & choco install innosetup -y
            }
        }
        $iscc = Get-IsccPath
    }
    if (-not $iscc) {
        Write-Host "Inno Setup not found. Install it or set INNO_SETUP to its folder."
        exit 1
    }
    Write-Host "Using ISCC: $iscc"
    & $iscc "installer.iss"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installer build failed. Check the ISCC output above."
        exit 1
    }
    $installerExe = Join-Path $root "dist\\installer\\PurpleSky-Setup.exe"
    if (Test-Path $installerExe) {
        Write-Host "Installer created: $installerExe"
    } else {
        Write-Host "Installer build finished but output not found at $installerExe"
        exit 1
    }
}
