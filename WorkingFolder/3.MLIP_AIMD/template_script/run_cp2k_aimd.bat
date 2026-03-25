@echo off
REM ====================================================================
REM  Terminal Logging Wrapper
REM ====================================================================
if "%1"=="--no-log-wrapper" (
    shift
    goto :main
)

echo [Info] Starting simulation with terminal logging to aimd_terminal.log...
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "& { .\%~nx0 --no-log-wrapper | Tee-Object -FilePath 'aimd_terminal.log' }"
exit /b

:main
REM ====================================================================
REM  Windows batch script to launch a GPU-accelerated CP2K AIMD
REM  simulation for gypsum slab (CaSO4-2H2O) using Docker
REM ====================================================================

REM --------------------------------------------------------------------
REM 0. Display startup information
REM --------------------------------------------------------------------
echo ====================================================================
echo  CP2K AIMD Simulation Runner
echo  Project: Gypsum Slab (CaSO4-2H2O) AIMD Simulation
echo ====================================================================
echo.

REM --------------------------------------------------------------------
REM 1. Check required files exist
REM --------------------------------------------------------------------
echo [Step 1.0] Checking required files...

if not exist "aimd.inp" (
    echo ERROR: aimd.inp not found. Aborting.
    pause
    exit /b 1
)

if not exist "geoopt_optimized_structure_extxyz_wrap.xyz" (
    echo ERROR: Structure file geoopt_optimized_structure_extxyz_wrap.xyz not found. Aborting.
    pause
    exit /b 1
)

if not exist "docker-compose-aimd.yml" (
    echo ERROR: docker-compose-aimd.yml not found. Aborting.
    pause
    exit /b 1
)

echo    - aimd.inp: OK
echo    - geoopt_optimized_structure_extxyz_wrap.xyz: OK
echo    - docker-compose-aimd.yml: OK
echo.

REM --------------------------------------------------------------------
REM 1.1 Extract last frame from trajectory and update coordinate file
REM --------------------------------------------------------------------
set TRAJ_FILE=gypsum_slab_nh4_geoopt-pos-1.xyz
set STRUCT_FILE=geoopt_optimized_structure_extxyz_wrap.xyz

if exist "%TRAJ_FILE%" (
    echo [Step 1.1] Extracting last frame from %TRAJ_FILE%...
    %SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "$traj = Get-Content '%TRAJ_FILE%'; $atomCount = [int]$traj[0].Trim(); $totalLines = $traj.Count; $lastFrameStart = $totalLines - ($atomCount + 2); if ($lastFrameStart -ge 0) { $coords = $traj[$($lastFrameStart+2)..($totalLines-1)]; $header = Get-Content '%STRUCT_FILE%' -TotalCount 2; $newContent = $header + $coords; $utf8NoBom = New-Object System.Text.UTF8Encoding($false); [IO.File]::WriteAllLines((Join-Path $PWD '%STRUCT_FILE%'), [string[]]$newContent, $utf8NoBom); Write-Host '   Structure coordinates updated from last trajectory frame.' -ForegroundColor Green } else { Write-Host '   ERROR: Could not find complete last frame in trajectory.' -ForegroundColor Red }"
) else (
    echo [Step 1.1] WARNING: Trajectory file %TRAJ_FILE% not found. Skipping coordinate update.
)

echo.

REM --------------------------------------------------------------------
REM 1.2 Update cell vectors in aimd.inp from extxyz Lattice
REM --------------------------------------------------------------------
echo [Step 1.2] Syncing cell vectors from optimized structure...

%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "$xyz = Get-Content 'geoopt_optimized_structure_extxyz_wrap.xyz' | Select-Object -Index 1; $match = [regex]::Match($xyz, 'Lattice=[\x22]([^\x22]+)[\x22]'); if ($match.Success) { $vals = $match.Groups[1].Value -split ' '; $A = '      A   ' + $vals[0] + ' ' + $vals[1] + ' ' + $vals[2]; $B = '      B   ' + $vals[3] + ' ' + $vals[4] + ' ' + $vals[5]; $C = '      C   ' + $vals[6] + ' ' + $vals[7] + ' ' + $vals[8]; $inp = Get-Content 'aimd.inp'; $newInp = @(); foreach ($line in $inp) { if ($line -match '^\s*A\s+') { $newInp += $A } elseif ($line -match '^\s*B\s+') { $newInp += $B } elseif ($line -match '^\s*C\s+') { $newInp += $C } else { $newInp += $line } }; $utf8NoBom = New-Object System.Text.UTF8Encoding($false); [IO.File]::WriteAllLines((Join-Path $PWD 'aimd.inp'), [string[]]$newInp, $utf8NoBom); Write-Host '   Cell vectors updated:' -ForegroundColor Green; Write-Host ('   A: ' + $vals[0] + ' ' + $vals[1] + ' ' + $vals[2]); Write-Host ('   B: ' + $vals[3] + ' ' + $vals[4] + ' ' + $vals[5]); Write-Host ('   C: ' + $vals[6] + ' ' + $vals[7] + ' ' + $vals[8]) } else { Write-Host '   WARNING: Could not parse Lattice from extxyz file' -ForegroundColor Yellow }"

echo.

REM --------------------------------------------------------------------
REM 2. Clean up any previous Docker containers
REM --------------------------------------------------------------------
echo [Step 2] Cleaning up previous Docker containers...
docker compose -f docker-compose-aimd.yml down --remove-orphans 2>nul
echo.

REM --------------------------------------------------------------------
REM 3. Launch CP2K in Docker with GPU support
REM --------------------------------------------------------------------
echo [Step 3] Launching CP2K AIMD simulation...
echo    Image: mycp2k-rtx5080:master_mpich_native_cuda_A100_psmp
echo    Input: aimd.inp
echo    Output: aimd.log
echo.
echo Starting CP2K container...
echo ====================================================================
echo.

docker compose -f docker-compose-aimd.yml up --abort-on-container-exit

REM --------------------------------------------------------------------
REM 4. Post-run information
REM --------------------------------------------------------------------
echo.
echo ====================================================================
echo  CP2K job completed.
echo  Check aimd.log for output details.
echo ====================================================================

pause
