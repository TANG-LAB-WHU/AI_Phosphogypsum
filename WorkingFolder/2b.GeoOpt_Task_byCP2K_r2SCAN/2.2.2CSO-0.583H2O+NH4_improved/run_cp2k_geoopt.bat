@echo off
REM ====================================================================
REM  Windows batch script to launch a GPU-accelerated CP2K geometry
REM  optimization for gypsum slab (CaSO4-2H2O) using Docker
REM ====================================================================

REM --------------------------------------------------------------------
REM 0. Display startup information
REM --------------------------------------------------------------------
echo ====================================================================
echo  CP2K Geometry Optimization Runner
echo  Project: Gypsum Slab (CaSO4-2H2O) Geometry Optimization
echo ====================================================================
echo.

REM --------------------------------------------------------------------
REM 1. Check required files exist
REM --------------------------------------------------------------------
echo [Step 1] Checking required files...

if not exist "geo_opt.inp" (
    echo ERROR: geo_opt.inp not found. Aborting.
    pause
    exit /b 1
)

if not exist "optimized_structure_extxyz_wrap.xyz" (
    echo ERROR: Structure file optimized_structure_extxyz_wrap.xyz not found. Aborting.
    pause
    exit /b 1
)

if not exist "docker-compose-cp2k.yml" (
    echo ERROR: docker-compose-cp2k.yml not found. Aborting.
    pause
    exit /b 1
)

echo    - geo_opt.inp: OK
echo    - optimized_structure_extxyz_wrap.xyz: OK
echo    - docker-compose-cp2k.yml: OK
echo.

REM --------------------------------------------------------------------
REM 1.5 Update cell vectors in geo_opt.inp from extxyz Lattice
REM --------------------------------------------------------------------
echo [Step 1.5] Syncing cell vectors from optimized structure...

powershell -NoProfile -Command "$xyz = Get-Content 'optimized_structure_extxyz_wrap.xyz' | Select-Object -Index 1; $match = [regex]::Match($xyz, 'Lattice=[\x22]([^\x22]+)[\x22]'); if ($match.Success) { $vals = $match.Groups[1].Value -split ' '; $A = '      A   ' + $vals[0] + ' ' + $vals[1] + ' ' + $vals[2]; $B = '      B   ' + $vals[3] + ' ' + $vals[4] + ' ' + $vals[5]; $C = '      C   ' + $vals[6] + ' ' + $vals[7] + ' ' + $vals[8]; $inp = Get-Content 'geo_opt.inp'; $newInp = @(); foreach ($line in $inp) { if ($line -match '^\s*A\s+') { $newInp += $A } elseif ($line -match '^\s*B\s+') { $newInp += $B } elseif ($line -match '^\s*C\s+') { $newInp += $C } else { $newInp += $line } }; $newInp | Set-Content 'geo_opt.inp' -Encoding UTF8; Write-Host '   Cell vectors updated:' -ForegroundColor Green; Write-Host ('   A: ' + $vals[0] + ' ' + $vals[1] + ' ' + $vals[2]); Write-Host ('   B: ' + $vals[3] + ' ' + $vals[4] + ' ' + $vals[5]); Write-Host ('   C: ' + $vals[6] + ' ' + $vals[7] + ' ' + $vals[8]) } else { Write-Host '   WARNING: Could not parse Lattice from extxyz file' -ForegroundColor Yellow }"

echo.

REM --------------------------------------------------------------------
REM 2. Clean up any previous Docker containers
REM --------------------------------------------------------------------
echo [Step 2] Cleaning up previous Docker containers...
docker compose -f docker-compose-cp2k.yml down --remove-orphans 2>nul
echo.

REM --------------------------------------------------------------------
REM 3. Launch CP2K in Docker with GPU support
REM --------------------------------------------------------------------
echo [Step 3] Launching CP2K geometry optimization...
echo    Image: mycp2k-rtx5080:master_mpich_native_cuda_A100_psmp
echo    Input: geo_opt.inp
echo    Output: geo_opt.log
echo.
echo Starting CP2K container...
echo ====================================================================
echo.

docker compose -f docker-compose-cp2k.yml up --abort-on-container-exit

REM --------------------------------------------------------------------
REM 4. Post-run information
REM --------------------------------------------------------------------
echo.
echo ====================================================================
echo  CP2K job completed.
echo  Check geo_opt.log for output details.
echo ====================================================================

pause
