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

if not exist "conventional_cell_slab_020_L1_2x2.xyz" (
    echo ERROR: Structure file conventional_cell_slab_020_L1_2x2.xyz not found. Aborting.
    pause
    exit /b 1
)

if not exist "docker-compose-cp2k.yml" (
    echo ERROR: docker-compose-cp2k.yml not found. Aborting.
    pause
    exit /b 1
)

echo    - geo_opt.inp: OK
echo    - conventional_cell_slab_020_L1_2x2.xyz: OK
echo    - docker-compose-cp2k.yml: OK
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
