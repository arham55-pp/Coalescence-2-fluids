# Parameter sweep for beta and Pe - Robust version
# Runs coalescence simulations and postprocessing across a parameter grid
# Usage: .\sweep_beta_Pe.ps1 [-ProcessOnly] [-ContinueFromBatch INT]

param(
    [switch]$ProcessOnly,
    [int]$ContinueFromBatch = 0
)

$ErrorActionPreference = "Continue"  # Continue on errors so we can handle them

# Get absolute path to script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SCRIPT_FULL_PATH = $MyInvocation.MyCommand.Path

# Parameter values
$BETA_VALUES = @(0, 0.1, 0.5, 0.9)
$PE_VALUES = @(1, 10, 100, 1000, 10000)
$BASE_DIR = Join-Path $ScriptDir "sweep_beta_Pe_results"
$BATCH_SIZE = 1  # Run 1 simulation at a time to avoid DLL locking with parallel compilation

$TOTAL_CASES = $BETA_VALUES.Count * $PE_VALUES.Count

Write-Host "=============================================="
Write-Host "Parameter Sweep: beta and Pe (ROBUST VERSION)"
Write-Host "beta values: $($BETA_VALUES -join ', ')"
Write-Host "Pe values: $($PE_VALUES -join ', ')"
Write-Host "Total cases: $TOTAL_CASES"
Write-Host "Script directory: $ScriptDir"
Write-Host "Batch size: $BATCH_SIZE (serialized to avoid DLL locking)"
if ($ProcessOnly) {
    Write-Host "Mode: Post-processing only"
}
Write-Host "=============================================="
Write-Host ""

if (-not $ProcessOnly) {
    # Create base output directory if it doesn't exist
    if (-not (Test-Path $BASE_DIR)) {
        Write-Host "Creating output directory: $BASE_DIR"
        New-Item $BASE_DIR -ItemType Directory -Force | Out-Null
    }

    Write-Host ""
    Write-Host "Launching simulations..."
    Write-Host "----------------------------------------------"

    $case_num = 0
    $batch_num = 0
    $jobs = @()
    $failed_cases = @()
    
    foreach ($beta in $BETA_VALUES) {
        foreach ($Pe in $PE_VALUES) {
            $case_num++
            
            # Skip cases if continuing from a batch
            if ($case_num -lt ($ContinueFromBatch * $BATCH_SIZE + 1)) {
                continue
            }
            
            $OUTDIR = Join-Path $BASE_DIR "beta_${beta}_Pe_${Pe}"
            $LOGFILE = Join-Path $BASE_DIR "beta_${beta}_Pe_${Pe}_log.txt"

            # Clean up DLL files to avoid locking issues
            $ccodedir = Join-Path $OUTDIR "_ccode"
            if (Test-Path $ccodedir) {
                Remove-Item (Join-Path $ccodedir "*.dll") -Force -ErrorAction SilentlyContinue
                Remove-Item (Join-Path $ccodedir "*.o") -Force -ErrorAction SilentlyContinue
            }

            # Create output directory
            New-Item $OUTDIR -ItemType Directory -Force | Out-Null

            Write-Host "Case $case_num/${TOTAL_CASES}: beta=$beta, Pe=$Pe"
            Write-Host "  Output: $OUTDIR"
            Write-Host "  Log: $LOGFILE"

            # Run simulation in background
            $job = Start-Job -ScriptBlock {
                param($ScriptDir, $beta, $Pe, $OUTDIR, $LOGFILE)
                Push-Location $ScriptDir
                try {
                    & ".\.venv\Scripts\python.exe" coalescence.py `
                        --beta $beta `
                        --Pe $Pe `
                        --output-dir $OUTDIR `
                        >> $LOGFILE 2>&1
                    Write-Output "SUCCESS"
                }
                catch {
                    Write-Output "FAILED: $_"
                }
                finally {
                    Pop-Location
                }
            } -ArgumentList $ScriptDir, $beta, $Pe, $OUTDIR, $LOGFILE

            $jobs += @{Job=$job; CaseNum=$case_num; Beta=$beta; Pe=$Pe; OutDir=$OUTDIR; LogFile=$LOGFILE}

            # Wait for batch to complete
            if ($jobs.Count -ge $BATCH_SIZE -or $case_num -eq $TOTAL_CASES) {
                $batch_num++
                Write-Host "  Waiting for batch $batch_num ($($jobs.Count) jobs) to complete..."
                
                foreach ($jobInfo in $jobs) {
                    $result = $jobInfo.Job | Wait-Job
                    $output = $result | Receive-Job -ErrorAction SilentlyContinue
                    
                    if ($result.State -eq "Completed" -and $output -match "SUCCESS") {
                        Write-Host "    [OK] Case $($jobInfo.CaseNum) completed (beta=$($jobInfo.Beta), Pe=$($jobInfo.Pe))"
                    }
                    else {
                        Write-Host "    [RETRY] Case $($jobInfo.CaseNum) needs retry (beta=$($jobInfo.Beta), Pe=$($jobInfo.Pe))"
                        $failed_cases += @{CaseNum=$jobInfo.CaseNum; Beta=$jobInfo.Beta; Pe=$jobInfo.Pe; OutDir=$jobInfo.OutDir; LogFile=$jobInfo.LogFile}
                    }
                    
                    # Clean up job
                    $result | Remove-Job -ErrorAction SilentlyContinue
                }
                
                $jobs = @()
                Write-Host ""
            }
        }
    }

    # Retry failed cases
    if ($failed_cases.Count -gt 0) {
        Write-Host ""
        Write-Host "Retrying $($failed_cases.Count) failed case(s)..."
        Write-Host "----------------------------------------------"
        
        foreach ($case in $failed_cases) {
            Write-Host "Retry Case $($case.CaseNum): beta=$($case.Beta), Pe=$($case.Pe)"
            
            # Clean up completely
            Remove-Item $case.OutDir -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item $case.LogFile -Force -ErrorAction SilentlyContinue
            New-Item $case.OutDir -ItemType Directory -Force | Out-Null
            
            $job = Start-Job -ScriptBlock {
                param($ScriptDir, $beta, $Pe, $OUTDIR, $LOGFILE)
                Push-Location $ScriptDir
                try {
                    & ".\.venv\Scripts\python.exe" coalescence.py `
                        --beta $beta `
                        --Pe $Pe `
                        --output-dir $OUTDIR `
                        >> $LOGFILE 2>&1
                    Write-Output "SUCCESS"
                }
                catch {
                    Write-Output "FAILED: $_"
                }
                finally {
                    Pop-Location
                }
            } -ArgumentList $ScriptDir, $case.Beta, $case.Pe, $case.OutDir, $case.LogFile
            
            $result = $job | Wait-Job
            $output = $result | Receive-Job -ErrorAction SilentlyContinue
            
            if ($result.State -eq "Completed" -and $output -match "SUCCESS") {
                Write-Host "  [OK] Retry succeeded"
            }
            else {
                Write-Host "  [FAILED] Retry failed - check log at $($case.LogFile)"
            }
            
            $result | Remove-Job -ErrorAction SilentlyContinue
        }
    }

    Write-Host ""
    Write-Host "All simulations completed!"
}
else {
    # Check that simulation data exists
    if (-not (Test-Path $BASE_DIR)) {
        Write-Host "Error: $BASE_DIR not found. Run simulations first without -ProcessOnly flag."
        exit 1
    }
}

Write-Host ""
Write-Host "Post-processing results..."
Write-Host "----------------------------------------------"

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "Warning: Python venv not found at $PythonExe, using system python"
    $PythonExe = "python"
}

$postproc_count = 0
$postproc_failed = @()

foreach ($beta in $BETA_VALUES) {
    foreach ($Pe in $PE_VALUES) {
        $SIMDIR = Join-Path $BASE_DIR "beta_${beta}_Pe_${Pe}"
        if (Test-Path $SIMDIR) {
            $postproc_count++
            $PLOTSDIR = "${SIMDIR}_plots"
            
            # Skip if plots already exist and are recent
            if ((Test-Path $PLOTSDIR) -and ((Get-ChildItem $PLOTSDIR -Filter "*.png" -ErrorAction SilentlyContinue | Measure-Object).Count -ge 4)) {
                Write-Host "[$postproc_count/${TOTAL_CASES}] SKIPPED (plots exist): beta=$beta, Pe=$Pe"
                continue
            }
            
            Write-Host "[$postproc_count/${TOTAL_CASES}] Processing: beta=$beta, Pe=$Pe"
            Write-Host "  Input: $SIMDIR"
            
            try {
                Push-Location $ScriptDir
                $output = & $PythonExe postprocess.py $SIMDIR 2>&1
                
                # Check if successful
                if (Test-Path $PLOTSDIR) {
                    $plotcount = (Get-ChildItem $PLOTSDIR -Filter "*.png" -ErrorAction SilentlyContinue | Measure-Object).Count
                    if ($plotcount -ge 4) {
                        Write-Host "  [OK] Plots saved to: $PLOTSDIR ($plotcount PNG files)"
                    }
                    else {
                        Write-Host "  [ERROR] Only $plotcount plots generated (expected >= 4)"
                        $postproc_failed += "beta=$beta, Pe=$Pe"
                    }
                }
                else {
                    Write-Host "  [ERROR] Plots directory not created"
                    $postproc_failed += "beta=$beta, Pe=$Pe"
                }
                Pop-Location
            }
            catch {
                Write-Host "  [ERROR] Post-processing failed: $_"
                $postproc_failed += "beta=$beta, Pe=$Pe"
                Pop-Location
            }
        }
        else {
            Write-Host "Warning: $SIMDIR not found, skipping post-processing"
        }
    }
}

Write-Host ""
Write-Host "=============================================="
Write-Host "FINAL SUMMARY"
Write-Host "=============================================="
Write-Host "Total simulations: $TOTAL_CASES"
Write-Host "Post-processed: $postproc_count"

if ($postproc_failed.Count -gt 0) {
    Write-Host "Failed post-processing: $($postproc_failed.Count)"
    foreach ($fail in $postproc_failed) {
        Write-Host "  - $fail"
    }
}
else {
    Write-Host "All post-processing successful!"
}

Write-Host "Results location: $BASE_DIR"
Write-Host "=============================================="

