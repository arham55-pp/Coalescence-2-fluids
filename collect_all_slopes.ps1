# Collect slope results from parameter sweep
# Combines results from all beta and Pe combinations into a single array
# Usage: .\collect_all_slopes.ps1

param(
    [string]$BaseDir = "sweep_beta_Pe_results"
)

$BETA_VALUES = @(0, 0.1, 0.5, 0.9)
$PE_VALUES = @(1, 10, 100, 1000, 10000)

Write-Host "=============================================="
Write-Host "Collecting slope results from parameter sweep"
Write-Host "=============================================="

# Array to store all results
$all_results = @()

# Get absolute path to script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

foreach ($beta in $BETA_VALUES) {
    foreach ($Pe in $PE_VALUES) {
        $folder_name = "beta_${beta}_Pe_${Pe}"
        $folder_path = Join-Path $BaseDir $folder_name
        
        Write-Host "Processing: $folder_name" -NoNewline
        
        # Call Python script to extract slopes
        $python_output = & python (Join-Path $ScriptDir "collect_sweep_results.py") $folder_path $beta $Pe 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            $result = $python_output | Select-Object -First 1
            Write-Host " -> $result" -ForegroundColor Green
            
            # Parse result and add to array
            if ($result) {
                $all_results += @($result)
            }
        } else {
            Write-Host " -> FAILED" -ForegroundColor Red
        }
    }
}

# Save results to CSV file
$csv_file = Join-Path $BaseDir "slope_results.csv"
Write-Host ""
Write-Host "Saving results to: $csv_file"

# Create header
$header = "beta,Pe,slope_h0,slope_x0"

# Write to CSV
$header | Out-File -FilePath $csv_file -Encoding UTF8
$all_results | ForEach-Object {
    $_ -replace ' ', ',' | Out-File -FilePath $csv_file -Append -Encoding UTF8
}

Write-Host "Complete! Results saved to $csv_file"
Write-Host ""
Write-Host "Summary of results:"
Write-Host $header
$all_results | ForEach-Object { Write-Host $_ }
