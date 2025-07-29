# Cleanup script for Anima project
# This script removes test files, logs, and other unnecessary files

# Test files
$testFiles = @(
    "memory_integration_test.py",
    "validate_voice_emotion_pipeline.py",
    "silent_startup.py",
    "suppress_output.py",
    "voice_ml_results.txt"
)

# Log files
$logFiles = @(
    "memory_diagnostic.log",
    "memory_integration_test_20250725_191133.log",
    "voice_emotion_test.log",
    "voice_ml_test.log",
    "voice_pipeline_validation.log"
)

# Remove test files
foreach ($file in $testFiles) {
    $path = Join-Path "e:\Anima_v4" $file
    if (Test-Path $path) {
        Write-Host "Removing $file"
        Remove-Item -Path $path -Force
    }
}

# Remove log files
foreach ($file in $logFiles) {
    $path = Join-Path "e:\Anima_v4" $file
    if (Test-Path $path) {
        Write-Host "Removing $file"
        Remove-Item -Path $path -Force
    }
}

Write-Host "Cleanup completed."
