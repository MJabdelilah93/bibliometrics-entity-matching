# run_smoke.ps1 — Quick smoke test for the BEM pipeline.
#
# Prerequisites:
#   1. Create bem/.env containing:  ANTHROPIC_API_KEY=sk-ant-api03-...
#   2. Ensure configs/run_config.yaml has:
#        verification.smoke_test: true
#        smoke_pairs_per_task: 5        (or any small number)
#        llm.backend: "anthropic_api"
#
# Usage (from the repo root in PowerShell):
#   .\scripts\run_smoke.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host ""
Write-Host "=== BEM Smoke Test ===" -ForegroundColor Cyan
Write-Host "Config : configs/run_config.yaml"
Write-Host "Mode   : smoke_test (small pair cap — check run_config.yaml)"
Write-Host ""

python -m bem --config configs/run_config.yaml
$exit_code = $LASTEXITCODE

Write-Host ""
if ($exit_code -eq 0) {
    Write-Host "Smoke test completed successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "Outputs to inspect:"
    Write-Host "  runs/<run_id>/logs/llm_decisions_and.jsonl"
    Write-Host "  runs/<run_id>/logs/llm_decisions_ain.jsonl"
    Write-Host "  runs/<run_id>/logs/routing_log_and.parquet"
    Write-Host "  runs/<run_id>/logs/routing_log_ain.parquet"
    Write-Host "  runs/<run_id>/manifests/thresholds_tuned_dev.json"
    Write-Host ""
    Write-Host "Next step: set verification.smoke_test: false in configs/run_config.yaml"
    Write-Host "           then re-run:  python -m bem --config configs/run_config.yaml"
} else {
    Write-Host "Smoke test FAILED (exit code $exit_code)." -ForegroundColor Red
    Write-Host ""
    Write-Host "Check for errors in:"
    $runs = Get-ChildItem runs -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($runs) {
        Write-Host "  $($runs.FullName)\logs\"
    } else {
        Write-Host "  runs/<latest_run_id>/logs/"
    }
    exit $exit_code
}
