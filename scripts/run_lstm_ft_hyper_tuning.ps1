$ErrorActionPreference = "Stop"

$Configs = @(
    "configs/tuning/lstm_ft_lr1e4.yaml",
    "configs/tuning/lstm_ft_lr5e5.yaml",
    "configs/tuning/lstm_ft_lr3e5.yaml",
    "configs/tuning/lstm_ft_lr5e5_lstm256.yaml",
    "configs/tuning/lstm_ft_lr5e5_layer3_layer4.yaml",
    "configs/tuning/lstm_ft_lr5e5_nsteps1024.yaml"
)

foreach ($Config in $Configs) {
    Write-Host "========================================"
    Write-Host "Running $Config"
    Write-Host "========================================"

    python -m baselines.ppo_resnet18_lstm_ft_baseline $Config

    Write-Host "Finished $Config"
}
