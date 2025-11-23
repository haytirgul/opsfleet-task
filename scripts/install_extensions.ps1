# PowerShell script to install required VS Code extensions

Write-Host "Installing required VS Code extensions..." -ForegroundColor Cyan

$extensions = @(
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "charliermarsh.ruff",
    "ms-python.mypy-type-checker",
    "ms-python.debugpy"
)

foreach ($ext in $extensions) {
    Write-Host "Installing: $ext" -ForegroundColor Yellow
    code --install-extension $ext --force
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully installed $ext" -ForegroundColor Green
    } else {
        Write-Host "Failed to install $ext" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Extension installation complete!" -ForegroundColor Green
Write-Host "Please reload VS Code window to activate extensions." -ForegroundColor Cyan
