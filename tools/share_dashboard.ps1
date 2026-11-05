# Share the dashboard through an ngrok link.
#   Run from the project folder:   powershell -ExecutionPolicy Bypass -File tools\share_dashboard.ps1
# It checks the dashboard password, starts the dashboard if needed, starts ngrok and prints the public link.
# Stop sharing: close the ngrok window (or press Ctrl+C in it).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$port = 8501

# 1. a public link must be protected by the dashboard password
$envFile = Join-Path $root ".env"
$pw = ""
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*DASHBOARD_PASSWORD\s*=\s*(.*)$') { $pw = $Matches[1].Trim().Trim('"').Trim("'") }
    }
}
if (-not $pw) {
    Write-Host ""
    Write-Host "STOP: DASHBOARD_PASSWORD is empty in .env." -ForegroundColor Red
    Write-Host "A public link without a password lets anyone see the camera pictures and press Stop."
    Write-Host "Add a line like   DASHBOARD_PASSWORD=choose-a-strong-password   to .env, then run this script again."
    exit 1
}

# 2. ngrok must be installed
if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ngrok is not installed. Install it with:   winget install ngrok.ngrok" -ForegroundColor Yellow
    Write-Host "Then close and reopen PowerShell and run this script again."
    exit 1
}

# 3. start the dashboard if nothing is listening on the port yet
$listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Write-Host "Starting the dashboard in a new window ..."
    $cmd = "Set-Location '$root'; .\.venv\Scripts\Activate.ps1; streamlit run dashboard\app.py --server.port $port --server.headless true"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 1
        if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { break }
    }
} else {
    Write-Host "The dashboard is already running on port $port."
}

# 4. start ngrok in its own window
Write-Host "Starting ngrok ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ngrok http $port"

# 5. read the public link from ngrok's local API
$url = $null
for ($i = 0; $i -lt 20 -and -not $url; $i++) {
    Start-Sleep -Seconds 1
    try {
        $t = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -ErrorAction Stop
        $url = ($t.tunnels | Where-Object { $_.public_url -like "https://*" } | Select-Object -First 1).public_url
    } catch { }
}
Write-Host ""
if ($url) {
    Write-Host "Your public link:" -ForegroundColor Green
    Write-Host "  $url" -ForegroundColor Green
    Write-Host ""
    Write-Host "Anyone with this link sees the login screen. Share the password separately."
    Write-Host "The first visit may show an ngrok notice page: click 'Visit Site'."
    Write-Host "To stop sharing, close the ngrok window."
} else {
    Write-Host "Could not read the link. Look in the ngrok window: the line starting with 'Forwarding' has it." -ForegroundColor Yellow
    Write-Host "If it says the authtoken is missing, run:  ngrok config add-authtoken YOUR_TOKEN"
}
