@echo off
REM Setup Windows Firewall rule for Qwen LLM Server
REM Run this script as Administrator

echo Configuring Windows Firewall for Qwen LLM Server...
echo.

netsh advfirewall firewall add rule name="Qwen LLM Server" dir=in action=allow protocol=TCP localport=8080 profile=private,domain description="Allow local network access to Qwen3-Coder-Next LLM server running in WSL"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Firewall rule created successfully!
    echo Port 8080 is now accessible from your local network.
    echo.
    netsh advfirewall firewall show rule name="Qwen LLM Server"
) else (
    echo.
    echo ERROR: Failed to create firewall rule.
    echo Make sure you are running this script as Administrator.
    echo Right-click the .bat file and select "Run as administrator"
)

echo.
pause
