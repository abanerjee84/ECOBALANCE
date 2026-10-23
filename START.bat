@echo off
title Emergent Economies - Vite Dev Server
cd /d "%~dp0"

if not exist "node_modules\vite" (
  echo [START] Installing dependencies, this happens only once...
  call npm install
  if errorlevel 1 (
    echo [START] npm install failed. Check your network / Node.js install.
    pause
    exit /b 1
  )
)

echo [START] Launching Vite dev server...
echo [START] Open http://localhost:5173/  (Ctrl+C here stops the server)
echo.
call npm run dev
pause
