@echo off
REM Quick Vercel Deployment Script for Windows
REM Run this from the project root

echo.
echo ========================================
echo   Vercel Full-Stack Deployment
echo ========================================
echo.

REM Step 1: Check if Vercel CLI is installed
where vercel >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [!] Vercel CLI not found
    echo [*] Installing Vercel CLI...
    call npm i -g vercel
    echo [+] Vercel CLI installed
) else (
    echo [+] Vercel CLI found
)

echo.

REM Step 2: Copy Vercel chat config
echo [*] Setting up chat widget configuration...
copy /Y VERCEL_chat-config.js humanoid-robotics-textbook\static\js\chat-config.js
echo [+] Chat config updated

echo.

REM Step 3: Test frontend build
echo [*] Testing frontend build...
cd humanoid-robotics-textbook
call npm install
call npm run build
cd ..
if %ERRORLEVEL% NEQ 0 (
    echo [!] Frontend build failed!
    pause
    exit /b 1
)
echo [+] Frontend build successful

echo.

REM Step 4: Deploy to Vercel
echo [*] Deploying to Vercel...
echo.
call vercel --prod

echo.
echo ========================================
echo   Deployment Complete!
echo ========================================
echo.
echo [!] IMPORTANT: Set environment variables in Vercel Dashboard:
echo    - OPENROUTER_API_KEY
echo    - COHERE_API_KEY
echo    - QDRANT_URL
echo    - QDRANT_API_KEY
echo    - QDRANT_COLLECTION_NAME
echo.
echo [?] See VERCEL_DEPLOYMENT.md for complete instructions
echo.
pause
