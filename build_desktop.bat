@echo off
REM ============================================================
REM  Smart Grading Kopi — Build Desktop Application (EXE)
REM  Menggunakan PyInstaller untuk mengemas semua dependensi.
REM ============================================================
setlocal

echo.
echo ============================================================
echo   Smart Grading Kopi — Desktop Build
echo ============================================================
echo.

REM Pastikan PyInstaller terinstall
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyInstaller tidak ditemukan!
    echo Jalankan: pip install pyinstaller
    pause
    exit /b 1
)

echo [1/3] Membersihkan folder build sebelumnya...
if exist "build" rmdir /s /q "build"
if exist "dist\SmartGradingKopi" rmdir /s /q "dist\SmartGradingKopi"

echo [2/3] Memulai build dengan PyInstaller...
python -m PyInstaller smart_grading_kopi.spec --noconfirm

if errorlevel 1 (
    echo.
    echo [ERROR] Build gagal! Periksa pesan error di atas.
    pause
    exit /b 1
)

echo [3/3] Build selesai!
echo.
echo ============================================================
echo   Output: dist\SmartGradingKopi\SmartGradingKopi.exe
echo ============================================================
echo.
echo Untuk menjalankan:
echo   cd dist\SmartGradingKopi
echo   SmartGradingKopi.exe
echo.

pause