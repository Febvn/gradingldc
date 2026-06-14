@echo off
echo ============================================================
echo  Training Script - AMD RX 6700 XT
echo ============================================================
set TF_USE_LEGACY_KERAS=1
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONUTF8=1
set TF_ENABLE_ONEDNN_OPTS=0

echo [1/2] Mencoba GPU training (venv_gpu + DirectML)...
"%~dp0venv_gpu\Scripts\python.exe" "%~dp0train_poc.py" %*
if %ERRORLEVEL% equ 0 goto done

echo.
echo ============================================================
echo  GPU training gagal. Fallback ke CPU training...
echo  (tensorflow-directml-plugin 0.4.0.dev memiliki bug kernel)
echo  Untuk GPU AMD yang benar: gunakan PyTorch + torch-directml
echo ============================================================
echo [2/2] CPU training dengan Python sistem...
python "%~dp0train_poc.py" %*

:done
pause
