@echo off
echo ============================================================
echo  Training dengan GPU AMD RX 6700 XT (DirectML)
echo ============================================================
set TF_USE_LEGACY_KERAS=1
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONUTF8=1
set TF_ENABLE_ONEDNN_OPTS=0
"%~dp0venv_gpu\Scripts\python.exe" "%~dp0train_poc.py" %*
pause
