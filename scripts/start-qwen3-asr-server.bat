@echo off
rem Start the Qwen3-ASR server with the CUDA-enabled venv python.
rem Adjust QWEN3_VENV if your venv lives elsewhere.
set "QWEN3_VENV=C:\mt\qwen3-test\venv"
"%QWEN3_VENV%\Scripts\python.exe" "%~dp0qwen3-asr-server.py" %*
