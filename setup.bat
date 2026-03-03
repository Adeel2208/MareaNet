@echo off
REM Quick setup script for MAREA-Net (Windows)

echo ==================================
echo MAREA-Net Setup
echo ==================================

REM Check Python version
echo.
echo Checking Python version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo.
echo Creating directories...
if not exist "data\train\images" mkdir data\train\images
if not exist "data\train\masks" mkdir data\train\masks
if not exist "data\test\images" mkdir data\test\images
if not exist "data\test\masks" mkdir data\test\masks
if not exist "models" mkdir models
if not exist "results" mkdir results

echo.
echo ==================================
echo Setup complete!
echo ==================================
echo.
echo Next steps:
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Download the SUIM dataset:
echo    python scripts\download_dataset.py
echo.
echo 3. Start training:
echo    python train.py --data_dir data --output_dir models
echo.
echo For more information, see README.md
echo ==================================

pause
