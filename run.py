import os
import platform
import subprocess
import sys
from pathlib import Path
import shutil

# --- Detect the correct system Python ---
def find_python_command():
    # On Windows, usually 'python' works
    if platform.system() == "Windows":
        return "python"

    # On macOS/Linux, test 'python3' first, fallback to 'python'
    for cmd in ("python3", "python"):
        try:
            subprocess.run([cmd, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return cmd
        except Exception:
            continue
    sys.exit("No suitable Python interpreter found (need Python 3.8+).")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
IS_WINDOWS = platform.system() == "Windows"
PYTHON_BIN = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

# --- Core steps ---
def create_virtualenv():
    if not PYTHON_BIN.exists():
        print("Creating virtual environment...")
        python_cmd = find_python_command()
        subprocess.check_call([python_cmd, "-m", "venv", str(VENV_DIR)])
    else:
        print("Virtual environment already exists.")

def install_requirements():
    print("Installing dependencies...")
    subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip"])
    if (BASE_DIR / "requirements.txt").exists():
        subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("No requirements.txt found — skipping dependency installation.")

def run_streamlit():
    #print("Running migrations and starting Django server...")
    #subprocess.check_call([str(PYTHON_BIN), "manage.py", "migrate"])
    subprocess.check_call([str(PYTHON_BIN), "-m", "streamlit", "run", "app.py"])


# --- Main ---
if __name__ == "__main__":
    create_virtualenv()
    install_requirements()
    run_streamlit()