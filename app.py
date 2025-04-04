import subprocess
import sys

def run_dashboard():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard.py"])

if __name__ == "__main__":
    run_dashboard()