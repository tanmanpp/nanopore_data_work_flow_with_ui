# run_ui.py
import os
import time
import subprocess
import shlex
from pathlib import Path

APP_MODULE = "main:app"
HOST = "0.0.0.0"
PORT = 8000

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
EN_HTML = "v1.3_20250812_en.html"
ZH_HTML = "v1.3_20250812_zh.html"

def get_wsl_ip() -> str:
    """Return the first value from `hostname -I` output (no filtering)."""
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=3).strip()
        if out:
            return out.split()[0]
    except Exception:
        pass
    return "127.0.0.1"

def main():
    # Start uvicorn backend
    uv_cmd = f"uvicorn {APP_MODULE} --host {HOST} --port {PORT}"
    print("🚀 Starting backend:", uv_cmd)
    proc = subprocess.Popen(shlex.split(uv_cmd), cwd=str(ROOT))

    # Get WSL IP
    wsl_ip = get_wsl_ip()
    base = f"http://{wsl_ip}:{PORT}"

    en_url = f"{base}/ui/{EN_HTML}"
    zh_url = f"{base}/ui/{ZH_HTML}"

    print("\n✅ Service is running")
    print(f"🔎 Detected IP: {wsl_ip}")
    print("\n— UI entry (English):")
    print(f"   {en_url}")
    print("\n— UI entry (Chinese):")
    print(f"   {zh_url}")
    print("\n📌 Copy and paste one of the above URLs into your Windows browser.")
    print("   Press Ctrl+C to stop the service.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down uvicorn ...")
        proc.terminate()

if __name__ == "__main__":
    main()
