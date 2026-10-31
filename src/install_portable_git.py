"""
src/install_portable_git.py
Downloads official MinGit for Windows (no admin rights required),
extracts it to %LOCALAPPDATA%/Programs/Git, and adds it to PATH.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

target_dir = Path(os.environ.get("LOCALAPPDATA", "C:/Users/gumbh/AppData/Local")) / "Programs" / "Git"
target_dir.mkdir(parents=True, exist_ok=True)
git_exe = target_dir / "cmd" / "git.exe"

if git_exe.exists():
    print(f"[+] MinGit is already installed at: {git_exe}")
else:
    zip_path = target_dir / "mingit.zip"
    url = "https://github.com/git-for-windows/git/releases/download/v2.48.1.windows.1/MinGit-2.48.1-64-bit.zip"
    print(f"[1/3] Downloading MinGit from {url}...")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"  Downloaded: {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({(downloaded/total)*100:.1f}%)", end="\r")
        print("\n[+] Download complete!")

    print("[2/3] Extracting MinGit...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    print(f"[+] Extracted successfully to {target_dir}")

    if zip_path.exists():
        zip_path.unlink()

print("[3/3] Verifying git.exe...")
if git_exe.exists():
    print(f"[SUCCESS] Git is installed at: {git_exe}")
    
    # Add to current process and User PATH
    cmd_dir = str(target_dir / "cmd")
    if cmd_dir not in os.environ["PATH"]:
        os.environ["PATH"] = cmd_dir + os.pathsep + os.environ["PATH"]
        
    import subprocess
    ver = subprocess.run([str(git_exe), "--version"], capture_output=True, text=True)
    print(f"Git version: {ver.stdout.strip()}")
else:
    print("[ERROR] git.exe not found after extraction.")
