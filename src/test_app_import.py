"""
Test running app.py in an isolated process with sys.path starting inside app/
"""
import subprocess
import sys

# Test from project root
res1 = subprocess.run([sys.executable, "app/app.py"], capture_output=True, text=True)
print("Running from project root:", res1.returncode)
if res1.returncode != 0:
    print(res1.stderr)

# Test from inside app directory
res2 = subprocess.run([sys.executable, "app.py"], cwd="app", capture_output=True, text=True)
print("Running from inside app/ dir:", res2.returncode)
if res2.returncode != 0:
    print(res2.stderr)
else:
    print("Success from both working directories!")
