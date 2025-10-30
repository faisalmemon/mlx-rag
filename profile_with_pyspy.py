#!/usr/bin/env python3
"""
Helper script for profiling create_vdb.py with py-spy
"""
import subprocess
import sys
import time
import threading
import os

def run_target_script():
    """Run the target script"""
    cmd = [
        sys.executable, 
        "create_vdb.py", 
        "--pdf", "flash_attention.pdf", 
        "--vdb", "vdb.npz"
    ]
    return subprocess.run(cmd, capture_output=True, text=True)

def run_pyspy_profiling():
    """Run py-spy profiling in a separate process"""
    # Start the target script
    target_process = subprocess.Popen([
        sys.executable, 
        "create_vdb.py", 
        "--pdf", "flash_attention.pdf", 
        "--vdb", "vdb.npz"
    ])
    
    # Give it a moment to start
    time.sleep(1)
    
    # Run py-spy to profile the process
    pyspy_cmd = [
        "py-spy", "record", 
        "-o", "profile_pyspy.svg",
        "-d", "60",  # Duration in seconds
        "-p", str(target_process.pid)
    ]
    
    try:
        subprocess.run(pyspy_cmd)
        print("✅ py-spy profiling completed - check profile_pyspy.svg")
    except FileNotFoundError:
        print("❌ py-spy not found. Install with: pip install py-spy")
    except Exception as e:
        print(f"❌ py-spy failed: {e}")
    
    # Wait for target to complete
    target_process.wait()

if __name__ == "__main__":
    print("🔍 Starting py-spy profiling...")
    run_pyspy_profiling()