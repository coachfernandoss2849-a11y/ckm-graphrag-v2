#!/usr/bin/env python3
"""
CKM Graph RAG v2 — Launch Script
Run: python launch.py
Or:  streamlit run app.py --server.port 8501
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  CKM Graph RAG v2 — Intelligent Clinical Decision Support")
print("=" * 60)
print()
print("Starting Streamlit server on http://localhost:8501")
print("Press Ctrl+C to stop.")
print()

subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.headless", "false",
    "--browser.gatherUsageStats", "false",
])
