# Demo target: intentionally weak patterns for offline blue-team analysis.
# Defensive review only — not production code.

import os
import pickle
import sqlite3
import subprocess


def login(username: str, password: str) -> bool:
    # CWE-89: string-built SQL
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    query = f"SELECT * FROM users WHERE user='{username}' AND pass='{password}'"
    cur.execute(query)
    return cur.fetchone() is not None


def run_diag(host: str) -> str:
    # CWE-78: shell interpolation
    return subprocess.check_output(f"ping -c 1 {host}", shell=True, text=True)


def load_session(path: str):
    # CWE-502: unsafe deserialize
    with open(path, "rb") as f:
        return pickle.load(f)


def fetch_url(url: str) -> bytes:
    # SSRF-shaped helper using requests (dependency risk surface)
    import requests

    return requests.get(url, timeout=5).content


DEBUG = True
SECRET = os.environ.get("APP_SECRET", "hardcoded-demo-secret")
