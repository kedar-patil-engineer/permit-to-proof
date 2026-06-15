#!/usr/bin/env bash
# ===================================================================
#  Permit-to-Proof  -  one-click launcher for macOS / Linux
#  Mirrors run.bat: venv + install (only when changed) + open browser.
# ===================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- 1. Find Python 3 --------------------------------------------------
if command -v python3 >/dev/null 2>&1; then PYEXE=python3
elif command -v python >/dev/null 2>&1; then PYEXE=python
else
  echo "ERROR: Python 3 was not found. Install it from https://www.python.org/downloads/"
  exit 1
fi

# --- 2. Virtual environment -------------------------------------------
[ -d .venv ] || "$PYEXE" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 3. Install / update deps only when changed -----------------------
if ! cmp -s requirements.txt .venv/requirements.lock 2>/dev/null; then
  echo "Installing dependencies (first run or requirements changed)..."
  python -m pip install --upgrade pip -q
  python -m pip install -q -r requirements.txt
  cp requirements.txt .venv/requirements.lock
fi

# --- 4. First-run secrets file ----------------------------------------
[ -f .env ] || { [ -f .env.example ] && cp .env.example .env; }

# --- 5. Open the browser shortly after the server starts --------------
(
  sleep 4
  if command -v open >/dev/null 2>&1; then open http://localhost:8501
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open http://localhost:8501
  fi
) &

# --- 6. Launch the app ------------------------------------------------
echo "Permit-to-Proof is starting at http://localhost:8501 (Ctrl+C to stop)."
streamlit run app/main.py
