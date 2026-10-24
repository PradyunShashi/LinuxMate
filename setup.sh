#!/usr/bin/env bash
# LinuxMate - install requirements and run the app (WSL / Linux)
#   bash setup.sh          install + start      ->  http://localhost:5000
#   bash setup.sh check    test setup only
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv || { echo "Run:  sudo apt install python3-venv python3-pip"; exit 1; }
source .venv/bin/activate
pip install -q flask || exit 1
if [ "${1:-}" = "check" ]; then exec python linuxmate.py --check; fi
exec python linuxmate.py
