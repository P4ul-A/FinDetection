#!/bin/zsh
set -e

cd -- "$(dirname "$0")"

REQUIREMENTS_FILE="requirements.txt"

show_error_and_pause() {
  echo ""
  echo "Setup failed. Please copy the messages above if you need help."
  echo "Press any key to close this window."
  read -k 1
}

trap show_error_and_pause ERR

if [ -x "venv/bin/python" ]; then
  PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 was not found. Install Python 3, then open this launcher again."
    exit 1
  fi
  echo "Creating local Python environment..."
  python3 -m venv venv
  PYTHON="venv/bin/python"
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Could not find $REQUIREMENTS_FILE in this folder."
  exit 1
fi

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "Setting up pip..."
  "$PYTHON" -m ensurepip --upgrade
fi

if ! "$PYTHON" - <<'PY'
import sys

try:
    import tkinter
except Exception as exc:
    print("This Python installation cannot open Tkinter apps.")
    print(exc)
    sys.exit(1)
PY
then
  exit 1
fi

if ! "$PYTHON" - "$REQUIREMENTS_FILE" <<'PY'
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

requirements_path = Path(sys.argv[1])


def version_parts(value):
    return tuple(int(part) for part in re.findall(r"\d+", value.split("+", 1)[0]))


def is_too_old(installed, minimum):
    installed_parts = version_parts(installed)
    minimum_parts = version_parts(minimum)
    length = max(len(installed_parts), len(minimum_parts))
    installed_parts += (0,) * (length - len(installed_parts))
    minimum_parts += (0,) * (length - len(minimum_parts))
    return installed_parts < minimum_parts


needs_install = []
for line in requirements_path.read_text(encoding="utf-8").splitlines():
    requirement = line.strip()
    if not requirement or requirement.startswith("#"):
        continue

    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*>=\s*([^;\s]+)", requirement)
    if not match:
        continue

    package_name, minimum_version = match.groups()
    try:
        installed_version = version(package_name)
    except PackageNotFoundError:
        needs_install.append(f"{package_name}>={minimum_version} missing")
        continue

    if is_too_old(installed_version, minimum_version):
        needs_install.append(
            f"{package_name}>={minimum_version} installed {installed_version}"
        )

if needs_install:
    print("Requirements need install/update:")
    for item in needs_install:
        print("  " + item)
    sys.exit(1)
PY
then
  echo "Installing required packages. This can take a few minutes the first time..."
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$REQUIREMENTS_FILE"
fi

if ! "$PYTHON" - <<'PY'
import importlib.util
import sys

required_modules = (
    "PIL",
    "fastapi",
    "matplotlib",
    "multipart",
    "pandas",
    "rawpy",
    "requests",
    "ultralytics",
    "uvicorn",
    "yaml",
)

missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    print("Still missing required Python modules: " + ", ".join(missing))
    sys.exit(1)
PY
then
  exit 1
fi

exec "$PYTHON" findetection_app.py
