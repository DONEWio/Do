uv venv -p python3.11
source .venv/bin/activate
uv pip install pip
which pip
uv pip install -e ".[dev,kg]"