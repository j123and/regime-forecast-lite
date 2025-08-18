.PHONY: init lint fmt test run
init:
python3 -m venv .venv
. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]" && pre-commit install
lint:
. .venv/bin/activate && ruff check .
fmt:
. .venv/bin/activate && ruff check . --fix && ruff format .
test:
. .venv/bin/activate && pytest -q
run:
. .venv/bin/activate && uvicorn service.app:app --reload --host 0.0.0.0 --port 8000
