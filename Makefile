# ---------- config ----------
VENV        ?= .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
UVICORN     := $(VENV)/bin/uvicorn
PYTEST      := $(VENV)/bin/pytest
RUFF        := $(VENV)/bin/ruff
MYPY        := $(VENV)/bin/mypy

IMAGE       ?= regime-forecast-lite:latest
PORT        ?= 8000
WORKERS     ?= 1

.PHONY: help venv init install lint fmt typecheck test run run-prod ci \
        docker-build docker-build-dev docker-run docker-run-dev readme clean

help:
	@echo "Targets:"
	@echo "  init            - venv + install dev/service/plot/market extras + pre-commit (if present)"
	@echo "  run             - dev server (uvicorn --reload)"
	@echo "  run-prod        - prod server (no reload, --workers=$(WORKERS))"
	@echo "  docker-build    - build lean image"
	@echo "  docker-build-dev- build image incl. dev deps"
	@echo "  docker-run      - run image on port $(PORT)"
	@echo "  docker-run-dev  - run image mounting source, with reload"
	@echo "  lint/fmt/typecheck/test/ci/readme/clean"

venv:
	test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,service,plot,market,backtest]"
	-$(VENV)/bin/pre-commit install

init: install

lint:
	$(RUFF) check .

fmt:
	$(RUFF) check . --fix
	$(RUFF) format .

typecheck:
	$(MYPY) .

test:
	$(PYTEST) -q

run:
	$(UVICORN) service.app:app --reload --host 0.0.0.0 --port $(PORT)

run-prod:
	$(UVICORN) service.app:app --host 0.0.0.0 --port $(PORT) --workers $(WORKERS)

ci:
	./scripts/ci_checks.sh

readme:
	./scripts/readme_run.sh

docker-build:
	docker build --build-arg INSTALL_EXTRAS=service -t $(IMAGE) .

docker-build-dev:
	docker build --build-arg INSTALL_EXTRAS="service,plot,market,backtest,dev" -t $(IMAGE) .


docker-run:
	docker run --rm -p $(PORT):8000 $(IMAGE)

docker-run-dev:
	docker run --rm -it -p $(PORT):8000 -v "$(CURDIR)":/app $(IMAGE) \
	  uvicorn service.app:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache build dist *.egg-info .coverage coverage.xml
