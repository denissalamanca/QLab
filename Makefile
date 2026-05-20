.PHONY: install sync lock test lint format type fix clean redis-up redis-down phase0 phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase9 integration

# --- Setup ----------------------------------------------------------------------
install:
	uv sync --all-groups

sync:
	uv sync

lock:
	uv lock

# --- Quality gates --------------------------------------------------------------
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests

fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests

type:
	uv run mypy src tests

test:
	uv run pytest -q

# --- Phase gates: each runs lint + type + that phase's tests --------------------
phase0: lint type
	uv run pytest -q -m phase0

phase1: lint type
	uv run pytest -q -m phase1

phase2: lint type
	uv run pytest -q -m phase2

phase3: lint type
	uv run pytest -q -m phase3

phase4: lint type
	uv run pytest -q -m phase4

phase5: lint type
	uv run pytest -q -m phase5

phase6: lint type
	uv run pytest -q -m phase6

phase7: lint type
	uv run pytest -q -m phase7

phase8: lint type
	uv run pytest -q -m phase8

phase9: lint type
	uv run pytest -q -m phase9

# Operational milestones (post-Phase-9; see docs/OPERATIONS_ROADMAP.md).
m0: lint type
	uv run pytest -q -m m0

m1: lint type
	uv run pytest -q -m m1

# Cross-phase integration tests (AFML 0-4 audit clearance + Phase 1→4 end-to-end).
integration: lint type
	uv run pytest -q -m integration

# --- Infra ---------------------------------------------------------------------
redis-up:
	docker compose up -d redis

redis-down:
	docker compose down

# --- Housekeeping ---------------------------------------------------------------
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis .coverage htmlcov dist build *.egg-info artifacts
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
