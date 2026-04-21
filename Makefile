SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------- setup ----------

.PHONY: venv
venv:  ## Create venv
	python3.11 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -U pip wheel

.PHONY: install
install: venv  ## Install base deps
	$(PIP) install -e ".[dev]"

.PHONY: install-ml
install-ml:  ## Install ML deps (torch, transformers, YOLO, HF datasets)
	$(PIP) install -e ".[ml]"

.PHONY: env
env:  ## Copy .env.example to .env (if not exists)
	@test -f .env || cp .env.example .env
	@echo "✓ .env ready"

# ---------- infra ----------

.PHONY: up
up:  ## Start infra (Qdrant, Redis, Prometheus, Grafana)
	docker compose up -d
	@echo "Qdrant:    http://localhost:6333/dashboard"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:   http://localhost:3000 (anonymous)"

.PHONY: down
down:  ## Stop infra
	docker compose down

.PHONY: nuke
nuke:  ## Stop infra + wipe volumes
	docker compose down -v

.PHONY: ps
ps:  ## Infra status
	docker compose ps

# ---------- app ----------

.PHONY: dev
dev:  ## Run app (uvicorn, reload)
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

.PHONY: run
run:  ## Run app (production mode)
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --workers 2

# ---------- data / eval ----------

.PHONY: ingest
ingest:  ## Ingest HF dataset into Qdrant (100 properties)
	$(PY) -m scripts.ingest_hf --limit 100

.PHONY: eval
eval:  ## Eval on labeled subset (real metrics)
	$(PY) -m evals.run_eval \
		--queries evals/queries_subset.labeled.yaml \
		--endpoint http://127.0.0.1:8002 \
		--top-k 10 --concurrency 2 --json-out

.PHONY: eval-full
eval-full:  ## Eval on full 15-query set (mostly unlabeled — latency/cost benchmark)
	$(PY) -m evals.run_eval \
		--queries evals/queries.yaml \
		--endpoint http://127.0.0.1:8002 \
		--top-k 10 --concurrency 2 --json-out

.PHONY: search
search:  ## Ad-hoc search: make search Q="family with kids and dog"
	@curl -s -X POST http://localhost:8002/search \
		-H 'Content-Type: application/json' \
		-d '{"query": "$(Q)"}' | jq .

# ---------- quality ----------

.PHONY: fmt
fmt:  ## Format code
	$(VENV)/bin/ruff format app evals scripts tests

.PHONY: lint
lint:  ## Lint code
	$(VENV)/bin/ruff check app evals scripts tests

.PHONY: typecheck
typecheck:  ## Run mypy
	$(VENV)/bin/mypy app

.PHONY: test
test:  ## Run pytest
	$(VENV)/bin/pytest

.PHONY: check
check: lint typecheck test  ## All quality gates
