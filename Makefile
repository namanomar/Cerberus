# ──────────────────────────────────────────────────────────────────────────────
# Fraud Detection System — Makefile
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   make install       – install dependencies
#   make preprocess    – run data preprocessing
#   make train         – run full training pipeline
#   make evaluate      – evaluate trained models
#   make api           – start the FastAPI server
#   make dashboard     – start the Streamlit dashboard
#   make stream-sim    – run Kafka consumer in simulation mode
#   make docker-up     – start all services via Docker Compose
#   make docker-down   – stop all Docker services
#   make test          – run unit tests
#   make lint          – run code linting
#   make clean         – remove generated artifacts
# ──────────────────────────────────────────────────────────────────────────────

PYTHON       ?= python3
PIP          ?= pip3
CONFIG       ?= config/config.yaml
PORT_API     ?= 8000
PORT_DASH    ?= 8501
COMPOSE      ?= docker-compose

.PHONY: all install preprocess train evaluate api dashboard \
        stream-produce stream-consume stream-sim \
        docker-up docker-down docker-build docker-logs \
        test lint clean help

# ── Default ───────────────────────────────────────────────────────────────────
all: help

# ── Environment ───────────────────────────────────────────────────────────────
install:
	@echo "Installing Python dependencies …"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Done."

.env:
	cp .env.example .env
	@echo "Created .env from .env.example — edit it with your settings."

# ── Data & Training ───────────────────────────────────────────────────────────
preprocess:
	@echo "Running data preprocessing …"
	$(PYTHON) -m src.preprocessing.clean_data $(CONFIG)

train:
	@echo "Running full training pipeline …"
	$(PYTHON) scripts/train_pipeline.py --config $(CONFIG)

train-fast:
	@echo "Running training pipeline (skip graph) …"
	$(PYTHON) scripts/train_pipeline.py --config $(CONFIG) --skip-graph

evaluate:
	@echo "Evaluating trained models …"
	$(PYTHON) scripts/evaluate_model.py --config $(CONFIG)

# ── Services ──────────────────────────────────────────────────────────────────
api:
	@echo "Starting FastAPI on port $(PORT_API) …"
	$(PYTHON) -m uvicorn src.api.app:app \
		--host 0.0.0.0 \
		--port $(PORT_API) \
		--reload \
		--log-level info

dashboard:
	@echo "Starting Streamlit dashboard on port $(PORT_DASH) …"
	streamlit run dashboard/streamlit_app.py \
		--server.port $(PORT_DASH) \
		--server.address 0.0.0.0

mlflow-server:
	@echo "Starting MLflow tracking server …"
	mlflow server \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri ./mlruns

# ── Kafka Streaming ───────────────────────────────────────────────────────────
stream-produce:
	@echo "Starting Kafka producer (10 TPS) …"
	$(PYTHON) -m streaming.kafka_producer --tps 10 --config $(CONFIG)

stream-consume:
	@echo "Starting Kafka consumer …"
	$(PYTHON) -m streaming.kafka_consumer --config $(CONFIG)

stream-sim:
	@echo "Running streaming consumer in simulation mode (no Kafka needed) …"
	FRAUD_API_URL=http://localhost:$(PORT_API) $(PYTHON) -m streaming.kafka_consumer --config $(CONFIG)

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up -d
	@echo ""
	@echo "Services started:"
	@echo "  Fraud API      → http://localhost:8000"
	@echo "  API Docs       → http://localhost:8000/docs"
	@echo "  Dashboard      → http://localhost:8501"
	@echo "  MLflow         → http://localhost:5000"
	@echo "  Kafka          → localhost:9092"

docker-down:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f fraud-api fraud-dashboard

docker-restart-api:
	$(COMPOSE) restart fraud-api

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	@echo "Running tests …"
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	@echo "Running fast tests (skip slow integration tests) …"
	$(PYTHON) -m pytest tests/ -v -m "not slow"

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	@echo "Running linting …"
	$(PYTHON) -m ruff check src/ streaming/ dashboard/ scripts/ || true
	$(PYTHON) -m mypy src/ --ignore-missing-imports || true

format:
	$(PYTHON) -m ruff format src/ streaming/ dashboard/ scripts/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning generated files …"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done."

clean-models:
	@echo "Removing trained models …"
	rm -rf data/models/*.pkl data/models/*.yaml data/models/*.csv
	@echo "Done."

clean-processed:
	@echo "Removing processed data …"
	rm -rf data/processed/*.parquet data/processed/*.pkl
	@echo "Done."

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Graph-Enhanced Fraud Detection System"
	@echo "────────────────────────────────────────────────────────"
	@echo "  make install       Install Python dependencies"
	@echo "  make preprocess    Preprocess raw IEEE-CIS data"
	@echo "  make train         Run full training pipeline"
	@echo "  make train-fast    Train without graph (faster)"
	@echo "  make evaluate      Evaluate trained models"
	@echo "  make api           Start FastAPI server (port 8000)"
	@echo "  make dashboard     Start Streamlit dashboard (port 8501)"
	@echo "  make mlflow-server Start MLflow tracking server (port 5000)"
	@echo "  make stream-sim    Simulate live transaction stream"
	@echo "  make docker-up     Start all services via Docker Compose"
	@echo "  make docker-down   Stop Docker services"
	@echo "  make test          Run unit tests with coverage"
	@echo "  make lint          Run linters"
	@echo "  make clean         Remove temporary files"
	@echo ""
