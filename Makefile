.PHONY: install test lint typecheck train predict backtest fmb clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/stockgpt --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/ scripts/

lint-fix:
	ruff check --fix src/ tests/ scripts/

typecheck:
	mypy src/stockgpt scripts/

format:
	ruff format src/ tests/ scripts/

quality: lint typecheck test

prepare:
	python scripts/prepare_crsp.py

train:
	python scripts/train_stockgpt.py

predict:
	python scripts/predict.py

backtest:
	python scripts/backtest_daily.py

fmb:
	python scripts/eval_fmb.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
