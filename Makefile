PYTHON ?= python3
CONFIG ?= config.yaml

.PHONY: install test lint run evaluate clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q src tests

run:
	$(PYTHON) -m src.fraud.train --config $(CONFIG)

evaluate:
	$(PYTHON) -m src.fraud.evaluate --config $(CONFIG) --model artifacts/model.pkl

clean:
	rm -rf data artifacts .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
