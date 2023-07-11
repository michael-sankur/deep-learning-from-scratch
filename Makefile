
PROJECT = neuralnetwork
PYTHON_VERSION = 3.11

PYTEST_ARGS = -x -p no:warnings
PYTEST_COVERAGE = --cov-report term-missing --cov=${PROJECT}

.PHONY: test

test:
	## PYTHONPATH =. pytest
	## pytest ${PYTEST_COVERAGE} ${PYTEST_ARGS}
	pytest -v -x tests/test_NN.py

clean:
	rm -rf .pytest_cache
	rm -rf .tests/pytest_cache
	rm -rf .__pycache__/
	rm -rf ./tests/__pycache__/
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete