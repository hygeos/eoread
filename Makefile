samples:
	python -m tests.test_products

test:
	pytest tests

.PHONY: samples test