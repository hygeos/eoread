.PHONY: samples tests

samples:
	python -m tests.test_products

tests:
	pytest tests
