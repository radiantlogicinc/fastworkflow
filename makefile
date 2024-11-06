SHELL = /bin/bash

.EXPORT_ALL_VARIABLES:

.PHONY: gen-env lint publish-testpypi publish

gen-env:
	chmod +x ./gen-env.sh
	./gen-env.sh
    include ./.env

lint:
	py3clean .
	isort .
	black .
	flake8 . --ignore E501,E122,W503,E402,F401
	pylint --recursive=y .
	mypy --install-types --non-interactive .
	mypy .
	bandit -c pyproject.toml -r .

publish-testpypi: gen-env
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry config pypi-token.testpypi $(TESTPYPI_ACCESS_TOKEN)
	poetry build
	poetry publish --repository testpypi

publish: gen-env
	poetry config pypi-token.pypi $(PYPI_ACCESS_TOKEN)
	poetry build
	poetry publish
