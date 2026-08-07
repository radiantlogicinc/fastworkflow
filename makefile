SHELL = /bin/bash

.EXPORT_ALL_VARIABLES:

.PHONY: gen-env lint audit audit-json publish-testpypi publish install-gpu-torch

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

# Explicit CUDA torch opt-in. Default `poetry install` pins CPU-only wheels
# (see pyproject.toml `pytorch-cpu` source). Poetry cannot lock torch from
# both the CPU and CUDA indexes at once, so GPU users run:
#   poetry install --with gpu && make install-gpu-torch
# Verifies CUDA is usable before returning success. Constrains fsspec after
# the torch reinstall so the optional `datasets` (training) extra keeps working.
install-gpu-torch:
	source .venv/bin/activate && \
	pip install --upgrade --force-reinstall 'torch>=2.7.1' \
		--index-url https://download.pytorch.org/whl/cu130 && \
	pip install --upgrade 'fsspec[http]>=2023.1.0,<=2026.2.0' && \
	python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch installed but torch.cuda.is_available() is False'; print('GPU torch OK:', torch.__version__, torch.cuda.get_device_name(0))"

# Local dependency vulnerability scan (the same advisory data Dependabot uses).
# Audits the FULL locked dependency graph (all groups + extras) exported from
# poetry.lock, so results mirror the GitHub default-branch Dependabot alerts.
# Requires the local virtualenv (.venv) to be present.
audit:
	source .venv/bin/activate && \
	pip install --quiet pip-audit && \
	poetry export --without-hashes --with dev,test,aws --all-extras \
		-f requirements.txt -o .audit-requirements.txt && \
	pip-audit --requirement .audit-requirements.txt --desc; \
	status=$$?; rm -f .audit-requirements.txt; exit $$status

# Same scan, emitting machine-readable JSON to audit-report.json for tooling.
audit-json:
	source .venv/bin/activate && \
	pip install --quiet pip-audit && \
	poetry export --without-hashes --with dev,test,aws --all-extras \
		-f requirements.txt -o .audit-requirements.txt && \
	pip-audit --requirement .audit-requirements.txt --format json \
		--output audit-report.json; \
	status=$$?; rm -f .audit-requirements.txt; exit $$status

publish-testpypi: gen-env
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry config pypi-token.testpypi $(TESTPYPI_ACCESS_TOKEN)
	poetry build
	poetry publish --repository testpypi

publish: gen-env
	poetry config pypi-token.pypi $(PYPI_ACCESS_TOKEN)
	poetry build
	poetry publish
