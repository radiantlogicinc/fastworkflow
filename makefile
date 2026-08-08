SHELL = /bin/bash

.EXPORT_ALL_VARIABLES:

.PHONY: gen-env lint audit audit-json publish-testpypi publish install-gpu-torch \
	security-pip-audit security-syft security-grype security-trivy \
	security-snyk security-scout security-scout-image security-dockle security-dive \
	security-scan security-scan-deps security-scan-image security-tools-check

# Directory for machine-readable CVE / SBOM outputs (gitignored).
SECURITY_REPORT_DIR ?= security-reports
# OpenVEX document for accepted / not-affected findings (tracked in git).
VEX_FILE ?= security/vex/fastworkflow.openvex.json
# Optional container image ref for image-layer scanners (dive/dockle/scout image mode).
# Example: make security-scan-image IMAGE=myrepo/fastworkflow:dev
IMAGE ?=
# Optional Snyk org / severity gate.
SNYK_SEVERITY_THRESHOLD ?= medium

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

# ---------------------------------------------------------------------------
# Local CVE / SBOM scanners
#
# Prefer lockfile / SBOM scans over `dir:.` — scanning the working tree picks up
# `.venv` and other local pollution that is NOT what Dependabot / PyPI users see.
#
# Tools that run without a container image (dependency graph):
#   pip-audit, syft, grype, trivy, snyk (needs SNYK_TOKEN), docker scout fs://
# Tools that need IMAGE=... (image / Dockerfile consumers):
#   dive, dockle, docker scout image://
#
# Install hints (tools not vendored by this repo):
#   trivy:  curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b ~/.local/bin
#   grype/syft: https://github.com/anchore/{grype,syft}/releases
#   dockle: https://github.com/goodwithtech/dockle/releases
#   dive:   https://github.com/wagoodman/dive/releases (or apt)
#   snyk:   npm i -g snyk && snyk auth
#   scout:  docker scout (Docker Desktop / docker-scout plugin)
# ---------------------------------------------------------------------------

$(SECURITY_REPORT_DIR):
	mkdir -p $(SECURITY_REPORT_DIR)

# Export the same locked graph Dependabot / make audit use.
$(SECURITY_REPORT_DIR)/requirements.txt: $(SECURITY_REPORT_DIR) poetry.lock pyproject.toml
	poetry export --without-hashes --with dev,test,aws --all-extras \
		-f requirements.txt -o $(SECURITY_REPORT_DIR)/requirements.txt

security-tools-check:
	@echo "=== Local scanner availability ==="
	@for t in pip-audit poetry syft grype trivy snyk dockle dive docker; do \
		if command -v $$t >/dev/null 2>&1; then echo "OK   $$t: $$(command -v $$t)"; \
		else echo "MISS $$t"; fi; \
	done
	@if docker scout version >/dev/null 2>&1; then echo "OK   docker scout"; \
	else echo "MISS docker scout (plugin)"; fi
	@echo "IMAGE=$(if $(IMAGE),$(IMAGE),<unset — required for dive/dockle/scout-image>)"

# pip-audit (OSV / PyPI advisories) — same data source family as Dependabot.
security-pip-audit: $(SECURITY_REPORT_DIR) $(SECURITY_REPORT_DIR)/requirements.txt
	source .venv/bin/activate && \
	pip install --quiet pip-audit && \
	pip-audit --requirement $(SECURITY_REPORT_DIR)/requirements.txt --desc \
		| tee $(SECURITY_REPORT_DIR)/pip-audit.txt; \
	status=$$?; \
	pip-audit --requirement $(SECURITY_REPORT_DIR)/requirements.txt \
		--format json --output $(SECURITY_REPORT_DIR)/pip-audit.json; \
	cp $(SECURITY_REPORT_DIR)/pip-audit.json audit-report.json; \
	exit $$status

# Anchore Syft — SBOM from poetry.lock (no .venv pollution).
security-syft: $(SECURITY_REPORT_DIR)
	@command -v syft >/dev/null 2>&1 || { echo "syft not installed; see makefile header"; exit 1; }
	syft poetry.lock \
		-o json=$(SECURITY_REPORT_DIR)/syft-poetry.json \
		-o table=$(SECURITY_REPORT_DIR)/syft-poetry.txt \
		-o spdx-json=$(SECURITY_REPORT_DIR)/syft-poetry.spdx.json
	@echo "Wrote $(SECURITY_REPORT_DIR)/syft-poetry.{json,txt,spdx.json}"

# Anchore Grype — vulns against the Syft SBOM (regenerates SBOM if missing).
security-grype: security-syft
	@command -v grype >/dev/null 2>&1 || { echo "grype not installed; see makefile header"; exit 1; }
	grype sbom:$(SECURITY_REPORT_DIR)/syft-poetry.json \
		--vex $(VEX_FILE) \
		-o table | tee $(SECURITY_REPORT_DIR)/grype-poetry.txt
	grype sbom:$(SECURITY_REPORT_DIR)/syft-poetry.json \
		--vex $(VEX_FILE) \
		-o json --file $(SECURITY_REPORT_DIR)/grype-poetry.json
	@echo "Wrote $(SECURITY_REPORT_DIR)/grype-poetry.{txt,json} (VEX=$(VEX_FILE))"

# Aqua Trivy — filesystem / lockfile scan (excludes .venv via skip-dirs).
security-trivy: $(SECURITY_REPORT_DIR)
	@command -v trivy >/dev/null 2>&1 || { echo "trivy not installed; see makefile header"; exit 1; }
	trivy fs --scanners vuln \
		--skip-dirs .venv,.git,dist,.mypy_cache,.pytest_cache,.ruff_cache \
		--vex $(VEX_FILE) \
		--format table \
		--output $(SECURITY_REPORT_DIR)/trivy-fs.txt .
	trivy fs --scanners vuln \
		--skip-dirs .venv,.git,dist,.mypy_cache,.pytest_cache,.ruff_cache \
		--vex $(VEX_FILE) \
		--format json \
		--output $(SECURITY_REPORT_DIR)/trivy-fs.json .
	trivy fs --scanners vuln --vex $(VEX_FILE) --format table \
		--output $(SECURITY_REPORT_DIR)/trivy-poetry.txt poetry.lock
	@echo "Wrote $(SECURITY_REPORT_DIR)/trivy-{fs,poetry}.{txt,json} (VEX=$(VEX_FILE))"

# Snyk — optional; requires `snyk auth` / SNYK_TOKEN.
security-snyk: $(SECURITY_REPORT_DIR) $(SECURITY_REPORT_DIR)/requirements.txt
	@command -v snyk >/dev/null 2>&1 || { \
		echo "snyk not installed (npm i -g snyk && snyk auth). Skipping."; exit 0; }; \
	if [ -z "$$SNYK_TOKEN" ] && [ ! -f "$$HOME/.config/configstore/snyk.json" ]; then \
		echo "snyk not authenticated (export SNYK_TOKEN or run snyk auth). Skipping."; exit 0; \
	fi; \
	snyk test --file=$(SECURITY_REPORT_DIR)/requirements.txt --package-manager=pip \
		--severity-threshold=$(SNYK_SEVERITY_THRESHOLD) \
		--json-file-output=$(SECURITY_REPORT_DIR)/snyk.json \
		| tee $(SECURITY_REPORT_DIR)/snyk.txt || true; \
	echo "Wrote $(SECURITY_REPORT_DIR)/snyk.{txt,json} (non-zero exit from snyk is expected when vulns exist)"

# Docker Scout — filesystem mode against the exported requirements / project root.
security-scout: $(SECURITY_REPORT_DIR) $(SECURITY_REPORT_DIR)/requirements.txt
	@if ! docker scout version >/dev/null 2>&1; then \
		echo "docker scout plugin not available. Skipping."; exit 0; \
	fi; \
	docker scout cves fs://$(SECURITY_REPORT_DIR)/requirements.txt \
		--format sarif --output $(SECURITY_REPORT_DIR)/scout-reqs.sarif || true; \
	docker scout cves fs://. \
		--only-package-types python \
		--format sarif --output $(SECURITY_REPORT_DIR)/scout-fs.sarif || true; \
	echo "Wrote $(SECURITY_REPORT_DIR)/scout-*.sarif"

# Image-only scanners -------------------------------------------------------

security-dockle:
	@command -v dockle >/dev/null 2>&1 || { echo "dockle not installed; see makefile header"; exit 1; }; \
	if [ -z "$(IMAGE)" ]; then \
		echo "No Dockerfile/image in this repo by default. Set IMAGE=... to scan a built image."; \
		echo "Example: make security-dockle IMAGE=fastworkflow:local"; \
		exit 0; \
	fi; \
	mkdir -p $(SECURITY_REPORT_DIR); \
	dockle --exit-code 0 --format json --output $(SECURITY_REPORT_DIR)/dockle.json $(IMAGE); \
	dockle --exit-code 0 $(IMAGE) | tee $(SECURITY_REPORT_DIR)/dockle.txt

security-dive:
	@command -v dive >/dev/null 2>&1 || { echo "dive not installed; see makefile header"; exit 1; }; \
	if [ -z "$(IMAGE)" ]; then \
		echo "dive needs a built image. Set IMAGE=... (CI=true for non-interactive)."; \
		echo "Example: CI=true make security-dive IMAGE=fastworkflow:local"; \
		exit 0; \
	fi; \
	mkdir -p $(SECURITY_REPORT_DIR); \
	CI=true dive $(IMAGE) 2>&1 | tee $(SECURITY_REPORT_DIR)/dive.txt

# Scout against a built image (in addition to fs:// mode above).
security-scout-image:
	@if ! docker scout version >/dev/null 2>&1; then \
		echo "docker scout plugin not available. Skipping."; exit 0; \
	fi; \
	if [ -z "$(IMAGE)" ]; then \
		echo "Set IMAGE=... for image scout. Example: make security-scout-image IMAGE=fastworkflow:local"; \
		exit 0; \
	fi; \
	mkdir -p $(SECURITY_REPORT_DIR); \
	docker scout cves image://$(IMAGE) \
		--format sarif --output $(SECURITY_REPORT_DIR)/scout-image.sarif || true

# Aggregate dependency-graph scanners (no container image required).
security-scan-deps: security-tools-check security-pip-audit security-syft security-grype security-trivy security-snyk security-scout
	@echo ""
	@echo "Dependency scan reports under $(SECURITY_REPORT_DIR)/"
	@ls -la $(SECURITY_REPORT_DIR) | sed 's/^/  /'

# Aggregate image scanners (no-op with guidance when IMAGE unset).
security-scan-image: security-dockle security-dive security-scout-image
	@echo "Image scan complete (IMAGE=$(if $(IMAGE),$(IMAGE),unset))."

# Full local security report suite.
security-scan: security-scan-deps security-scan-image
	@echo ""
	@echo "All requested local scanners finished. Review $(SECURITY_REPORT_DIR)/"
	@echo "Primary Python CVE sources: pip-audit.json, grype-poetry.json, trivy-fs.json"
