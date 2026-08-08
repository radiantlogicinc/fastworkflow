# Security HTML reports

Tracked outputs from the multi-scanner CVE automation:

- `cve-report.html` — consolidated scanner status and findings
- `human-review.html` — FIX/IGNORE recommendations for items that need code or process changes

Machine-readable JSON/SBOM artifacts are written to gitignored `security-reports/` locally via `make security-scan`.
OpenVEX acceptances live in `security/vex/fastworkflow.openvex.json`.
