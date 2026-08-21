# Security reports

Tracked HTML summaries from the multi-scanner CVE automation:

- `cve-report.html` — executive scan status across Trivy, Grype, Syft, osv-scanner, Dockle, Dive (Snyk/Scout when authenticated)
- `human-review.html` — findings that need FIX vs IGNORE decisions with rationale

Machine-readable raw outputs live under gitignored `security-reports/` at the repo root.
OpenVEX statements: `security/vex/fastworkflow.openvex.json`.
