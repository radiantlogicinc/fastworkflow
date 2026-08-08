# Security scan reports

Generated 2026-08-08 18:58 UTC on branch `cursor/cve-report-and-remediation-6925`.

| Report | Description |
|--------|-------------|
| [cve-report.html](cve-report.html) | Multi-scanner CVE summary (Trivy, Grype, Syft, Snyk, Scout, Dockle, Dive, osv-scanner) |
| [human-review.html](human-review.html) | Findings needing code/process changes with FIX or IGNORE recommendations |

Machine-readable scanner outputs remain local/gitignored under `security-reports/` (see makefile `security-scan` targets). Accepted Python finding: `diskcache` CVE-2025-69872 via [`../vex/fastworkflow.openvex.json`](../vex/fastworkflow.openvex.json).
