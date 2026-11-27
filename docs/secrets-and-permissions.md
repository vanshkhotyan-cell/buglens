# GitHub Secrets & Permissions Inventory

- GITHUB_TOKEN: Built-in workflow token (contents: read in CI; contents: write, pull-requests: write in Release Please; packages: write, security-events: write in GHCR release).
- ENABLE_PYPI_PUBLISH (Repository Variable): Set to 	rue once PyPI Trusted Publishing (pypi environment) is configured.
- Local Secrets (Never Committed): .env and .dvc/config.local hold personal DagsHub tokens.
