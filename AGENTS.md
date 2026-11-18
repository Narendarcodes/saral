# AGENTS.md

Guidance for coding agents (and people) working in this repository. Gnani is a standalone voice-agent project. Keep Gnani's product identity, core logic, data, configuration and documentation self-contained; do not import or describe unrelated projects, organizations, teams or coursework as part of Gnani. Treat Gnani-specific source, tests and documentation as the source of truth.

## Rules

- **Synthetic data only.** Use synthetic data in tests and examples. Never add real personal details, customer records, call recordings or transcripts to the repository.
- **No secrets in Git.** Never commit `.env`, API keys, credentials, local runtime state or caller-directory exports. Load required keys, including `GNANI_API_KEY`, from the environment at runtime; never write secret values to logs.
- **Voice privacy.** Keypad digits and sensitive caller information must never reach the LLM, transcripts or logs. Treat caller-ID and demo verification as demonstration-only, not production authentication.
- **Standalone boundary.** Gnani's core path must run and be documented without requiring another product's code, data, configuration or infrastructure. Keep optional integrations isolated and do not make them part of Gnani's identity.
- **Commits.** The pre-commit hook in `.githooks/` runs ruff and the test suite. For docs-only commits, or on a low-memory machine, `git commit --no-verify` is acceptable if the applicable CI checks are green before merge. Never force-push `main`.

## Repository map

| Path | What it is |
|---|---|
| `apps/voice/` | Gnani voice implementation and worker code |
| `scripts/voice/` | Gnani voice utilities and round-trip harness |
| `tests/` | Python test suite, with JS and browser tests under their own subdirectories |
| `docs/` | Project documentation; keep Gnani-specific material self-contained |

## Run

Python 3.14. Use an isolated virtual environment and provide `GNANI_API_KEY` through the shell environment.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r apps/voice/requirements.txt
# Gnani voice round-trip, without requiring an external product stack
python scripts/voice/gnani_roundtrip.py --out /tmp/gnani_rt
```

## Test

```bash
python -m ruff check .          # lint (rules pinned in pyproject.toml)
python -m pytest tests/ -q      # project test suite
```

The full suite can be heavy on a small machine: run relevant individual test files while iterating, then the complete suite before a PR. CI is configured in `.github/workflows/ci.yml`; verify its required checks before merging.

## Pull requests

Branch from `main`, keep one topic per PR, and describe what changed and how it was checked. Merge only after the required CI checks are green. Do not claim an unmeasured number (latency, accuracy, coverage); mark it as not yet measured.
