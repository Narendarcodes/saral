# AGENTS.md

Guide for coding agents (and people) working in this repository. Saral is a loan advisory agent for Indian
personal-loan borrowers: it **explains** a decision that the ZEN rules engine already made, in English, Hindi or
Telugu. It never decides. Product overview: [README.md](README.md). Design reasoning: [APPROACH.md](APPROACH.md).

## Rules

- **Synthetic data only.** Never add a real person, PAN, Aadhaar number, phone number, employer, bank account or
  email. Synthetic PANs start `ZZ`, Aadhaar numbers start with 0 or 1, emails use `example.invalid`. Generate data
  with `synthetic_data/` and seed it with `ops/saral_seed/`.
- **No secrets in git.** Never commit `.env`, API keys, Frappe `site_config.json`, `creds.json`, `mapping.json`,
  `storage.sqlite`, trace exports, seeded login files or caller directories. `.env.example` lists variable names
  only. Keys (LLM gateway, Groq, Gnani, LiveKit, Frappe, HMAC secrets) come from the environment at run time.
- **ZEN decides, the LLM explains.** Do not let the agent compute, change or override a decision, show a cut-off or a
  rule number, or give a borrower an internal reason. Fraud/list hits go to a person, never to an automatic decline.
- **Copyrighted policy texts stay out of the repo.** The policy index is built under `$LOAN_POLICY_DIR`, outside git.
- **Phone verification is demo-grade.** Caller ID plus the last 4 digits of one factor is a demonstration, not
  production authentication. Keypad digits must never reach the LLM, transcripts or logs.
- **Own work only.** This repo holds the owner's build of a former team capstone; do not re-add teammates' modules
  (document OCR, customer segmentation, the functional write-up).
- **Commits.** The pre-commit hook in `.githooks/` runs ruff and the whole test suite. For docs-only commits, or on a
  low-memory machine, `git commit --no-verify` is acceptable as long as CI (`ruff + pytest`) is green before merge.
  Never force-push `main`.

## Repository map

| Path | What it is |
|---|---|
| `apps/loan_advisory_portal/` | Frappe app: Saral borrower portal, doctypes, assertion minting (`api/`), Loan Officer desk |
| `apps/api/src/loan_advisory/agent/` | Advisory agent: LangGraph graph, guards, tools, prompts, `server.py` (standard-library HTTP) |
| `apps/bureau/` | Synthetic bureau client code |
| `apps/runtime/` | AG-UI / CopilotKit runtime (Node, TypeScript) between the chat island and the agent |
| `apps/web/assistant/` | CopilotKit chat island, built into the portal app |
| `apps/web/chat/` | Standalone chat and voice test pages |
| `apps/voice/` | LiveKit Agents voice worker and Frappe assertion exchange. On `feat/gnani-voice`: Gnani Prisma/Timbre speech (`gnani_speech.py`) and keypad caller verification (`caller_verification.py`) |
| `decision_engine/` | ZEN rules, catalogue, evidence and decision code |
| `contracts/` | Shared data contracts, schemas and field maps |
| `config/` | Product, lender, what-if and RAG configuration |
| `rag/` | Policy retrieval (dense read, routing, multi-query) |
| `ops/` | Decision runner, synthetic bureau server, synthetic KYC + AA bank API, seeding |
| `synthetic_data/` | Synthetic applicant generator |
| `data/` | Evaluation sets and synthetic fixtures (no real data) |
| `scripts/` | Policy corpus build, agent eval, golden sets, voice harnesses |
| `infra/` | Frappe compose and bootstrap, Langfuse, proxy, voice side stack |
| `tests/` | pytest suite (`tests/`), JS tests (`tests/js/`), browser e2e (`tests/e2e/`) |
| `docs/` | Architecture, development, deployment, testing, policy and evaluation docs ([index](docs/README.md)) |

## Run

Python 3.14. Export `.env` into each shell (`set -a; . ./.env; set +a`); no service reads it by itself.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-test.txt
# Agent alone, no Docker or Frappe: built-in synthetic records, no policy search
PYTHONPATH=.:apps/api/src python -m loan_advisory.agent.server --offline --no-rag --port 8095
curl http://127.0.0.1:8095/healthz
```

Full stack (Frappe in Docker, synthetic services, decision runner, agent, runtime, chat island): README
"Run it locally" and [docs/development/setup.md](docs/development/setup.md). Variables:
[docs/development/environment-variables.md](docs/development/environment-variables.md).

Voice (on `feat/gnani-voice`): `pip install -r apps/voice/requirements.txt`, set `GNANI_API_KEY`, then
`python scripts/voice/gnani_roundtrip.py --out /tmp/gnani_rt`. Phone verification needs `CALLER_VERIFY_HMAC_KEY` and a
directory built with `scripts/voice/seed_caller_directory.py`. Details:
[docs/architecture/livekit-integration.md](docs/architecture/livekit-integration.md).

## Test

```bash
python -m ruff check .          # lint (rules pinned in pyproject.toml)
python -m pytest tests/ -q      # what CI runs; live-service tests skip without their services
python -m pytest tests/test_agent_guards.py -q   # one file while iterating
cd apps/runtime && npm ci && npm test           # runtime (Node)
```

The full suite is heavy on a small machine: run single files while iterating and the whole suite once before a PR.
CI is `.github/workflows/ci.yml`, check name **`ruff + pytest`**. Evaluation method and results: README
"Evaluation setup and results" and [docs/saral/FINAL_EVAL_20261104.md](docs/saral/FINAL_EVAL_20261104.md).

## Pull requests

Branch from `main`, keep one topic per PR, write what changed and how it was checked, and merge with a merge commit
after `ruff + pytest` is green. Do not claim a number (latency, accuracy, coverage) that was not measured; mark it as
not yet measured instead.
