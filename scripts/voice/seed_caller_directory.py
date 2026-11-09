"""Build the phone-caller directory for keypad verification (apps/voice/caller_verification.py).

    CALLER_VERIFY_HMAC_KEY=$(openssl rand -hex 32) \
    python scripts/voice/seed_caller_directory.py --out /srv/saral/voice/caller_directory.json

Every applicant in data/synthetic/applicants.json gets a synthetic phone number (+91 0000 ......, never a real
mobile) and a synthetic last-4 of its loan application number, from a fixed RNG seed so the demo is repeatable, plus
the language its call opens in (call flow v2). Languages are dealt from DEMO_LANGUAGES in order (en-IN, hi-IN, te-IN,
...), so a demo covers several; --language APPLICANT_ID=CODE overrides one (repeatable). The output holds
HMAC-SHA256 hashes (per-record salt, key from CALLER_VERIFY_HMAC_KEY) and the plaintext language code only: no phone
number and no digit. Point the worker at it with CALLER_DIRECTORY and the SAME key.

--show-demo-values prints the synthetic phone numbers, digits and languages to stdout for whoever runs the demo. It is
opt-in and never writes them to a file or a log.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "voice"))
from caller_verification import KEY_ENV, build_seed, load_key  # noqa: E402
from prompt_texts import LANGUAGES  # noqa: E402
from synthetic_data.saral_kyc_bank.ids import synthetic_phone  # noqa: E402

APPLICANTS = ROOT / "data" / "synthetic" / "applicants.json"


DEMO_LANGUAGES = LANGUAGES  # dealt in order: en-IN, hi-IN, te-IN, ta-IN, ...


def demo_entries(seed: int = 20261106, applicants_path: Path = APPLICANTS, languages: dict | None = None) -> list[dict]:
    rng = random.Random(seed)
    cases = json.loads(applicants_path.read_text(encoding="utf-8"))["cases"]
    languages = languages or {}
    return [{"phone": synthetic_phone(rng), "applicant_id": c["applicant_id"], "application_id": c["application_id"],
             "application_last4": f"{rng.randint(0, 9999):04d}",
             "language": languages.get(c["applicant_id"], DEMO_LANGUAGES[i % len(DEMO_LANGUAGES)])}
            for i, c in enumerate(cases)]


def _language_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs or []:
        applicant, _, code = pair.partition("=")
        if not applicant or code not in LANGUAGES:
            raise ValueError(f"--language wants APPLICANT_ID=CODE with CODE one of {', '.join(LANGUAGES)}")
        out[applicant] = code
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="where to write the hashed directory (JSON)")
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--language", action="append", default=[], metavar="APPLICANT_ID=CODE",
                    help="the language one applicant's call opens in (repeatable)")
    ap.add_argument("--show-demo-values", action="store_true", help="print synthetic phones and digits to stdout")
    a = ap.parse_args(argv)
    key = load_key(os.environ)
    if key is None:
        print(f"{KEY_ENV} is missing or shorter than 16 bytes", file=sys.stderr)
        return 2
    try:
        overrides = _language_overrides(a.language)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    entries = demo_entries(a.seed, languages=overrides)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_seed(key, entries), indent=1) + "\n", encoding="utf-8")
    os.chmod(out, 0o600)
    print(f"wrote {len(entries)} hashed caller records to {out}")
    if a.show_demo_values:
        for e in entries:
            print(f"{e['applicant_id']}  {e['phone']}  application ..{e['application_last4']}  "
                  f"language {e['language']}  (SYNTHETIC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
