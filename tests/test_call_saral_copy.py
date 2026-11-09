"""The portal's Call Saral line matches call flow v2 (press 9 for language, then the last 4 of the application), opens
the web phone (/borrower/call) and still shows the real number to dial."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "apps" / "loan_advisory_portal" / "loan_advisory_portal" / "www" / "borrower" / "assistant.html"


def test_call_saral_copy_matches_the_phone_flow():
    html = PAGE.read_text(encoding="utf-8")
    assert '<a class="callsaral__link" href="/borrower/call">Call Saral</a>' in html
    assert 'or dial <a href="tel:{{ call_saral.tel }}">{{ call_saral.display }}</a>' in html
    assert "Press 9 to change language, then key the last 4 digits of your application number." in html
    assert "bank account" not in html and "Press 1 for your application" not in html
