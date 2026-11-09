"""/borrower/call: the web phone. A phone-call screen in the browser that runs the same flow as dialling Saral
(api/web_phone.py has the security notes). The page only shows the masked caller number; the token endpoint picks
the number again on the server when the call starts."""

from pathlib import Path

from loan_advisory_portal.api import web_phone
from loan_advisory_portal.portal.errors import PortalError, ServiceUnavailable
from loan_advisory_portal.portal.pages import borrower_page, is_guest, public_page
from loan_advisory_portal.services import livekit_token
from loan_advisory_portal.www.borrower.assistant import _call_saral

no_cache = 1
base_template_path = "templates/portal/base.html"

# The portal already ships livekit-client inside the assistant island's build (a self-contained ES module chunk).
# The web phone imports that same file, so there is one copy of the SDK and nothing is fetched from a CDN.
CHUNKS = Path(__file__).resolve().parents[2] / "public" / "js" / "assistant" / "chunks"
ASSET_BASE = "/assets/loan_advisory_portal/js/assistant/chunks/"


def livekit_module() -> str | None:
	found = sorted(CHUNKS.glob("livekit-client.esm-*.js"))
	return ASSET_BASE + found[0].name if found else None


def _configured() -> bool:
	try:
		livekit_token.configuration()
	except ServiceUnavailable:
		return False
	return True


def get_context(context):
	if is_guest() and web_phone.demo_number():
		context = public_page(context, "assistant", "Call Saral")
		number, demo = web_phone.demo_number(), True
	else:
		context = borrower_page(context, "assistant", "Call Saral")
		if context.get("blocked"):
			return context
		demo = False
		try:
			number = web_phone.registered_number(context["borrower"])
		except PortalError:
			number = None
	reason = None
	if not number:
		reason = "no_number"
	elif not _configured() or not livekit_module():
		reason = "unavailable"
	context["web_phone"] = {
		"available": reason is None,
		"reason": reason,
		"caller": web_phone.masked(number),
		"demo": demo,
		"livekit": livekit_module(),
	}
	context["call_saral"] = _call_saral()
	return context
