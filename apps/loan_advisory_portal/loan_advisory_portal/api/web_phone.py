"""Web phone: the phone-call flow in the browser (/borrower/call).

The demo cannot always use the real inbound number (+1 484 295 0295: no free Indian number, and calling the US costs
the borrower), so the portal offers a phone-style page that joins LiveKit over WebRTC and runs the SAME flow as a SIP
call: caller ID -> greeting in the directory language -> press 9 for the language menu -> last 4 digits of the loan
application number -> verified -> the live agent in the chosen language.

How it stays a phone call and not voice mode:
- The room is new for every call and named `webcall-<random>` (voice mode rooms are `loan-...`).
- The token carries LiveKit participant attributes `saral.mode = web_phone` and `sip.phoneNumber = <registered
  number>`, and denies `canUpdateOwnMetadata`, so the browser can neither choose nor change them. The caller ID comes
  ONLY from here: the borrower's own profile number (or, on a demo site, the configured demo number). Nothing the
  browser sends is read as a phone number.
- No `Voice Session` row and no `session_id` metadata are created, so the worker's voice-mode path refuses this
  participant; the worker routes it to the keypad flow (apps/voice/call_routing.py) and the borrower still has to key
  the last 4 digits before anything about an application is said. Phone vouching (`voice_phone_assertions`) must be
  on, exactly as for SIP calls.
- The worker is dispatched the same way as for voice mode (automatic dispatch to new rooms; no agent name).

Limits: 10 calls an hour and one call at a time per account (per address for the demo caller). The number is never
returned to the browser in full and never logged.
"""

import hashlib
import re
import secrets
import uuid

import frappe
from frappe import _

from loan_advisory_portal.portal import limits
from loan_advisory_portal.portal.errors import Conflict, NotFound, SignInRequired
from loan_advisory_portal.portal.identity import borrower_context
from loan_advisory_portal.services import livekit_token

ROOM_PREFIX = "webcall-"
IDENTITY_PREFIX = "webphone-"
MODE_ATTRIBUTE = "saral.mode"
MODE = "web_phone"
CALLER_ID_ATTRIBUTE = "sip.phoneNumber"
# Site config: an E.164 number a signed-out visitor's web call presents as its caller ID (a synthetic demo applicant).
# Unset (the default) means the web phone is for signed-in borrowers only.
DEMO_NUMBER_FLAG = "saral_web_phone_demo_number"
# Site config: "dtmf" (default, livekit-client publishDtmf -> the worker's sip_dtmf_received) or "data" (topic "dtmf").
KEYPAD_FLAG = "saral_web_phone_keypad"
KEYPAD_MODES = ("dtmf", "data")
CALLS_PER_HOUR = 10
LOCK_SECONDS = livekit_token.TOKEN_TTL_SECONDS
E164 = re.compile(r"^\+\d{8,15}$")
ROOM = re.compile(r"^webcall-[0-9a-f]{16}$")


def normalize(raw) -> str | None:
	"""E.164 from a stored phone value ('+91 98765 43210', '+91-98765-43210'), or None. A bare 10-digit number is
	taken as Indian, the way the worker reads caller IDs."""
	s = re.sub(r"[\s\-().]", "", str(raw or ""))
	if re.fullmatch(r"\d{10}", s):
		s = "+91" + s
	elif re.fullmatch(r"91\d{10}", s):
		s = "+" + s
	return s if E164.match(s) else None


def masked(number: str | None) -> str | None:
	"""For the page: '+91 ••••• •••10'. The full number never leaves the server."""
	if not number:
		return None
	return f"{number[:3]} ••••• •••{number[-2:]}"


def registered_number(ctx) -> str | None:
	"""The signed-in borrower's own number: Borrower Profile `phone`, else the Customer's `mobile_no`."""
	number = normalize(frappe.db.get_value("Borrower Profile", ctx.profile, "phone"))
	if number:
		return number
	try:
		return normalize(frappe.db.get_value("Customer", ctx.customer, "mobile_no"))
	except Exception:  # a Customer without the field: no fallback number
		return None


def demo_number() -> str | None:
	return normalize(frappe.conf.get(DEMO_NUMBER_FLAG))


def keypad_mode() -> str:
	mode = str(frappe.conf.get(KEYPAD_FLAG) or "dtmf")
	return mode if mode in KEYPAD_MODES else "dtmf"


def _is_guest() -> bool:
	return frappe.session.user in (None, "", "Guest")


def _guest_owner() -> str:
	ip = str(getattr(frappe.local, "request_ip", "") or "unknown")
	return "webcall-guest:" + hashlib.sha256(ip.encode("utf-8")).hexdigest()[:24]


def caller() -> dict:
	"""{owner, number, name, demo} for whoever is asking. Signed-in borrowers use their own number; a signed-out
	visitor gets the demo number only when the site configured one."""
	if not _is_guest():
		ctx = borrower_context()
		name = frappe.db.get_value("Borrower Profile", ctx.profile, "full_name") or ctx.user
		return {"owner": ctx.user, "number": registered_number(ctx), "name": name, "demo": False}
	number = demo_number()
	if not number:
		raise SignInRequired(_("Sign in to call Saral from the browser."))
	return {"owner": _guest_owner(), "number": number, "name": "Web caller", "demo": True}


def _lock_key(owner: str) -> str:
	return frappe.cache.make_key(f"saral-web-call:{owner}")


@frappe.whitelist(methods=["POST"], allow_guest=True)
def start_web_call() -> dict:
	"""Mint a join token for a new `webcall-` room. The caller ID is the server's, never the browser's."""
	who = caller()
	if not who["number"]:
		raise NotFound(_("No phone number on your profile."))
	limits.enforce("web-call", CALLS_PER_HOUR, 3600, identity=who["owner"])
	key = _lock_key(who["owner"])
	if frappe.cache.get(key):
		raise Conflict(_("You already have a call in progress. Hang up first, or try again in a few minutes."))

	config = livekit_token.configuration()
	room = f"{ROOM_PREFIX}{secrets.token_hex(8)}"
	identity = f"{IDENTITY_PREFIX}{uuid.uuid4().hex}"
	token = livekit_token.create_token(
		api_key=config["api_key"],
		api_secret=config["api_secret"],
		identity=identity,
		name=who["name"],
		room=room,
		metadata={},  # no session_id: this participant can never take the voice-mode path
		attributes={MODE_ATTRIBUTE: MODE, CALLER_ID_ATTRIBUTE: who["number"]},
	)
	frappe.cache.setex(key, LOCK_SECONDS, room)
	return {
		"token": token,
		"url": config["url"],
		"room": room,
		"identity": identity,
		"keypad": keypad_mode(),
		"caller": masked(who["number"]),
	}


@frappe.whitelist(methods=["POST"], allow_guest=True)
def end_web_call(room: str | None = None) -> dict:
	"""Release the caller's one-call lock (the worker ends the room itself). Idempotent; another caller's room is
	ignored."""
	try:
		who = caller()
	except SignInRequired:
		return {"ended": False}
	key = _lock_key(who["owner"])
	held = frappe.cache.get(key)
	if isinstance(held, bytes):
		held = held.decode("utf-8", "ignore")
	if held and isinstance(room, str) and ROOM.match(room) and held == room:
		frappe.cache.delete(key)  # the raw key: limits-style keys are already made
		return {"ended": True}
	return {"ended": False}
