"""Voice bridge: a LiveKit session becomes an assertion.

Frappe mints the LiveKit token and writes the `Voice Session` row before the
browser joins, so the row is the authorisation. The voice worker authenticates as
an integration account (`Agent Integration` role), presents the session id with the
room and participant identity it is actually serving, and receives a fresh
60-second assertion for that borrower and application.

The worker never holds a borrower cookie or a long-lived agent credential, and the
browser never receives an assertion: participant metadata carries identifiers only.

Phone (SIP) callers have no `Voice Session` row: the worker matches the caller ID to
an applicant and the caller proves one factor on the keypad. Only then does it call
`phone_assertion` with the applicant and application that record names. Frappe cannot
see the keypad, so the integration account vouches for the verification; Frappe
re-checks that phone vouching is switched on for the site, that the application
exists and belongs to that applicant, and that the borrower account is not closed.
The assertion has the browser path's shape and names that one applicant and
application. No phone number or digit is ever sent here, and nothing is logged.
"""

import hashlib
import re

import frappe
from frappe import _
from frappe.utils import now_datetime

from loan_advisory_portal.install import AGENT_ROLE
from loan_advisory_portal.portal import limits
from loan_advisory_portal.portal.errors import Conflict, Forbidden, NotFound
from loan_advisory_portal.services import assertion, livekit_token

SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
TERMINAL = ("Ended", "Expired")

# Phone path. Record ids are Frappe document names (Customer, Loan Application); the room and
# identity come from LiveKit. All are refused, not cleaned, when they fall outside these shapes.
# A SIP participant identity can embed the caller's number (LiveKit's default is `sip_<number>`),
# so it is only ever compared and hashed here: never stored, returned or logged.
PHONE_FLAG = "voice_phone_assertions"
RECORD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,139}$")
LIVEKIT_NAME = re.compile(r"^[A-Za-z0-9_+][A-Za-z0-9_.:@+\-]{0,127}$")
VERIFIED_FACTORS = ("application", "bank_account")
INACTIVE_PROFILE = ("Suspended", "Closed")


def _require_service() -> None:
	user = frappe.session.user
	if user in (None, "", "Guest") or AGENT_ROLE not in frappe.get_roles(user):
		raise Forbidden(_("This method is reserved for the advisory service."))


def _live_session(session_id: str, room: str, identity: str) -> dict:
	# The caller is a service, not a person, so a shape violation is a refusal like any other.
	if not isinstance(session_id, str) or not SESSION_ID.match(session_id):
		raise NotFound(_("Voice session not found."))

	row = frappe.db.get_value(
		"Voice Session",
		{"session_id": session_id},
		[
			"name",
			"session_id",
			"room",
			"identity",
			"user",
			"customer",
			"application",
			"expires_at",
			"status",
			"consent_version",
		],
		as_dict=True,
	)
	# A session id presented with the wrong room or identity is answered exactly like a
	# missing one: the worker has no business probing for which part it got wrong.
	if not row or row.room != (room or "") or row.identity != (identity or ""):
		raise NotFound(_("Voice session not found."))
	# A row without the current notice version was not consented to (or predates this notice):
	# the worker gets no assertion, so no turn can be processed for it.
	if row.consent_version != livekit_token.CONSENT_VERSION:
		raise Forbidden(_("This voice session has no recorded consent."))
	if row.status in TERMINAL:
		raise Conflict(_("This voice session has ended."))
	if row.expires_at <= now_datetime():
		frappe.db.set_value("Voice Session", row.name, "status", "Expired", update_modified=False)
		raise Conflict(_("This voice session has expired."))
	return row


def _preferences(user: str) -> tuple[str | None, str | None]:
	profile = frappe.db.get_value(
		"Borrower Profile", {"user": user}, ["literacy", "preferred_locale"], as_dict=True
	)
	return (profile.literacy, profile.preferred_locale) if profile else (None, None)


@frappe.whitelist(methods=["POST"])
def voice_assertion(session_id: str, room: str, identity: str) -> dict:
	"""One assertion per call: the worker asks again for every turn, so none outlives its turn."""
	_require_service()
	limits.enforce("voice-assertion", 600, 3600)
	row = _live_session(session_id, room, identity)
	if row.status == "Created":
		frappe.db.set_value("Voice Session", row.name, "status", "Joined", update_modified=False)

	literacy, locale = _preferences(row.user)
	thread = f"voice-{row.session_id}"
	result = assertion.mint(row.user, row.customer, row.application, thread, literacy, locale)
	result.update(
		{
			"application": row.application,
			"customer": row.customer,
			"thread": thread,
			"session_id": row.session_id,
		}
	)
	return result


def _phone_enabled() -> bool:
	# Off unless the site turns it on: a site that never decided gives phone callers nothing.
	return frappe.conf.get(PHONE_FLAG) in (1, True, "1", "true", "True")


def _phone_subject(applicant: str, application: str, room: str, identity: str, factor: str) -> tuple[str, dict]:
	"""(sub, preferences row or None). Every refusal about *which* applicant is the same NotFound."""
	missing = NotFound(_("Applicant not found."))
	if factor not in VERIFIED_FACTORS:
		raise Forbidden(_("This caller has not been verified."))
	for value, shape in ((applicant, RECORD_ID), (application, RECORD_ID), (room, LIVEKIT_NAME), (identity, LIVEKIT_NAME)):
		if not isinstance(value, str) or not shape.fullmatch(value):
			raise missing

	row = frappe.db.get_value(
		"Loan Application", application, ["name", "applicant_type", "applicant"], as_dict=True
	)
	if not row or row.applicant_type != "Customer" or row.applicant != applicant:
		raise missing
	if frappe.db.get_value("Customer", applicant, "disabled") in (None, 1, "1"):
		# None: no such Customer (a dangling link); 1: disabled. Either way, nobody to speak for.
		raise missing

	profile = frappe.db.get_value(
		"Borrower Profile",
		{"customer": applicant},
		["user", "status", "literacy", "preferred_locale"],
		as_dict=True,
	)
	if profile and profile.status in INACTIVE_PROFILE:
		raise missing
	if profile and profile.user:
		return profile.user, profile
	# A seeded applicant with no portal account yet: the subject names the phone channel and the
	# customer, never the caller's number.
	return f"phone:{applicant}", profile


@frappe.whitelist(methods=["POST"])
def phone_assertion(
	applicant: str, application: str, room: str, identity: str, verified_factor: str
) -> dict:
	"""One assertion per spoken turn for a phone caller the worker has verified on the keypad.

	Same authentication as `voice_assertion` (an `Agent Integration` API key), same response shape,
	scoped to `applicant` and `application` only. Unknown, mismatched or unverified -> no assertion.
	"""
	_require_service()
	if not _phone_enabled():
		raise Forbidden(_("Phone callers are not enabled on this site."))
	limits.enforce("phone-assertion", 600, 3600)
	sub, profile = _phone_subject(applicant, application, room, identity, verified_factor)
	# A second budget per applicant: one compromised worker cannot drain a single file at the service rate.
	limits.enforce("phone-assertion-applicant", 240, 3600, identity=f"phone:{applicant}")

	literacy, locale = (profile.literacy, profile.preferred_locale) if profile else (None, None)
	call = hashlib.sha256(f"{room}|{identity}".encode("utf-8")).hexdigest()[:32]
	thread = f"phone-{call}"
	result = assertion.mint(sub, applicant, application, thread, literacy, locale)
	result.update(
		{
			"application": application,
			"customer": applicant,
			"thread": thread,
			"session_id": call,
		}
	)
	return result
