"""Which door a LiveKit participant came in by, decided once when the worker joins a room. No LiveKit imports, so the
precedence is unit-testable on its own.

Saral has three ways to talk, and the worker serves two of them (text chat never touches LiveKit):

    PHONE      a phone call. Either a real SIP participant (LiveKit telephony, participant kind SIP), or the portal's
               WEB PHONE (/borrower/call): a browser participant whose LiveKit token Frappe minted with the attribute
               `saral.mode = web_phone` and the caller's registered number in `sip.phoneNumber`, in a room named
               `webcall-<random>`. Both run the same keypad flow (caller ID -> greeting -> press 9 -> last 4 digits).
    VOICE      in-app voice mode: a browser participant whose metadata names a Frappe `Voice Session` row
               (`session_id`, 32 hex); the row is the authorisation.
    REFUSE     anything else. The worker leaves the room without a greeting.

Why the attribute can be trusted: participant attributes set in an access token are written by whoever holds the
LiveKit API secret, i.e. Frappe. A browser can only change its own attributes with the `canUpdateOwnMetadata` grant,
which no Saral token carries (the web-phone token sets it to false explicitly; checked against livekit-server 1.13:
a participant's own `set_attributes` is refused and the token values stay). So `saral.mode`, `sip.phoneNumber` and the
`webcall-` room are all server-chosen; the browser never picks its caller ID.

Precedence (first match wins):
    1. kind SIP                                                              -> PHONE
    2. `saral.mode == web_phone` AND room starts `webcall-` AND no session metadata -> WEB_PHONE (phone flow)
    3. any other `saral.mode`, a web_phone attribute outside a `webcall-` room, a `webcall-` room without the
       attribute, or a web_phone participant that also carries a session id                 -> REFUSE
    4. a well-formed `session_id` in metadata                                -> VOICE
    5. otherwise                                                             -> REFUSE
so a web-phone participant can never reach voice mode (no Voice Session row, and webcall rooms never route there)
and a voice-mode participant can never reach the keypad flow (its token has no attributes and its room is `loan-...`).
"""
from __future__ import annotations

import json
import re

PHONE, WEB_PHONE, VOICE, REFUSE = "phone", "web_phone", "voice", "refuse"
MODE_ATTRIBUTE = "saral.mode"
WEB_PHONE_MODE = "web_phone"
WEB_PHONE_ROOM_PREFIX = "webcall-"
SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


def session_id(metadata) -> str | None:
    """The Voice Session id from participant metadata (a JSON string), or None when it is absent or malformed."""
    try:
        meta = json.loads(metadata or "{}")
    except (TypeError, ValueError):
        return None
    sid = meta.get("session_id") if isinstance(meta, dict) else None
    return sid if isinstance(sid, str) and SESSION_ID.match(sid) else None


def route(*, is_sip: bool, attributes: dict | None, metadata, room: str | None) -> str:
    """PHONE, WEB_PHONE, VOICE or REFUSE for one participant (see the module docstring for the precedence)."""
    if is_sip:
        return PHONE
    attributes = dict(attributes or {})
    mode = attributes.get(MODE_ATTRIBUTE)
    webcall_room = str(room or "").startswith(WEB_PHONE_ROOM_PREFIX)
    sid = session_id(metadata)
    if mode == WEB_PHONE_MODE and webcall_room and sid is None:
        return WEB_PHONE
    if mode is not None or webcall_room:
        return REFUSE
    return VOICE if sid is not None else REFUSE


def is_phone_flow(decision: str) -> bool:
    return decision in (PHONE, WEB_PHONE)
