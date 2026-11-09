"""Web phone (/borrower/call): the phone-call flow in the browser.

- Routing precedence in the voice worker (apps/voice/call_routing.py, no LiveKit): SIP -> phone; a server-set
  `saral.mode=web_phone` attribute in a `webcall-` room -> phone; Voice Session metadata -> voice mode; anything else,
  including every mix of the two, -> refuse.
- The Frappe token endpoint (api/web_phone.py, against an in-memory frappe stand-in): the caller ID and the room
  prefix come from the server, the token can't rewrite its own attributes, there is no session id, one call at a time.
- The keypad data-packet fallback is accepted only from the web-phone participant, on topic "dtmf", one key.
- The page, the assistant's Call Saral line and the vendored livekit-client the page imports.
"""
import base64
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_phone_assertion import FakeCache, FakeDB, _dict, _fake_frappe, _ours

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "apps" / "voice"
APP = ROOT / "apps" / "loan_advisory_portal"
PORTAL = APP / "loan_advisory_portal"
BORROWER = "borrower@example.invalid"
SESSION = "ab" * 16


@pytest.fixture
def routing(monkeypatch):
    monkeypatch.syspath_prepend(str(VOICE))
    sys.modules.pop("call_routing", None)
    import call_routing
    return call_routing


@pytest.fixture
def cv(monkeypatch):
    monkeypatch.syspath_prepend(str(VOICE))
    import caller_verification
    return caller_verification


# ------------------------------------------------------------------------------------------------ routing precedence
def meta(session_id=SESSION):
    return json.dumps({"session_id": session_id, "user": BORROWER, "customer": "CUS-1", "application": "APP-1"})


WEB = {"saral.mode": "web_phone", "sip.phoneNumber": "+919876500001"}


@pytest.mark.parametrize("kwargs, expected", [
    # 1. SIP always takes the phone flow, whatever else it carries
    (dict(is_sip=True, attributes={"sip.phoneNumber": "+919876500001"}, metadata="", room="saral-call-x"), "phone"),
    (dict(is_sip=True, attributes={}, metadata=meta(), room="loan-abc"), "phone"),
    # 2. the web phone: server-set attribute AND a webcall- room AND no session id
    (dict(is_sip=False, attributes=WEB, metadata="{}", room="webcall-0123456789abcdef"), "web_phone"),
    (dict(is_sip=False, attributes=WEB, metadata="", room="webcall-0123456789abcdef"), "web_phone"),
    # 3. voice mode: a Voice Session id in metadata, a loan- room, no attributes
    (dict(is_sip=False, attributes={}, metadata=meta(), room="loan-abc-1234"), "voice"),
    (dict(is_sip=False, attributes=None, metadata=meta(), room="loan-abc-1234"), "voice"),
    # 4. nothing -> refuse
    (dict(is_sip=False, attributes={}, metadata="", room="loan-abc"), "refuse"),
    (dict(is_sip=False, attributes={}, metadata=meta("not-hex"), room="loan-abc"), "refuse"),
    (dict(is_sip=False, attributes={}, metadata="not json", room="loan-abc"), "refuse"),
    # 5. no crossing between the two browser doors
    (dict(is_sip=False, attributes=WEB, metadata=meta(), room="webcall-0123456789abcdef"), "refuse"),
    (dict(is_sip=False, attributes=WEB, metadata="{}", room="loan-abc-1234"), "refuse"),
    (dict(is_sip=False, attributes=WEB, metadata=meta(), room="loan-abc-1234"), "refuse"),
    (dict(is_sip=False, attributes={}, metadata=meta(), room="webcall-0123456789abcdef"), "refuse"),
    (dict(is_sip=False, attributes={}, metadata="{}", room="webcall-0123456789abcdef"), "refuse"),
    (dict(is_sip=False, attributes={"saral.mode": "voice"}, metadata=meta(), room="loan-abc"), "refuse"),
    (dict(is_sip=False, attributes={"saral.mode": "phone"}, metadata="{}", room="webcall-0123456789abcdef"), "refuse"),
])
def test_routing_precedence(routing, kwargs, expected):
    assert routing.route(**kwargs) == expected


def test_only_the_two_phone_doors_take_the_keypad_flow(routing):
    assert routing.is_phone_flow(routing.PHONE) and routing.is_phone_flow(routing.WEB_PHONE)
    assert not routing.is_phone_flow(routing.VOICE) and not routing.is_phone_flow(routing.REFUSE)


def test_worker_entrypoint_routes_through_call_routing():
    src = (VOICE / "agent_worker.py").read_text(encoding="utf-8")
    entry = src[src.index("async def entrypoint"):src.index("async def phone_entrypoint")]
    assert "participant_route(participant, ctx.room.name)" in entry
    assert "call_routing.is_phone_flow(route)" in entry
    assert "web_phone=route == call_routing.WEB_PHONE" in entry
    # voice mode still needs a well-formed session id and refuses otherwise, before any greeting
    assert entry.index("raise RuntimeError") < entry.index("Assertions(")
    assert "route != call_routing.VOICE or session_id is None" in entry


def test_data_keypad_fallback_is_wired_for_the_web_phone_only():
    src = (VOICE / "agent_worker.py").read_text(encoding="utf-8")
    phone = src[src.index("async def phone_entrypoint"):]
    assert 'ctx.room.on("sip_dtmf_received", cv.dtmf_handler(participant.identity, phone.keypad))' in phone
    assert ('    if web_phone:\n        ctx.room.on("data_received", cv.dtmf_data_handler(participant.identity, '
            'phone.keypad))') in phone
    assert src.count("dtmf_data_handler") == 1


def test_worker_participant_route_reads_kind_attributes_metadata(monkeypatch):
    pytest.importorskip("livekit.agents")
    pytest.importorskip("livekit.plugins.gnani")
    monkeypatch.setenv("GNANI_API_KEY", "test-dummy-gnani-key")
    monkeypatch.syspath_prepend(str(VOICE))
    sys.modules.pop("agent_worker", None)
    import agent_worker
    from livekit import rtc
    std, sip = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    p = lambda kind, attrs, md: SimpleNamespace(kind=kind, attributes=attrs, metadata=md)  # noqa: E731
    assert agent_worker.participant_route(p(sip, {}, ""), "x") == "phone"
    assert agent_worker.participant_route(p(std, WEB, "{}"), "webcall-0123456789abcdef") == "web_phone"
    assert agent_worker.participant_route(p(std, {}, meta()), "loan-a") == "voice"
    assert agent_worker.participant_route(p(std, WEB, meta()), "webcall-0123456789abcdef") == "refuse"


# ------------------------------------------------------------------------------------------------ DTMF data fallback
def packet(data, topic="dtmf", identity="webphone-abc"):
    return SimpleNamespace(data=data, topic=topic, participant=SimpleNamespace(identity=identity) if identity else None)


def test_data_fallback_accepts_one_key_from_the_web_phone_caller_only(cv):
    pressed = []
    h = cv.dtmf_data_handler("webphone-abc", SimpleNamespace(press=pressed.append))
    h(packet(b'{"digit": "4"}'))
    h(packet(b'{"digit": "#"}'))
    h(packet(b'{"digit": "4"}', identity="someone-else"))      # another participant
    h(packet(b'{"digit": "4"}', identity=None))                # server-sent
    h(packet(b'{"digit": "4"}', topic="advisor"))              # another topic
    h(packet(b'{"digit": "12"}'))                              # more than one key
    h(packet(b'{"digit": ""}'))
    h(packet(b'{"digit": 5}'))
    h(packet(b'not json'))
    h(packet(b'\xff\xfe'))
    h(packet(b'["4"]'))
    assert pressed == ["4", "#"]
    cv.dtmf_data_handler("webphone-abc", None)(packet(b'{"digit": "1"}'))  # no keypad: nothing happens


def test_data_fallback_never_logs(cv):
    src = (VOICE / "caller_verification.py").read_text(encoding="utf-8")
    fn = src[src.index("def dtmf_data_handler"):]
    assert "log." not in fn and "print(" not in fn


# ------------------------------------------------------------------------------------------------ token endpoint
class Cache(FakeCache):
    def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture
def portal(monkeypatch):
    saved = {k: v for k, v in sys.modules.items() if _ours(k)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(APP))
    frappe, mods = _fake_frappe()
    sys.modules.update(mods)
    try:
        frappe.conf.update({"livekit_api_key": "APIkey", "livekit_api_secret": "x" * 40,
                            "livekit_url": "wss://saral.livekit.cloud"})
        frappe.cache = Cache()
        frappe.local.request_ip = "203.0.113.9"
        frappe.db = FakeDB({
            "Borrower Profile": [_dict(name="BP-1", user=BORROWER, customer="CUS-TEST-001", status="Active",
                                       full_name="Asha Rao", phone="+91 98765 00001"),
                                 _dict(name="BP-2", user="nophone@example.invalid", customer="CUS-TEST-002",
                                       status="Active", full_name="No Phone", phone=None)],
            "Customer": [_dict(name="CUS-TEST-001", mobile_no=None), _dict(name="CUS-TEST-002", mobile_no=None)],
        })
        web = importlib.import_module("loan_advisory_portal.api.web_phone")
        errors = importlib.import_module("loan_advisory_portal.portal.errors")
        token_mod = importlib.import_module("loan_advisory_portal.services.livekit_token")

        def ctx():
            row = next(r for r in frappe.db.tables["Borrower Profile"] if r.user == frappe.session.user)
            return SimpleNamespace(user=row.user, customer=row.customer, profile=row.name)
        monkeypatch.setattr(web, "borrower_context", ctx)
        frappe.session.user = BORROWER
        yield SimpleNamespace(web=web, errors=errors, token=token_mod, frappe=frappe)
    finally:
        for k in [k for k in sys.modules if _ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(APP))


def claims(token):
    body = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def test_token_sets_the_server_side_caller_id_and_a_webcall_room(portal):
    out = portal.web.start_web_call()
    c = claims(out["token"])
    assert out["room"].startswith("webcall-") and c["video"]["room"] == out["room"]
    assert portal.web.ROOM.match(out["room"])
    assert c["attributes"] == {"saral.mode": "web_phone", "sip.phoneNumber": "+919876500001"}
    assert c["video"]["canUpdateOwnMetadata"] is False  # the browser cannot rewrite its caller ID or mode
    assert json.loads(c["metadata"]) == {}  # no session id: never voice mode
    assert c["sub"] == out["identity"] and out["identity"].startswith("webphone-")
    assert out["url"] == "wss://saral.livekit.cloud" and out["keypad"] == "dtmf"
    # the browser gets the masked number only
    assert out["caller"] == "+91 ••••• •••01" and "9876500001" not in json.dumps({k: v for k, v in out.items()
                                                                                    if k != "token"})


def test_the_endpoint_takes_no_number_from_the_browser(portal):
    import inspect
    assert list(inspect.signature(portal.web.start_web_call).parameters) == []
    with pytest.raises(TypeError):
        portal.web.start_web_call(phone="+15550000000")


def test_one_call_at_a_time_and_hang_up_releases_it(portal):
    first = portal.web.start_web_call()
    with pytest.raises(portal.errors.Conflict):
        portal.web.start_web_call()
    assert portal.web.end_web_call(room="webcall-ffffffffffffffff") == {"ended": False}  # not this caller's room
    assert portal.web.end_web_call(room="../../etc") == {"ended": False}
    assert portal.web.end_web_call(room=first["room"]) == {"ended": True}
    second = portal.web.start_web_call()
    assert second["room"] != first["room"]


def test_calls_are_rate_limited_per_account(portal):
    for _ in range(portal.web.CALLS_PER_HOUR):
        out = portal.web.start_web_call()
        portal.web.end_web_call(room=out["room"])
    with pytest.raises(portal.errors.TooManyRequests):
        portal.web.start_web_call()


def test_no_number_on_the_profile_is_refused(portal):
    portal.frappe.session.user = "nophone@example.invalid"
    with pytest.raises(portal.errors.NotFound, match="No phone number on your profile"):
        portal.web.start_web_call()


def test_customer_mobile_is_the_fallback_number(portal):
    portal.frappe.session.user = "nophone@example.invalid"
    portal.frappe.db.tables["Customer"][1]["mobile_no"] = "9876500002"
    assert claims(portal.web.start_web_call()["token"])["attributes"]["sip.phoneNumber"] == "+919876500002"


def test_guests_need_the_demo_flag_and_then_get_the_demo_number(portal):
    portal.frappe.session.user = "Guest"
    with pytest.raises(portal.errors.SignInRequired):
        portal.web.start_web_call()
    portal.frappe.conf["saral_web_phone_demo_number"] = "+91 98765 00003"
    out = portal.web.start_web_call()
    assert claims(out["token"])["attributes"]["sip.phoneNumber"] == "+919876500003"
    with pytest.raises(portal.errors.Conflict):  # one call per address too
        portal.web.start_web_call()


def test_keypad_mode_is_a_site_choice(portal):
    portal.frappe.conf["saral_web_phone_keypad"] = "data"
    assert portal.web.start_web_call()["keypad"] == "data"
    portal.frappe.conf["saral_web_phone_keypad"] = "anything"
    assert portal.web.keypad_mode() == "dtmf"


def test_voice_mode_token_is_unchanged(portal):
    t = portal.token.create_token(api_key="k", api_secret="s" * 32, identity="applicant-x", name="A", room="loan-x",
                                  metadata={"session_id": SESSION})
    c = claims(t)
    assert "attributes" not in c and "canUpdateOwnMetadata" not in c["video"]
    assert set(c["video"]) == {"room", "roomJoin", "canPublish", "canSubscribe", "canPublishData"}


@pytest.mark.parametrize("raw, expected", [("+91 98765 43210", "+919876543210"), ("9876543210", "+919876543210"),
                                          ("919876543210", "+919876543210"), ("+1-484-295-0295", "+14842950295"),
                                          ("", None), (None, None), ("call me", None), ("+12", None)])
def test_profile_numbers_normalise_to_e164(portal, raw, expected):
    assert portal.web.normalize(raw) == expected


# ------------------------------------------------------------------------------------------------ page and copy
def test_call_page_is_a_phone_screen():
    html = (PORTAL / "www" / "borrower" / "call.html").read_text(encoding="utf-8")
    for key in "123456789*0#":
        assert f'("{key}",' in html
    for needed in ('id="webphone-call"', 'id="webphone-hangup"', 'id="webphone-mute"', 'id="webphone-speaker"',
                   'id="webphone-timer"', 'id="webphone-status"', "saral-mark.svg",
                   "No phone number on your profile",
                   "Press 9 to change language, then key the last 4 digits of your application number.",
                   'or dial <a href="tel:{{ call_saral.tel }}">{{ call_saral.display }}</a>',
                   "/assets/loan_advisory_portal/css/web_phone.css",
                   '<script type="module" src="/assets/loan_advisory_portal/js/pages/web_phone.js">'):
        assert needed in html, needed


def test_web_phone_script_sends_rfc4733_codes_and_never_a_number():
    js = (PORTAL / "public" / "js" / "pages" / "web_phone.js").read_text(encoding="utf-8")
    assert '"*": 10, "#": 11' in js and "publishDtmf(DTMF_CODES[key], key)" in js
    assert 'topic: "dtmf"' in js and "JSON.stringify({ digit: key })" in js
    assert "console." not in js and "phoneNumber" not in js and "sip." not in js
    assert "start_web_call\", {})" in js  # no arguments: the caller ID is the server's


def test_page_imports_the_livekit_client_the_portal_already_ships():
    chunks = sorted((PORTAL / "public" / "js" / "assistant" / "chunks").glob("livekit-client.esm-*.js"))
    assert len(chunks) == 1
    src = chunks[0].read_text(encoding="utf-8")
    assert "publishDtmf(" in src and "export {" in src and " as Room" in src and " as RoomEvent" in src
    py = (PORTAL / "www" / "borrower" / "call.py").read_text(encoding="utf-8")
    assert 'glob("livekit-client.esm-*.js")' in py and "http" not in py


def test_assistant_call_saral_line_opens_the_web_phone():
    html = (PORTAL / "www" / "borrower" / "assistant.html").read_text(encoding="utf-8")
    assert '<a class="callsaral__link" href="/borrower/call">Call Saral</a>' in html
    assert 'or dial <a href="tel:{{ call_saral.tel }}">{{ call_saral.display }}</a>' in html


# ------------------------------------------------------------------------------------------------ seed directory
def test_seed_phone_override_puts_the_profile_number_in_the_directory_and_keeps_the_rest(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "voice"))
    monkeypatch.syspath_prepend(str(VOICE))
    import seed_caller_directory as sd
    base = sd.demo_entries()
    first = base[0]["applicant_id"]
    moved = sd.demo_entries(phones={first: "+919876500001"})
    assert moved[0]["phone"] == "+919876500001"
    assert moved[0]["application_last4"] == base[0]["application_last4"]
    assert moved[1:] == base[1:]
