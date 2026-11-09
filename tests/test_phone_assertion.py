"""`loan_advisory_portal.api.voice.phone_assertion`: the Frappe side of the phone (SIP) path.

Pure python: a small in-memory stand-in for `frappe` (db, cache, conf, session) replaces the real one while the
portal modules are imported fresh, and everything is put back afterwards so other tests' stubs are untouched.
The worker only calls this after keypad verification; Frappe re-checks the applicant/application pairing, that
phone vouching is on for the site, and that the caller is the integration account, and never logs a number.
"""
import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "loan_advisory_portal"
SECRET = "s" * 48
SERVICE, BORROWER = "worker@example.invalid", "borrower@example.invalid"
SIP_NUMBER = "919876543210"
SIP_IDENTITY = f"sip_+{SIP_NUMBER}"
ROOM = "saral-call-_+919876543210_AbCd"  # LiveKit SIP dispatch rooms can carry the number too


class _dict(dict):
    __getattr__ = dict.get


class FakeDB:
    def __init__(self, tables):
        self.tables, self.calls, self.writes = tables, [], []

    def _match(self, row, filters):
        if isinstance(filters, str):
            return row.get("name") == filters
        return all(row.get(k) == v for k, v in filters.items())

    def get_value(self, doctype, filters, fields="name", as_dict=False, **_):
        self.calls.append((doctype, filters))
        row = next((r for r in self.tables.get(doctype, []) if self._match(r, filters)), None)
        if row is None:
            return None
        if isinstance(fields, str):
            return row.get(fields)
        picked = _dict({f: row.get(f) for f in fields})
        return picked if as_dict else tuple(picked.values())

    def set_value(self, doctype, name, field, value, **_):
        self.writes.append((doctype, name, field, value))
        for r in self.tables.get(doctype, []):
            if r.get("name") == name:
                r[field] = value


class FakeCache:
    def __init__(self):
        self.store = {}

    def make_key(self, key):
        return key

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, seconds, value):
        self.store[key] = value

    def incrby(self, key, n):
        self.store[key] = int(self.store.get(key) or 0) + n
        return self.store[key]

    def ttl(self, key):
        return 3600


def _fake_frappe():
    f = types.ModuleType("frappe")
    f.__path__ = []  # a package, so `frappe.utils` and friends resolve to the stubs below
    f._ = lambda s, *a, **k: s
    f.whitelist = lambda **kw: (lambda fn: fn)
    f._dict = _dict
    for name in ("ValidationError", "AuthenticationError", "PermissionError", "DoesNotExistError",
                 "RateLimitExceededError"):
        setattr(f, name, type(name, (Exception,), {}))
    f.flags = SimpleNamespace()
    f.local = SimpleNamespace()
    f.session = SimpleNamespace(user="Guest")
    f.roles = {SERVICE: ["Agent Integration"], BORROWER: ["Loan Borrower"]}
    f.get_roles = lambda user=None: list(f.roles.get(user, []))
    f.conf = {"agent_assertion_secret": SECRET, "voice_phone_assertions": 1}
    f.cache = FakeCache()
    f.db = FakeDB({})
    f.logger = lambda *a, **k: logging.getLogger("frappe.fake")

    utils = types.ModuleType("frappe.utils")
    utils.now_datetime = datetime.now
    f.utils = utils
    mods = {"frappe": f, "frappe.utils": utils}
    for dotted in ("frappe.custom", "frappe.custom.doctype", "frappe.custom.doctype.custom_field"):
        m = types.ModuleType(dotted)
        m.__path__ = []
        mods[dotted] = m
    cf = types.ModuleType("frappe.custom.doctype.custom_field.custom_field")
    cf.create_custom_fields = lambda *a, **k: None
    mods[cf.__name__] = cf
    return f, mods


def _ours(name):
    return name == "frappe" or name.startswith("frappe.") or name == "loan_advisory_portal" \
        or name.startswith("loan_advisory_portal.")


@pytest.fixture
def portal():
    saved = {k: v for k, v in sys.modules.items() if _ours(k)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(APP))
    frappe, mods = _fake_frappe()
    sys.modules.update(mods)
    try:
        voice = importlib.import_module("loan_advisory_portal.api.voice")
        assertion = importlib.import_module("loan_advisory_portal.services.assertion")
        errors = importlib.import_module("loan_advisory_portal.portal.errors")
        frappe.db = FakeDB({
            "Customer": [_dict(name="CUS-TEST-001", disabled=0), _dict(name="CUS-TEST-002", disabled=0),
                         _dict(name="CUS-OFF", disabled=1), _dict(name="CUS-SEED", disabled=0)],
            "Loan Application": [
                _dict(name="APP-TEST-001", applicant_type="Customer", applicant="CUS-TEST-001"),
                _dict(name="APP-TEST-002", applicant_type="Customer", applicant="CUS-TEST-002"),
                _dict(name="APP-EMP", applicant_type="Employee", applicant="CUS-TEST-001"),
                _dict(name="APP-OFF", applicant_type="Customer", applicant="CUS-OFF"),
                _dict(name="APP-SEED", applicant_type="Customer", applicant="CUS-SEED"),
                _dict(name="APP-GHOST", applicant_type="Customer", applicant="CUS-GONE"),
            ],
            "Borrower Profile": [
                _dict(name="BP-1", user=BORROWER, customer="CUS-TEST-001", status="Active",
                      literacy="low", preferred_locale="te"),
                _dict(name="BP-2", user="other@example.invalid", customer="CUS-TEST-002", status="Active",
                      literacy="high", preferred_locale="hi"),
            ],
            "Voice Session": [],
        })
        frappe.session.user = SERVICE
        yield SimpleNamespace(voice=voice, assertion=assertion, errors=errors, frappe=frappe)
    finally:
        for k in [k for k in sys.modules if _ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(APP))


def call(p, **override):
    values = {"applicant": "CUS-TEST-001", "application": "APP-TEST-001", "room": ROOM,
              "identity": SIP_IDENTITY, "verified_factor": "application"}
    values.update(override)
    return p.voice.phone_assertion(**values)


# ------------------------------------------------------------------------------------------------ happy path
def test_a_verified_caller_gets_an_assertion_for_that_applicant_only(portal):
    result = call(portal)
    claims = portal.assertion.verify(result["assertion"])
    assert claims["customer"] == "CUS-TEST-001"
    assert claims["application"] == "APP-TEST-001"
    assert claims["sub"] == BORROWER
    assert (claims["literacy"], claims["locale"]) == ("low", "te")
    assert claims["thread"] == result["thread"] and result["thread"].startswith("phone-")
    assert claims["exp"] - claims["iat"] == portal.assertion.TTL_SECONDS
    assert result["customer"] == "CUS-TEST-001" and result["application"] == "APP-TEST-001"
    assert result["expires_in"] == portal.assertion.TTL_SECONDS and result["jti"] == claims["jti"]


def test_the_response_has_the_browser_paths_shape(portal):
    f = portal.frappe
    f.db.tables["Voice Session"].append(_dict(
        name="VS-1", session_id="a" * 32, room="loan-x", identity="applicant-x", user=BORROWER,
        customer="CUS-TEST-001", application="APP-TEST-001", expires_at=datetime.now() + timedelta(minutes=5),
        status="Joined", consent_version="voice-consent/1"))
    browser = portal.voice.voice_assertion("a" * 32, "loan-x", "applicant-x")
    phone = call(portal)
    assert set(phone) == set(browser)
    assert set(portal.assertion.verify(phone["assertion"])) == set(portal.assertion.verify(browser["assertion"]))


def test_each_turn_gets_a_fresh_assertion_on_the_same_thread(portal):
    first, second = call(portal), call(portal)
    assert first["jti"] != second["jti"] and first["assertion"] != second["assertion"]
    assert first["thread"] == second["thread"] and first["session_id"] == second["session_id"]
    other_call = call(portal, room="saral-call-other")
    assert other_call["thread"] != first["thread"]


def test_both_keypad_factors_are_accepted(portal):
    assert call(portal, verified_factor="bank_account")["assertion"]


def test_a_seeded_applicant_without_a_portal_account_gets_a_phone_subject_and_defaults(portal):
    claims = portal.assertion.verify(call(portal, applicant="CUS-SEED", application="APP-SEED")["assertion"])
    assert claims["sub"] == "phone:CUS-SEED"
    assert claims["customer"] == "CUS-SEED" and claims["application"] == "APP-SEED"
    assert (claims["literacy"], claims["locale"]) == ("medium", "en")


# ------------------------------------------------------------------------------------------- authentication
def test_only_the_integration_account_can_call_it(portal):
    f, errors = portal.frappe, portal.errors
    for user in ("Guest", "", None, BORROWER, "stranger@example.invalid"):
        f.session.user = user
        f.db.calls.clear()
        with pytest.raises(errors.Forbidden):
            call(portal)
        assert f.db.calls == []  # refused before any record is read


def test_the_site_must_switch_phone_vouching_on(portal):
    f, errors = portal.frappe, portal.errors
    for value in (None, 0, "0", "", "yes-please", False):
        f.conf["voice_phone_assertions"] = value
        f.db.calls.clear()
        with pytest.raises(errors.Forbidden):
            call(portal)
        assert f.db.calls == []
    del f.conf["voice_phone_assertions"]
    with pytest.raises(errors.Forbidden):
        call(portal)


def test_no_signing_secret_means_no_assertion(portal):
    portal.frappe.conf["agent_assertion_secret"] = "short"
    with pytest.raises(portal.errors.ServiceUnavailable):
        call(portal)


# ------------------------------------------------------------------------------------------------ fail safe
@pytest.mark.parametrize("factor", [None, "", "voice", "1234", "APPLICATION", "verified", 1])
def test_an_unverified_caller_gets_nothing(portal, factor):
    with pytest.raises(portal.errors.Forbidden):
        call(portal, verified_factor=factor)
    assert portal.frappe.db.calls == []


@pytest.mark.parametrize("override", [
    {"application": "APP-NOPE"},                                   # no such application
    {"application": "APP-TEST-002"},                               # someone else's application
    {"applicant": "CUS-TEST-002"},                                 # someone else's applicant id
    {"applicant": "CUS-NOPE"},
    {"application": "APP-EMP"},                                    # not a Customer applicant
    {"applicant": "CUS-OFF", "application": "APP-OFF"},            # disabled customer
    {"applicant": "CUS-GONE", "application": "APP-GHOST"},         # dangling customer link
])
def test_unknown_or_mismatched_applicants_are_all_the_same_not_found(portal, override):
    with pytest.raises(portal.errors.NotFound) as caught:
        call(portal, **override)
    assert str(caught.value) == "Applicant not found."


@pytest.mark.parametrize("status", ["Suspended", "Closed"])
def test_a_suspended_or_closed_borrower_gets_nothing(portal, status):
    portal.frappe.db.tables["Borrower Profile"][0]["status"] = status
    with pytest.raises(portal.errors.NotFound):
        call(portal)


@pytest.mark.parametrize("override", [
    {"applicant": None}, {"application": None}, {"room": None}, {"identity": None},
    {"applicant": ""}, {"application": ["APP-TEST-001"]}, {"applicant": {"name": "CUS-TEST-001"}},
    {"applicant": "CUS-TEST-001' OR '1'='1"}, {"application": "APP-TEST-001%"}, {"applicant": "a" * 141},
    {"room": "room with spaces"}, {"identity": "x" * 129}, {"identity": "id\n"},
])
def test_malformed_references_are_refused_without_a_lookup(portal, override):
    with pytest.raises(portal.errors.NotFound):
        call(portal, **override)
    assert portal.frappe.db.calls == []


def test_the_phone_path_never_writes(portal):
    call(portal)
    assert portal.frappe.db.writes == []


def test_a_single_applicant_has_its_own_budget(portal):
    for _ in range(240):
        call(portal)
    with pytest.raises(portal.errors.TooManyRequests):
        call(portal)
    # Another applicant on the same worker is unaffected by the first one's budget.
    assert call(portal, applicant="CUS-TEST-002", application="APP-TEST-002")["assertion"]


def test_the_service_budget_still_applies(portal):
    portal.frappe.cache.store["portal-budget:phone-assertion:" + SERVICE] = 600
    with pytest.raises(portal.errors.TooManyRequests):
        call(portal)


# ------------------------------------------------------------------------------------------ no number leaks
def test_no_phone_number_or_digit_reaches_logs_or_the_response(portal, caplog):
    caplog.set_level(logging.DEBUG)
    result = call(portal)
    claims = portal.assertion.verify(result["assertion"])
    for refused in ({"application": "APP-TEST-002"}, {"verified_factor": None}):
        with pytest.raises(Exception):
            call(portal, **refused)
    blob = json.dumps(result) + json.dumps(claims) + caplog.text
    assert SIP_NUMBER not in blob and SIP_IDENTITY not in blob and ROOM not in blob
    for _, filters in portal.frappe.db.calls:  # nor is the number used to look anything up
        assert SIP_NUMBER not in json.dumps(filters)


def test_error_messages_do_not_echo_the_caller(portal):
    with pytest.raises(portal.errors.NotFound) as caught:
        call(portal, application="APP-TEST-002")
    assert SIP_NUMBER not in str(caught.value) and SIP_NUMBER not in str(vars(portal.frappe.flags))


# ------------------------------------------------------------------------------------- worker contract
def _load_exchange():
    spec = importlib.util.spec_from_file_location(
        "voice_assertion_exchange_phone", ROOT / "apps" / "voice" / "voice_assertion_exchange.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload):
        self.status, self._payload = 200, payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_the_worker_body_is_exactly_what_frappe_accepts(portal):
    """PhoneAssertions posts to this method; its body must bind to the whitelisted signature, nothing more."""
    exchange = _load_exchange()
    sent = {}

    class HTTP:
        def post(self, url, *, headers=None, json=None, timeout=None):
            sent.update(url=url, body=json)
            return _Response({"message": portal.voice.phone_assertion(**json)})

    pa = exchange.PhoneAssertions(HTTP(), "CUS-TEST-001", "APP-TEST-001", ROOM, SIP_IDENTITY,
                                  frappe_url="http://frappe.test", api_key="k", api_secret="s", timeout_s=5)
    message = asyncio.run(pa.fresh("application"))
    assert sent["url"].endswith("/api/method/loan_advisory_portal.api.voice.phone_assertion")
    assert set(sent["body"]) == set(inspect.signature(portal.voice.phone_assertion).parameters)
    assert portal.assertion.verify(message["assertion"])["customer"] == "CUS-TEST-001"
