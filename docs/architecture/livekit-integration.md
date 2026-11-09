# LiveKit integration

Status: **implemented and unit-tested end to end across Frappe, the worker contract and the island UI, including explicit consent and session lifecycle. A live LiveKit room has not been exercised — the configured URL is `wss://livekit.invalid` and no LiveKit server or voice worker container is running, so the live test is BLOCKED (see below).**

## What existed before this phase (legacy worktree, historical)

| Step | Where | Detail |
| --- | --- | --- |
| Session start | `apps/web/chat/voice.html:74-97` | `POST /api/session` (demo sign-in), then `POST /api/livekit-token {literacy, locale}`, then `room.connect(url, token)` |
| Token minting | `apps/api/src/loan_advisory/agent/server.py:427-447` | requires a selected application; 503 unless `LIVEKIT_API_KEY/SECRET/URL` are set; room `loan-<sha256(app)[:10]>-<uuid6>`; participant identity `applicant-<uuid8>` |
| Credential | same | JWT metadata carried a freshly signed **1-hour advisory bearer token** plus `literacy`/`locale` |
| Worker | `apps/voice/agent_worker.py` | reads participant metadata, builds `AdvisorAgent`, sends the bearer to `/api/chat/stream`, `conversation_id = voice-<session_id>` |

The legacy page had no consent gate or record, and its UI was a synthetic demo retired with the static frontend. Both are replaced below.

## Current design

Frappe issues the LiveKit token; the `Voice Session` row is the authorisation.

```
/borrower/assistant (voice panel) ──POST create_voice_session {application, consent: 1}──► Frappe
        │                                                       │ writes Voice Session row
        │◄── {token, url, room, identity, session_id, expires_at} ┘   (consent_version + consent_at)
        ▼
LiveKit room ── participant metadata {session_id, user, customer, application} ──► worker
                                                                                   │
                                        verify Voice Session row (session_id, room,
                                        identity, consent_version, not expired, status)
                                                                                   ▼
                                                    mint/obtain assertion → FastAPI /api/chat/stream
```

### Frappe side (implemented)

- `POST api/livekit.create_voice_session(application?, consent)` (`api/livekit.py:31`):
  - borrower context required; the application comes from the argument or the cached selection and is ownership-checked **before** consent, so a foreign id is a 404 whether or not the notice was ticked;
  - **without `consent` the call is refused with `403` and no row is written**; the notice version (`services/livekit_token.CONSENT_VERSION = "voice-consent/1"`) and `consent_at` are stored on the row;
  - rate limit `voice-session` 10/hour applies only to consented starts;
  - room `loan-<sha256(customer:application)[:16]>-<8 hex>`, identity `applicant-<uuid4>`, `session_id` 32 hex;
  - metadata is `{session_id, user, customer, application}` — **no bearer token**;
  - writes the `Voice Session` row (`session_id`, `room`, `identity`, `user`, `customer`, `application`, `expires_at`, `status`, `created_at`, `consent_version`, `consent_at`) *before* returning the token;
  - the response adds `session_id` so the browser can close its own session; it is an identifier, not a credential — the assertion still requires the worker's service role.
- `POST api/livekit.end_voice_session(session_id)` (`api/livekit.py:95`):
  - borrower context required; another borrower's or an unknown id is a 404;
  - a live session becomes `Ended` with `ended_at`; a terminal session is reported unchanged, so retries from a closing tab and a second tab converge on one state;
  - no rate limit: the effect is idempotent and bounded to the caller's own row.
- `services/livekit_token.py`: HS256 JWT built without the LiveKit SDK (`iss`, `sub`, `name`, `nbf`, `iat`, `exp`, `jti`, `video` grant with join/publish/subscribe/data), 600 s TTL, secret read from site config (`livekit_api_key`, `livekit_api_secret`, `livekit_url`), `ServiceUnavailable` when any is missing.
- `POST api/voice.voice_assertion(session_id, room, identity)` (`api/voice.py:80`): reserved for the `Agent Integration` service role. Looks up the row by `session_id`, requires room and identity to match, **refuses a row whose `consent_version` is not the current notice (`403`)**, marks Created → Joined, refuses `Ended`/`Expired` with `409` and marks a past-`expires_at` row `Expired`. Returns a fresh 60-second assertion (thread `voice-<session_id>`) plus `{application, customer, thread, session_id}`.
- Page context (`www/borrower/assistant.py`): the island receives `voice: {enabled, consent_version}`. `enabled` reflects configuration only, so an unreachable service is reported by the panel as a failed join rather than hiding a feature the site offers.

### Island UI (implemented)

`apps/web/assistant/src/voice-panel.tsx`, mounted in the assistant aside:

- **Explicit consent** — a checkbox with the notice text and version; Start is disabled until it is ticked. Consent is sent with the create request and recorded server-side; the browser never receives an assertion.
- **Real room states** — `connecting`, `connected`, `reconnecting`, `ended`, `failed` come from LiveKit's `ConnectionStateChanged`/`Disconnected` events (`voice-transport.ts`), not from UI guesses.
- **Microphone** — enabled only after `room.connect`; a Mute/Unmute toggle calls `setMicrophoneEnabled` and reflects that state. Blocked or missing microphones are explained.
- **Audio** — remote audio tracks are attached to media elements on `TrackSubscribed` and detached on `TrackUnsubscribed`/disconnect; the LiveKit SDK is a lazy chunk (≈658 kB, loaded only when a call starts).
- **Transcript controls** — the worker's per-turn JSON on the `advisor` topic (`transcript` = recognised question, `answer` = spoken answer) is appended to the panel. Hide/Show toggles it with `aria-pressed`; Clear removes it from this page only.
- **Lifecycle** — End session disconnects the room and calls `end_voice_session` with `keepalive`; the panel also ends itself at the server-provided `expires_at` and on unmount (navigation), so no live room is left behind. A new session can be started afterwards.
- **Failure** — create refusals, 401 redirects to `/login`, connect/microphone failures and reconnects each have their own message; the text chat stays usable throughout.

### The worker exchange

`apps/voice/agent_worker.py`:

1. reads `{session_id, user, customer, application}` from participant metadata and validates `session_id` against `^[0-9a-f]{32}$`; missing metadata leaves the room instead of starting a turn;
2. calls Frappe with the worker's own API key (`Authorization: token <key>:<secret>`, env `FRAPPE_URL` / `FRAPPE_API_KEY` / `FRAPPE_API_SECRET`) to obtain an assertion — the browser never supplies a credential;
3. posts each turn to `/api/chat/stream` with `Authorization: Bearer <assertion>` and `conversation_id = voice-<session_id>`, refreshing the assertion per turn so a long call never reuses one;
4. publishes `{answer, intent, action, abstained, citations, decision_card, improvement_plan, what_if, handoff_ref, transcript}` on the `advisor` data topic; the turn log records metadata and omits the transcript;
5. a session without recorded consent, wrong room/identity, an ended session or an expired one ends the call before any turn is processed.

There is no local fallback: without the Frappe credential the worker raises before greeting.

## Languages

Saral's voice runs in all 10 languages Gnani Prisma and Timbre support (English, Hindi, Telugu, Tamil, Kannada,
Malayalam, Marathi, Bengali, Gujarati, Punjabi). Evaluations and native-speaker checks were done in Telugu and Hindi
only; the other languages are supported but not yet evaluated.

Before Timbre speaks a reply in an Indian language, rupee amounts, percentages and plain numbers in the text are
written out as words in that language ("₹4,100" -> "నాలుగు వేల వంద రూపాయలు" / "चार हज़ार एक सौ रुपये";
`apps/voice/spoken_numbers.py`); IDs, phone numbers and codes stay as digits, English is left to Timbre, and only the
TTS request changes (transcripts, logs and the LLM context keep the digits). The Telugu and Hindi forms were written
for native review; the other seven languages' tables are not yet native-reviewed. `GNANI_TTS_VERBALISE_NUMBERS=off`
turns it off.

## Phone callers: keypad verification (SIP + DTMF)

Call flow v2, "language first, then 4 digits". A phone call joins the room as a LiveKit SIP participant, so there is
no `Voice Session` row and no browser login. The worker (`apps/voice/agent_worker.py`, `phone_entrypoint`) takes the
phone path instead (`apps/voice/caller_verification.py`, lines in `apps/voice/prompt_texts.py`):

1. **Caller ID.** The from-number is read from the participant attributes (`sip.phoneNumber`, then `sip.from`,
   `sip.fromUser`, `sip.callerId`, `sip.callerNumber`, then a `sip_<number>` identity) and normalised to E.164:
   `tel:`/`sip:` prefixes, `@host`, `;params`, spaces, dashes and brackets are dropped, and `+91XXXXXXXXXX`,
   `91XXXXXXXXXX`, `0XXXXXXXXXX` and a bare 10-digit Indian number are the same key. It is looked up, by its HMAC, in a
   directory of synthetic applicants (`CALLER_DIRECTORY`). Each record names the language its call opens in (one of
   the ten codes below, default `en-IN`, stored in plaintext). The worker logs which reason fired, with the number
   masked to its last 2 digits: key missing, `CALLER_DIRECTORY` not set, directory unreadable (error class only), no
   caller ID, no match, or a match with its language. With no key or directory nobody can be verified (fails closed).
2. **Greeting, in the record's language, with no name or account data.** "Welcome to Saral. To change language, press
   nine." An unknown caller (or no directory) hears it in en-IN.
3. **Language menu (press 9).** A 9 keyed during the greeting or within 3 s after it opens the menu: each option is said
   in its own language, in key order: 1 English (en-IN), 2 Hindi (hi-IN), 3 Telugu (te-IN), 4 Tamil (ta-IN), 5 Kannada
   (kn-IN), 6 Malayalam (ml-IN), 7 Marathi (mr-IN), 8 Bengali (bn-IN), 9 Gujarati (gu-IN), 0 Punjabi (pa-IN). An invalid
   key or no key in 8 s replays it once, then the call continues in the current language. Once the enter-digits prompt
   has started, 9 is an ordinary digit (a last-4 can start with 9).
4. **`verify_caller` (keypad only, one factor).** "Please enter the last four digits of your loan application number."
   No factor menu, no bank-account path and no read-back: a wrong entry hears "That didn't match. Please try again."
   Keys come from the room event `sip_dtmf_received` (`rtc.SipDTMF`: `digit`, `code`, `participant`; only the caller's
   own presses count), never from speech; `GetDtmfTask` is not used. `*` clears the entry; `#` is ignored.
5. **Check.** HMAC-SHA256(`CALLER_VERIFY_HMAC_KEY`, salt|application|last4), compared in constant time; the directory
   holds hashes only (`scripts/voice/seed_caller_directory.py`). Three mismatches lock verification: the locked line
   plays once and the call ends. No keypress for 8 s plays the keypad nudge; 15 s ends the attempt (neither counts as a
   mismatch). Saying something after a timeout starts one more attempt; a second timeout ends the call.
6. **Speech before verification is ignored.** `llm_node` never calls the advisory server (`VerifiedGate.fresh()`
   raises) and speaks nothing for a caller's "hello": the only reaction is the keypad nudge, at most once per prompt.
   Every pre-verification clip is played with `allow_interruptions=False`, so speech cannot cut a line off and make it
   restart. A refusal (unknown caller, unavailable, locked) is spoken once, then the worker deletes the room (hang-up).
   The LLM side only ever sees an outcome word; digits are never logged (`****`).
7. **Verified.** The pre-recorded "Thank you, you're verified." plays in the chosen language. Then the session fixes
   that language: Prisma STT is swapped (`Agent.update_options(stt=...)`) for one with `language_code`
   "<language>,en-IN" (en-IN alone for English), Timbre TTS speaks the language, and every turn sends
   `reply_language` to the advisory server, whose LLM is told to write in it (`prompts.answer_language`; the graph's
   fixed templates exist in en / hi / te, so the other seven languages keep English templates). The live agent then
   greets the caller by first name (live TTS); the name comes from the phone assertion (`first_name`), which is only
   requested after verification. The tone engine starts only now; its closed-set lines exist in en / hi / te and are
   skipped for the other seven languages.

**Barge-in on the phone.** SIP sessions pass `min_interruption_duration=0.8`, `false_interruption_timeout=1.5` and
`resume_false_interruption=True` (only the keywords the installed livekit-agents accepts), so a short echo of our own
TTS on the line does not cut the agent off after verification.

**Pre-recorded auth phase.** Every line a phone caller hears before verification succeeds (greeting, language-menu
option, enter, cleared, nudge, no-match, verified, locked, timeout, unknown caller, unavailable, needs-verification) is
a clip rendered once with Timbre v3.0, voice Tarika, 16 kHz mono, in each of the ten languages (12 clips x 10), and
committed under `apps/voice/prompts/<locale>/` with a `manifest.json` (text, sha256 of the text, duration). Numbers read
aloud are written as words. `apps/voice/auth_prompts.py` looks a clip up by key (or exact text) per language; the
worker plays it as LiveKit audio frames and never calls the advisory graph or live TTS in this phase. A clip that is
missing, unreadable or rendered from different text (hash mismatch) is spoken with live TTS of the same fixed text in
its own language and logs a warning (never a digit). Re-render after changing any line (Gnani allows one active TTS
request per key, so the script renders sequentially):

```bash
GNANI_API_KEY=... python scripts/voice/render_auth_prompts.py --prune          # all ten languages, changed clips only
GNANI_API_KEY=... python scripts/voice/render_auth_prompts.py --locale te-IN --force --only enter
```

**Tone after verification.** Once a caller is verified (phone: keypad; browser: the `Voice Session` row), the worker
starts the tone engine (`apps/voice/tone_wiring.py` -> `tone_hook.py`). Each turn it plans a tier from live signals
(confusion, anger, distress, barge-in, language), sets the Timbre speed inside 0.90-1.05, speaks closed-set
acknowledgement / hand-over lines from `tone_cards.py` around the graph answer, and speaks only the whole graph
sentences that fit the tier's word and sentence budget (deferred sentences are never cut or rewritten; the page still
gets the full answer card). On a safety turn the answer is withheld and the unapproved safety line is never spoken.
Nothing in the auth phase reaches the tone engine. Not wired yet: decision prefetch from ZEN (no decision-record port
in the worker, so the hook runs without a record), `respond()` templates, and the silence / close plans.

**SIP configuration.** The provider trunk must send DTMF as RFC 4733 / RFC 2833 `telephone-event` (not in-band audio);
LiveKit SIP relays those tones to the room as `sip_dtmf_received`. If your trunk or dispatch rule has a DTMF
forwarding setting, it must be on, and the dispatch rule must not hide the caller's number (`hide_phone_number`
off), or every caller is treated as unknown.

**Phone assertion.** After verification the worker asks Frappe for an assertion at
`loan_advisory_portal.api.voice.phone_assertion` (`PhoneAssertions`) with the applicant and application the
directory record names, the room, the participant identity and the factor that was verified (`application`, never a
digit; the v1 `bank_account` factor is refused). The response adds the Customer's `first_name` when it has a usable
one, for the post-verification greeting. It
authenticates exactly like `voice_assertion` (an `Agent Integration` API key) and returns the same shape: a 60-second
assertion for that one customer and application, with thread `phone-<hash of room and identity>`. Frappe cannot see
the keypad, so the integration account vouches for the verification. Frappe refuses unless the site sets
`voice_phone_assertions: 1` in `site_config.json` (off by default), and answers an unknown application, one that
belongs to another applicant, a disabled customer or a suspended/closed borrower with the same 404. An applicant with
no portal account gets `sub = phone:<customer>`. The method is read-only, logs nothing, and only hashes the SIP
identity (which can embed the caller's number). Budgets: 600/hour per service account, 240/hour per applicant. Until a
site turns the flag on, a verified caller hears the "couldn't reach the advisory service" line (fails closed). The
"connect you to a member of our team" offer is spoken only; no SIP transfer is wired.

**Demo-grade, synthetic data only.** Four digits are 10,000 combinations and caller ID can be spoofed. This shows the
flow on synthetic applicants; it is not production authentication.

## Privacy rules that hold

- The LiveKit API secret never leaves the server; the browser gets a scoped, short-lived JWT only.
- Participant metadata carries identifiers, not credentials.
- A session exists only after the borrower accepted the notice; the version and time are on the row, and the worker refuses a row that lacks them.
- Transcripts stay on the data channel and in the borrower's open page; the audit log records turn metadata only.
- Room names are derived from a hash of customer + application, so a leaked room name does not expose who applied.
- No permanent audio retention policy is defined in this document; nothing here promises one.

## Tests (current run, 2026-10-31)

| Suite | Count | Evidence |
| --- | --- | --- |
| `tests/test_livekit.py` | 9 | signed/scoped token, `session_id` returned, consent required and recorded, foreign application 404, missing config 503, owner end + idempotent retry, expiry reported as closed, foreign/unknown end 404 |
| `tests/test_voice_assertion.py` | 11 | per-turn scoped assertion, borrower/guest refused, wrong room/identity/unknown id, expired marked, ended refused, **missing or older consent version refused without flipping status** |
| `test_authorization_matrix.py` | 8 | `end_voice_session` listed, POST-only, guest/plain-user refused |
| `test_assistant_page.py` | 8 | `voice.enabled` true when configured, false when not, version always sent |
| Island `voice.test.ts` | 5 | consent payload, keepalive end, typed refusals, redirect = signed out, borrower-facing messages |
| Island `voice-panel.test.tsx` | 7 | disabled state, consent gating + join, transcript hide/clear, end in room + server, blocked microphone closes the session, reconnect state, 401 sign-in |

Full app run: `Ran 154 tests … OK`; island `33 passed`; typecheck clean.

## Live test: BLOCKED

No successful live call is claimed. Observed on 2026-10-31:

- site config `livekit_url` is `wss://livekit.invalid` (a deliberately non-routable host);
- `docker ps` shows no LiveKit server container and no voice-worker container;
- the backend container has no worker API credential environment.

With no reachable SFU there is no room to join, so the browser, audio, transcript and hang-up paths are verified only by the unit suites above. The live test stays BLOCKED until valid credentials and a running LiveKit server/worker exist; nothing was fabricated or invented.

## Unresolved

- **Live room** — requires a real LiveKit deployment with credentials; see BLOCKED above.
- **Consent copy** — engineering now records consent and version; the exact wording is still a product/legal decision. Changing it means bumping `CONSENT_VERSION`, which invalidates older rows by design.
- **Retention** — no permanent retention policy for audio or voice transcripts is defined; the text transcript lives in the open page only, and the server-side conversation row is bounded like text chat (Phase F/H review).
- LiveKit's room-level permissions for the agent participant are unchanged from the legacy service.
