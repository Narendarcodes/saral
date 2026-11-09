/* Web phone (/borrower/call): a phone call to Saral over LiveKit WebRTC.
 *
 * The server mints the token (api/web_phone.start_web_call) for a fresh `webcall-` room with the caller ID set as a
 * participant attribute, so this page never sends a phone number. The voice worker then runs the same flow as a real
 * phone call: greeting, press 9 for the language menu, the last 4 digits of the application number, then the agent.
 *
 * Keypad presses go out as telephone events: livekit-client `publishDtmf(code, digit)` (RFC 4733 codes: 0-9 -> 0-9,
 * * -> 10, # -> 11), which the worker receives as `sip_dtmf_received`, exactly like a SIP caller's. A site can switch
 * to the fallback (`keypad: "data"`): a reliable data packet on topic "dtmf" with {"digit"}. Keyed digits are shown
 * as dots, never logged, and never sent anywhere else. A short dual-tone plays locally for each press.
 *
 * livekit-client is the copy the assistant island already ships (an ES module chunk), loaded only on "Call".
 */
const DTMF_CODES = { "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "*": 10, "#": 11 };
const DTMF_TONES = {
	"1": [697, 1209], "2": [697, 1336], "3": [697, 1477],
	"4": [770, 1209], "5": [770, 1336], "6": [770, 1477],
	"7": [852, 1209], "8": [852, 1336], "9": [852, 1477],
	"*": [941, 1209], "0": [941, 1336], "#": [941, 1477],
};
const ANSWER_TIMEOUT_MS = 25000;
const METHOD = "loan_advisory_portal.api.web_phone.";

const config = JSON.parse(document.getElementById("web-phone-context").textContent || "{}");
const screen = document.getElementById("webphone");
const statusLine = document.getElementById("webphone-status");
const timerLine = document.getElementById("webphone-timer");
const digitsLine = document.getElementById("webphone-digits");
const keypad = document.getElementById("webphone-keypad");
const callButton = document.getElementById("webphone-call");
const hangupButton = document.getElementById("webphone-hangup");
const muteButton = document.getElementById("webphone-mute");
const keysButton = document.getElementById("webphone-keys");
const speakerButton = document.getElementById("webphone-speaker");

let room = null;
let roomName = null;
let lk = null;
let audioContext = null;
let ringback = null;
let answerTimer = 0;
let tick = 0;
let startedAt = 0;
let muted = false;
let loud = true;
let keypadMode = "dtmf";

function setState(state, text) {
	screen.dataset.state = state;
	if (text) statusLine.textContent = text;
	const live = state === "connected";
	for (const key of keypad.querySelectorAll("[data-key]")) key.disabled = !live;
	muteButton.disabled = !live;
	speakerButton.disabled = !live;
	const inCall = state === "connecting" || state === "ringing" || state === "connected";
	callButton.hidden = inCall;
	hangupButton.hidden = !inCall;
	callButton.disabled = false;
}

function ensureAudio() {
	if (!audioContext) {
		const Ctx = window.AudioContext || window.webkitAudioContext;
		if (Ctx) audioContext = new Ctx();
	}
	if (audioContext && audioContext.state === "suspended") audioContext.resume();
	return audioContext;
}

function tone(freqs, seconds, gain = 0.08) {
	const ctx = ensureAudio();
	if (!ctx) return null;
	const out = ctx.createGain();
	out.gain.value = gain;
	out.connect(ctx.destination);
	const oscillators = freqs.map((f) => {
		const o = ctx.createOscillator();
		o.frequency.value = f;
		o.connect(out);
		o.start();
		if (seconds) o.stop(ctx.currentTime + seconds);
		return o;
	});
	return () => {
		for (const o of oscillators) {
			try {
				o.stop();
			} catch (err) {
				/* already stopped */
			}
		}
		out.disconnect();
	};
}

/* Indian ring-back cadence (400 Hz, 0.4 s on, 0.2 s off, 0.4 s on, 2 s off) while the call is being answered. */
function startRingback() {
	stopRingback();
	let stopped = false;
	let current = null;
	const pattern = [[400, 0.4], [0, 0.2], [400, 0.4], [0, 2.0]];
	let i = 0;
	const step = () => {
		if (stopped) return;
		const [f, s] = pattern[i % pattern.length];
		i += 1;
		if (f) current = tone([f, f * 1.0625], s, 0.05);
		window.setTimeout(step, s * 1000);
	};
	step();
	ringback = () => {
		stopped = true;
		if (current) current();
	};
}

function stopRingback() {
	if (ringback) ringback();
	ringback = null;
}

function startTimer() {
	startedAt = Date.now();
	const render = () => {
		const s = Math.floor((Date.now() - startedAt) / 1000);
		timerLine.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
	};
	render();
	tick = window.setInterval(render, 1000);
}

function stopTimer() {
	if (tick) window.clearInterval(tick);
	tick = 0;
}

function remoteAudio() {
	return document.querySelectorAll("[data-webphone-remote]");
}

function answered() {
	if (screen.dataset.state === "connected") return;
	window.clearTimeout(answerTimer);
	stopRingback();
	setState("connected", "Connected");
	startTimer();
}

async function startCall() {
	callButton.disabled = true;
	ensureAudio();
	digitsLine.textContent = "";
	setState("connecting", "Calling…");
	let grant;
	try {
		grant = await window.Portal.call(METHOD + "start_web_call", {});
	} catch (err) {
		setState("idle", err && err.message ? err.message : "The call could not be started.");
		return;
	}
	keypadMode = grant.keypad === "data" ? "data" : "dtmf";
	roomName = grant.room;
	try {
		lk = lk || (await import(config.livekit));
		room = new lk.Room({
			audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
		});
		room.on(lk.RoomEvent.TrackSubscribed, (track) => {
			if (track.kind !== lk.Track.Kind.Audio) return;
			const element = track.attach();
			element.dataset.webphoneRemote = "audio";
			element.volume = loud ? 1 : 0.5;
			document.body.append(element);
			answered();
		});
		room.on(lk.RoomEvent.TrackUnsubscribed, (track) => {
			for (const element of track.detach()) element.remove();
		});
		room.on(lk.RoomEvent.Disconnected, () => finish("Call ended"));
		setState("ringing", "Ringing…");
		startRingback();
		answerTimer = window.setTimeout(() => {
			if (screen.dataset.state !== "connected") hangUp("Saral did not answer. Try again in a moment.");
		}, ANSWER_TIMEOUT_MS);
		await room.connect(grant.url, grant.token);
		await room.startAudio();
		await room.localParticipant.setMicrophoneEnabled(true);
	} catch (err) {
		const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
		await hangUp(denied ? "Allow the microphone to call Saral." : "The call could not connect.");
	}
}

async function hangUp(text) {
	const current = room;
	room = null;
	finish(text || "Call ended");
	if (current) {
		try {
			await current.disconnect();
		} catch (err) {
			/* already gone */
		}
	}
}

function release() {
	if (!roomName) return;
	const name = roomName;
	roomName = null;
	// keepalive so the one-call lock is released even when the tab is closing.
	fetch("/api/method/" + METHOD + "end_web_call", {
		method: "POST",
		keepalive: true,
		credentials: "same-origin",
		headers: {
			"Content-Type": "application/json",
			Accept: "application/json",
			"X-Frappe-CSRF-Token": (window.Portal && window.Portal.boot.csrf_token) || "",
		},
		body: JSON.stringify({ room: name }),
	}).catch(() => {});
}

function finish(text) {
	window.clearTimeout(answerTimer);
	stopRingback();
	stopTimer();
	for (const element of remoteAudio()) element.remove();
	if (screen.dataset.state !== "idle" && screen.dataset.state !== "ended") {
		setState("ended", text);
		tone([480, 620], 0.25, 0.05);
	} else if (text) {
		statusLine.textContent = text;
	}
	muted = false;
	muteButton.setAttribute("aria-pressed", "false");
	release();
}

function press(key) {
	if (!(key in DTMF_CODES) || screen.dataset.state !== "connected" || !room) return;
	tone(DTMF_TONES[key], 0.12);
	digitsLine.textContent = (digitsLine.textContent + "•").slice(-12);
	const button = keypad.querySelector(`[data-key="${CSS.escape(key)}"]`);
	if (button) {
		button.setAttribute("data-pressed", "");
		window.setTimeout(() => button.removeAttribute("data-pressed"), 120);
	}
	const participant = room.localParticipant;
	const sent =
		keypadMode === "data"
			? participant.publishData(new TextEncoder().encode(JSON.stringify({ digit: key })), {
					reliable: true,
					topic: "dtmf",
				})
			: participant.publishDtmf(DTMF_CODES[key], key);
	Promise.resolve(sent).catch(() => {
		statusLine.textContent = "That key did not go through. Press it again.";
	});
}

callButton.addEventListener("click", startCall);
hangupButton.addEventListener("click", () => hangUp("Call ended"));
keypad.addEventListener("click", (event) => {
	const button = event.target.closest("[data-key]");
	if (button) press(button.dataset.key);
});
document.addEventListener("keydown", (event) => {
	if (event.ctrlKey || event.metaKey || event.altKey) return;
	if (event.key in DTMF_CODES) press(event.key);
});
muteButton.addEventListener("click", async () => {
	if (!room) return;
	muted = !muted;
	muteButton.setAttribute("aria-pressed", String(muted));
	await room.localParticipant.setMicrophoneEnabled(!muted);
	statusLine.textContent = muted ? "Muted" : "Connected";
});
keysButton.addEventListener("click", () => {
	keypad.hidden = !keypad.hidden;
	keysButton.setAttribute("aria-pressed", String(!keypad.hidden));
});
speakerButton.addEventListener("click", () => {
	loud = !loud;
	speakerButton.setAttribute("aria-pressed", String(loud));
	for (const element of remoteAudio()) element.volume = loud ? 1 : 0.5;
});
window.addEventListener("pagehide", () => {
	if (room) room.disconnect();
	release();
});
