#!/usr/bin/env bash
# Bring up the voice stack (LiveKit bound to 127.0.0.1).
#   livekit-server: self-hosted LiveKit 1.13 (dev keys generated locally, never committed) :7880 / rtc tcp 7881 / udp 50000-50200
#   voice worker  : apps/voice/agent_worker.py inside hx-loan-agent (host network), drives the advisory graph via :8095
# Speech is Gnani's hosted API (Prisma STT + Timbre TTS through livekit-plugins-gnani); there is no local STT/TTS
# container and no GPU use for speech. The worker needs GNANI_API_KEY in its environment.
# Prereqs: hx-loan-agent container running the chat server (see docs/agent/README.md) with
#   pip install "livekit-agents[silero,gnani]==1.8.11" "livekit-plugins-gnani==1.8.11"
set -euo pipefail
ROOT=${ROOT:-/root/hx/loan}
REPO=${REPO:-$ROOT/repo-agent}
V=$ROOT/voice
mkdir -p "$V"

# --- LiveKit keys + config (local only)
if [ ! -f "$ROOT/agent/livekit.env" ]; then
  K="lk$(head -c 6 /dev/urandom | xxd -p)"; S="$(head -c 32 /dev/urandom | xxd -p -c 64)"
  printf 'LIVEKIT_URL=ws://127.0.0.1:7880\nLIVEKIT_API_KEY=%s\nLIVEKIT_API_SECRET=%s\n' "$K" "$S" > "$ROOT/agent/livekit.env"
  chmod 600 "$ROOT/agent/livekit.env"
fi
set -a; . "$ROOT/agent/livekit.env"; set +a
cat > "$V/livekit.yaml" <<EOF
port: 7880
bind_addresses: ["127.0.0.1"]
rtc:
  tcp_port: 7881
  port_range_start: 50000
  port_range_end: 50200
  use_external_ip: false
  node_ip: 127.0.0.1
keys:
  $LIVEKIT_API_KEY: $LIVEKIT_API_SECRET
logging:
  level: info
EOF
chmod 600 "$V/livekit.yaml"
pkill -f "livekit-server --config" || true
nohup "$V/livekit-server" --config "$V/livekit.yaml" > "$ROOT/logs/livekit.log" 2>&1 &

# --- voice worker (needs livekit-agents[silero,gnani] in hx-loan-agent)
# The worker exchanges its Voice Session for an assertion, so it needs the Frappe service credential, and it calls
# Gnani for speech, so it needs GNANI_API_KEY. Values are passed by name (-e VAR), never echoed.
for name in FRAPPE_URL FRAPPE_API_KEY FRAPPE_API_SECRET GNANI_API_KEY; do
  if [ -z "${!name:-}" ]; then
    echo "$name must be set for the voice worker" >&2
    exit 1
  fi
done
docker exec hx-loan-agent pkill -f agent_worker.py || true
docker exec -d -e FRAPPE_URL -e FRAPPE_API_KEY -e FRAPPE_API_SECRET -e GNANI_API_KEY \
  -e GNANI_STT_MODEL -e GNANI_STT_LANGUAGE -e GNANI_STT_FORMAT -e GNANI_TTS_MODEL -e GNANI_TTS_VOICE \
  -e GNANI_TTS_LANGUAGE -e GNANI_TTS_SPEED -e GNANI_TTS_SAMPLE_RATE -e GNANI_TTS_METHOD hx-loan-agent bash -lc "cd $REPO/apps/voice && set -a && . $ROOT/agent/livekit.env && set +a && \
  AGENT_TRACING=${AGENT_TRACING:-1} OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:6006 OTEL_SERVICE_NAME=loan-voice-worker \
  VOICE_TURN_LOG=$V/turns.jsonl exec python agent_worker.py start > $ROOT/logs/voice_worker.log 2>&1"
sleep 5
echo "voice stack up (LiveKit :7880; speech via Gnani)"
