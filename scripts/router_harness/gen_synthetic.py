"""Generate the SYNTHETIC router benchmark utterances with LLMs on Groq (offline; resumable).

Two independent sets so the test set never shares a generator call (or a generator model) with training:
  A (train + calibration): openai/gpt-oss-120b and openai/gpt-oss-20b, one chunk each per (intent, bucket), chat style
  B (test):                qwen/qwen3.8-27b, one chunk per (intent, bucket), voice-transcript style (ASR-like)

Every row is synthetic and carries `synthetic: true` plus its generator model. Raw replies are cached per cell under
--cache so the script can be re-run until all cells are done (each run stops before --budget seconds).

    GROQ_API_KEY is injected by the sandbox proxy (never printed or stored).
    python scripts/router_harness/gen_synthetic.py --cache /path/cache --budget 100
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import threading
import time
import urllib.error
import urllib.request

from intents import BUCKETS, DEFS, LABELS

URL = "https://api.groq.com/openai/v1/chat/completions"
PLAN = [  # (set, model, chunk id, n per cell, style)
    ("A", "openai/gpt-oss-120b", 0, 15, "chat"),
    ("A", "openai/gpt-oss-20b", 1, 15, "chat"),
    ("B", "qwen/qwen3.8-27b", 0, 15, "voice"),
]
STYLE = {
    "chat": "They are typed in a chat window. Vary length (2 to 30 words), tone (polite, blunt, anxious, casual), "
            "with and without question marks, some with small typos.",
    "voice": "They are what a speech recogniser transcribed from a PHONE CALL: mostly lowercase, little punctuation, "
             "spoken fillers (umm, actually, ji, andi, haan, sir, madam), occasional repeated words. Vary length (2 to "
             "30 words) and tone.",
}


def prompt(label, bucket, n, style):
    others = "; ".join(f"{k}: {v}" for k, v in DEFS.items() if k != label)
    return (
        "You write SYNTHETIC test data for a loan-advisory voice/chat assistant called Saral, used by personal-loan "
        "applicants in India (salaried, mostly first-time borrowers).\n"
        f"Write {n} DIFFERENT messages an applicant might send whose intent is exactly '{label}': {DEFS[label]}.\n"
        f"Language: {BUCKETS[bucket]}.\n{STYLE[style]}\n"
        "Make them realistic and varied: different openings, wordings, situations, amounts in rupees/lakh/k where "
        "relevant. Do not number them. Each must clearly belong to this intent and NOT to any of these other "
        f"intents: {others}.\n"
        'Return only JSON: {"utterances": ["...", "..."]}'
    )


class Bucket:
    """Per-model token pacing (Groq free tier: 8k tokens/min per model)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_ok = 0.0

    def wait(self):
        while True:
            with self.lock:
                d = self.next_ok - time.time()
            if d <= 0:
                return
            time.sleep(min(d, 2))

    def backoff(self, s):
        with self.lock:
            self.next_ok = max(self.next_ok, time.time() + s)


def _secs(v):
    if not v:
        return 0.0
    m = re.findall(r"([\d.]+)(ms|m|s)", v)
    return sum(float(x) * {"ms": 0.001, "s": 1, "m": 60}[u] for x, u in m)


def call(model, text, pacer, deadline):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 1.0,
            "max_tokens": 3000, "response_format": {"type": "json_object"}}
    if model.startswith("openai/"):
        body["reasoning_effort"] = "low"
    else:
        body["reasoning_effort"] = "none"
    for attempt in range(3):
        pacer.wait()
        if time.time() > deadline:
            return None
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": "saral-router-harness/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                rem = int(r.headers.get("x-ratelimit-remaining-tokens") or 8000)
                if rem < 2500:
                    pacer.backoff(_secs(r.headers.get("x-ratelimit-reset-tokens")) + 0.5)
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                pacer.backoff(_secs(e.headers.get("retry-after")) or 15)
                continue
            if e.code == 400 and "reasoning_effort" in body:  # model does not take the param
                body.pop("reasoning_effort")
                continue
            raise
    return None


def parse(content):
    try:
        d = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content or "", re.S)
        d = json.loads(m.group(0)) if m else {}
    out = d.get("utterances") if isinstance(d, dict) else d
    return [u.strip() for u in (out or []) if isinstance(u, str) and u.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--budget", type=float, default=100)
    a = ap.parse_args()
    cache = pathlib.Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.budget
    pacers = {m: Bucket() for _, m, _, _, _ in PLAN}
    jobs = []
    for st, model, chunk, n, style in PLAN:
        for label in LABELS:
            for bucket in BUCKETS:
                f = cache / f"{st}__{model.replace('/', '_')}__{chunk}__{label}__{bucket}.json"
                if not f.exists():
                    jobs.append((st, model, chunk, n, style, label, bucket, f))
    order = {(l, b): i for i, (l, b) in enumerate((l, b) for l in LABELS for b in BUCKETS)}
    jobs.sort(key=lambda j: (order[(j[5], j[6])], j[1]))  # interleave models so every rate bucket stays busy
    print(f"{len(jobs)} cells to do", flush=True)

    def run(job):
        st, model, chunk, n, style, label, bucket, f = job
        if time.time() > deadline:
            return 0
        content = call(model, prompt(label, bucket, n, style), pacers[model], deadline)
        if content is None:
            return 0
        utts = parse(content)
        if len(utts) < max(3, n // 3):
            return 0
        f.write_text(json.dumps({"set": st, "model": model, "chunk": chunk, "style": style, "label": label,
                                 "bucket": bucket, "utterances": utts}, ensure_ascii=False))
        return 1

    done = 0
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(lambda j: _safe(run, j), jobs):
            done += r
    left = len(jobs) - done
    print(f"done {done}, left {left}", flush=True)


def _safe(fn, j):
    try:
        return fn(j)
    except Exception as e:  # one failed cell is retried on the next run
        print("cell error", j[1], j[5], j[6], type(e).__name__, str(e)[:120], flush=True)
        return 0


if __name__ == "__main__":
    main()
