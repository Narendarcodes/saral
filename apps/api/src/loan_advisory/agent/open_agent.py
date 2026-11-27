"""Open advisory agent v2: no intent classifier, the whole customer file in context.

Differences from v1 (2026-11-27):
- real token streaming: the answer's sentences are emitted as the model writes them, each one
  checked against the grounded-number / leak guards before it is sent (SentenceGate semantics)
- conversation history survives a server restart: it lives in a JSON file per conversation,
  not only in process memory
- every turn appends an audit row in the same shape the intent-routed graph writes
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

import requests

from loan_advisory.agent import guards as G
from loan_advisory.agent import facts as F
from loan_advisory.agent.contracts import AgentRequest, AgentResponse, Session
from loan_advisory.agent.tools import HandoffQueue, InvalidWhatIf, NotAuthorized
from loan_advisory.state_paths import AUDIT_LOG, append_line, resolve

DISPUTE = re.compile(r"\b(dispute|wrong|incorrect|not mine|false information)\b", re.I)
FOREIGN_REF = re.compile(r"\b(?:SYN-LA|APP-TEST|ACC-LOAP-\d{4})-\d+\b", re.I)
SENTENCE = re.compile(r"[^.!?]+[.!?]+")
MAX_TOOL_ROUNDS = 3
HISTORY_TURNS = 12
SESSION_DIR = Path("/data/home/.saral_agent_sessions")

SYSTEM = """You are Saral, a loan advisor that EXPLAINS a recorded decision and never decides one. The decision was already made by a deterministic rules engine; you only explain it.

The customer's complete file (JSON) is below. Every fact you may state is in it. your_recorded_inputs are the customer's own figures (income, existing EMIs, credit score, employment) — you may quote them freely. The lender's thresholds, cut-offs, rule names and rule IDs are deliberately NOT in the file: never state, hint at, or confirm any of them, even when asked "what cut-off did I miss" — say the exact cut-off is internal. Never state a number that is not in the file, in a tool result, or in the customer's own message. Never mention rule names, rule IDs, internal codes, or system names. Never promise approval, waive a charge, or change anything.

Style: answer in {lang}. {style}Warm, plain, respectful; short sentences. When the decision is a rejection, acknowledge it before explaining. If something is not in the file, say you cannot see it.

Tools: call run_what_if ONLY when the customer explicitly proposes changed numbers; improvement_plan when they ask what to do next; search_policy for general policy questions; request_human_review when they ask for a person or you cannot help. Otherwise answer directly.

Customer file:
{context}"""

LITERACY_STYLE = {
    "low": "Very simple words. One idea per sentence. No jargon at all.",
    "medium": "Plain everyday words. Explain any term you must use.",
    "high": "Clear and direct; the customer knows loan terms.",
}
LANG_NAME = {"en": "English", "hi": "Hindi (Devanagari)", "te": "Telugu"}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "run_what_if",
        "description": "Re-estimate the decision with changed borrower inputs. ONLY these fields are supported: monthly_income and monthly_obligations (existing EMIs). Changing loan_amount or tenure is NOT available for this application; if the customer asks for that, say it is not possible.",
        "parameters": {"type": "object", "properties": {
            "changes": {"type": "object", "description": "e.g. {\"monthly_income\": 80000} or {\"monthly_obligations\": 12000}"},
        }, "required": ["changes"]}}},
    {"type": "function", "function": {
        "name": "improvement_plan",
        "description": "Get the concrete steps that could change this decision.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_policy",
        "description": "Search lender policy documents for general policy questions. Returns cited passages.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "request_human_review",
        "description": "Escalate to a human reviewer. Use when the customer asks for a person, disputes the data, or you cannot help.",
        "parameters": {"type": "object", "properties": {
            "note": {"type": "string", "description": "what the reviewer should look at"}}, "required": []}}},
]


class SentenceGate:
    """Splits streamed text into sentences and emits only the ones the guards pass.

    A sentence is held back when it carries a number that is not in the grounded set (the model
    must never speak a figure the decision record does not contain) or when it trips a leak
    pattern (rule ids, internal codes). Held-back text is never lost: it stays in the buffer and
    either a later tool result grounds it or the final draft replaces the whole answer.
    """

    def __init__(self, allowed: set[str], on_sentence):
        self.allowed = allowed
        self.on_sentence = on_sentence
        self.buffer = ""
        self.emitted = 0

    def feed(self, piece: str) -> None:
        self.buffer += piece
        for m in SENTENCE.finditer(self.buffer):
            sentence = m.group(0).strip()
            rest = self.buffer[m.end():]
            if G.ungrounded_numbers(sentence, self.allowed):
                continue  # grounded nowhere: hold it back, the final validation decides
            if G.leaked_terms(sentence):
                continue
            self.buffer = rest.lstrip()
            self.emitted += 1
            try:
                self.on_sentence(sentence)
            except Exception:
                pass
            return  # one sentence per feed: the stream paces itself

    def flush(self) -> str:
        """What never got emitted (the tail after the last clean sentence)."""
        return self.buffer.strip()


class OpenAgent:
    """One run per turn: full file in context, tools on demand, guards on the way out."""

    def __init__(self, tools, llm, policy_tools=None, profile=None, handoff=None,
                 audit_path=None, session_dir=None):
        self.tools = tools
        self.llm = llm
        self.policy_tools = policy_tools
        self.profile = profile
        self.handoff = handoff or HandoffQueue()
        self.audit_path = resolve(audit_path, "AGENT_AUDIT_LOG", AUDIT_LOG, "agent_out/turns.jsonl")
        self.session_dir = Path(session_dir) if session_dir else SESSION_DIR
        self._history = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ full customer context
    def _context(self, req: AgentRequest, record: dict | None, summary: dict, allowed: set) -> dict:
        """Everything on file for this customer, in one JSON block for the system prompt."""
        context = {"application": summary}
        values = ((record or {}).get("inputs") or {}).get("values") or {}
        if values:  # the borrower's OWN recorded data: income, EMIs, score, employment
            context["your_recorded_inputs"] = values
        if record:
            context["decision"] = {
                "outcome": F.outcome_plain(record, req.locale),
                "main_reason": reason["plain"] if (reason := F.main_reason(record, req.locale)) else None,
                "reasons": F.reason_categories(record),
                "offer": F.offer_plain(record),
            }
        else:
            context["decision"] = {"outcome": "No recorded decision is available for this application yet."}
        if allowed:
            context["numbers_you_may_state"] = sorted(allowed)
        return context

    def _fallback(self, record, req) -> str:
        """Deterministic template when the loop cannot produce a valid draft."""
        outcome = F.outcome_plain(record, req.locale) if record else \
            "I cannot see a recorded decision for this application right now."
        return (f"{outcome}\nI can share general steps that help, answer questions, "
                f"or ask a person to review it.")

    # ------------------------------------------------------------------ history (restart-safe)
    def _conv_file(self, conv_id: str) -> Path:
        return self.session_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', conv_id)}.json"

    def _remember(self, conv_id, role, text):
        with self._lock:
            h = self._history.setdefault(conv_id, [])
            h.append({"role": role, "content": text})
            del h[:-HISTORY_TURNS]
            try:
                self.session_dir.mkdir(parents=True, exist_ok=True)
                self._conv_file(conv_id).write_text(json.dumps(h, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            return list(h)

    def _recall(self, conv_id):
        with self._lock:
            if conv_id in self._history:
                return list(self._history[conv_id])
        try:
            h = json.loads(self._conv_file(conv_id).read_text(encoding="utf-8"))
            with self._lock:
                self._history[conv_id] = h
            return list(h)
        except (OSError, ValueError):
            return []

    # ------------------------------------------------------------------ tools the model may call
    def _exec_tool(self, name, args, req, session, record):
        ref = req.application_ref
        try:
            if name == "run_what_if":
                out = self.tools.run_what_if(ref, session, args.get("changes") or {}, record=record)
                return {"result": F.decision_card_borrower(out.get("record") or out) or out}
            if name == "improvement_plan":
                return {"result": self.tools.improvement_plan(ref, session, record or {})}
            if name == "search_policy":
                if self.policy_tools is None:
                    return {"result": None, "note": "policy search is not available; say you cannot quote policy and offer human review"}
                passages, _meta = self.policy_tools.search(args.get("query") or "")
                return {"result": [{"text": p["text"][:600], "citation": p["citation"].label,
                                    "title": p["citation"].title} for p in passages[:4]]}
            if name == "request_human_review":
                handoff_ref = self.handoff.create("human_review", [ref], args.get("note") or req.message,
                                                  request_id=req.request_id)
                return {"result": {"handoff_ref": handoff_ref}}
            return {"error": f"unknown tool {name}"}
        except InvalidWhatIf as exc:
            return {"error": f"invalid what-if: {exc}"}
        except NotAuthorized:
            return {"error": "not authorized"}
        except Exception as exc:  # tool failures go back to the model, never crash the turn
            return {"error": str(exc)[:200]}

    def _stream_call(self, messages, deadline, gate):
        """One streamed model call. Sentences go through the gate as they complete; tool-call
        deltas are collected and returned as an assistant message dict."""
        body = {"model": self.llm.model, "messages": messages, "tools": TOOLS_SPEC,
                "tool_choice": "auto", "max_tokens": 900, "temperature": 0.2, "stream": True}
        headers = {"Authorization": f"Bearer {self.llm._secret}", "Content-Type": "application/json"}
        content, tool_acc = [], {}
        t0 = time.perf_counter()
        with requests.post(f"{self.llm.base}/chat/completions", json=body, headers=headers,
                           timeout=max(5.0, deadline - time.perf_counter()), stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content.append(piece)
                    gate.feed(piece)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    acc = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc["name"] = fn["name"]
                    if fn.get("arguments"):
                        acc["arguments"] += fn["arguments"]
        took = round(time.perf_counter() - t0, 3)
        message = {"role": "assistant", "content": "".join(content)}
        calls = [{"id": a["id"], "type": "function",
                  "function": {"name": a["name"], "arguments": a["arguments"]}}
                 for a in tool_acc.values() if a["name"]]
        if calls:
            message["tool_calls"] = calls
        return message, {"tag": "open_turn", "model": self.llm.model, "ttft_s": None,
                         "first_token_any_s": None, "total_s": took}

    def _audit(self, resp: AgentResponse, req: AgentRequest, llm_calls, guard_flags):
        row = {"request_id": resp.request_id, "trace_id": resp.trace_id, "intent": resp.intent,
               "action": resp.action, "answer": resp.answer[:2000],
               "citations": [c.document_id for c in resp.citations],
               "evidence_refs": resp.evidence_refs, "guard_flags": guard_flags,
               "handoff_ref": resp.handoff_ref, "events": [], "timings": {"llm_calls": llm_calls}}
        try:
            append_line(self.audit_path, json.dumps(row, default=str) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ the turn
    def run(self, req: AgentRequest, session: Session, trace_context=None, on_sentence=None) -> AgentResponse:
        t0 = time.perf_counter()
        deadline = t0 + 75.0
        llm_calls: list[dict] = []
        trace_id = str(uuid.uuid4())
        emitted: list[str] = []

        # 1. deterministic boundary pre-checks (never the LLM's call)
        flags = G.request_flags(req.message, req.application_ref)
        if req.application_ref and DISPUTE.search(req.message) and not G.is_informational_question(req.message) \
                and not {"injection", "other_applicant", "misrepresentation"} & set(flags):
            flags = ["dispute"]
        if req.application_ref and FOREIGN_REF.search(req.message) \
                and re.search(re.escape(req.application_ref), req.message, re.I) is None \
                and "other_applicant" not in flags:
            flags = ["other_applicant"] + flags
        if flags:
            resp = AgentResponse(request_id=req.request_id, trace_id=trace_id, intent="boundary",
                                 action="refuse", guard_flags=flags,
                                 answer="I can't help with that request. I can explain your own loan decision, "
                                        "answer general policy questions, or ask a person to review your case.",
                                 timings={"total_ms": round((time.perf_counter() - t0) * 1000, 1), "llm_calls": []})
            self._audit(resp, req, llm_calls, flags)
            return resp

        # 2. load the whole file once (ownership re-checked inside every tool call)
        try:
            summary = self.tools.get_application_summary(req.application_ref, session) if req.application_ref else {}
            record = self.tools.get_decision_evidence(req.application_ref, session) if req.application_ref else None
        except NotAuthorized:
            resp = AgentResponse(request_id=req.request_id, trace_id=trace_id, intent="other", action="refuse",
                                 answer="I can't open that application with your current sign-in.")
            self._audit(resp, req, llm_calls, flags)
            return resp
        except Exception as exc:
            resp = AgentResponse(request_id=req.request_id, trace_id=trace_id, intent="other", action="refuse",
                                 answer="The advisory service is unavailable right now. Please try again shortly.",
                                 error=str(exc)[:200])
            self._audit(resp, req, llm_calls, flags)
            return resp

        # 3. the tool loop, streamed
        history = self._recall(req.conversation_id) + [{"role": "user", "content": req.message}]
        allowed = set(G.numbers_in(req.message))
        if record:
            allowed |= set(F.grounded_numbers(record))
            allowed |= {str(v) for v in ((record.get("inputs") or {}).get("values") or {}).values()
                        if isinstance(v, (int, float))}
        if summary:
            allowed |= set(G.numbers_in(json.dumps(summary, default=str)))
        system = SYSTEM.format(lang=LANG_NAME.get(req.locale, "English"),
                               style=LITERACY_STYLE.get(req.literacy, LITERACY_STYLE["medium"]),
                               context=json.dumps(self._context(req, record, summary, allowed),
                                                  ensure_ascii=False, indent=1))
        messages = [{"role": "system", "content": system}] + history
        gate = SentenceGate(allowed, on_sentence)

        what_if_out, plan, handoff_ref, citations = None, None, None, []
        draft, guard_flags = None, []
        messages_now = list(messages)

        for _round in range(MAX_TOOL_ROUNDS + 1):
            if time.perf_counter() > deadline - 2.0:
                break
            try:
                msg, info = self._stream_call(messages_now, deadline, gate)
            except Exception as exc:
                llm_calls.append({"tag": "open_turn", "model": getattr(self.llm, "model", "?"),
                                  "error": str(exc)[:120], "total_s": None, "ttft_s": None})
                break
            llm_calls.append(info)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                draft = (msg.get("content") or "").strip()
                break
            messages_now.append(msg)
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except ValueError:
                    args = {}
                out = self._exec_tool(name, args, req, session, record)
                if name == "run_what_if" and out.get("result"):
                    what_if_out = {"changes": args.get("changes") or {}, **(out["result"] or {})}
                    if isinstance(what_if_out.get("offer"), dict):
                        allowed |= set(F.grounded_numbers(None, extra=[what_if_out["offer"].get(k)
                                                                       for k in ("amount", "emi", "tenure_months")]))
                    allowed |= set(F.grounded_numbers(None, extra=list((args.get("changes") or {}).values())))
                    gate.allowed = allowed  # tool results may ground numbers the record lacked
                if name == "improvement_plan" and out.get("result"):
                    plan = out["result"]
                if name == "request_human_review" and out.get("result"):
                    handoff_ref = out["result"].get("handoff_ref")
                if name == "search_policy":
                    for p in out.get("result") or []:
                        allowed |= set(G.numbers_in(p["text"]))
                        citations.append(p["citation"])
                messages_now.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": json.dumps(out, ensure_ascii=False)})

        # 4. deterministic guards on the final draft
        if draft:
            guard_flags = G.answer_flags(draft)
            bad = G.ungrounded_numbers(draft, allowed)
            if bad:
                guard_flags.append("ungrounded_numbers:" + ",".join(bad[:5]))
            guard_flags += [f for f in G.output_violations(draft, decision_turn=bool(record))
                            if f not in guard_flags]
            if guard_flags:  # one retry with the violations named
                messages_now.append({"role": "user", "content": (
                    "Your draft violated these rules: " + ", ".join(guard_flags) +
                    ". Rewrite it: plain language, only numbers from the customer file or tool results, "
                    "no internal codes.")})
                try:
                    msg2, info2 = self._stream_call(messages_now, deadline, gate)
                    draft2 = (msg2.get("content") or "").strip()
                    flags2 = G.answer_flags(draft2) + [
                        f"ungrounded_numbers:{','.join(G.ungrounded_numbers(draft2, allowed)[:5])}"]
                    flags2 += [f for f in G.output_violations(draft2, decision_turn=bool(record))]
                    llm_calls.append(info2)
                    if not flags2:
                        draft, guard_flags = draft2, []
                    elif not any(f.startswith("ungrounded_numbers") for f in flags2):
                        draft, guard_flags = draft2, flags2
                except Exception:
                    pass
        tail = gate.flush()
        if tail and not draft:
            draft = tail
        if not draft or any(f.startswith("ungrounded_numbers") for f in guard_flags):
            draft, guard_flags = self._fallback(record, req), guard_flags

        if on_sentence is not None and gate.emitted == 0:
            try:
                on_sentence(draft)
                emitted.append(draft)
            except Exception:
                pass

        self._remember(req.conversation_id, "assistant", draft)
        card = F.decision_card_borrower(record) if record else None
        intent = "explain" if card else "other"
        resp = AgentResponse(request_id=req.request_id, trace_id=trace_id, intent=intent,
                             action="handoff" if handoff_ref else "answer",
                             answer=draft, citations=citations,
                             evidence_refs=[(record or {}).get("record_id")] if record else [],
                             handoff_ref=handoff_ref, decision_card=card,
                             improvement_plan=(plan.get("steps") if isinstance(plan, dict) else plan) or [],
                             what_if=what_if_out,
                             guard_flags=guard_flags,
                             timings={"total_ms": round((time.perf_counter() - t0) * 1000, 1),
                                      "llm_calls": llm_calls})
        self._audit(resp, req, llm_calls, guard_flags)
        return resp


def build_open_agent(offline: bool = True, policy_tools=None, profile=None) -> OpenAgent:
    from loan_advisory.agent.llm import ChatLLM
    if offline:
        from loan_advisory.agent.runtime import decide_offline
        from loan_advisory.infrastructure.frappe_adapter import MockStore
        from loan_advisory.agent.tools import DecisionTools, InMemoryEvidenceSource
        store = MockStore()
        sink = decide_offline(store, list(store._apps))
        tools = DecisionTools(store.reader(), InMemoryEvidenceSource(sink))
        return OpenAgent(tools, ChatLLM.from_env(), policy_tools=policy_tools, profile=profile)
    from loan_advisory.agent.runtime import build_live_agent
    _agent, _store, _sink = build_live_agent(policy_tools=policy_tools, profile=profile)
    return OpenAgent(_agent.tools, _agent.llm, policy_tools=policy_tools, profile=profile)
