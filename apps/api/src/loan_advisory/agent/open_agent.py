"""Open advisory agent: no intent classifier, the whole customer file in context, one LLM tool loop.

Design (2026-11-27, replaces intent-routed graph for deployments that opt in with --design open):

    receive -> authenticate (assertion, unchanged) -> boundary pre-checks (deterministic)
            -> load the customer's FULL file once (application, recorded decision, offer, reasons,
               grounded numbers) -> one OpenAI-style tool loop (run_what_if, improvement_plan,
               search_policy, request_human_review) -> deterministic guards -> answer.

Why: a closed intent set routes turn shapes, not meaning; anything it did not expect fell to
templates. Here the model sees the same facts the rules engine recorded and asks for tools when
the turn needs them. The guarantees that matter stay deterministic and outside the model:
ownership (owned_doc re-checks on every tool call), grounded numbers (validate rejects any number
not in the record), refusal boundaries (regex pre-checks, never the LLM), and fail-safe templates
when the loop cannot produce a valid draft.

The LLM still never decides: every fact it may phrase comes from the recorded decision, and
guards.validate rejects drafts that invent numbers or leak internals.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import uuid

import requests

from loan_advisory.agent import guards as G
from loan_advisory.agent import facts as F
from loan_advisory.agent.contracts import AgentRequest, AgentResponse, Session
from loan_advisory.agent.tools import HandoffQueue, InvalidWhatIf, NotAuthorized

DISPUTE = re.compile(r"\b(dispute|wrong|incorrect|not mine|false information)\b", re.I)
FOREIGN_REF = re.compile(r"\b(?:SYN-LA|APP-TEST|ACC-LOAP-\d{4})-\d+\b", re.I)
MAX_TOOL_ROUNDS = 3
HISTORY_TURNS = 12

SYSTEM = """You are Saral, a loan advisor that EXPLAINS a recorded decision and never decides one. The decision was already made by a deterministic rules engine; you only explain it.

The customer's complete file (JSON) is below. Every fact you may state is in it. Never state a number that is not in the file, in a tool result, or in the customer's own message. Never mention rule names, rule IDs, internal codes, thresholds, or system names. Never promise approval, waive a charge, or change anything.

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
            "changes": {"type": "object", "description": "e.g. {\"monthly_income\": 80000} or {\"loan_amount\": 250000}"},
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


class OpenAgent:
    """One run per turn: full file in context, tools on demand, guards on the way out."""

    def __init__(self, tools, llm, policy_tools=None, profile=None, handoff=None, checkpointer=None):
        self.tools = tools
        self.llm = llm
        self.policy_tools = policy_tools
        self.profile = profile
        self.handoff = handoff or HandoffQueue()
        self._history = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ history
    def _remember(self, conv_id, role, text):
        with self._lock:
            h = self._history.setdefault(conv_id, [])
            h.append({"role": role, "content": text})
            del h[:-HISTORY_TURNS]
            return list(h)

    def _context(self, req: AgentRequest, session: Session, record: dict | None, summary: dict) -> dict:
        ctx = {"application": summary, "policy_note": F.SYNTHETIC_LABEL}
        if record:
            ctx["outcome"] = F.outcome_plain(record, req.locale)
            reason = F.main_reason(record, req.locale)
            if reason:
                ctx["main_reason"] = reason["plain"]
            offer = F.offer_plain(record)
            if offer:
                ctx["offer"] = offer
            card = F.decision_card_borrower(record)
            if card:
                ctx["decision_card"] = card
            cats = F.reason_categories(record)
            if cats:
                ctx["reasons"] = cats
        else:
            ctx["outcome"] = "No recorded decision is available for this application yet."
        return ctx

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

    def _call_llm(self, messages, deadline):
        body = {"model": self.llm.model, "messages": messages, "tools": TOOLS_SPEC,
                "tool_choice": "auto", "max_tokens": 900, "temperature": 0.2}
        headers = {"Authorization": f"Bearer {self.llm._secret}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        r = requests.post(f"{self.llm.base}/chat/completions", json=body, headers=headers,
                          timeout=max(5.0, deadline - time.perf_counter()))
        took = round(time.perf_counter() - t0, 3)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        usage = r.json().get("usage") or {}
        info = {"tag": "open_turn", "model": self.llm.model, "ttft_s": None, "total_s": took,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        return msg, info

    def _fallback(self, record, req) -> str:
        outcome = F.outcome_plain(record, req.locale) if record else \
            "I cannot see a recorded decision for this application right now."
        return f"{outcome}\nI can share general steps that help, answer questions, or ask a person to review it."

    # ------------------------------------------------------------------ the turn
    def run(self, req: AgentRequest, session: Session, trace_context=None, on_sentence=None) -> AgentResponse:
        t0 = time.perf_counter()
        deadline = t0 + 75.0
        llm_calls: list[dict] = []
        trace_id = str(uuid.uuid4())

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
            return AgentResponse(request_id=req.request_id, trace_id=trace_id, intent="boundary",
                                 action="refuse", guard_flags=flags,
                                 answer="I can't help with that request. I can explain your own loan decision, "
                                        "answer general policy questions, or ask a person to review your case.",
                                 timings={"total_ms": round((time.perf_counter() - t0) * 1000, 1), "llm_calls": []})

        # 2. load the whole file once (ownership re-checked inside every tool call)
        try:
            summary = self.tools.get_application_summary(req.application_ref, session) if req.application_ref else {}
            record = self.tools.get_decision_evidence(req.application_ref, session) if req.application_ref else None
        except NotAuthorized:
            return AgentResponse(request_id=req.request_id, trace_id=trace_id, intent="other", action="refuse",
                                 answer="I can't open that application with your current sign-in.")
        except Exception as exc:
            return AgentResponse(request_id=req.request_id, trace_id=trace_id, intent="other", action="refuse",
                                 answer="The advisory service is unavailable right now. Please try again shortly.",
                                 error=str(exc)[:200])

        # 3. the tool loop
        history = self._remember(req.conversation_id, "user", req.message)
        system = SYSTEM.format(lang=LANG_NAME.get(req.locale, "English"),
                               style=LITERACY_STYLE.get(req.literacy, LITERACY_STYLE["medium"]),
                               context=json.dumps(self._context(req, session, record, summary),
                                                  ensure_ascii=False, indent=1))
        messages = [{"role": "system", "content": system}] + history
        what_if_out, plan, handoff_ref, citations = None, None, None, []
        allowed = set(G.numbers_in(req.message))
        if record:
            allowed |= set(F.grounded_numbers(record))
        if summary:  # application-level figures (amount, tenure) are the customer's own numbers too
            allowed |= set(G.numbers_in(json.dumps(summary, default=str)))
        draft = None

        for _round in range(MAX_TOOL_ROUNDS + 1):
            if time.perf_counter() > deadline - 2.0:
                break
            try:
                msg, info = self._call_llm(messages, deadline)
            except Exception as exc:
                llm_calls.append({"tag": "open_turn", "model": getattr(self.llm, "model", "?"),
                                  "error": str(exc)[:120], "total_s": None, "ttft_s": None})
                break
            llm_calls.append(info)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                draft = (msg.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": [{"id": tc["id"], "type": "function",
                                             "function": {"name": tc["function"]["name"],
                                                          "arguments": tc["function"]["arguments"]}}
                                            for tc in tool_calls]})
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
                if name == "improvement_plan" and out.get("result"):
                    plan = out["result"]
                if name == "request_human_review" and out.get("result"):
                    handoff_ref = out["result"].get("handoff_ref")
                if name == "search_policy":
                    for p in out.get("result") or []:
                        allowed |= set(G.numbers_in(p["text"]))
                        citations.append(p["citation"])
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out, ensure_ascii=False)})

        # 4. deterministic guards on the draft
        guard_flags: list[str] = []
        if draft:
            guard_flags += G.answer_flags(draft)
            bad = G.ungrounded_numbers(draft, allowed)
            if bad:
                guard_flags.append("ungrounded_numbers:" + ",".join(bad[:5]))
            guard_flags += [f for f in G.output_violations(draft, decision_turn=bool(record)) if f not in guard_flags]
        if draft and guard_flags:  # one retry with the violations named
            retry_note = ("Your draft violated these rules: " + ", ".join(guard_flags) +
                          ". Rewrite it: plain language, only numbers from the customer file or tool results, "
                          "no internal codes.")
            messages.append({"role": "user", "content": retry_note})
            try:
                msg, info = self._call_llm(messages, deadline)
                draft2 = (msg.get("content") or "").strip()
                flags2 = G.answer_flags(draft2) + [
                    f"ungrounded_numbers:{','.join(G.ungrounded_numbers(draft2, allowed)[:5])}"]
                flags2 += [f for f in G.output_violations(draft2, decision_turn=bool(record))]
                llm_calls.append(info)
                if not [f for f in flags2 if f]:
                    draft, guard_flags = draft2, []
                elif not any(f.startswith("ungrounded_numbers") for f in flags2):
                    draft, guard_flags = draft2, flags2
            except Exception:
                pass
        if not draft or any(f.startswith("ungrounded_numbers") for f in guard_flags):
            draft, guard_flags = self._fallback(record, req), guard_flags

        if on_sentence is not None:  # NDJSON stream: the whole answer as one grounded chunk
            try:
                on_sentence(draft)
            except Exception:
                pass

        self._remember(req.conversation_id, "assistant", draft)
        card = F.decision_card_borrower(record) if record else None
        intent = "explain" if card else "other"
        return AgentResponse(request_id=req.request_id, trace_id=trace_id, intent=intent,
                             action="handoff" if handoff_ref else "answer",
                             answer=draft, citations=citations,
                             evidence_refs=[(record or {}).get("record_id")] if record else [],
                             handoff_ref=handoff_ref, decision_card=card,
                             improvement_plan=plan or [], what_if=what_if_out,
                             guard_flags=guard_flags,
                             timings={"total_ms": round((time.perf_counter() - t0) * 1000, 1),
                                      "llm_calls": llm_calls})


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
