"""Newest-first admission for vLLM's priority scheduler (loaded in-process via --middleware).

Under sustained overload, first-come-first-served makes the queue grow until every new request
misses its TTFT SLO (0.5 s + 0.25 s per 1K prompt tokens). The only requests that can still be on
time are the ones that just arrived, so we admit those first; requests already stuck in the queue
are late either way and finish later (e.g. in the quiet phase). Nothing is dropped and nothing
the client sent is changed except vLLM's own `priority` field (lower = scheduled earlier).

Used with: --scheduling-policy priority --middleware newest_first.NewestFirst

Organizers confirmed on Discord (Oct 24) that server-side request prioritization is allowed.
"""
import json
import os
import time

# Optional, off by default: cap reasoning at N tokens via vLLM's thinking_token_budget. Thinking stays on;
# the server closes </think> after N tokens and the model writes its answer. The organizers' GPU mentor
# confirmed on Discord (Oct 25): "Yes, you can cap the thinking budget." Thinking is never disabled.
THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "0"))
# PRIORITY_MODE=shortest: serve shorter prompts first instead of newest first. Once the backlog is gone (INT8),
# the grader's TTFT deadline (0.5 s + 0.25 s per 1K prompt tokens) is tightest for short prompts.
PRIORITY_MODE = os.environ.get("PRIORITY_MODE", "newest")


class NewestFirst:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not (scope["type"] == "http" and scope["method"] == "POST"
                and scope["path"].endswith("/chat/completions")):
            return await self.app(scope, receive, send)

        body, more = b"", True
        while more:
            msg = await receive()
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
        try:
            req = json.loads(body)
            if PRIORITY_MODE == "shortest":
                req["priority"] = len(json.dumps(req.get("messages", []))) // 4  # ~prompt tokens; ties: arrival
            else:
                req["priority"] = -(time.monotonic_ns() // 1_000_000)  # newer arrival -> lower value
            if THINK_BUDGET > 0 and req.get("thinking_token_budget") is None:
                req["thinking_token_budget"] = THINK_BUDGET
            body = json.dumps(req).encode()
            scope = dict(scope, headers=[(k, v) for k, v in scope["headers"] if k != b"content-length"]
                         + [(b"content-length", str(len(body)).encode())])
        except (ValueError, TypeError):
            pass  # not JSON: pass through untouched and let vLLM report the error

        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
