"""Tiny OpenAI-style SSE mock: thinks, then answers; calls a tool once if tools are given."""
import json, re, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send_json({"data": [{"id": "mock-model"}]})

    def _send_json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        def ev(o):
            self.wfile.write(f"data: {json.dumps(o)}\n\n".encode()); self.wfile.flush()
        msgs = body["messages"]
        ev({"choices": [{"delta": {"role": "assistant"}, "index": 0}]})
        time.sleep(0.05)
        for _ in range(5):
            ev({"choices": [{"delta": {"reasoning_content": "hmm "}, "index": 0}]}); time.sleep(0.01)
        if body.get("tools") and msgs[-1]["role"] == "user":
            ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": ""}}]}, "index": 0}]})
            ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\": \"Oslo\"}"}}]}, "index": 0}]})
            fin, text = "tool_calls", None
        else:
            q = msgs[0]["content"]
            if msgs[-1]["role"] == "tool":
                text = "The answer is " + str(json.loads(msgs[-1]["content"]).get("temp_c"))
            elif "budget code" in q:
                nm = re.search(r"project (\w+)\?", q).group(1)
                text = re.search(rf"project {nm} is (\d+)", q).group(1)
            elif "Answer with the letter" in q:
                text = "Answer: B"
            else:
                text = "42"
            fin = "stop"
        if text:
            for w in text.split(" "):
                ev({"choices": [{"delta": {"content": w + " "}, "index": 0}]}); time.sleep(0.01)
        ev({"choices": [{"delta": {}, "finish_reason": fin, "index": 0}]})
        ev({"choices": [], "usage": {"prompt_tokens": len(json.dumps(msgs)) // 4, "completion_tokens": 12}})
        self.wfile.write(b"data: [DONE]\n\n")


ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
