#!/usr/bin/env python3
"""Mock OpenAI API server for testing the evaluation pipeline."""

import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            messages = body.get("messages", [])
            last_msg = messages[-1]["content"] if messages else ""

            # Detect VQA vs caption based on prompt content
            is_vqa = False
            prompt_text = ""
            if isinstance(last_msg, list):
                for part in last_msg:
                    if part.get("type") == "text":
                        prompt_text = part["text"]
                        break
            else:
                prompt_text = last_msg

            if "Index:" in prompt_text and "Answer:" in prompt_text:
                is_vqa = True

            if is_vqa:
                reply = 'Index: 0, Answer: "dummy answer"'
            else:
                reply = "This is a mock caption for testing the evaluation pipeline."

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[mock] {args[0]}")


if __name__ == "__main__":
    port = 8642
    server = HTTPServer(("127.0.0.1", port), MockOpenAIHandler)
    print(f"Mock OpenAI server running at http://127.0.0.1:{port}")
    print(f"\nUsage:")
    print(f"  export OPENAI_API_KEY=test-key")
    print(f"  export OPENAI_BASE_URL=http://127.0.0.1:{port}/v1")
    print(f"  bash run_all.sh")
    server.serve_forever()
