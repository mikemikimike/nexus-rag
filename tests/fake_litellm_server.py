"""A local HTTP server speaking the OpenAI-compatible /embeddings and LiteLLM /rerank endpoints."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nexus_rag.fakes import FakeModelGateway
from nexus_rag.sparse import tokenize


class FakeLiteLLM:
    """Embeds with a FakeModelGateway; rerank scores by query-word overlap.

    Set `fail_with = (status, body)` to make every request fail (or, with `fail_for = n`,
    only the next n), `garbage = True` to answer with a body that is not JSON. Every request
    is recorded in `requests`.
    """

    def __init__(self, embedder: FakeModelGateway | None = None, api_key: str = "sk-test"):
        self.embedder = embedder or FakeModelGateway()
        self.api_key = api_key
        self.requests: list[dict] = []
        self.fail_with: tuple[int, str] | None = None
        self.fail_for: int | None = None  # requests left to fail; None fails them all
        self.garbage = False
        self.rerank_results: list[dict] | None = None  # overrides the computed results
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def start(self) -> "FakeLiteLLM":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(
                    {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
                )
                if fake.fail_with:
                    status, text = fake.fail_with
                    if fake.fail_for is not None:
                        fake.fail_for -= 1
                        if fake.fail_for <= 0:
                            fake.fail_with = fake.fail_for = None
                    return self._send(status, text.encode())
                if fake.garbage:
                    return self._send(200, b"<html>not json</html>")
                if self.headers.get("Authorization") != f"Bearer {fake.api_key}":
                    return self._send(401, b'{"error": "bad key"}')
                if self.path == "/v1/embeddings":
                    return self._json(fake._embeddings(body))
                if self.path == "/v1/rerank":
                    return self._json(fake._rerank(body))
                self._send(404, b'{"error": "unknown endpoint"}')

            def _json(self, payload):
                self._send(200, json.dumps(payload).encode())

            def _send(self, status, data):
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def _embeddings(self, body):
        inputs = body["input"]
        vectors = self.embedder.embed(inputs)
        # Answer out of order, as a real server is allowed to: clients must use "index".
        data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]
        return {"object": "list", "model": body["model"], "data": data[::-1]}

    def _rerank(self, body):
        if self.rerank_results is not None:
            return {"results": self.rerank_results}
        query = set(tokenize(body["query"]))
        results = [
            {"index": i, "relevance_score": float(len(query & set(tokenize(doc))))}
            for i, doc in enumerate(body["documents"])
        ]
        results.sort(key=lambda r: -r["relevance_score"])  # best first, like Cohere
        return {"results": results}
