"""Stand-ins for the infrastructure the acceptance suite is run against, and how to run it."""

import io

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from local_rag import acceptance, acceptance_corpus
from local_rag.fakes import FakeModelGateway

QDRANT_KEY = "qdrant-secret-key"
MCP_TOKEN = "mcp-secret-token"
GATEWAY_KEY = "gateway-secret-key"
SECRETS = (QDRANT_KEY, MCP_TOKEN, GATEWAY_KEY, "second-secret-token")

ENVIRONMENT = {
    "LITELLM_BASE_URL": "http://gateway.example.test:4000/v1",
    "LITELLM_API_KEY": GATEWAY_KEY,
    "RAG_EMBED_MODEL": "embed-model",
    "QDRANT_URL": "http://qdrant.example.test:6333",
    "QDRANT_API_KEY": QDRANT_KEY,
    "RAG_COLLECTION": "knowledge",
    "MCP_TOKENS": f"{MCP_TOKEN},second-secret-token",
}


def probe_synonyms() -> dict[str, set[str]]:
    """What a real embedding model knows: each probe's paraphrase means its Document."""
    return {
        f"concept-{number}": set(words) for number, words in enumerate(acceptance_corpus.MEANINGS)
    }


def good_gateway() -> FakeModelGateway:
    return FakeModelGateway(synonyms=probe_synonyms())


class RefusingQdrant:
    """A Qdrant client whose server answers every request with 403 (a wrong API key)."""

    def __getattr__(self, name):
        def refuse(*args, **kwargs):
            raise UnexpectedResponse(403, "Forbidden", b"Invalid api-key", {})

        return refuse


class Qdrant:
    """One in-process Qdrant standing in for a server that checks the API key."""

    def __init__(self, key: str = QDRANT_KEY, *, accept_any_key: bool = False):
        self.client = QdrantClient(":memory:")
        self.key, self.accept_any_key = key, accept_any_key
        self.connections: list[str | None] = []

    def connect(self, location: str, api_key: str | None):
        self.connections.append(api_key)
        if self.accept_any_key or api_key == self.key:
            return self.client
        return RefusingQdrant()

    def names(self) -> set[str]:
        return {c.name for c in self.client.get_collections().collections}


class Host:
    """Stands in for the commands the suite runs (confluence, systemctl, docker)."""

    def __init__(self, answers=None, failing=()):
        self.answers = dict(answers or {})
        self.failing = set(failing)
        self.calls: list[list[str]] = []

    def __call__(self, command):
        self.calls.append(list(command))
        joined = " ".join(command)
        for needle in self.failing:
            if needle in joined:
                raise RuntimeError(f"{command[0]} failed: {needle}")
        for needle, answer in self.answers.items():
            if needle in joined:
                return answer
        return ""


def run_suite(*argv, environ=None, gateway=None, qdrant=None, host=None, no_gateway=False):
    """Run the suite offline; returns (exit code, stdout, stderr).

    ``no_gateway`` leaves the suite to build the Model gateway itself, as a real run does."""
    out, err = io.StringIO(), io.StringIO()
    qdrant = qdrant or Qdrant()
    code = acceptance.main(
        list(argv),
        {**ENVIRONMENT, **(environ or {})},
        gateway=None if no_gateway else gateway or good_gateway(),
        connect=qdrant.connect,
        run=host or Host(),
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def line(output: str, name: str) -> str:
    """The one line of the output that reports the check ``name``."""
    found = [l for l in output.splitlines() if f" {name}:" in l]
    assert len(found) == 1, (name, output)
    return found[0]


TIMERS = (
    "Thu 2026-10-01 13:00:00 UTC  40min left Thu 2026-10-01 12:00:01 UTC  19min ago {sync}.timer {sync}.service\n"
    "Sun 2026-10-04 03:00:00 UTC  2 days left n/a                          n/a      {full}.timer {full}.service\n"
)


def systemd_host(**kwargs) -> Host:
    return Host(
        {
            "stat -c %a": "640\n",
            "list-timers": TIMERS.format(sync="local-rag-sync", full="local-rag-full"),
        },
        **kwargs,
    )


def container_host(**kwargs) -> Host:
    return Host(
        {
            "stat -c %a": "600\n",
            "docker ps": "mcp Up 3 hours (healthy)\nqdrant Up 3 hours\n",
            "list-timers": TIMERS.format(
                sync="local-rag-container-sync", full="local-rag-container-full"
            ),
        },
        **kwargs,
    )
