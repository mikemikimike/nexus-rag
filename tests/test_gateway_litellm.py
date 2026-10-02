import pytest

from fake_litellm_server import FakeLiteLLM
from local_rag import Settings
from local_rag.fakes import FakeModelGateway
from local_rag.gateway_litellm import GatewayError, LiteLLMGateway

from conftest import SYNONYMS


@pytest.fixture
def server():
    fake = FakeLiteLLM(FakeModelGateway(synonyms=SYNONYMS)).start()
    yield fake
    fake.stop()


def make_gateway(server, **overrides):
    settings = dict(gateway_base_url=server.base_url, gateway_api_key=server.api_key,
                    embed_model="embed", rerank_model="rerank", embed_batch_size=2)
    settings.update(overrides)
    return LiteLLMGateway.from_settings(Settings(**settings))


def test_embed_returns_one_vector_per_text_in_order(server):
    gateway = make_gateway(server)
    texts = ["battery is dead", "ORA-00942 table missing", "car will not start"]

    vectors = gateway.embed(texts)

    assert vectors == server.embedder.embed(texts)


def test_embed_sends_texts_in_batches_of_the_configured_size(server):
    gateway = make_gateway(server, embed_batch_size=2)

    gateway.embed(["a1", "b2", "c3", "d4", "e5"])

    sent = [r["body"]["input"] for r in server.requests]
    assert sent == [["a1", "b2"], ["c3", "d4"], ["e5"]]


def test_embed_names_the_model_and_authenticates(server):
    gateway = make_gateway(server, embed_model="local-embed")

    gateway.embed(["hello"])

    [request] = server.requests
    assert request["path"] == "/v1/embeddings"
    assert request["body"]["model"] == "local-embed"
    assert request["auth"] == "Bearer sk-test"


def test_embed_of_nothing_makes_no_request(server):
    assert make_gateway(server).embed([]) == []
    assert server.requests == []


def test_embed_query_returns_the_query_vector(server):
    gateway = make_gateway(server)

    assert gateway.embed_query("vehicle refuses to ignite") == \
        server.embedder.embed_query("vehicle refuses to ignite")


def test_gateway_error_reports_status_and_message_without_the_api_key(server):
    server.fail_with = (500, '{"error": "model overloaded"}')
    gateway = make_gateway(server)

    with pytest.raises(GatewayError) as error:
        gateway.embed(["hello"])

    assert "500" in str(error.value)
    assert "model overloaded" in str(error.value)
    assert "sk-test" not in str(error.value)


def test_wrong_api_key_is_a_clear_error(server):
    gateway = make_gateway(server, gateway_api_key="sk-wrong")

    with pytest.raises(GatewayError, match="401"):
        gateway.embed(["hello"])


def test_non_json_answer_is_a_gateway_error(server):
    server.garbage = True

    with pytest.raises(GatewayError, match="not valid JSON"):
        make_gateway(server).embed(["hello"])


def test_unreachable_gateway_is_a_gateway_error(server):
    gateway = make_gateway(server)
    server.stop()

    with pytest.raises(GatewayError, match="unreachable"):
        gateway.embed(["hello"])


def test_answer_with_the_wrong_number_of_embeddings_is_a_gateway_error(server):
    gateway = make_gateway(server)
    server.embedder = type("Short", (), {"embed": lambda self, texts: [[1.0]] * (len(texts) - 1)})()

    with pytest.raises(GatewayError, match="expected 2 embeddings"):
        gateway.embed(["a1", "b2"])


def test_rerank_returns_scores_aligned_with_the_texts(server):
    gateway = make_gateway(server)
    texts = ["nothing relevant", "the battery is dead", "battery"]

    scores = gateway.rerank("dead battery", texts)

    assert scores == [0.0, 2.0, 1.0]
    [request] = server.requests
    assert request["path"] == "/v1/rerank"
    assert request["body"]["model"] == "rerank"
    assert request["body"]["query"] == "dead battery"


def test_rerank_truncates_very_long_texts(server):
    make_gateway(server).rerank("q", ["x" * 10_000, "short"])

    [request] = server.requests
    assert len(request["body"]["documents"][0]) < 10_000
    assert request["body"]["documents"][1] == "short"


def test_rerank_failure_raises_so_the_caller_can_fall_back(server):
    server.fail_with = (503, "rerank backend down")

    with pytest.raises(GatewayError, match="503"):
        make_gateway(server).rerank("q", ["a", "b"])


def test_rerank_answer_missing_a_text_raises(server):
    server.rerank_results = [{"index": 0, "relevance_score": 0.9}]

    with pytest.raises(GatewayError, match="rerank"):
        make_gateway(server).rerank("q", ["a", "b"])


def test_rerank_without_a_configured_model_raises_without_calling_out(server):
    gateway = make_gateway(server, rerank_model=None)

    with pytest.raises(GatewayError, match="no rerank model"):
        gateway.rerank("q", ["a"])
    assert server.requests == []


def test_count_tokens_needs_no_network_and_is_conservative(server):
    gateway = make_gateway(server)

    assert gateway.count_tokens("ORA-00942: table missing.") == 7  # ORA - 00942 : table missing .
    assert gateway.count_tokens("") == 0
    assert server.requests == []


def test_base_url_must_be_http(server):
    with pytest.raises(ValueError, match="http"):
        make_gateway(server, gateway_base_url="file:///etc/passwd")


def test_gateway_ignores_proxy_environment_variables(server, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    gateway = make_gateway(server)

    vectors = gateway.embed(["battery is dead"])

    assert vectors == server.embedder.embed(["battery is dead"])


def test_a_query_instruction_goes_in_front_of_the_query_in_the_trained_format(server):
    gateway = make_gateway(server, query_instruction="Find the passage that answers the question")

    gateway.embed_query("vehicle refuses to ignite")

    [request] = server.requests
    assert request["body"]["input"] == [
        "Instruct: Find the passage that answers the question\nQuery: vehicle refuses to ignite"]


def test_document_embeddings_never_get_the_query_instruction(server):
    gateway = make_gateway(server, query_instruction="Find the passage that answers the question")

    gateway.embed(["battery is dead", "car will not start"])

    [request] = server.requests
    assert request["body"]["input"] == ["battery is dead", "car will not start"]


def test_without_a_query_instruction_the_query_is_sent_unchanged(server):
    make_gateway(server).embed_query("vehicle refuses to ignite")

    [request] = server.requests
    assert request["body"]["input"] == ["vehicle refuses to ignite"]
