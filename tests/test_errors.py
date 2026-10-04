from nexus_rag.errors import describe


def test_an_error_is_described_by_its_message():
    assert describe(ValueError("no such page")) == "no such page"


def test_an_error_without_a_message_is_described_by_its_type():
    assert describe(TimeoutError()) == "TimeoutError"
