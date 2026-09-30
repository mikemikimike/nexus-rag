import pytest

from local_rag import KnowledgeIndex, Settings
from local_rag.corpus import Document, InMemoryCorpus
from local_rag.fakes import FakeModelGateway

SYNONYMS = {
    "vehicle": {"vehicle", "car", "automobile"},
    "start": {"start", "ignite", "ignition", "crank"},
    "broken": {"broken", "dead", "failed", "refuses"},
}

DOCUMENTS = [
    Document(
        id="dba/ora-00942",
        title="Fixing ORA-00942",
        url="https://wiki.example.test/dba/ora-00942",
        space="DBA",
        version="1",
        text="ORA-00942: table or view does not exist. Check the schema owner and grants.",
    ),
    Document(
        id="dba/ora-12154",
        title="Fixing ORA-12154",
        url="https://wiki.example.test/dba/ora-12154",
        space="DBA",
        version="1",
        text="ORA-12154: TNS could not resolve the connect identifier. Check tnsnames.",
    ),
    Document(
        id="garage/battery",
        title="Car battery checks",
        url="https://wiki.example.test/garage/battery",
        space="GARAGE",
        version="1",
        text="When the car will not start, the battery is dead more often than not.",
    ),
]


@pytest.fixture
def gateway():
    return FakeModelGateway(synonyms=SYNONYMS)


@pytest.fixture
def index(gateway):
    return KnowledgeIndex(Settings(), gateway)


@pytest.fixture
def corpus():
    return InMemoryCorpus(DOCUMENTS)
