"""States of a Knowledge index in Qdrant that its interface never produces: damage, and the
layouts earlier versions wrote. Tests reach them through the Qdrant client they share with the
index, naming collections through ``IndexNames`` rather than spelling them out."""
from qdrant_client import QdrantClient, models

from local_rag import Settings
from local_rag.chunker import Piece
from local_rag.corpus import Document
from local_rag.gateway import ModelGateway
from local_rag.generations import Generations
from local_rag.index_format import IndexFormat
from local_rag.index_names import IndexNames
from local_rag.passage_store import PassageStore, Written
from local_rag.sparse import SparseEncoder

NAMES = IndexNames(Settings().collection)


def collections(client: QdrantClient) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def generations_in(client: QdrantClient) -> set[str]:
    return {name for name in collections(client) if NAMES.is_generation(name)}


def legacy_index(client: QdrantClient, gateway: ModelGateway,
                 documents: list[Document]) -> None:
    """The plain collection earlier versions kept as the whole index: no generations, no
    record of them or of the format, and Documents owned by no Corpus."""
    store, sparse = PassageStore(client, payload_indexes=False), SparseEncoder(Settings().sparse)
    contents = [f"{document.title}\n\n{document.text}" for document in documents]
    vectors = gateway.embed(contents)
    store.create(NAMES.plain, dimension=len(vectors[0]))
    for document, content, vector in zip(documents, contents, vectors):
        store.write(NAMES.plain, document,
                    [Written(Piece(document.text), sparse.encode_document(content), vector)])


def record_format(client: QdrantClient, generation: str, index_format: IndexFormat) -> None:
    """Overwrite what the registry says a generation was built with."""
    Generations(client, NAMES).record(generation, index_format)


def registry_payload(client: QdrantClient) -> dict:
    return client.retrieve(NAMES.registry, ids=[1])[0].payload


def overwrite_registry(client: QdrantClient, payload: dict) -> None:
    client.upsert(NAMES.registry, points=[
        models.PointStruct(id=1, vector=[1.0], payload=payload)], wait=True)


def lose_the_registry(client: QdrantClient) -> None:
    client.delete_collection(NAMES.registry)


def lose_the_collection(client: QdrantClient, generation: str) -> None:
    """What losing a generation's data in Qdrant leaves behind (its aliases go with it)."""
    client.delete_collection(generation)


def replace_with_a_non_index(client: QdrantClient, generation: str) -> None:
    """The Live index's alias pointing at a collection of the same name that is not an index."""
    client.delete_collection(generation)
    client.create_collection(generation, vectors_config=models.VectorParams(
        size=1, distance=models.Distance.COSINE))
    client.update_collection_aliases(change_aliases_operations=[
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name=generation, alias_name=NAMES.live))])


def strip_from_every_point(client: QdrantClient, field: str) -> None:
    """Points of the Live index that lost a field every version writes."""
    client.delete_payload(NAMES.live, keys=[field], points=models.Filter())


def overwrite_the_last_sync(client: QdrantClient, payload: dict) -> None:
    client.upsert(NAMES.last_sync, points=[
        models.PointStruct(id=1, vector=[1.0], payload=payload)], wait=True)


def take_over_the_lock(client: QdrantClient) -> None:
    """What another run taking over a stale index-wide lock does to its holder."""
    client.delete_collection(NAMES.lock)
