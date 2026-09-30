"""Qdrant helpers. Search always goes through an ALIAS so rebuilds swap atomically."""
import uuid
import warnings

import config
from qdrant_client import QdrantClient, models


def get_client() -> QdrantClient:
    if config.QDRANT_URL.startswith("http://") and any(
            h in config.QDRANT_URL for h in ("127.0.0.1", "localhost")):
        warnings.filterwarnings("ignore", message=".*insecure connection.*")
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None, timeout=60)


def resolve(client: QdrantClient):
    """Real collection name behind the alias, or None if there is no index yet."""
    for a in client.get_aliases().aliases:
        if a.alias_name == config.COLLECTION:
            return a.collection_name
    return None


def create_collection(client: QdrantClient, name: str, dim: int) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("source_id", "space"):
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def swap_alias(client: QdrantClient, new: str, had_alias: bool) -> None:
    ops = []
    if had_alias:
        ops.append(models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name=config.COLLECTION)))
    ops.append(models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=new, alias_name=config.COLLECTION)))
    client.update_collection_aliases(change_aliases_operations=ops)


def point_id(source_id: str, i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}#{i}"))


def delete_source(client: QdrantClient, coll: str, source_id: str) -> None:
    client.delete(
        collection_name=coll,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))])),
        wait=True,
    )


def upsert(client: QdrantClient, coll: str, points: list, batch: int = 64) -> None:
    for i in range(0, len(points), batch):
        client.upsert(collection_name=coll, points=points[i:i + batch], wait=True)
