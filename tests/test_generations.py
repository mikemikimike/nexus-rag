"""Generations record a new collection before it exists, so cleanup always knows it."""

import pytest
from qdrant_client import models

from local_rag import IndexFormat, Settings
from local_rag.generations import Generations
from local_rag.index_names import IndexNames


@pytest.fixture
def generations(client):
    return Generations(client, IndexNames("knowledge"))


def test_a_new_generation_is_recorded_with_its_format_before_it_is_made(generations):
    generation, written = generations.fresh_name(), IndexFormat.written_by(Settings(), 8)
    seen = []

    generations.create(
        generation, written, make=lambda: seen.append(generations.format_of(generation))
    )

    assert seen == [written]


def test_a_generation_whose_making_failed_is_still_swept_up(client, generations):
    generation = generations.fresh_name()

    def half_made():
        client.create_collection(
            generation, vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
        )
        raise ConnectionError("Qdrant went away")

    with pytest.raises(ConnectionError):
        generations.create(generation, IndexFormat.written_by(Settings(), 8), make=half_made)
    generations.prune()

    assert not client.collection_exists(generation)
