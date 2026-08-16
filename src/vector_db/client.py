"""
Chroma client.

Persistent and in-process: no server to start, and the index survives between
runs by writing to data/chroma/. Everything else in the package should get its
client from here rather than constructing its own.
"""

import os

import chromadb

CHROMA_DIR = os.path.join("data", "chroma")

_client = None


def get_client(path=CHROMA_DIR):
    """Return the shared client, creating it on first use."""
    global _client
    if _client is None:
        os.makedirs(path, exist_ok=True)
        _client = chromadb.PersistentClient(path=path)
    return _client


def reset(path=CHROMA_DIR):
    """Drop every collection. Used when reindexing from scratch."""
    client = get_client(path)
    for collection in client.list_collections():
        client.delete_collection(collection.name)
    return client
