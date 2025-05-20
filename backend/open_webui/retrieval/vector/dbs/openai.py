import openai
from typing import List, Optional
from open_webui.retrieval.vector.main import VectorDBBase, VectorItem, SearchResult


class OpenAIStore(VectorDBBase):
    """
    VectorDB driver for OpenAI Vector Store.
    Automatically creates a new Vector Store per knowledge base via OpenAI API.
    """
    TYPE = "openai"

    def __init__(self):
        # API-Key & optional API-Base aus Konfiguration laden
        openai.api_key = self._get_config("OPENAI_API_KEY")
        base = self._get_config("OPENAI_API_BASE", None)
        if base:
            openai.api_base = base
        self.index: Optional[str] = None

    def create_collection(self, name: str) -> None:
        """
        Create a new Vector Store for the KB named `name` and store its ID.
        """
        resp = openai.request(
            method="POST",
            url="/vector_stores",
            json={
                "name": name,
                "description": f"Vector store for KB '{name}'"
            }
        )
        self.index = resp.get("id")

    def add_items_to_collection(self, name: str, items: List[VectorItem]) -> None:
        """
        Generate embeddings and upsert into the OpenAI Vector Store.
        Automatically creates the store if it does not exist yet.
        """
        if not self.index:
            self.create_collection(name)

        texts = [item.text for item in items]
        embed_resp = openai.Embedding.create(
            model=self._get_config("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=texts
        )

        vectors = []
        for item, embed in zip(items, embed_resp.data):
            vectors.append({
                "id": item.id,
                "values": embed.embedding,
                "metadata": item.metadata or {}
            })

        openai.Vector.upsert(
            index=self.index,
            vectors=vectors
        )

    def search(self, name: str, vectors: List[List[float]], limit: int) -> Optional[SearchResult]:
        """
        Search the Vector Store for the query vector.
        """
        if not self.index:
            return SearchResult(ids=[], distances=[], metadatas=[])

        resp = openai.Vector.search(
            index=self.index,
            vector=vectors[0],
            top_k=limit,
            include=["metadata"]
        )
        ids = [hit["id"] for hit in resp["data"]]
        distances = [hit.get("score", 0.0) for hit in resp["data"]]
        metadatas = [hit.get("metadata") for hit in resp["data"]]
        return SearchResult(ids=ids, distances=distances, metadatas=metadatas)

    def delete_collection(self, name: str) -> None:
        """
        Delete all items in this Vector Store.
        """
        if not self.index:
            return
        # list up to 1000 items
        resp = openai.Vector.search(
            index=self.index,
            vector=[0.0] * self._get_config("DUMMY_DIM", 1536),
            top_k=1000,
            include=["metadata"]
        )
        ids = [hit["id"] for hit in resp["data"]]
        if ids:
            openai.Vector.delete(
                index=self.index,
                ids=ids
            )

    def list_items(self, name: str) -> List[VectorItem]:
        """
        List all items in this Vector Store.
        """
        if not self.index:
            return []

        resp = openai.Vector.search(
            index=self.index,
            vector=[0.0] * self._get_config("DUMMY_DIM", 1536),
            top_k=1000,
            include=["values", "metadata"]
        )
        items: List[VectorItem] = []
        for hit in resp["data"]:
            items.append(VectorItem(
                id=hit["id"],
                vector=hit.get("values"),
                metadata=hit.get("metadata")
            ))
        return items


class OpenAIClient(OpenAIStore):
    """Alias for OpenAIStore to match VectorDB client naming conventions."""
    pass
