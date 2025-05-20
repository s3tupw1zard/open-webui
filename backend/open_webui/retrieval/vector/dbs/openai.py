import openai
from typing import Any, Dict, List, Optional
from open_webui.retrieval.vector.main import VectorDBBase, VectorItem, GetResult, SearchResult


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

    def create_collection(self, collection_name: str) -> None:
        """
        Create a new Vector Store for the KB named `collection_name` and store its ID.
        """
        resp = openai.request(
            method="POST",
            url="/vector_stores",
            json={
                "name": collection_name,
                "description": f"Vector store for KB '{collection_name}'"
            }
        )
        self.index = resp.get("id")

    def add_items_to_collection(self, collection_name: str, items: List[VectorItem]) -> None:
        """
        Generate embeddings and upsert into the OpenAI Vector Store.
        Automatically creates the store if it does not exist yet.
        """
        if not self.has_collection(collection_name):
            self.create_collection(collection_name)

        texts = [item["text"] for item in items]
        embed_resp = openai.Embedding.create(
            model=self._get_config("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=texts
        )

        vectors = []
        for item, embed in zip(items, embed_resp.data):
            md: Dict[str, Any] = item.get("metadata", {}) or {}
            md["text"] = item.get("text")
            vectors.append({
                "id": item["id"],
                "values": embed.embedding,
                "metadata": md,
            })

        openai.Vector.upsert(
            index=self.index,
            vectors=vectors
        )

    def search(self, collection_name: str, vectors: List[List[float]], limit: int) -> Optional[SearchResult]:
        """
        Search the Vector Store for the query vector.
        """
        if not self.has_collection(collection_name):
            return None

        resp = openai.Vector.search(
            index=self.index,
            vector=vectors[0],
            top_k=limit,
            include=["metadata"]
        )
        ids: List[str] = [hit["id"] for hit in resp.get("data", [])]
        distances: List[float] = [hit.get("score", 0.0) for hit in resp.get("data", [])]
        documents: List[Any] = []
        metadatas: List[Any] = []
        for hit in resp.get("data", []):
            md = hit.get("metadata") or {}
            text = md.get("text")
            documents.append(text)
            metadatas.append({k: v for k, v in md.items() if k != "text"})
        return SearchResult(
            ids=[ids],
            distances=[distances],
            documents=[documents],
            metadatas=[metadatas],
        )

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete all items in this Vector Store.
        """
        if not self.has_collection(collection_name):
            return
        # list up to 1000 items
        resp = openai.Vector.search(
            index=self.index,
            vector=[0.0] * self._get_config("DUMMY_DIM", 1536),
            top_k=1000,
            include=["metadata"]
        )
        ids = [hit.get("id") for hit in resp.get("data", [])]
        if ids:
            openai.Vector.delete(index=self.index, ids=ids)

    def list_items(self, collection_name: str) -> List[VectorItem]:
        """
        List all items in this Vector Store.
        """
        if not self.has_collection(collection_name):
            return []

        resp = openai.Vector.search(
            index=self.index,
            vector=[0.0] * self._get_config("DUMMY_DIM", 1536),
            top_k=1000,
            include=["values", "metadata"]
        )
        items: List[VectorItem] = []
        for hit in resp.get("data", []):
            md = hit.get("metadata") or {}
            items.append(VectorItem(
                id=hit.get("id"),
                text=md.get("text"),
                vector=hit.get("values"),
                metadata={k: v for k, v in md.items() if k != "text"},
            ))
        return items

    def has_collection(self, collection_name: str) -> bool:
        """Check if a vector store with the given name exists and set its index."""
        resp = openai.request(method="GET", url="/vector_stores")
        for entry in resp.get("data", []):
            if entry.get("name") == collection_name:
                self.index = entry.get("id")
                return True
        return False

    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        """Insert new items (or upsert) into the vector store."""
        self.add_items_to_collection(collection_name, items)

    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        """Insert or update items in the vector store."""
        self.add_items_to_collection(collection_name, items)

    def query(
        self, collection_name: str, filter: Dict[str, Any], limit: Optional[int] = None
    ) -> Optional[GetResult]:
        """Query items by metadata filter from the vector store."""
        if not self.has_collection(collection_name):
            return None
        items = self.list_items(collection_name)
        matched = [it for it in items if all(it.metadata.get(k) == v for k, v in filter.items())]
        if limit is not None:
            matched = matched[:limit]
        if not matched:
            return None
        ids = [it.id for it in matched]
        docs = [it.text for it in matched]
        metadatas = [it.metadata for it in matched]
        return GetResult(ids=[ids], documents=[docs], metadatas=[metadatas])

    def get(self, collection_name: str) -> Optional[GetResult]:
        """Retrieve all items from the vector store."""
        if not self.has_collection(collection_name):
            return None
        items = self.list_items(collection_name)
        if not items:
            return None
        ids = [it.id for it in items]
        docs = [it.text for it in items]
        metadatas = [it.metadata for it in items]
        return GetResult(ids=[ids], documents=[docs], metadatas=[metadatas])

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete items by ID or metadata filter from the vector store."""
        if not self.has_collection(collection_name):
            return
        if ids:
            openai.Vector.delete(index=self.index, ids=ids)
        elif filter:
            openai.Vector.delete(index=self.index, filter={"metadata": filter})

    def reset(self) -> None:
        """Delete all vector stores."""
        resp = openai.request(method="GET", url="/vector_stores")
        for entry in resp.get("data", []):
            vid = entry.get("id")
            if vid:
                openai.request(method="DELETE", url=f"/vector_stores/{vid}")


class OpenAIClient(OpenAIStore):
    """Alias for OpenAIStore to match VectorDB client naming conventions."""
    pass
