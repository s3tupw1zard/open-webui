import openai
from typing import List, Optional
from open_webui.retrieval.vector.main import VectorDBBase, SearchResult, Document

class OpenAIClient(VectorDBBase):
    TYPE = "openai"

    def __init__(self):
        openai.api_key = self._get_config("OPENAI_API_KEY")
        self.index = self._get_config("OPENAI_VECTOR_INDEX")

    def create_collection(self, name: str) -> None:
        return

    def add_items_to_collection(self, name: str, items: List[Document]) -> None:
        for doc in items:
            resp = openai.Embedding.create(
                model="text-embedding-3-small",
                input=doc.text,
            )
            vec = resp.data[0].embedding
            openai.Vector.upsert(
                index=self.index,
                vectors=[{
                    "id": doc.id,
                    "values": vec,
                    "metadata": {"kb": name, **doc.metadata},
                }],
            )

    def search(self, name: str, vectors: List[List[float]], limit: int) -> Optional[SearchResult]:
        resp = openai.Vector.search(
            index=self.index,
            vector=vectors[0],
            top_k=limit,
            include=["metadata"]
        )
        ids   = [item["id"]       for item in resp["data"]]
        dists = [item["score"]    for item in resp["data"]]
        mets  = [item["metadata"] for item in resp["data"]]
        return SearchResult(ids=ids, distances=dists, metadatas=mets)

    def delete_collection(self, name: str) -> None:
        openai.Vector.delete(
            index=self.index,
            ids=[doc.id for doc in self.list_items(name)]
        )

    def list_items(self, name: str) -> List[Document]:
        resp = openai.Vector.search(
            index=self.index,
            vector=[0.0]*1536,
            top_k=1000,
            include=["metadata"]
        )
        docs: List[Document] = []
        for item in resp["data"]:
            if item["metadata"].get("kb") == name:
                docs.append(Document(
                    id=item["id"],
                    text="",
                    metadata=item["metadata"],
                ))
        return docs
