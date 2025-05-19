import openai
from typing import Optional, List
from open_webui.retrieval.vector.base import VectorDB, SearchResult, Document
from open_webui.env import get_persistent_config

class OpenAIStore(VectorDB):
    TYPE = "openai"

    def __init__(self):
        key = get_persistent_config("OPENAI_API_KEY")
        self.index = get_persistent_config("OPENAI_VECTOR_INDEX")
        openai.api_key = key

    def create_collection(self, name: str) -> None:
        pass

    def add_items_to_collection(self, name: str, items: List[Document]) -> None:
        for doc in items:
            resp = openai.Embedding.create(
                model="text-embedding-3-small",
                input=doc.text,
                user="open_webui",
            )

    def search(self, name: str, vectors: List[List[float]], limit: int) -> Optional[SearchResult]:
        resp = openai.Embedding.create(
            model="text-embedding-3-small",
            input=[...],
        )
        return SearchResult(ids=[...], distances=[...])

    def delete_collection(self, name: str) -> None:
        pass
