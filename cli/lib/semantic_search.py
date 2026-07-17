import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies


class SemanticSearch:
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not (text := text.strip()):
            raise ValueError("The text must not be empty")
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self, documents):
        self.documents = documents
        docs_strrep = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            docs_strrep.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(docs_strrep, show_progress_bar=True)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / "movie_embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if (CACHE_DIR / "movie_embeddings.npy").exists():
            with open(CACHE_DIR / "movie_embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        score_doc_list = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            score_doc_list.append((similarity_score, self.documents[i]))
        score_doc_list = sorted(score_doc_list, key=lambda x: -x[0])
        results = []

        for score, doc in score_doc_list:
            results.append(
                {
                    "score": score,
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
            if len(results) >= limit:
                return results

        return results


def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    return ss.search(query, limit)


def verify_model():
    ss = SemanticSearch()
    print(f"Model Loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
