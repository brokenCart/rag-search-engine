import math
import pickle
import string
import sys
from collections import Counter, defaultdict
from typing import Any

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH, load_movies


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies = defaultdict(Counter)
        self.__index_path = CACHE_DIR / "index.pkl"
        self.__docmap_path = CACHE_DIR / "docmap.pkl"
        self.__tf_path = CACHE_DIR / "term_frequencies.pkl"

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str):
        return self.term_frequencies[doc_id].get(term, 0)

    def get_idf(self, term_token: str):
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term_token])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id: int, term_token: str):
        return self.get_tf(doc_id, term_token) * self.get_idf(term_token)

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with open(self.__index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.__docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.__tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open(self.__index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.__docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.__tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)


def search_command(query: str) -> list[dict[str, Any]]:
    inv_idx = InvertedIndex()
    try:
        inv_idx.load()
    except FileNotFoundError:
        print(
            f"Both {CACHE_DIR / 'index.pkl'} and {CACHE_DIR / 'docmap.pkl'} should exist."
        )
        sys.exit(1)

    results = []
    query_tokens = tokenize_text(query)

    for token in query_tokens:
        doc_ids = inv_idx.get_documents(token)
        for doc_id in doc_ids:
            results.append(inv_idx.docmap[doc_id])
            if len(results) >= DEFAULT_SEARCH_LIMIT:
                return results

    return results


def build_command():
    inv_idx = InvertedIndex()
    inv_idx.build()
    inv_idx.save()


def tf_command(doc_id: int, term: str) -> int:
    inv_idx = InvertedIndex()
    inv_idx.load()

    try:
        term_token = tokenize_single_term(term)
    except ValueError as e:
        print(e)

    return inv_idx.get_tf(doc_id, term_token)


def idf_command(term: str) -> float:
    inv_idx = InvertedIndex()
    inv_idx.load()

    try:
        term_token = tokenize_single_term(term)
    except ValueError as e:
        print(e)

    return inv_idx.get_idf(term_token)


def tfidf_command(doc_id: int, term: str) -> float:
    inv_idx = InvertedIndex()
    inv_idx.load()

    try:
        term_token = tokenize_single_term(term)
    except ValueError as e:
        print(e)

    return inv_idx.get_tfidf(doc_id, term_token)


def preprocess_text(text: str) -> str:
    # lowercase
    text = text.lower()

    # remove punctuations
    table = str.maketrans("", "", string.punctuation)
    text = text.translate(table)

    return text


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH) as f:
        data = [preprocess_text(word) for word in f.read().splitlines()]
    return data


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()

    # removing stopwords
    filtered_tokens = list(filter(lambda x: x not in STOPWORDS, tokens))

    # stemming
    stemmer = PorterStemmer()
    stemmed_tokens = list(map(stemmer.stem, filtered_tokens))

    return stemmed_tokens


def tokenize_single_term(term: str):
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("Term must be a single token.")
    return tokens[0]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
