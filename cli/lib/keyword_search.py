import string
from nltk.stem import PorterStemmer

from .search_utils import STOPWORDS_PATH, load_movies


def search_command(query: str):
    movies = load_movies()

    results = []
    query_tokens = tokenize_text(query)

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)

    return results


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


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
