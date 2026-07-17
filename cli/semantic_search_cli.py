import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embeddings model")
    subparsers.add_parser("verify_embeddings", help="Verify the movie embeddings")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embeddings for a given text"
    )
    embed_text_parser.add_argument(
        "text", help="Text for which the embedding has to be generated"
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate embeddings for a given query"
    )
    embed_query_parser.add_argument(
        "query", help="Query for which the embedding has to be generated"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of documents to show",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            print(f"Searching for: {args.query}")
            results = search(args.query, args.limit)
            for idx, item in enumerate(results):
                print(
                    f"{idx + 1}. ({item['id']}) {item['title']} (score: {item['score']:.2f})"
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
