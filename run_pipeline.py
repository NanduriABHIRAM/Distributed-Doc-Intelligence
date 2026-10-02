"""Terminal entrypoint for the midterm document-intelligence demonstration."""

import argparse
from pathlib import Path
import sys

from pipeline import DocumentIntelligencePipeline

# Typing "exit" must always end the session, but a question about exiting a
# process ("how do I exit a deadlock?") must not be swallowed by the check.
_EXIT_WORDS = {"exit", "quit", "q", ":q", "bye"}


def _is_exit(text: str) -> bool:
    """Whether *text* is a request to leave rather than a question."""
    return text.strip().strip(".!?").lower() in _EXIT_WORDS


def display_results(response: dict) -> None:
    """Print a compact demonstration of retrieval and grounded answering."""
    for name in ("bm25", "vector", "hybrid"):
        print(f"\n[{name.upper()}]")
        for index, result in enumerate(response["retrieval"][name], 1):
            print(f"{index}. {result['metadata']['filename']}  Score: {result['score']:.4f}")
    print(f"\nAnswer:\n{response['answer']}")
    strategy = response.get("strategy")
    if strategy:
        # Surfaced because the strategy explains the answer: a templated
        # definition and a LexRank extract are equally valid, but only one of
        # them is a direct quote of a defining sentence.
        print(f"\nStrategy: {strategy}")
    if response.get("abstained"):
        print("Note: the retrieved context did not support an answer.")
    if response["sources"]:
        print("\nSources:")
        seen: set[str] = set()
        for source in response["sources"]:
            if source["document_name"] in seen:
                continue
            seen.add(source["document_name"])
            print(f"[{len(seen)}] {source['document_name']}")


def main() -> None:
    """Build indexes and answer a supplied terminal question."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Distributed Document Intelligence")
    parser.add_argument("--data-dir", default=str(Path("data") / "raw"))
    parser.add_argument(
        "--processed-chunks",
        help="Path to Scala Spark output/chunks_json; skips local ingestion and chunking.",
    )
    parser.add_argument("--question", help="Answer a single question, then exit.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep asking questions until 'exit'. This is the default when no --question is given.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    pipeline = DocumentIntelligencePipeline(args.data_dir, processed_chunks_path=args.processed_chunks).build()
    stats = pipeline.stats()
    print("=" * 40)
    print("Distributed Document Intelligence")
    print("=" * 40)
    print(f"Documents loaded     : {stats['documents']}")
    print(f"Chunks generated    : {stats['chunks']}")
    print(f"Embeddings generated: {stats['embeddings']}")
    print(f"Embedding model      : {stats['embedding_model']}")
    if not stats["documents"]:
        print("Add .txt, .md, .pdf, .docx, .pptx, or .ppsx files to data/raw and run again.")
        return

    # Indexing is the expensive part and does not depend on the question, so a
    # session builds once and then reuses the indexes for every question.
    if args.question:
        display_results(pipeline.answer(args.question, args.top_k))
        return

    print("\nType a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nEnter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C or Ctrl+D ends the session cleanly rather than dumping a
            # traceback, which matters for a terminal demo.
            print("\nExiting.")
            return
        if not question:
            continue
        if _is_exit(question):
            print("Exiting.")
            return
        display_results(pipeline.answer(question, args.top_k))


if __name__ == "__main__":
    main()
