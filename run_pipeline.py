"""Terminal entrypoint for the midterm document-intelligence demonstration."""

import argparse
from pathlib import Path

from pipeline import DocumentIntelligencePipeline


def display_results(response: dict) -> None:
    """Print a compact demonstration of retrieval and grounded answering."""
    for name in ("bm25", "vector", "hybrid"):
        print(f"\n[{name.upper()}]")
        for index, result in enumerate(response["retrieval"][name], 1):
            print(f"{index}. {result['metadata']['filename']}  Score: {result['score']:.4f}")
    print(f"\nAnswer:\n{response['answer']}")
    if response["sources"]:
        print("\nSources:")
        for index, source in enumerate(response["sources"], 1):
            print(f"[{index}] {source['document_name']}")


def main() -> None:
    """Build indexes and answer a supplied terminal question."""
    parser = argparse.ArgumentParser(description="Distributed Document Intelligence")
    parser.add_argument("--data-dir", default=str(Path("data") / "raw"))
    parser.add_argument("--question", help="Question to ask the indexed documents")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    pipeline = DocumentIntelligencePipeline(args.data_dir).build()
    stats = pipeline.stats()
    print("=" * 40)
    print("Distributed Document Intelligence")
    print("=" * 40)
    print(f"Documents loaded     : {stats['documents']}")
    print(f"Chunks generated    : {stats['chunks']}")
    print(f"Embeddings generated: {stats['embeddings']}")
    if not stats["documents"]:
        print("Add .txt, .md, or .pdf files to data/raw and run again.")
        return

    question = args.question or input("\nEnter your question: ").strip()
    if question:
        display_results(pipeline.answer(question, args.top_k))


if __name__ == "__main__":
    main()
