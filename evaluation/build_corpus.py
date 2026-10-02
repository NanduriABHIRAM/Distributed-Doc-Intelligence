"""Build the cached corpus the benchmark loads.

The benchmark deliberately never re-reads `data/raw`. PDF and slide extraction
is slow and, worse, is not deterministic across library versions -- the same
file can yield slightly different text after a Tika or pypdf upgrade. Caching
the extracted text once means every later benchmark run scores the identical
corpus, so a metric change is attributable to the pipeline rather than to the
parser.

This is the single point where raw files are read. In production that work is
done by the Spark job (`spark-scala/.../DocumentBatchJob.scala`), which reads
the same directory and writes the same chunk schema. This script is the local
equivalent, used for development and for grading runs that must be repeatable.

Usage:
    python -m evaluation.build_corpus
    python -m evaluation.build_corpus --limit 20     # quick smoke test
    python -m evaluation.build_corpus --workers 8     # parallel extraction
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.document_loader import SUPPORTED_SUFFIXES  # noqa: E402
from ingestion.ingestion_pipeline import run_ingestion  # noqa: E402

DEFAULT_DATA_DIR = Path("data") / "raw"
DEFAULT_OUTPUT = Path(__file__).with_name("corpus_cache.json")


def discover_files(data_dir: Path | str) -> list[Path]:
    """Every readable document under *data_dir*, sorted for reproducibility."""
    root = Path(data_dir)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _extract_one(path_str: str) -> dict:
    """Extract a single file in a worker process.

    Runs in a subprocess so that a malformed or corrupt document raises inside
    its own process rather than killing the whole build. A single bad PDF is
    common in a 130-file coursework corpus and must not abort the run.
    """
    try:
        documents = run_ingestion(path_str)
    except Exception as error:  # noqa: BLE001 - one bad file must not stop the build
        return {"error": f"{Path(path_str).name}: {type(error).__name__}: {error}"}
    return {
        "docs": [
            {
                "document_id": document["document_id"],
                "source": document["source"],
                "text": document["clean_text"],
                # Chunkers read `clean_text`; the cache keeps `text` as the
                # canonical extracted field, so both names are populated here
                # rather than making every consumer cope with the difference.
                "clean_text": document["clean_text"],
                "metadata": document["metadata"],
            }
            for document in documents
        ]
    }


def build(data_dir: Path | str, output: Path | str, limit: int | None = None, workers: int = 4) -> dict:
    """Extract *data_dir* into the benchmark's corpus cache."""
    output = Path(output)
    files = discover_files(data_dir)
    if limit:
        files = files[:limit]
    if not files:
        print(f"No supported files under {data_dir}. Supported: {sorted(SUPPORTED_SUFFIXES)}")
        return {"documents": 0, "errors": []}

    print(f"Extracting {len(files)} files from {data_dir} using {workers} worker(s)...")
    documents: list[dict] = []
    errors: list[str] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_one, str(path)): path for path in files}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if "error" in result:
                    errors.append(result["error"])
                else:
                    documents.extend(result["docs"])
                if completed % 20 == 0 or completed == len(files):
                    print(f"  {completed}/{len(files)} files, {len(documents)} documents", flush=True)
    else:
        for index, path in enumerate(files, 1):
            result = _extract_one(str(path))
            if "error" in result:
                errors.append(result["error"])
            else:
                documents.extend(result["docs"])
            if index % 20 == 0 or index == len(files):
                print(f"  {index}/{len(files)} files, {len(documents)} documents", flush=True)

    # Sorted by document_id so chunk indices, and therefore the corpus the
    # benchmark scores, do not depend on filesystem ordering.
    documents.sort(key=lambda document: document["document_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(documents, handle, ensure_ascii=False)

    print(f"\nCached {len(documents)} documents -> {output}")
    if errors:
        print(f"\n{len(errors)} file(s) failed extraction and are NOT in the cache:")
        for error in errors:
            print(f"  {error}")
        print("Queries labelled against these files will be reported as unresolved.")
    return {"documents": len(documents), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the benchmark corpus cache.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Only extract the first N files (smoke test).")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    build(args.data_dir, args.output, args.limit, args.workers)


if __name__ == "__main__":
    main()
