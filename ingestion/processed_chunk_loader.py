"""Read JSON chunk records emitted by the Scala Spark batch job."""

import json
from pathlib import Path

from text_analytics.preprocessing import strip_surrogates


def load_processed_chunks(processed_path: str | Path) -> list[dict]:
    """Load Spark ``chunks_json`` output into the retrieval chunk schema.

    ``processed_path`` must be the ``chunks_json`` directory written by
    ``DocumentBatchJob``. Spark creates one or more ``part-*.json`` files;
    each line in them is a complete chunk record.
    """
    source = Path(processed_path)
    if not source.exists():
        raise FileNotFoundError(f"Processed chunk output does not exist: {source}")

    files = [source] if source.is_file() else sorted(source.glob("part-*.json"))
    if not files:
        raise FileNotFoundError(f"No Spark JSON part files found in: {source}")

    chunks = []
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    # Spark writes JSON with \udXXX escapes, which survive
                    # json.loads as lone surrogates that cannot be encoded or
                    # re-serialized. Scrubbed here so the cache fingerprint and
                    # any later JSON write do not fail on them.
                    for key in ("text", "clean_text"):
                        if isinstance(record.get(key), str):
                            record[key] = strip_surrogates(record[key])
                    record["metadata"] = record.get("metadata") or {"filename": Path(record["source"]).name}
                    record["keywords"] = record.get("keywords") or []
                    chunks.append(record)
    return chunks
