# Scala Spark batch pipeline

This module owns the project's distributed, offline workload: document text
extraction, normalization, keyword extraction, and overlapping chunk creation.
It uses Spark SQL with Scala 2.12 and Apache Tika, so PDF, Office, Markdown,
and text files can be processed by Spark executors.

Run from this directory with a local Spark installation or cluster settings:

```powershell
sbt run -- --input ..\data\raw\* --output ..\data\processed --chunk-size 180 --overlap 30 --partitions 8
```

For a cluster submission, first package the application and submit the JAR with
the Spark version available on the cluster:

```powershell
sbt package
spark-submit --class org.distributeddocintelligence.spark.DocumentBatchJob target\scala-2.12\distributed-document-intelligence-spark_2.12-0.1.0.jar --input hdfs:///ddi/raw/* --output hdfs:///ddi/processed --partitions 128
```

The job writes these datasets below `--output`:

- `documents_parquet` — normalized document records for Spark SQL/analytics.
- `chunks_parquet` — query-ready chunks for distributed downstream jobs.
- `chunks_json` — the same chunk schema for the Python retrieval service.

Serve the generated records locally with:

```powershell
python run_pipeline.py --processed-chunks data\processed\chunks_json --question "Your question"
```

The Python application remains the online RAG service: it creates embeddings,
retrieves candidates, reranks them, and produces answers. This separation keeps
Spark work distributed and avoids starting a Spark context for every question.
