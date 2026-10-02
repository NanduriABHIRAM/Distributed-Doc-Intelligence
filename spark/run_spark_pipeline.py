"""Compatibility notice for the legacy PySpark batch entry point.

The production batch job is now implemented in ``spark-scala``. Run it with
``sbt run -- --input <raw-path> --output <processed-path>`` and pass the
resulting ``chunks_json`` directory to ``run_pipeline.py --processed-chunks``.
"""


def run_spark_pipeline(*_args, **_kwargs):
    """Raise an actionable message instead of running the retired PySpark job."""
    raise RuntimeError(
        "The Spark batch pipeline is implemented in Scala. "
        "See spark-scala/README.md for the sbt command."
    )
