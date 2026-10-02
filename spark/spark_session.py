"""Compatibility module for the retired PySpark implementation.

The distributed workload belongs to the Scala Spark project in ``spark-scala``.
Keeping this small module gives callers an explicit migration message instead
of silently starting a separate Python Spark implementation.
"""


def create_spark_session(*_args, **_kwargs):
    """Direct callers to the Scala Spark batch job."""
    raise RuntimeError(
        "Spark processing is implemented in Scala. "
        "Run the job in spark-scala with sbt or spark-submit."
    )
