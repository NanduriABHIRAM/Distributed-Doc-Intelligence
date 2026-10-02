"""Compatibility module for the retired PySpark transformations.

Normalization and chunking are performed by the Scala Spark batch job so the
project has one distributed implementation and one output schema.
"""


def preprocess_documents(*_args, **_kwargs):
    """Direct callers to the Scala Spark batch job."""
    raise RuntimeError(
        "Distributed preprocessing is implemented in spark-scala, not PySpark. "
        "See spark-scala/README.md."
    )
