"""Spark session creation and configuration."""


def create_spark_session(app_name: str = "DistributedDocumentIntelligence"):
    """Create a local-or-cluster Spark session when PySpark is installed."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as error:
        raise RuntimeError("Spark support requires `pip install pyspark` and a Java runtime.") from error
    return SparkSession.builder.appName(app_name).getOrCreate()
