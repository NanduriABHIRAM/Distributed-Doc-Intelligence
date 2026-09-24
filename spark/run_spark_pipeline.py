"""Entrypoint for the distributed preprocessing job."""

from spark.spark_session import create_spark_session


def run_spark_pipeline(input_path: str, output_path: str):
    """Read text files with Spark, normalize whitespace, and write JSON output."""
    spark = create_spark_session()
    frame = spark.read.text(input_path).toDF("text")
    from spark.distributed_preprocessing import preprocess_documents

    preprocess_documents(frame).write.mode("overwrite").json(output_path)
