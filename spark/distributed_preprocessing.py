"""Distributed document preprocessing transformations."""

from text_analytics.preprocessing import preprocess_text


def preprocess_documents(documents):
    """Normalize local document dictionaries or a Spark DataFrame's ``text`` column."""
    if hasattr(documents, "withColumn"):
        from pyspark.sql.functions import col, regexp_replace, trim

        return documents.withColumn("clean_text", trim(regexp_replace(col("text"), r"\\s+", " ")))
    return [dict(document, clean_text=preprocess_text(document["text"])) for document in documents]
