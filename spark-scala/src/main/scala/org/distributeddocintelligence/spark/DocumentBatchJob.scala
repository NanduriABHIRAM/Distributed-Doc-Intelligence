package org.distributeddocintelligence.spark

import java.io.ByteArrayInputStream
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

import scala.collection.JavaConverters._
import scala.util.Try

import org.apache.spark.sql.{Dataset, Encoders, Row, SparkSession}
import org.apache.tika.metadata.Metadata
import org.apache.tika.parser.AutoDetectParser
import org.apache.tika.parser.ParseContext
import org.apache.tika.sax.BodyContentHandler

/**
 * Distributed extraction, cleaning, and chunking job.
 *
 * Input is a directory (or glob) of TXT, Markdown, PDF, DOCX, PPTX, or PPSX
 * files. Spark distributes files with the binaryFile source and Apache Tika
 * extracts their text on executors. The job writes both Parquet for analytics
 * and JSON records that the Python online retrieval service can load directly.
 */
object DocumentBatchJob {
  private val SupportedExtensions = Set("txt", "md", "pdf", "docx", "pptx", "ppsx")
  private val StopWords = Set("a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with")

  final case class SourceDocument(
      document_id: String,
      source: String,
      filename: String,
      text: String,
      clean_text: String,
      metadata: Map[String, String]
  )

  final case class ChunkRecord(
      chunk_id: String,
      document_id: String,
      source: String,
      text: String,
      metadata: Map[String, String],
      keywords: Seq[String]
  )

  final case class JobConfig(input: String = "", output: String = "", chunkSize: Int = 180, overlap: Int = 30, partitions: Option[Int] = None)

  def main(args: Array[String]): Unit = {
    val config = parseArguments(args)
    require(config.input.nonEmpty, "--input is required")
    require(config.output.nonEmpty, "--output is required")
    require(config.chunkSize > 0 && config.overlap >= 0 && config.overlap < config.chunkSize, "chunk size must be positive and overlap smaller than chunk size")

    val spark = SparkSession.builder().appName("DistributedDocumentIntelligence").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val binaryFiles = spark.read.format("binaryFile").load(config.input)
    val documents = binaryFiles.select("path", "content").as[Row].mapPartitions { rows =>
      rows.flatMap(row => extractDocument(row.getAs[String]("path"), row.getAs[Array[Byte]]("content")))
    }(Encoders.product[SourceDocument]).filter(_.clean_text.nonEmpty)

    val chunks = documents.flatMap(document => chunkDocument(document, config.chunkSize, config.overlap))(Encoders.product[ChunkRecord])
    val outputChunks = config.partitions.map(chunks.repartition).getOrElse(chunks)

    documents.write.mode("overwrite").parquet(s"${config.output}/documents_parquet")
    outputChunks.write.mode("overwrite").parquet(s"${config.output}/chunks_parquet")
    outputChunks.write.mode("overwrite").json(s"${config.output}/chunks_json")
    spark.stop()
  }

  private def extractDocument(path: String, bytes: Array[Byte]): Option[SourceDocument] = {
    val filename = path.split('/').lastOption.getOrElse(path)
    val extension = filename.split('.').lastOption.map(_.toLowerCase).getOrElse("")
    if (!SupportedExtensions.contains(extension)) return None

    val extracted = Try {
      if (extension == "txt" || extension == "md") new String(bytes, StandardCharsets.UTF_8)
      else {
        val parser = new AutoDetectParser()
        val handler = new BodyContentHandler(-1)
        parser.parse(new ByteArrayInputStream(bytes), handler, new Metadata(), new ParseContext())
        handler.toString
      }
    }.toOption.map(cleanText).filter(_.nonEmpty)

    extracted.map { text =>
      SourceDocument(
        document_id = sha256(path).take(24),
        source = path,
        filename = filename,
        text = text,
        clean_text = text,
        metadata = Map("filename" -> filename, "file_type" -> extension)
      )
    }
  }

  private def chunkDocument(document: SourceDocument, chunkSize: Int, overlap: Int): Seq[ChunkRecord] = {
    val words = document.clean_text.split("\\s+").filter(_.nonEmpty)
    val step = chunkSize - overlap
    words.indices.by(step).map { start =>
      val text = words.slice(start, math.min(start + chunkSize, words.length)).mkString(" ")
      val index = start / step
      ChunkRecord(
        chunk_id = s"${document.document_id}-$index",
        document_id = document.document_id,
        source = document.source,
        text = text,
        metadata = document.metadata,
        keywords = keywords(text)
      )
    }.toSeq
  }

  private def cleanText(text: String): String = text.replaceAll("\\s+", " ").trim

  private def keywords(text: String, limit: Int = 12): Seq[String] =
    text.toLowerCase.split("[^a-z0-9_-]+").iterator
      .filter(token => token.length > 1 && !StopWords.contains(token))
      .toSeq.groupBy(identity).toSeq.sortBy { case (token, values) => (-values.size, token) }
      .take(limit).map(_._1)

  private def sha256(value: String): String =
    MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)).map("%02x".format(_)).mkString

  private def parseArguments(args: Array[String]): JobConfig = {
    @annotation.tailrec
    def loop(remaining: List[String], config: JobConfig): JobConfig = remaining match {
      case "--input" :: value :: tail => loop(tail, config.copy(input = value))
      case "--output" :: value :: tail => loop(tail, config.copy(output = value))
      case "--chunk-size" :: value :: tail => loop(tail, config.copy(chunkSize = value.toInt))
      case "--overlap" :: value :: tail => loop(tail, config.copy(overlap = value.toInt))
      case "--partitions" :: value :: tail => loop(tail, config.copy(partitions = Some(value.toInt)))
      case Nil => config
      case option :: _ => throw new IllegalArgumentException(s"Unknown or incomplete option: $option")
    }
    loop(args.toList, JobConfig())
  }
}
