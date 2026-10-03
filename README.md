# Distributed Document Intelligence System with Hybrid Retrieval

**TA and BDA Project** - a distributed document intelligence platform that
processes large collections of documents and lets users ask natural-language
questions. The system retrieves relevant evidence and generates answers that
are grounded in the retrieved document chunks.

## Core idea

The project combines four complementary capabilities:

- **Text Analytics (TA):** document cleaning, tokenization, NLP, entity
  extraction, and summarization.
- **Big Data Analytics (BDA):** distributed processing and storage for large
  document collections.
- **Hybrid Retrieval:** keyword/BM25 search combined with semantic/vector
  search for stronger retrieval quality.
- **Retrieval-Augmented Generation (RAG):** retrieves relevant chunks first,
  then supplies them to an LLM to generate evidence-based answers.

## Objectives

- Ingest documents from PDF, DOCX, TXT, Markdown, PPTX, and PPSX sources.
- Clean, normalize, and enrich text with useful NLP metadata.
- Process large document batches using a distributed framework such as Apache
  Spark.
- Build both a BM25 lexical index and a vector similarity index.
- Fuse retrieval results to provide accurate, relevant context to the RAG
  pipeline.
- Expose document search and question-answering through an API and frontend.
- Evaluate retrieval quality, answer quality, latency, and scalability.

## System architecture

```text
Documents
   |
   v
Ingestion -> Text Analytics -> Chunking -> Embeddings
   |              |                |             |
   v              v                v             v
Raw storage    Metadata store    BM25 index   Vector index
                                      \          /
                                       v        v
                                     Hybrid Retrieval
                                             |
                                             v
                                    RAG / LLM Answering
                                             |
                                             v
                                      API and Frontend
```

## Processing workflow

1. **Ingest** documents and extract their text and metadata.
2. **Clean and analyze** text: remove noise, tokenize, identify entities, and
   optionally summarize documents.
3. **Chunk** documents into smaller overlapping passages while retaining source
   metadata. This offline, high-volume step is implemented in Scala/Spark in
   [`spark-scala`](spark-scala/README.md).
4. **Index** chunks twice: once for BM25 keyword retrieval and once as vector
   embeddings for semantic retrieval.
5. **Retrieve** candidate chunks with both methods, then merge and rerank them.
6. **Generate** an answer from the highest-ranked chunks, including citations
   back to the source documents.
7. **Evaluate** the pipeline using test queries and relevant-document labels.

## Project layout

```text
distributed-document-intelligence/
├── data/             # Source documents and local datasets
├── ingestion/        # Document loading, parsing, and normalization
├── spark/            # Distributed Spark ETL and indexing jobs
├── text_analytics/   # Cleaning, tokenization, NLP, NER, summarization
├── chunking/         # Document chunking strategies
├── embeddings/       # Embedding generation and vector-store integration
├── retrieval/        # BM25, vector, and hybrid retrieval modules
├── rag/              # Context construction and LLM answer generation
├── evaluation/       # Retrieval, answer-quality, latency, scale evaluation
├── api/              # Backend API endpoints
├── frontend/         # Search and question-answering user interface
├── requirements.txt  # Python dependencies
└── README.md
```

## Hybrid retrieval

Hybrid retrieval uses two different signals:

- **BM25 retriever** finds passages that closely match the query terms. It is
  effective for exact names, identifiers, rare keywords, and phrases.
- **Vector retriever** finds passages with similar meaning, even when they use
  different words from the query.
- **Hybrid retriever** combines the two ranked result sets, for example through
  weighted scores or Reciprocal Rank Fusion (RRF), before returning the best
  chunks.

The retrieval modules are located in `retrieval/`:

- `bm25_retriever.py`
- `vector_retriever.py`
- `hybrid_retriever.py`

## Suggested technology stack

| Area | Suggested options |
| --- | --- |
| Distributed processing | Apache Spark with Scala |
| Document storage | HDFS, S3, MinIO, or local storage for development |
| Text analytics | spaCy, NLTK, Hugging Face Transformers |
| Lexical search | Elasticsearch/OpenSearch or `rank-bm25` |
| Vector database | FAISS, Chroma, Qdrant, Weaviate, or Milvus |
| Embeddings and LLM | Sentence Transformers and an LLM provider or local model |
| API | FastAPI |
| Frontend | React, Streamlit, or another web UI |
| Evaluation | RAGAS, custom relevance metrics, and latency benchmarks |

## Evaluation plan

Measure the system with a small labelled query set and report:

- **Retrieval relevance:** Precision@K, Recall@K, MRR, and nDCG.
- **Answer quality:** groundedness, correctness, relevance, and citation
  quality.
- **System performance:** indexing time, query latency, throughput, and the
  effect of increasing document volume or worker count.
- **Comparison:** BM25-only, vector-only, and hybrid retrieval results.

## Development roadmap

1. Implement basic document ingestion and store extracted text with metadata.
2. Add cleaning, chunking, and a small sample dataset.
3. Implement and test BM25 retrieval.
4. Generate embeddings and implement vector retrieval.
5. Add rank fusion and compare hybrid retrieval against the individual methods.
6. Build the RAG pipeline with source citations.
7. Run the Scala/Spark batch job for large-scale extraction, cleaning, and chunking.
8. Build API/frontend interfaces and run the evaluation suite.

## Getting started

This repository currently contains the project scaffold. Add dependencies to
`requirements.txt`, install them with `pip install -r requirements.txt`, and
then implement the modules in the order described in the development roadmap.

### Enable LLM answer generation

The pipeline always retrieves and reranks document chunks first. When a
`GEMINI_API_KEY` environment variable is present, `pipeline.answer()` then uses
the Gemini API to synthesize a concise answer from only those chunks.
Every factual claim is instructed to cite its chunk as `[Source N]`; answers
without a valid citation are rejected. Without the key, the project stays fully
local and uses its existing extractive answer generator instead.

All local API and runtime settings are in [`.env.example`](.env.example). A
private `.env` template is created in the project root and is ignored by Git.
Open `.env` and add your Gemini key after `GEMINI_API_KEY=`. The application
loads it automatically, so no PowerShell environment command is required.

In PowerShell for the current terminal session:

```powershell
$env:GEMINI_API_KEY = "your_api_key"
$env:RAG_LLM_MODEL = "gemini-3.8-flash"  # optional; choose your deployed model
python run_pipeline.py --question "What is a deadlock?"
```

To compare approaches in code, pass `generator="llm"` to require generation,
or `generator="extractive"` to force the offline baseline. Do not put API keys
in source files or commit them; `.env` is already ignored by Git.

### Run the student question-answering session

Run without `--question` to keep the document index in memory and let students
ask multiple questions:

```powershell
python run_pipeline.py
```

Use `exit` when finished. The first run builds local chunk and embedding caches
under `data/processed`; later runs reuse them unless a source document changes.
`--question "..."` is deliberately a one-question command for scripts and
testing. Add `--interactive` after it if the session should continue.
