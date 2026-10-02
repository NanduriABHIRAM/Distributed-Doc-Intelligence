# Design: TA-Augmented Hybrid Retrieval

Status: proposal for discussion. No code changed yet.

## 1. The problem with the current system

The RAG pipeline is not failing because the architecture is wrong. It is
failing because three specific components are placeholders:

| Component | File | Current state | Effect |
| --- | --- | --- | --- |
| Embeddings | `embeddings/embedding_model.py` | 256-dim hashed bag-of-words | "Semantic" retrieval has no semantics. It is fuzzy word-matching with extra steps. |
| Generation | `rag/llm_client.py` | Raises `RuntimeError` | `answer_generator.py` falls back to picking the single best-matching sentence and dumping it verbatim. This is why answers feel weak. |
| Chunking | `chunking/chunker.py` | Flat 180-word split | Slices through definitions, lists, and tables. Breaks the one structure the answers depend on. |

Fixing these three is a large fraction of the available quality gain, and it
does not require changing the architecture. Everything below is built on top
of that baseline.

## 2. Why not "an LLM that does Text Analytics and BDA"

The initial instinct was to replace the RAG model with a single LLM
responsible for text analytics and big data analytics. This is not workable:

- Text analytics is already implemented, and correctly so, as a library of
  deterministic transformations in `text_analytics/`. That is the correct
  shape for it.
- Big data analytics is Spark and the storage layer. `spark/` and
  `spark-scala/` already hold this. An LLM cannot substitute for distributed
  execution.
- Folding both into one model produces something mediocre at both, and
  surrenders the two most demonstrable components of the project.

The framing we are adopting instead: the LLM is one component inside the
pipeline. The pipeline is the product.

## 3. Core idea

Use text-analytics output as first-class retrieval signal, and let Spark
produce that analytics at scale.

```
Documents
   |
   v
Ingestion (Spark, Scala)  -->  document-level TA: summary, entities, keywords
   |                                    |
   v                                    v
Chunking (structure-aware)         Coarse retrieval over document summaries
   |                                    |
   v                                    v
Embeddings (real model)  ------->  Fine retrieval over chunks
   |                                    |
   +----------- BM25 -------------------+
                        |
                        v
              RRF fusion + TA filters
                        |
                        v
              Local LLM generation with citations
```

The coarse-to-fine structure is the main new idea. Ranking whole documents by
summary first, then searching within only the winners, is cheap, gives a
natural place for document-level analytics to act, and sidesteps most of the
noise that kills flat chunk retrieval.

## 4. Workstreams

### WS1 — Real embeddings (prerequisite)

Replace `HashEmbeddingModel` with `sentence-transformers`, behind the same
`encode()` interface so nothing downstream changes. Use a small multilingual
model (`paraphrase-multilingual-MiniLM-L12-v2`) as the default: it is ~470 MB,
runs on CPU, and handles the code-switched English/math text that appears in
Indian university lecture material far better than an English-only MiniLM.

Store both the vector and the lexical BM25 index so hybrid fusion still works.

### WS2 — Structure-aware chunking

Replace the flat word-count split. Chunk on document structure first — headings,
`Slide N` markers, bullet blocks — and only fall back to size-based splitting
inside oversized sections. Preserve section titles in chunk metadata so a
retrieved chunk carries the context of where it came from, which matters most
for slides where a bullet has no meaning alone.

### WS3 — Upgrade the analytics primitives

This is a prerequisite for WS4, not an optional extra. The current
implementations would actively degrade retrieval if wired in as-is:

- `text_analytics/ner.py` matches every Title-Cased sequence and labels it
  `PROPER_NOUN`. On lecture material this returns "Slide 4", "Chapter 2",
  "Lecture Notes" — formatting artifacts, not entities. Needs a stoplist of
  structural terms and a filter for numeric/digit-bearing tokens, or a real
  model.
- `text_analytics/keyword_extraction.py` is raw token frequency. On technical
  text this surfaces "used", "method", "system" ahead of actual domain terms.
  Needs TF-IDF against the corpus rather than absolute frequency, plus
  bigrams so multi-word technical terms survive.

### WS4 — TA-augmented retrieval

Three concrete mechanisms, in increasing order of expected impact:

1. **Document-level pre-filter.** Rank documents by summary/keyword similarity
   to the query, keep the top N, and run chunk retrieval only within them.
2. **Entity boosting.** Chunks whose extracted entities intersect entities
   detected in the query get a score boost. Requires WS3 to be trustworthy.
3. **Query expansion.** Expand the query with corpus-derived synonyms and
   morphological variants before retrieval. Makes recall meaningfully better
   on technical vocabulary where the user and the document use different words.

### WS5 — Local generation

Under the no-API constraint, generation quality is the hard ceiling. Realistic
tiers, best first:

- **7B–8B instruct model via Ollama** (Qwen2.5, Llama 3.1). Viable on CPU,
  slow-ish, and good enough for extractive-with-citation answers.
- **3B model** as a floor if throughput matters more than answer quality.

Generation must stay extractive-leaning and strictly grounded: answer only from
supplied context, cite `[Source N]` inline, and emit an explicit
"I could not find this in the documents" when the context does not support an
answer. Without that last part a small model will confabulate, which is worse
than the current extractive fallback.

### WS6 — Evaluation (needs to come early)

There is currently one function, `evaluate_retrieval`, that computes
Precision@K, Recall@K, and reciprocal rank. There is no labelled query set. An
improvement cannot be claimed without one.

Build a small hand-labelled set — roughly 50 queries with relevant document
IDs, biased toward the hard cases: definitional questions ("what is a
deadlock"), acronym questions, and questions whose answer spans two
documents. Then report BM25-only vs vector-only vs hybrid vs TA-augmented on
the same set. This is also the single most valuable artifact for the writeup,
because it turns "I think this is better" into a table.

## 5. Sequencing

1. **Evaluation harness and query set.** Before any change, so every subsequent
   step is measurable.
2. **WS1 real embeddings.** Largest single gain, lowest risk.
3. **WS2 structure-aware chunking.** Cheap, independent.
4. **WS5 local generation.** Replaces the sentence-dumping fallback.
5. **WS3 analytics upgrades.** Prerequisite for WS4.
6. **WS4 TA-augmented retrieval.** The differentiating contribution.

WS6 should be re-run after steps 2, 3, and 4 so the gains are attributable.

## 6. Alternative worth benchmarking

This corpus is lecture PDFs and slide decks. Many documents fit in a modern
long context window. Storing the top-3 documents whole and asking directly,
with no chunking and no retrieval, is a strong baseline and sometimes beats
chunk-and-retrieve. It is worth measuring in step 1, because if it wins
outright, the honest answer is that the retrieval complexity is not earning its
cost on this corpus — and knowing that early is far cheaper than finding out
at the end.

## 7. Constraints and risks

- **Compute.** Local generation is the bottleneck. Everything above it
  (embedding, chunking, retrieval, analytics) is fast and cheap on CPU.
- **Cold start.** A 470 MB embedding model and a multi-GB LLM both download on
  first run. Pin versions in `requirements.txt`.
- **Confabulation risk.** Small local models are more prone to inventing
  plausible answers. The grounded-generation prompt and the explicit
  abstention path are load-bearing, not nice-to-haves.
- **NER quality.** WS3 is the most likely to consume more time than estimated.
  If the rule-based upgrade proves insufficient, the fallback is to restrict
  entity boosting to entities that survive a corpus-wide frequency filter,
  rather than attempting full NER.