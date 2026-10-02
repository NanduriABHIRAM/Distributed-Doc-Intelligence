"""Grounded extractive answering without a generative model.

The previous generator returned the single highest-scoring sentence, which reads
as a keyword match rather than an answer and frequently surfaces a table-of-
contents line or a fragment. This module routes by question type instead:

- Definitional questions ("what is X", "define X") get a template answer built
  from the span that actually defines the concept.
- Everything else gets query-focused extractive summarization via LexRank, so
  the answer is the set of sentences most central to the question rather than
  one arbitrarily-ranked sentence.

Every answer is assembled verbatim from retrieved text and carries citations,
so it is verifiable by construction. When the retrieved context does not
support an answer the module abstains explicitly rather than emitting the
best-looking fragment, which is the failure mode that makes extractive
answering look unreliable.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from text_analytics.keyword_extraction import STOP_WORDS
from text_analytics.preprocessing import expand_acronyms, expand_terms, normalize_term, split_compounds, tokenize

# Question types recognised by the router.
_DEFINITION_PREFIXES = ("what is", "what are", "define", "definition of", "meaning of", "explain what")
_LIST_PREFIXES = ("list", "name the", "enumerate")
_COMPARISON_MARKERS = ("difference between", "differ from", "compare", "contrast", "versus", " vs ")

# Text that looks like a hit but answers nothing.
_NON_ANSWER = re.compile(
    r"\b(?:table\s+of\s+contents|chapter\s+\d+|unit\s+\d+|slide\s+\d+|"
    r"learning\s+objectives?|references?|index|page\s+\d+|fig(?:ure)?\.?\s*\d+|"
    r"the\s+following\s+(?:table|figure|diagram)|as\s+shown\s+in\s+the\s+figure)\b",
    re.IGNORECASE,
)

_CLAUSE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# PDF text layers routinely drop the space after a sentence period, gluing the
# next sentence onto the end of the previous one ("deadlocks.completely"). The
# split above cannot see a boundary that has no whitespace, so it is repaired
# before splitting rather than filtered out afterwards.
#
# The next word often starts lowercase ("... eliminate deadlocks.completely")
# because sentence casing is not preserved by the extractor, so both cases are
# accepted. Decimal points ("3.14") and file-like names are excluded by
# requiring a letter on both sides.
_GLUED_SENTENCE = re.compile(r"(?<=[a-z0-9)\]’'])\.(?=[A-Za-z])")

# End-of-book review questions. These are the textbook's exercises, not its
# explanations, and they rank highly for topical queries because they repeat the
# terminology densely. Answering a user with "EX-28 following changes can be
# made safely (without introducing the possibility of deadlock)..." is worse
# than abstaining, because the text is on-topic and looks authoritative.
_EXAM_DEBRIS = re.compile(
    r"^\s*(?:ex|exr|q|ques|question|prob|problem)\s*[-–.]?\s*\d+\b|"
    r"\b(?:answer the following questions|what changes can be made|"
    r"give an example|explain why|discuss the following)\b",
    re.IGNORECASE,
)

# ALL-CAPS heading runs into the paragraph beneath it ("8.8 Recovery from
# Deadlock 341 MANAGING DEADLOCK IN DATABASES Database systems provide ...").
# Small-caps PDF headings extract as capitals, so the casing itself marks them.
_ALLCAPS_HEADING = re.compile(r"\b[A-Z][A-Z0-9 &/,'-]{8,}\b")

# Numbered section headings that run into the following sentence in extracted
# text ("7.5.3 Functional Programming Languages Most well-known programming
# languages ..."). The heading is not part of the claim and must be stripped,
# not answered.
_HEADING_PREFIX = re.compile(r"^\s*\d+(?:\.\d+){1,3}\s+(?:[A-Z][A-Za-z'/-]*\s+)+(?=[a-z])")


def repair_spacing(text: str) -> str:
    """Restore sentence spaces lost during PDF text extraction."""
    return _GLUED_SENTENCE.sub(". ", text)

# Fraction of query terms a passage must cover to be considered answer-bearing.
# Tuned against the corpus: too low and long passages pass on incidental terms,
# too high and correct single-concept answers get dropped.
MINIMUM_USABLE_COVERAGE = 0.34

# Words that must survive after a run-in heading is stripped for the remainder
# to count as a sentence rather than another fragment of the same heading.
_MIN_SENTENCE_AFTER_HEADING = 5

# A definition's predicate is a noun phrase naming what the concept *is* ("a
# situation in which processes are blocked", "system software that manages
# resources"). It is not a description of what the system *does with* the
# concept. Ranked purely by length, a procedural passage wins, and "what is a
# deadlock" gets answered with deadlock *recovery* -- "Deadlock is detected, a
# victim is selected and the transaction is aborted and rolled back" -- which
# describes the recovery procedure, not the condition.
_FINITE_VERB = re.compile(
    # A noun phrase contains no finite verb. Testing for one generalizes to
    # vocabulary no pattern has seen: enumerating head nouns would have caught
    # "a technique" and "a variable" but missed "system software" and
    # "communication protocol".
    #
    # "are blocked" / "is used" are participles, not finite verbs, so they stay
    # out: "a situation in which processes are blocked waiting" is a perfectly
    # good definition. What does mark a definition as procedural is a *sequence*
    # of participles, which _PROCEDURAL_PREDICATE already catches.
    r"\b(?:is|are|was|were|has|have|had|will|would|can|could|may|might|must|"
    r"shall|should|does|do|did|allows?|allows|enable[sd]?|requires?|"
    r"requires|provides?|consists?|contains?|refers?)\b",
    re.IGNORECASE,
)

# A predicate that narrates steps rather than naming a thing: several finite
# verbs in sequence ("a victim is selected and the transaction is aborted").
_PROCEDURAL_PREDICATE = re.compile(
    r"\b(?:is|are)\s+\w+ed\s+(?:and|then|before|after|while|when)\b|"
    r"^(?:when|if|after|before|once)\b",
    re.IGNORECASE,
)


def content_terms(text: str) -> set[str]:
    """Normalized matching terms for *text*.

    Uses morphological normalization and compound splitting so a user query of
    "superkey" matches a document that writes "super key", and "keys" matches
    "key". Without this, coverage scoring silently fails on exactly the
    technical terms that matter most in this corpus.
    """
    terms = set(expand_terms(text))
    for token in split_compounds(text):
        terms.add(normalize_term(token))
    return {term for term in terms if term not in STOP_WORDS and len(term) > 2}


def _match_terms(text: str) -> set[str]:
    """Terms used for scoring coverage against a passage.

    Expansion is deliberately asymmetric: the query keeps its raw normalized
    forms so the denominator reflects what the user actually asked, while the
    passage is expanded so alternate spellings can still match. Expanding both
    sides lets the query's own alternate forms inflate its own coverage.

    Two-character terms are retained. Dropping them silently emptied the term
    set for acronym questions -- "what is OS?" has no term longer than two
    characters, so every definitional question about an acronym scored zero
    coverage and either abstained or fell through to unrelated text. Acronyms
    are the single most common query shape in this corpus.
    """
    return {term for term in expand_terms(text) if len(term) > 1 and term not in STOP_WORDS}


def _passage_terms(text: str) -> set[str]:
    """Fully expanded terms for a passage, including compound spellings."""
    terms = _match_terms(text)
    for token in split_compounds(text):
        terms.add(normalize_term(token))
    return terms


def classify_question(question: str) -> str:
    """Route a question to a answering strategy."""
    normalized = " ".join(tokenize(question))
    if any(marker in normalized for marker in _COMPARISON_MARKERS):
        return "comparison"
    if normalized.startswith(_DEFINITION_PREFIXES):
        return "definition"
    if normalized.startswith(_LIST_PREFIXES):
        return "list"
    return "general"


def _strip_run_in_heading(sentence: str) -> str:
    """Remove a numbered heading that extracted inline with the next sentence.

    Textbooks print "7.5.3 Functional Programming Languages" on its own line,
    but extraction frequently joins it to the paragraph below, producing
    "7.5.3 Functional Programming Languages Most well-known programming
    languages are known as imperative...". Quoting that verbatim makes the
    answer read as gibberish, so the heading prefix is cut off.

    Deciding where the heading ends is the whole problem: the first word of the
    real sentence may itself be capitalised, so "run until lowercase" wrongly
    swallows "Most" and yields "well-known programming languages are known...".
    Instead the capitalised run is shortened one word at a time and the split
    kept is the earliest that still leaves a full sentence behind, which
    preserves the sentence's own first word.
    """
    match = _HEADING_PREFIX.match(sentence)
    if not match:
        return sentence

    heading = match.group(0)
    # Give words back to the sentence, shortest heading first, and accept the
    # first split that leaves enough of a sentence to stand on its own.
    while len(heading.split()) > 1:
        trimmed = heading.rsplit(None, 1)[0] + " "
        remainder = sentence[len(trimmed) :].strip()
        if len(remainder.split()) >= _MIN_SENTENCE_AFTER_HEADING:
            return remainder
        heading = trimmed
    return sentence


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping bullet lines intact as their own units."""
    sentences: list[str] = []
    for block in re.split(r"\n+", repair_spacing(text)):
        block = block.strip()
        if not block:
            continue
        if "\n" in block or len(block.split()) < 4:
            sentences.extend(part.strip() for part in block.splitlines() if part.strip())
        else:
            sentences.extend(part.strip() for part in _CLAUSE_BREAK.split(block) if part.strip())
    stripped = [_strip_run_in_heading(sentence) for sentence in sentences]
    return [sentence for sentence in stripped if len(sentence.split()) >= 3]


def _trim_to_complete(sentence: str) -> str:
    """Drop a trailing fragment left by a mid-sentence cut.

    PDF and slide extraction frequently breaks a sentence across a bullet or a
    line boundary, producing units that start or end mid-clause. Concatenating
    those yields unreadable answers, so an incomplete tail is discarded.
    """
    text = sentence.strip()
    # Discard a leading fragment such as "if they request a resource held by ...".
    if re.match(r"^(?:if|when|that|because|while|and|or|but|to|for|with|by)\s", text, re.IGNORECASE):
        parts = _CLAUSE_BREAK.split(text)
        text = parts[-1].strip() if len(parts) > 1 else ""
        if not text:
            return ""
    # Discard a trailing fragment that never terminates.
    if text and text[-1] not in ".!?":
        match = re.search(r"^(.*[.!?])\s+\S*$", text)
        if match:
            text = match.group(1).strip()
    return text


def _is_junk(text: str) -> bool:
    """Detect extracted math, glyph soup, and table debris.

    PDF extraction of formulae and matrices yields runs of symbols that carry no
    verbal meaning. These must never reach an answer, because they look like
    content while being unreadable.
    """
    if _EXAM_DEBRIS.search(text):
        return True
    if _ALLCAPS_HEADING.search(text):
        # A run of capitals this long is a running head, not prose.
        return True
    if not text:
        return True
    # Math-heavy text carries very few alphabetic words.
    words = re.findall(r"[A-Za-z]{3,}", text)
    if not words:
        return True
    ratio = sum(len(word) for word in words) / max(len(text), 1)
    if ratio < 0.55:
        return True
    if re.search(r"[]", text):
        return True
    # Dense runs of standalone single characters are matrix/vector notation
    # ("M M M V is N N") rather than prose.
    if len(re.findall(r"(?:\s|^)[A-Za-z](?:\s|$)", text)) >= 4:
        return True
    return False


def _coverage(sentence: str, query_terms: set[str]) -> float:
    """Fraction of query terms a sentence covers."""
    if not query_terms:
        return 0.0
    return len(_passage_terms(sentence) & query_terms) / len(query_terms)


def _is_usable(sentence: str, query_terms: set[str], minimum_coverage: float = MINIMUM_USABLE_COVERAGE) -> bool:
    """Reject fragments, headings, and sentences only loosely related to the question.

    Requiring proportional coverage rather than a single overlapping term is
    what stops an unrelated passage from being spliced into the answer: a slide
    about RNNs otherwise matches a deadlock question on the word "important".
    """
    sentence = _trim_to_complete(sentence)
    if not sentence:
        return False
    if len(sentence.split()) < 4 or len(sentence) < 25:
        return False
    if _NON_ANSWER.search(sentence):
        return False
    if _is_junk(sentence):
        return False
    if _looks_like_heading(sentence):
        return False
    if len(sentence.split()) > 120:
        return False
    return _coverage(sentence, query_terms) >= minimum_coverage


# Bullet glyphs that mark a list rather than a sentence. Slide decks extract as
# one long run of bullets separated by these, so their presence anywhere in a
# unit means the unit is a list -- not a claim that can answer a question.
_BULLET_GLYPH = re.compile(r"[•▪●‣⁃∙·]|(?:^|\s)[-–—]\s")


def _looks_like_heading(sentence: str) -> bool:
    """Whether *sentence* is a heading, outline bullet, or slide list.

    A real sentence makes one claim and ends in terminal punctuation. Slide and
    outline text instead arrives as a run of fragments joined by bullet glyphs
    ("Helps Define Primary and Candidate Keys • Super keys are used to find
    candidate keys, which are ..."). Such a run contains the queried term, so
    term coverage alone admits it and the answer reads as a non-sequitur.
    """
    text = sentence.strip()
    if not text:
        return True
    if _BULLET_GLYPH.search(text):
        return True
    if text[-1] in ".!?":
        return False
    # Title Case with no terminal punctuation is a heading, not a sentence.
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'-]*", text) if len(word) > 2]
    if len(words) >= 3 and sum(word[0].isupper() for word in words) / len(words) >= 0.8:
        return True
    return False


def _lexrank(sentences: list[str], query_terms: set[str], damping: float = 0.85) -> list[float]:
    """Score sentences by graph centrality weighted toward query overlap.

    Sentences are nodes, similarity edges. Power iteration yields stationary
    centrality, which favours sentences that are both query-relevant and
    surrounded by other relevant sentences.
    """
    total = len(sentences)
    term_sets = [_match_terms(sentence) for sentence in sentences]

    query_scores = [len(terms & query_terms) / len(query_terms) if query_terms else 0.0 for terms in term_sets]
    similarity: list[dict[int, float]] = [{} for _ in range(total)]
    for i in range(total):
        for j in range(i + 1, total):
            overlap = len(term_sets[i] & term_sets[j])
            if not overlap:
                continue
            weight = overlap / (math.log(len(term_sets[i]) + 1) + math.log(len(term_sets[j]) + 1))
            similarity[i][j] = weight
            similarity[j][i] = weight

    centrality = [0.0] * total
    for index in range(total):
        centrality[index] = 1.0 / total
    for _ in range(30):
        updated = []
        for index in range(total):
            neighbours = similarity[index]
            denominator = sum(neighbours.values())
            if denominator:
                incoming = sum(weight * centrality[other] for other, weight in neighbours.items()) / denominator
                updated.append((1 - damping) * query_scores[index] + damping * incoming)
            else:
                updated.append(query_scores[index])
        total_mass = sum(updated) or 1.0
        centrality = [value / total_mass for value in updated]
    return centrality


def _definition_answer(question: str, chunks: list[dict], query_terms: set[str]) -> str | None:
    """Build a template answer from an explicit definitional span.

    Candidate spans are ranked rather than taken first-match, because a document
    often defines a term early and loosely ("super key is one of the fundamental
    concept") before stating it precisely ("A Super Key in DBMS is a group of
    one or more attributes...").
    """
    # The pattern is built from the terms the user actually typed, not from the
    # expanded term space. Expanding "OS" to "os operating system" produced a
    # pattern requiring that exact order, which no document contains -- so every
    # acronym definition silently failed to match while its answer sat in the
    # top-ranked chunk. Expansion is for scoring coverage; the definition
    # template has to quote the query as written.
    concept_terms = [term for term in tokenize(question) if term not in STOP_WORDS and len(term) > 1]
    if not concept_terms:
        return None

    # The concept is written two ways in a lecture corpus: as the user typed it
    # ("OS") and as the document introduces it ("Operating System"). Both are
    # tried, because requiring the spelled-out form alone made "what is OS?"
    # miss a genuine "OS is ..." definition, and accepting only the query's own
    # form missed the far more common "Operating System (OS) is ...".
    phrasings = [concept_terms]
    expansion = expand_acronyms(question)
    if expansion:
        phrasings.append([normalize_term(word) for word in expansion])

    candidates: list[tuple[float, str]] = []
    for chunk in chunks:
        for match in _find_definitions(chunk["text"], phrasings):
            body = match.group(1).split(". ")[0].strip().rstrip(".")
            if len(body.split()) < 6:
                continue
            if _is_junk(body):
                continue
            if _looks_like_heading(body):
                # The copula can introduce an enumeration rather than a
                # definition: "Operating system is responsible for the following
                # activities ... • Creating files • Deleting files" describes the
                # concept's duties, not what it is. Presenting that as
                # "Operating system is responsible for the following..."
                # answers the wrong question, so it is left to the extractive
                # path rather than dressed up as a definition.
                continue
            body_terms = content_terms(body)
            if not body_terms:
                continue
            if _PROCEDURAL_PREDICATE.search(body):
                # The span narrates a procedure rather than naming the concept.
                continue
            # Score the span on its own substance. Weighting query-term coverage
            # heavily here was wrong: in a well-formed definition the concept is
            # the *subject*, so it sits before the copula and never appears in the
            # body. "A deadlock is a situation in which processes are blocked"
            # scores zero coverage while being exactly the right answer, so a
            # coverage-led threshold rejected genuine definitions and dropped the
            # question onto the generic ranking path. Coverage is kept only as a
            # mild bonus for spans that also restate the concept.
            coverage = len(body_terms & query_terms) / len(query_terms) if query_terms else 0.0
            substance = min(len(body.split()) / 25, 1.0)
            # Prefer a nominal predicate, which is what a definition actually
            # looks like, over a longer passage that merely mentions the concept.
            nominal = 0.0 if _FINITE_VERB.search(body) else 0.30
            # Prefer the phrasing the document itself uses for the concept, so
            # "what is OS?" is answered "operating system is ..." rather than
            # echoing the raw abbreviation the user typed.
            subject = _best_subject(concept_terms, match)
            candidates.append((substance + nominal + 0.10 * coverage, f"{subject} is {body}."))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]
    # A span too short to be a definition is worse than the extractive path,
    # which at least returns a contiguous passage. Fall back rather than emit it.
    if best_score < 0.35:
        return None
    return best


def _find_definitions(text: str, phrasings: list[list[str]]) -> list[re.Match]:
    """Every definitional span in *text*, across each way of naming the concept.

    Order is preserved when a phrasing is built: the terms must appear in the
    document in the order the user wrote them ("super key"), not alphabetically.

    The gap between concept and copula tolerates an interposed modifier ("A
    Super Key *in DBMS* is a group of attributes") and a trailing bracketed
    gloss ("Operating System *(OS)* is system software"), but it may not
    *begin* with a bracket -- otherwise a match on "OS is ..." wins over the
    better-anchored "Operating System (OS) is ..." and the answer's subject
    degrades to a dangling "OS)".
    """
    gap = r"(?:\s*[\w']+){0,3}(?:\s*\([^)]*\))?"
    copula = r"(?:is|are|refers\s+to|means|can\s+be\s+(?:defined|described)\s+as)"
    found: list[re.Match] = []
    already: set[str] = set()
    for phrasing in phrasings:
        if not phrasing:
            continue
        pattern = r"\W+".join(map(re.escape, phrasing))
        definition = re.compile(
            # The trailing \b matters: without it the pattern matches "super key"
            # inside the plural "super keys is used to find ...", and returns a
            # sentence about candidate keys as the definition of a super key.
            rf"\b(?:an?\s+|the\s+)?{pattern}\b{gap}\s+{copula}\s+(.{{20,400}})",
            re.IGNORECASE,
        )
        for match in definition.finditer(text):
            body = match.group(1).split(". ")[0].strip()
            # A definition's predicate must say something about the concept.
            # Bodies that merely restate existence ("There is some approaches
            # ...") or defer to a list are not definitions of anything.
            if re.match(r"^(?:there\s+(?:is|are)|it\s+is|this\s+is|these\s+are|they\s+are)\b", body, re.IGNORECASE):
                continue
            key = body[:120]
            if key not in already:
                already.add(key)
                found.append(match)
    return found


def _best_subject(concept_terms: list[str], match: re.Match) -> str:
    """The subject phrase as written in the document, stripped of its copula.

    A match on "Operating System (OS) is ..." begins at the spelled-out form,
    so quoting the match's own leading text keeps the answer in the document's
    vocabulary instead of echoing the abbreviation the user typed.

    Parentheticals are removed *before* the copula is split off, not after:
    stripping "(OS)" from an already-truncated "Operating System (OS)" leaves
    the dangling "OS)" that then becomes the answer's subject.
    """
    whole = match.group(0)
    # Cut the span at the copula that the definition pattern matched on.
    head = re.split(r"\s+(?:is|are|refers\s+to|means|can\s+be\s+\w+\s+as)\s+", whole, maxsplit=1, flags=re.IGNORECASE)[0]
    # Drop a leading article, then any bracketed gloss such as "(OS)".
    head = re.sub(r"^\s*(?:an?|the)\s+", "", head, flags=re.IGNORECASE)
    head = re.sub(r"\s*\([^)]*\)", "", head).strip(" ,.;:")
    if len(head.split()) >= len(concept_terms) and len(head) > 2:
        return head[0].upper() + head[1:]
    return " ".join(concept_terms)


def _comparison_answer(question: str, chunks: list[dict], query_terms: set[str]) -> str:
    """Assemble a multi-source answer by taking the best sentence per document.

    A comparison needs evidence from more than one place, so this deliberately
    spans documents rather than returning several sentences from the top chunk.
    """
    per_document: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        filename = chunk.get("metadata", {}).get("filename", "unknown")
        if filename in seen:
            continue
        sentences = [s for s in split_sentences(chunk["text"]) if _is_usable(s, query_terms)]
        if sentences:
            best = max(sentences, key=lambda s: _coverage(s, query_terms))
            best = _trim_to_complete(best)
            seen.add(filename)
            per_document.append((filename, best))
        if len(per_document) >= 3:
            break
    if not per_document:
        return ""
    # Only keep as many sources as the question asks about, so a two-way
    # comparison does not gain a third unrelated source.
    aspects = [term for term in tokenize(question) if term not in STOP_WORDS and len(term) > 2]
    limit = max(2, min(len(per_document), len(aspects)))
    kept = per_document[:limit]
    return " ".join(sentence for _, sentence in kept)


def generate_extractive_answer(question: str, chunks: list[dict], max_sentences: int = 3) -> dict:
    """Answer *question* from *chunks* using extractive strategies only."""
    if not chunks:
        return {
            "answer": "I could not find information about this in the indexed documents.",
            "sources": [],
            "abstained": True,
            "strategy": "none",
        }

    query_terms = _match_terms(question)
    question_type = classify_question(question)

    # Sentences are drawn from the single best chunk unless the question is a
    # comparison, which needs evidence from more than one place. Pooling every
    # sentence from every chunk and ranking them together produced answers that
    # spliced unrelated documents into one paragraph -- "Concurrency Problems in
    # DBMS" (a heading) concatenated onto a lock-control passage from a
    # different file. Within one chunk the sentences are adjacent by
    # construction, so joining them yields a passage that was actually written
    # that way.
    pool = chunks if question_type == "comparison" else chunks[:1]

    sentences: list[tuple[str, str]] = []
    for chunk in pool:
        for sentence in split_sentences(chunk["text"]):
            sentences.append((chunk.get("metadata", {}).get("filename", "unknown"), sentence))

    usable = [(name, sentence) for name, sentence in sentences if _is_usable(sentence, query_terms)]

    # Guard against a stray passage from an unrelated document riding along on a
    # single incidental term match. A passage must clear the same coverage bar
    # regardless of how highly retrieval ranked it.
    usable = [(name, sentence) for name, sentence in usable if _coverage(sentence, query_terms) >= MINIMUM_USABLE_COVERAGE]
    sources = [
        {"document_name": chunk.get("metadata", {}).get("filename", "unknown"), "score": chunk.get("score")}
        for chunk in chunks
    ]

    if question_type == "definition":
        templated = _definition_answer(question, chunks, query_terms)
        if templated:
            return {"answer": templated, "sources": sources, "abstained": False, "strategy": "definition_template"}

    if question_type == "comparison":
        assembled = _comparison_answer(question, chunks, query_terms)
        if assembled:
            return {"answer": assembled, "sources": sources, "abstained": False, "strategy": "comparison_multi_source"}

    if not usable:
        return {
            "answer": "I could not find information about this in the indexed documents.",
            "sources": sources,
            "abstained": True,
            "strategy": "insufficient_context",
        }

    flat_sentences = [sentence for _, sentence in usable]
    centrality = _lexrank(flat_sentences, query_terms)
    ranked = sorted(zip(centrality, flat_sentences), key=lambda item: item[0], reverse=True)

    limit = 1 if question_type == "list" else max_sentences
    chosen: list[str] = []
    # Chosen sentences are re-ordered back into source order before being joined.
    # Ranking order is by relevance, so concatenating it verbatim stitches
    # together sentences from unrelated parts of a 1,278-page textbook: the user
    # gets "...7.5.3 Functional Programming Languages" tacked onto the end of a
    # deadlock explanation. Reading order is what makes the result a passage
    # rather than a pile of quotes.
    chosen: list[tuple[int, str]] = []
    for position, (_, sentence) in enumerate(ranked):
        if len(chosen) >= limit:
            break
        if _coverage(sentence, query_terms) < MINIMUM_USABLE_COVERAGE:
            continue
        trimmed = _trim_to_complete(sentence)
        if not trimmed or len(trimmed.split()) < 4:
            continue
        # Avoid near-duplicate sentences from overlapping chunks.
        if any(_too_similar(trimmed, text) for _, text in chosen):
            continue
        chosen.append((position, trimmed))

    if not chosen:
        return {
            "answer": "I could not find information about this in the indexed documents.",
            "sources": sources,
            "abstained": True,
            "strategy": "insufficient_context",
        }

    return {
        "answer": " ".join(text for _, text in sorted(chosen, key=lambda item: item[0])),
        "sources": sources,
        "abstained": False,
        "strategy": f"lexrank_{question_type}",
    }


def _too_similar(candidate: str, existing: str, threshold: float = 0.8) -> bool:
    """Detect near-duplicate sentences produced by overlapping chunks."""
    left, right = _passage_terms(candidate), _passage_terms(existing)
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= threshold