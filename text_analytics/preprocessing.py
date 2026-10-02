"""Text normalization and tokenization utilities."""

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# Tokens that would otherwise match every document and dilute ranking. Without
# these excluded, any long passage accumulates enough of them to clear a
# coverage threshold regardless of actual relevance.
_NOISE_TERMS = {
    "a", "an", "the", "of", "and", "or", "but", "to", "in", "on", "at", "by",
    "for", "from", "with", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "there", "here",
    "he", "she", "they", "we", "you", "i", "not", "no", "so", "than", "then",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
    "if", "because", "while", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "also", "such", "into", "over", "under", "out", "up", "down", "about",
    "between", "among", "through", "during", "after", "before", "above",
    "below", "very", "more", "most", "other", "some", "any", "each", "both",
}


# Typographic ligatures that PDF text layers emit as single codepoints. Left
# alone they reach answers verbatim: "efficient" arrives as "efﬁcient", which
# both reads badly and stops the term from ever matching a query that spells it
# normally. Only these seven are replaced -- a blanket NFKC would also rewrite
# mathematical and technical symbols, which this corpus is full of.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
}
_LIGATURE_PATTERN = re.compile("|".join(_LIGATURES))

# Symbol and math fonts in PDFs frequently emit unpaired surrogates (U+D800 to
# U+DFFF) -- a surrogate escape that lost its partner during extraction, so it
# corresponds to no character. Python cannot UTF-8 encode these, so any code
# that hashes or serializes document text -- the embedding cache fingerprint,
# JSON writes -- raises UnicodeEncodeError on them. They are dropped here, at
# ingestion, rather than defended against at every consumer.
_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")


def strip_surrogates(text: str) -> str:
    """Drop unpaired surrogate codepoints that cannot be encoded or serialized."""
    return _SURROGATE_PATTERN.sub("", text)


def preprocess_text(text: str) -> str:
    """Normalize whitespace while retaining case for readable answers."""
    text = _SURROGATE_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", _LIGATURE_PATTERN.sub(lambda m: _LIGATURES[m.group(0)], text)).strip()


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens used for retrieval and analytics."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _strip_combining_marks(text: str) -> str:
    """Remove diacritics so terms match regardless of transliteration."""
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def normalize_term(term: str) -> str:
    """Normalize a single term for matching.

    Applies diacritic folding plus light singular folding so that a user's
    "keys" matches a document's "key", and plural/singular variants of a
    technical term still align. Kept deliberately shallow — an aggressive stemmer
    collapses distinct concepts such as "process" and "processing" that matter
    in this corpus.

    Acronyms are exempt from plural folding. Folding "DBMS" produced "dbm",
    which is not a word and matches nothing, so every query about the database
    management system silently lost its own acronym. Case is the only reliable
    signal that a token is an abbreviation rather than a plural.
    """
    # Acronyms are exempt from plural folding. Folding "DBMS" produced "dbm",
    # which is not a word and matches nothing, so every query about the database
    # management system silently lost its own acronym.
    #
    # Membership of ACRONYM_EXPANSIONS is the signal, not letter case:
    # `tokenize` lowercases before this function sees the term, so the original
    # casing that would distinguish "DBMS" from "keys" is already gone by here.
    if term in ACRONYM_EXPANSIONS:
        return term
    normalized = _strip_combining_marks(term).lower()
    if len(normalized) > 3 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if len(normalized) > 3 and normalized.endswith("es") and normalized[-3] in "sxzh":
        return normalized[:-2]
    if len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
        return normalized[:-1]
    return normalized


# Acronyms that this corpus spells out in prose. Lecture material introduces
# "operating system (OS)" once and then uses only the acronym, so a query of
# "what is OS?" shares almost no vocabulary with the sentence that defines it.
# Expanding the acronym on the query side is what lets those questions match.
#
# Deliberately limited to terms that genuinely appear both ways in this corpus.
# A larger, guessed table would manufacture matches that look confident and are
# wrong, which is worse than abstaining.
ACRONYM_EXPANSIONS = {
    "os": ["operating", "system"],
    "dbms": ["database", "management", "system"],
    "ds": ["data", "structure"],
    "sql": ["structured", "query", "language"],
    "tcp": ["transmission", "control", "protocol"],
    "ip": ["internet", "protocol"],
    "http": ["hypertext", "transfer", "protocol"],
    "dns": ["domain", "name", "system"],
    "ram": ["random", "access", "memory"],
    "cpu": ["central", "processing", "unit"],
    "gpu": ["graphics", "processing", "unit"],
    "osio": ["open", "systems", "interconnection", "model"],
    "fcfs": ["first", "come", "first", "served"],
    "sjf": ["shortest", "job", "first"],
    "rr": ["round", "robin"],
    "klr": ["kernel", "long", "running"],
    "rnn": ["recurrent", "neural", "network"],
    "lstm": ["long", "short", "term", "memory"],
    "gru": ["gated", "recurrent", "unit"],
    "svm": ["support", "vector", "machine"],
    "svd": ["singular", "value", "decomposition"],
    "hosvd": ["higher", "order", "singular", "value", "decomposition"],
    "ann": ["artificial", "neural", "network"],
    "cnn": ["convolutional", "neural", "network"],
    "pca": ["principal", "component", "analysis"],
    "pde": ["partial", "differential", "equation"],
    "ode": ["ordinary", "differential", "equation"],
    "nf": ["normal", "form"],
    "3nf": ["third", "normal", "form"],
    "2nf": ["second", "normal", "form"],
    "1nf": ["first", "normal", "form"],
    "bcnf": ["boyce", "codd", "normal", "form"],
    "fd": ["functional", "dependency"],
    "er": ["entity", "relationship"],
    "acd": ["atomicity", "consistency", "durability"],
    "acid": ["atomicity", "consistency", "isolation", "durability"],
}


def expand_acronyms(text: str) -> list[str]:
    """Add the spelled-out form of any acronym appearing in *text*.

    Returns the acronym plus its expansion, so a passage that writes
    "operating system" and a query of "OS" converge on the same terms. Only the
    query side needs expanding: the passage is scored on vocabulary it actually
    contains, and inflating it with expansions it does not use would let a
    query match text on terms that were never written.
    """
    tokens = tokenize(text)
    extras: list[str] = []
    for token in tokens:
        expansion = ACRONYM_EXPANSIONS.get(token)
        if expansion:
            extras.extend(expansion)
    return extras


def expand_terms(text: str) -> list[str]:
    """Tokenize *text* into normalized, deduplicated matching terms.

    Emits each normalized token once. This is the term space used for
    query/document term matching, where morphological variants must collapse to
    the same key. Use `tokenize` directly where raw occurrences matter, such as
    counting term frequencies for BM25.

    Acronyms are expanded into their component words, so "DBMS" contributes
    "database", "management" and "system" to the term space. A lecture deck
    introduces each abbreviation once and then uses only the acronym, so
    without this a query of "what is DBMS?" has no vocabulary in common with
    the sentence that actually defines it.
    """
    seen: dict[str, None] = {}
    for token in tokenize(text):
        normalized = normalize_term(token)
        if normalized and normalized not in _NOISE_TERMS:
            seen.setdefault(normalized, None)
    for expansion_word in expand_acronyms(text):
        normalized = normalize_term(expansion_word)
        if normalized and normalized not in _NOISE_TERMS:
            seen.setdefault(normalized, None)
    return list(seen)


def split_compounds(text: str) -> list[str]:
    """Split concatenated technical terms into their component words.

    A user searching "superkey" should match a document that writes
    "super key". Emits the original tokens plus, for known and unknown
    concatenations of two tokens, the split form — enough to bridge the
    spelling gap without requiring a dictionary.
    """
    expanded = tokenize(text)
    parts: list[str] = []
    for token in expanded:
        parts.append(token)
        pieces = _split_known_compound(token)
        parts.extend(pieces)
    return parts


def _split_known_compound(token: str) -> list[str]:
    """Best-effort split of a run-together word into two known terms.

    Only genuine orthographic concatenations are listed. Semantic near-pairs
    such as "deadlock" must NOT be split: both spellings already match
    themselves, and splitting would let the parts match unrelated questions.
    """
    for first, second in _COMPOUND_SPLITS:
        if token == first + second:
            return [first, second]
    return []


# Orthographic concatenations: the one-word form is not a real word, while the
# two-word form is. These are safe to split in both directions.
_COMPOUND_SPLITS = [
    ("super", "key"),
    ("primary", "key"),
    ("foreign", "key"),
    ("candidate", "key"),
    ("normal", "form"),
    ("multi", "thread"),
    ("back", "prop"),
    ("web", "socket"),
    ("file", "system"),
    ("data", "base"),
    ("superkey", "key"),
    ("multithread", "thread"),
]
