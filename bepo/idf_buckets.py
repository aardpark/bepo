"""Adaptive IDF Bucketing for PR Duplicate Detection.

Replaces hardcoded domain keyword buckets with corpus-learned TF-IDF weights.

WHY THIS EXISTS:
  Hardcoded domain buckets (auth, payments, ci, ...) fail on repos whose
  vocabulary doesn't match the predefined list. A Django repo's "migration"
  bucket gets zero entries; a Stripe repo's "payment" bucket collapses all PRs.
  IDF learns the repo's actual vocabulary — what terms discriminate in *this*
  corpus.

KEY INSIGHT:
  Terms with medium document frequency (1–50% of PRs) are the best bucket keys.
  - Too rare  (< 1%)  → noise, one-off typos, version numbers
  - Too common (> 50%) → generic stop words ("fix", "update", "add")
  - Sweet spot → "migration", "webhook", "oauth", "scheduler" etc.

ALGORITHM:
  1. fit(prs)         → compute df[term] and idf[term] from corpus
  2. top_terms(pr)    → top-k terms by TF-IDF for this PR
  3. bucket_prs(prs)  → dict[term → [prs that have it in top-k]]
  4. get_pairs(prs)   → deduplicated (pr_a, pr_b) pairs for comparison
                        + issue-ref cross-bucket index

INVARIANTS:
  I-DEDUP:     Every (a,b) pair yielded exactly once (stable pair key = min,max)
  I-FALLBACK:  corpus < min_corpus → all-pairs (no bucketing)
  I-STOP:      STOP_WORDS + too-common terms never become bucket keys
  I-CROSS:     PRs sharing an issue ref #NNN (N≥3 digits) always compared
  I-COVER:     A PR with no IDF terms goes into __misc__ bucket (still compared)
  I-COMPRESS:  For N≥50 PRs with clear topics, pair count < 25% of all-pairs

PERFORMANCE:
  O(N * top_k * B) where B = avg bucket size (≪ N for topically clustered repos)
  vs O(N²) for all-pairs.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterator

# ---------------------------------------------------------------------------
# Stop words — terms that should never be bucket keys
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset({
    # English function words
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "this", "that", "these", "those",
    "it", "its", "we", "you", "he", "she", "they", "them", "their",
    "not", "no", "nor", "all", "each", "some", "than", "very", "just",
    # Extremely generic PR words (corpus-agnostic stop words)
    "fix", "fixes", "fixed", "add", "adds", "added", "update", "updates",
    "updated", "change", "changes", "changed", "remove", "removes", "removed",
    "use", "uses", "used", "make", "makes", "made", "get", "set", "run",
    "build", "test", "new", "old", "also", "only", "same", "more",
    "support", "improve", "implement", "refactor", "cleanup", "clean",
    "issue", "pr", "branch", "commit", "merge", "pull", "request",
    "version", "release", "bump", "upgrade", "downgrade",
    # Common code tokens
    "def", "class", "import", "return", "self", "none", "true", "false",
    # File extensions (from path tokenization)
    "py", "js", "ts", "md", "txt", "json", "yaml", "yml", "toml",
    "rst", "cfg", "ini", "sh", "html", "css", "sql", "lock",
    # Very short
    "var", "val", "src", "lib", "bin", "pkg", "doc", "api",
})

# Document frequency bounds (as fraction of corpus size)
MIN_DF_RATIO: float = 0.01   # Must appear in ≥1% of PRs
MAX_DF_RATIO: float = 0.50   # Must appear in ≤50% of PRs

# Absolute minimum DF count (prevents singletons in tiny corpora)
MIN_DF_COUNT: int = 2

# Minimum corpus size to activate IDF (below this: all-pairs fallback)
MIN_CORPUS_DEFAULT: int = 20


# ---------------------------------------------------------------------------
# Text extraction & tokenization
# ---------------------------------------------------------------------------

_ISSUE_RE = re.compile(r"#(\d{3,})")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")


def _tokenize(text: str) -> list[str]:
    """Lowercase → extract alphanumeric tokens ≥3 chars → filter stop words."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def _pr_text(pr: dict) -> str:
    """Combine all searchable PR text into one string.

    Title is weighted 3× (repeated) because it's the most concise signal.
    Body is capped at 2000 chars to avoid description walls drowning signals.
    File paths are path-tokenized: 'auth/oauth2/handler.py' → auth oauth handler.
    """
    parts: list[str] = []

    title = pr.get("title", "")
    if title:
        parts += [title, title, title]  # 3× title weight

    body = pr.get("body") or ""
    if body:
        parts.append(body[:2000])

    for f in pr.get("files", []):
        parts.extend(_TOKEN_RE.findall(f.lower()))

    for lbl in pr.get("labels", []):
        parts.append(re.sub(r"[-_]", " ", lbl))

    return " ".join(parts)


def _pr_id(pr: dict) -> str:
    """Stable string key for a PR."""
    return str(pr.get("number", pr.get("id", id(pr))))


def _pair_key(a: dict, b: dict) -> tuple[str, str]:
    ka, kb = _pr_id(a), _pr_id(b)
    return (min(ka, kb), max(ka, kb))


def _issue_refs(pr: dict) -> set[str]:
    """Extract 3+-digit issue refs from title + body."""
    text = f"{pr.get('title', '')} {pr.get('body') or ''}"
    return set(_ISSUE_RE.findall(text))


# ---------------------------------------------------------------------------
# IDF model
# ---------------------------------------------------------------------------

@dataclass
class IDFModel:
    """Fitted IDF model from a PR corpus.

    Invariants:
    - I-FILTER:  idf only contains terms that passed df bounds
    - I-SMOOTH:  uses sklearn-style smoothed IDF: log((N+1)/(df+1)) + 1
    - I-STABLE:  same corpus → same idf dict (deterministic)
    """
    n: int                    # Total PRs in corpus
    df: dict[str, int]        # term → raw document frequency
    idf: dict[str, float]     # term → smoothed IDF score (filtered)

    def score_pr(self, pr: dict) -> dict[str, float]:
        """TF-IDF scores for all terms in a PR that have IDF coverage."""
        text = _pr_text(pr)
        tokens = _tokenize(text)
        if not tokens:
            return {}
        tf_raw = Counter(tokens)
        n_tok = len(tokens)
        return {
            term: (count / n_tok) * self.idf[term]
            for term, count in tf_raw.items()
            if term in self.idf
        }

    def top_terms(self, pr: dict, k: int = 3) -> list[str]:
        """Top-k terms by TF-IDF weight. Returns [] if no coverage."""
        scores = self.score_pr(pr)
        if not scores:
            return []
        return sorted(scores, key=scores.__getitem__, reverse=True)[:k]

    @property
    def vocab_size(self) -> int:
        return len(self.idf)

    def __repr__(self) -> str:
        return f"IDFModel(n={self.n}, vocab={self.vocab_size})"


def fit_idf(prs: list[dict]) -> IDFModel:
    """Compute IDF model from a list of PR dicts.

    Each PR dict should have at minimum: title (str), body (str|None),
    files (list[str]). Other fields (labels, number) are optional.

    Returns:
        IDFModel with idf scores for terms in the discriminating range.
    """
    n = len(prs)
    df: Counter[str] = Counter()

    for pr in prs:
        text = _pr_text(pr)
        # Use *set* of tokens so each term counts ≤1 per PR for df
        for token in set(_tokenize(text)):
            df[token] += 1

    min_df = max(MIN_DF_COUNT, math.ceil(n * MIN_DF_RATIO))
    max_df = int(n * MAX_DF_RATIO)

    idf: dict[str, float] = {}
    for term, count in df.items():
        if min_df <= count <= max_df:
            idf[term] = math.log((n + 1) / (count + 1)) + 1.0

    return IDFModel(n=n, df=dict(df), idf=idf)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def bucket_prs(
    prs: list[dict],
    model: IDFModel,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Group PRs into overlapping buckets by their top distinctive terms.

    Each PR is added to up to top_k buckets (one per top term).
    PRs with no IDF coverage go into __misc__.

    Overlapping buckets are intentional — a "django migrations" PR might
    appear in both the "migrations" and "django" buckets, maximising recall.

    Args:
        prs:    PR dicts
        model:  Fitted IDFModel
        top_k:  Max buckets per PR (overlap factor)

    Returns:
        dict[bucket_key → [PR, ...]]
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for pr in prs:
        terms = model.top_terms(pr, k=top_k)
        if not terms:
            buckets["__misc__"].append(pr)
        else:
            for term in terms:
                buckets[term].append(pr)
    return dict(buckets)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def get_pairs(
    prs: list[dict],
    model: IDFModel | None = None,
    top_k: int = 3,
    min_corpus: int = MIN_CORPUS_DEFAULT,
) -> Iterator[tuple[dict, dict]]:
    """Generate deduplicated (pr_a, pr_b) pairs for comparison.

    Invariants:
    - I-DEDUP:  every pair yielded at most once
    - I-CROSS:  PRs sharing issue ref #NNN (N≥3 digits) always compared
    - I-FALLBACK: corpus < min_corpus → all-pairs
    - I-COVER:  every PR is in ≥1 bucket (__misc__ if no IDF terms)

    Args:
        prs:        PR dicts
        model:      Pre-fitted IDFModel (fitted from prs if None)
        top_k:      Bucket overlap factor
        min_corpus: Threshold for IDF activation

    Yields:
        (pr_a, pr_b) tuples (each pair exactly once)
    """
    seen: set[tuple[str, str]] = set()

    def _emit(a: dict, b: dict) -> Iterator[tuple[dict, dict]]:
        key = _pair_key(a, b)
        if key not in seen:
            seen.add(key)
            yield (a, b)

    n = len(prs)

    # Fallback: too small for IDF → all-pairs
    if n < min_corpus or model is None:
        if model is None and n >= min_corpus:
            model = fit_idf(prs)
        else:
            for i, a in enumerate(prs):
                for b in prs[i + 1:]:
                    yield from _emit(a, b)
            return

    if model is None:
        model = fit_idf(prs)

    # Bucket-based pairs
    buckets = bucket_prs(prs, model, top_k=top_k)
    for bucket_prs_list in buckets.values():
        for i, a in enumerate(bucket_prs_list):
            for b in bucket_prs_list[i + 1:]:
                yield from _emit(a, b)

    # Issue ref cross-bucket index
    issue_index: dict[str, list[dict]] = defaultdict(list)
    for pr in prs:
        for ref in _issue_refs(pr):
            issue_index[ref].append(pr)

    for ref_prs in issue_index.values():
        if len(ref_prs) >= 2:
            for i, a in enumerate(ref_prs):
                for b in ref_prs[i + 1:]:
                    yield from _emit(a, b)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdaptiveIDFBucketer:
    """High-level interface for adaptive IDF bucketing.

    Usage::

        bucketer = AdaptiveIDFBucketer()
        bucketer.fit(prs)

        # Iterate pairs to compare
        for pr_a, pr_b in bucketer.get_pairs(prs):
            score = compare(pr_a, pr_b)

        # Inspect the model
        print(bucketer.model)
        print(bucketer.compression_ratio(prs))
        print(bucketer.top_bucket_terms(n=10))
    """

    def __init__(self, top_k: int = 3, min_corpus: int = MIN_CORPUS_DEFAULT):
        """
        Args:
            top_k:      Number of top IDF terms per PR (controls bucket overlap).
                        Higher → more pairs found, less compression.
            min_corpus: Min corpus size for IDF activation.
                        Below this → all-pairs fallback.
        """
        self.top_k = top_k
        self.min_corpus = min_corpus
        self.model: IDFModel | None = None

    def fit(self, prs: list[dict]) -> "AdaptiveIDFBucketer":
        """Fit IDF model from the PR corpus. Returns self for chaining."""
        if len(prs) >= self.min_corpus:
            self.model = fit_idf(prs)
        return self

    def get_pairs(self, prs: list[dict]) -> Iterator[tuple[dict, dict]]:
        """Yield deduplicated pairs to compare."""
        yield from get_pairs(
            prs,
            model=self.model,
            top_k=self.top_k,
            min_corpus=self.min_corpus,
        )

    def bucket_summary(self, prs: list[dict]) -> dict[str, int]:
        """Returns dict[bucket_key → bucket_size], sorted by size desc."""
        if self.model is None:
            return {"__all__": len(prs)}
        b = bucket_prs(prs, self.model, top_k=self.top_k)
        return dict(sorted(b.items(), key=lambda kv: -len(kv[1])))

    def top_bucket_terms(self, prs: list[dict] | None = None, n: int = 20) -> list[tuple[str, float]]:
        """Top-n IDF terms by score (the likely bucket keys).

        Returns list of (term, idf_score) sorted descending.
        """
        if self.model is None:
            return []
        return sorted(
            self.model.idf.items(),
            key=lambda kv: -kv[1],
        )[:n]

    def compression_ratio(self, prs: list[dict]) -> float:
        """Fraction of all-pairs generated. 0.1 = 90% reduction."""
        n = len(prs)
        all_pairs = n * (n - 1) // 2
        if all_pairs == 0:
            return 1.0
        generated = sum(1 for _ in self.get_pairs(prs))
        return generated / all_pairs

    def stats(self, prs: list[dict]) -> dict:
        """Summary statistics for reporting."""
        n = len(prs)
        all_pairs = n * (n - 1) // 2
        buckets = self.bucket_summary(prs)
        generated = sum(1 for _ in self.get_pairs(prs))
        return {
            "corpus_size": n,
            "all_pairs": all_pairs,
            "pairs_generated": generated,
            "compression_ratio": round(generated / all_pairs, 4) if all_pairs else 1.0,
            "bucket_count": len(buckets),
            "avg_bucket_size": round(sum(len(v) for v in buckets.values()) / len(buckets), 1) if buckets else 0,
            "vocab_size": self.model.vocab_size if self.model else 0,
            "top_buckets": [(k, len(v)) for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:10]],
        }
