"""PR fingerprinting and duplicate detection."""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

from .idf_buckets import AdaptiveIDFBucketer

# Splits file paths on common delimiters for word-boundary domain matching.
_PATH_SPLIT_RE = re.compile(r'[/._\-]')


@dataclass
class Fingerprint:
    """Fingerprint of a PR's changes."""
    pr_id: str
    title: str = ""
    files_touched: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    issue_refs: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    code_lines: list[str] = field(default_factory=list)  # normalized changed lines

    def similarity(
        self,
        other: "Fingerprint",
        idf_weights: dict[str, float] | None = None,
        valid_issue_refs: set[str] | None = None,
    ) -> float:
        """Compute similarity score between two fingerprints.

        Weights:
        - Issue ref overlap (10.0): Same issue = definite duplicate
        - Code content overlap (8.0): Same changes = definite duplicate
        - File path overlap (5.0): Same files = likely related
        - Domain overlap (3.0): Same feature area
        - Imports (1.0): Similar dependencies

        Args:
            other: Another fingerprint to compare against
            idf_weights: Optional IDF weights for code lines. If provided,
                        rare lines are weighted more heavily than common ones.
            valid_issue_refs: If provided, only these refs count as issue
                             overlap. Refs outside this set are ignored even
                             if shared. Used for per-pair parent-ref filtering.
        """
        scores = []
        weights = []

        # Issue reference overlap - strongest signal
        if self.issue_refs and other.issue_refs:
            r1, r2 = set(self.issue_refs), set(other.issue_refs)
            shared = r1 & r2
            if valid_issue_refs is not None:
                shared &= valid_issue_refs
            if shared:
                scores.append(1.0)
                weights.append(10.0)

        # Code content similarity - what actually changed
        code_sim = 0.0
        if self.code_lines and other.code_lines:
            c1, c2 = set(self.code_lines), set(other.code_lines)
            if c1 | c2:
                if idf_weights:
                    # IDF-weighted: rare lines matter more
                    code_sim = _idf_weighted_jaccard(c1, c2, idf_weights)
                else:
                    # Plain Jaccard
                    code_sim = len(c1 & c2) / len(c1 | c2)
                if code_sim > 0.1:  # only count if meaningful overlap
                    scores.append(code_sim)
                    weights.append(8.0)

        # Exact file overlap (strongest file signal)
        exact_files_a = set(self.files_touched)
        exact_files_b = set(other.files_touched)
        exact_overlap = exact_files_a & exact_files_b
        if exact_overlap and code_sim > 0.2:
            # Files AND code overlap together = very likely duplicate.
            # File overlap without code overlap is not a signal — concurrent
            # edits to the same hot file (community.md, CHANGELOG) are common
            # and coincidental.
            file_sim = len(exact_overlap) / len(exact_files_a | exact_files_b)
            scores.append(file_sim)
            weights.append(6.0)
        else:
            # Directory level similarity (weaker signal)
            def get_dirs(files):
                dirs = set()
                for f in files:
                    parts = f.split('/')
                    if len(parts) >= 2:
                        dirs.add('/'.join(parts[:-1]))
                return dirs
            f1, f2 = get_dirs(self.files_touched), get_dirs(other.files_touched)
            if f1 | f2:
                scores.append(len(f1 & f2) / len(f1 | f2))
                weights.append(2.0)  # lower weight for dir-only match

        # Domain similarity
        if self.domains or other.domains:
            d1, d2 = set(self.domains), set(other.domains)
            if d1 | d2:
                scores.append(len(d1 & d2) / len(d1 | d2))
                weights.append(3.0)

        # Import similarity
        if self.imports or other.imports:
            i1, i2 = set(self.imports), set(other.imports)
            if i1 | i2:
                scores.append(len(i1 & i2) / len(i1 | i2))
                weights.append(1.0)

        if not scores:
            return 0.0
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def _dominant_prefix(files: list[str], threshold: float = 0.8) -> str | None:
    """Return the deepest directory prefix covering ≥threshold of files.

    A PR is "concentrated" when most of its changed files live under one
    common subtree.  This function finds the deepest such prefix, but only
    considers prefixes with ≥2 path components (depth ≥2) — depth-1 entries
    like a top-level ``src/`` or ``tests/`` are too broad to be meaningful.

    Returns None when no depth-≥2 prefix covers ≥threshold of the files,
    indicating the PR is spread broadly across the repository.

    Examples::

        ['a/b/c.py', 'a/b/d.py', 'a/b/e.py']  →  'a/b'   (100%)
        ['a/b/c.py', 'a/b/d.py', 'a/x/e.py']  →  None    (67% < 80%)
        ['a/x.py', 'a/y.py']                   →  None    (depth-1 only)
    """
    if not files:
        return None
    n = len(files)
    prefix_counts: Counter[str] = Counter()
    for f in files:
        parts = f.split('/')
        for depth in range(2, len(parts)):  # depth ≥2 only
            prefix_counts['/'.join(parts[:depth])] += 1

    best: str | None = None
    best_depth = 0
    for prefix, count in prefix_counts.items():
        if count / n >= threshold:
            depth = prefix.count('/') + 1
            if depth > best_depth:
                best = prefix
                best_depth = depth
    return best


def _dominant_component_token(files: list[str], threshold: float = 0.6) -> str | None:
    """Return the deepest directory *name* (token) covering ≥threshold of files.

    Unlike ``_dominant_prefix``, this scans each depth level independently and
    looks for a single directory *name* — not a full path prefix — that appears
    in ≥threshold of the file paths at that depth.  This correctly handles
    repos where source and test files live at different top-level paths but
    share the same component name at a deeper level::

        'homeassistant/components/imou/__init__.py'  → depth-3 token: 'imou'
        'tests/components/imou/test_init.py'         → depth-3 token: 'imou'

    Both contribute to the same ``'imou'`` token, so it dominates at depth 3
    even though the full-path prefixes differ (``homeassistant/components/imou``
    vs ``tests/components/imou``).

    Only considers depths ≥2.  Returns the token from the deepest qualifying
    level, or None if no level has a dominant token.

    Examples::

        ['a/b/c.py', 'a/b/d.py', 'tests/b/e.py']  →  'b'    (depth-2, 100%)
        ['a/b/c.py', 'a/c/d.py', 'a/d/e.py']       →  None   (each 33%)
    """
    if not files:
        return None
    n = len(files)
    max_parts = max(len(f.split('/')) for f in files)
    best_token: str | None = None
    best_depth = 1  # must exceed depth 1 to qualify

    for depth in range(2, max_parts):  # 1-indexed depth ≥2
        token_counts: Counter[str] = Counter()
        for f in files:
            parts = f.split('/')
            # Only directory components: depth must be strictly less than the
            # number of parts so we do not count the filename itself.
            if depth < len(parts):
                token_counts[parts[depth - 1]] += 1  # 0-indexed access
        for token, count in token_counts.items():
            if count / n >= threshold and depth > best_depth:
                best_token = token
                best_depth = depth

    return best_token


def _is_concentrated_cross_component(fp_a: "Fingerprint", fp_b: "Fingerprint") -> bool:
    """True when both PRs are concentrated in distinct component subtrees.

    Uses depth-level token matching (``_dominant_component_token``) to identify
    the component each PR is primarily working in.  Token-based matching handles
    repos that mirror source and test files at different top-level paths::

        homeassistant/components/imou/  →  dominant token 'imou'
        tests/components/imou/          →  also contributes token 'imou'

    Returns True when:

    1. Both PRs have a dominant component token (≥60 % of files share one
       directory name at some depth ≥2).
    2. Those tokens differ — the PRs are working in different subtrees.

    A shared issue ref overrides this check at the call site.  Cross-component
    pairs that both reference the same issue are competing implementations of
    the same feature request — a genuine duplicate signal.
    """
    token_a = _dominant_component_token(fp_a.files_touched)
    token_b = _dominant_component_token(fp_b.files_touched)
    if token_a is None or token_b is None:
        return False
    return token_a != token_b


def _compute_idf_weights(fingerprints: list["Fingerprint"]) -> dict[str, float]:
    """Compute IDF weights for code lines across all fingerprints.

    IDF = log(N / df) where N is total documents and df is document frequency.
    Rare lines (appear in few PRs) get high weight.
    Common lines (appear in many PRs) get low weight.
    """
    n_docs = len(fingerprints)
    if n_docs == 0:
        return {}

    # Count how many fingerprints contain each code line
    df: Counter[str] = Counter()
    for fp in fingerprints:
        for line in set(fp.code_lines):  # set() to count each line once per doc
            df[line] += 1

    # Compute IDF: log(N / df)
    # Lines appearing in all docs get weight ~0, rare lines get high weight
    return {line: math.log(n_docs / count) for line, count in df.items()}


def _idf_weighted_jaccard(set_a: set[str], set_b: set[str], idf: dict[str, float]) -> float:
    """Compute IDF-weighted Jaccard similarity.

    Instead of plain |A∩B| / |A∪B|, weight each line by its IDF.
    Rare lines contribute more to similarity than common boilerplate.
    """
    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.0

    # Sum IDF weights (default to 1.0 for lines not in corpus)
    intersection_weight = sum(idf.get(line, 1.0) for line in intersection)
    union_weight = sum(idf.get(line, 1.0) for line in union)

    return intersection_weight / union_weight if union_weight > 0 else 0.0


@dataclass
class Duplicate:
    """A pair of similar PRs."""
    pr_a: str
    pr_b: str
    similarity: float
    shared_issues: list[str]
    shared_files: list[str]
    shared_domains: list[str]
    shared_code_lines: int  # count of overlapping code lines
    reason: str


# Domain keywords for categorizing PRs
DOMAINS = {
    # Messaging
    "telegram": "messaging", "whatsapp": "messaging", "discord": "messaging",
    "slack": "messaging", "matrix": "messaging", "twilio": "messaging",
    # Auth
    "auth": "auth", "login": "auth", "oauth": "auth", "jwt": "auth",
    "session": "auth", "token": "auth",
    # Data
    "cache": "cache", "redis": "cache", "database": "database", "db": "database",
    "postgres": "database", "mysql": "database", "mongo": "database",
    # API
    "api": "api", "endpoint": "api", "route": "api", "graphql": "api", "webhook": "api",
    # Scheduling
    "cron": "scheduling", "schedule": "scheduling", "job": "scheduling", "queue": "scheduling",
    # AI
    "llm": "ai", "model": "ai", "embedding": "ai", "openai": "ai", "anthropic": "ai",
    # Media
    "media": "media", "image": "media", "video": "media", "upload": "media",
    # Config
    "config": "config", "setting": "config", "env": "config",
    # Observability
    "log": "observability", "trace": "observability", "metric": "observability",
    # Plugins
    "plugin": "plugin", "extension": "plugin",
}


def _extract_files(diff: str) -> list[str]:
    """Extract file paths from diff."""
    files = []
    for line in diff.split('\n'):
        if line.startswith('+++ b/'):
            files.append(line[6:])
    return list(set(files))


def _extract_issues(text: str) -> list[str]:
    """Extract issue references (#123).

    Only captures 3+ digit numbers to avoid PR template noise like
    "step #1", "fix #2", or "#42" which are common boilerplate.
    """
    return list(set(m.group(1) for m in re.finditer(r'#(\d{3,})', text)))


def _extract_domains(files: list[str], code: str) -> list[str]:
    """Extract domains from file paths using word-boundary token matching.

    Splits each path on common delimiters (/ . - _) and checks whether any
    token *starts with* a domain keyword.  This prevents accidental substring
    matches such as 'log' in 'CHANGELOG.md' ('changelog' does not start with
    'log') while still recognising 'logger', 'logging', 'logfile', etc.

    Code content is not used: keywords like 'log', 'model', or 'token' appear
    in routine code across every domain and produced false assignments.
    """
    domains = set()
    for file_path in files:
        tokens = _PATH_SPLIT_RE.split(file_path.lower())
        for keyword, domain in DOMAINS.items():
            if domain not in domains and any(t.startswith(keyword) for t in tokens):
                domains.add(domain)
    return list(domains)


def _extract_imports(diff: str) -> list[str]:
    """Extract imports from diff."""
    imports = []
    for line in diff.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            line = line[1:]
            if line.lstrip().startswith('//'):  # skip JS/TS line comments
                continue
            m = re.search(r'import\s+.*\s+from\s+["\']([^"\']+)["\']', line)
            if m:
                imports.append(m.group(1))
            m = re.search(r'require\s*\(\s*["\']([^"\']+)["\']', line)
            if m:
                imports.append(m.group(1))
    return list(set(imports))


def _extract_new_code(diff: str) -> str:
    """Extract added lines from diff."""
    lines = []
    for line in diff.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            lines.append(line[1:])
    return '\n'.join(lines)


def _extract_code_lines(diff: str) -> list[str]:
    """Extract normalized code lines from diff for content comparison.

    Normalizes lines by stripping whitespace and filtering noise.
    Returns unique meaningful lines that represent actual code changes.
    """
    lines = set()
    for line in diff.split('\n'):
        # Get added and removed lines (both matter for comparison)
        if line.startswith('+') and not line.startswith('+++'):
            code = line[1:].strip()
        elif line.startswith('-') and not line.startswith('---'):
            code = line[1:].strip()
        else:
            continue

        # Skip empty lines and trivial changes
        if not code:
            continue
        if len(code) < 4:  # skip tiny fragments
            continue
        if code.startswith('//') or code.startswith('#') or code.startswith('*'):
            continue  # skip comments
        if code in ('{', '}', '(', ')', '[', ']', 'else', 'return', 'break', 'continue'):
            continue  # skip trivial syntax

        lines.add(code)

    return list(lines)


def fingerprint_pr(
    pr_id: str,
    diff: str,
    title: str = "",
    body: str = "",
) -> Fingerprint:
    """Create a fingerprint from a PR diff."""
    new_code = _extract_new_code(diff)
    files = _extract_files(diff)

    return Fingerprint(
        pr_id=pr_id,
        title=title,
        files_touched=files,
        domains=_extract_domains(files, new_code),
        issue_refs=_extract_issues(f"{title} {body} {diff}"),
        imports=_extract_imports(diff),
        code_lines=_extract_code_lines(diff),
    )



def find_duplicates(
    fingerprints: list[Fingerprint],
    threshold: float = 0.65,
    verbose: bool = False,
) -> list[Duplicate]:
    """Find duplicate PRs from a list of fingerprints.

    Uses domain bucketing to reduce comparisons: PRs are grouped by domain
    signal and only compared within the same bucket. Issue-ref matches
    (weight 10.0) always trigger cross-bucket comparison. PRs with no domain
    signals go into a catch-all bucket and are compared against everything.

    Uses IDF weighting for code similarity - rare lines matter more than
    common boilerplate like 'return null' or 'import React'.
    """
    if not fingerprints:
        return []

    idf_weights = _compute_idf_weights(fingerprints)

    # Build per-ref corpus: ref → list of fingerprints that reference it.
    ref_fps: dict[str, list[Fingerprint]] = {}
    for fp in fingerprints:
        for ref in fp.issue_refs:
            ref_fps.setdefault(ref, []).append(fp)

    # Global pre-filter: strip refs that appear in >3 PRs.
    # These are high-frequency release trackers or umbrella issues that would
    # flood the issue_index with spurious cross-bucket comparisons.
    # Refs with n ≤ 3 are left in fp.issue_refs and validated per-pair below.
    hot_refs = {ref for ref, fps in ref_fps.items() if len(fps) > 3}
    if hot_refs:
        for fp in fingerprints:
            fp.issue_refs = [ref for ref in fp.issue_refs if ref not in hot_refs]

    # Rebuild after stripping so ref_fps only contains n ≤ 3 refs.
    ref_fps = {}
    for fp in fingerprints:
        for ref in fp.issue_refs:
            ref_fps.setdefault(ref, []).append(fp)

    def _valid_issue_refs_for_pair(fp_a: Fingerprint, fp_b: Fingerprint) -> set[str]:
        """Return the subset of shared issue refs that are valid signals for this pair.

        A shared ref is valid only when there is corroborating evidence that
        the two PRs are genuinely working in the same area:

          1. Same real domain: both PRs have at least one domain keyword in
             common (e.g. both touch "config" or "messaging" files). This
             is the strongest structural signal.

          2. Both catchall + exclusive ref (n == 2): the ref was raised by
             exactly these two PRs and no others. Without domain signals,
             exclusivity is the best available indicator.

          3. Both catchall + code overlap (≥ 3 lines): shared code lines
             corroborate that the PRs touch the same code area even without
             detectable domain keywords.

        Cross-domain pairs (one or both have a real domain but they differ)
        always get zero ref credit — a shared ref between, say, a scheduling
        PR and a messaging PR is a parent/tracking issue regardless of how
        many PRs reference it.
        """
        shared_refs = set(fp_a.issue_refs) & set(fp_b.issue_refs)
        if not shared_refs:
            return set()

        d_a = set(fp_a.domains)
        d_b = set(fp_b.domains)

        # Case 1: both PRs have real domain signals with sufficient overlap.
        # A plain non-empty intersection is too weak — many PRs share generic
        # domains (config, api) that don't indicate related work.  Jaccard ≥
        # 0.25 requires the shared domains to be at least 25% of the union,
        # filtering out single-generic-domain overlap between unrelated PRs.
        if d_a and d_b:
            d_inter = d_a & d_b
            jaccard = len(d_inter) / len(d_a | d_b)
            if jaccard >= 0.25:
                return shared_refs
            return set()  # weak domain overlap → treat as cross-domain

        # Case 2 & 3 apply only when BOTH PRs are catchall (no domain info)
        if d_a or d_b:
            # One has real domains, the other is catchall → cross-domain
            return set()

        # Both catchall: validate each ref individually
        code_overlap = len(set(fp_a.code_lines) & set(fp_b.code_lines))
        valid: set[str] = set()
        for ref in shared_refs:
            n = len(ref_fps.get(ref, []))
            if n == 2 or code_overlap >= 3:  # exclusive or code-corroborated
                valid.add(ref)
        return valid

    # Best similarity per unordered pair; deduplicates cross-bucket hits.
    seen: dict[tuple[str, str], float] = {}

    def _compare_and_record(fp_a: Fingerprint, fp_b: Fingerprint) -> None:
        if fp_a.pr_id == fp_b.pr_id:
            return
        key = (min(fp_a.pr_id, fp_b.pr_id), max(fp_a.pr_id, fp_b.pr_id))
        valid_refs = _valid_issue_refs_for_pair(fp_a, fp_b)
        sim = fp_a.similarity(fp_b, idf_weights=idf_weights, valid_issue_refs=valid_refs)
        if sim >= threshold:
            if key not in seen or sim > seen[key]:
                seen[key] = sim

    # Build PR dicts for corpus-learned IDF bucketing.
    # Use pr_id as the "number" field so lookups into fp_by_id work directly.
    # Issue refs go into body so the bucketer's cross-bucket issue-ref index
    # guarantees that any pair sharing a ref is always compared (I-CROSS).
    pr_dicts = [
        {
            "number": fp.pr_id,
            "title": fp.title,
            "body": " ".join(f"#{ref}" for ref in fp.issue_refs),
            "files": fp.files_touched,
            "labels": [],
        }
        for fp in fingerprints
    ]

    bucketer = AdaptiveIDFBucketer(top_k=3)
    bucketer.fit(pr_dicts)

    if verbose:
        stats = bucketer.stats(pr_dicts)
        print(
            f"  IDF bucketing: {stats['pairs_generated']} pairs"
            f" from {stats['corpus_size']} PRs"
            f" ({stats['compression_ratio']:.0%} of all-pairs)",
            file=sys.stderr,
        )
        for bucket, size in stats["top_buckets"]:
            print(f"  {bucket} bucket: {size} PRs", file=sys.stderr)

    fp_by_id = {fp.pr_id: fp for fp in fingerprints}
    for pr_dict_a, pr_dict_b in bucketer.get_pairs(pr_dicts):
        _compare_and_record(fp_by_id[pr_dict_a["number"]], fp_by_id[pr_dict_b["number"]])

    # Build Duplicate objects from deduped seen pairs.
    def get_dirs(files: list[str]) -> set[str]:
        return set('/'.join(f.split('/')[:-1]) for f in files if '/' in f)

    fp_by_id = {fp.pr_id: fp for fp in fingerprints}
    duplicates = []

    for (pr_a_id, pr_b_id), sim in seen.items():
        fp_a = fp_by_id[pr_a_id]
        fp_b = fp_by_id[pr_b_id]

        shared_issues = list(set(fp_a.issue_refs) & set(fp_b.issue_refs))
        shared_domains = list(set(fp_a.domains) & set(fp_b.domains))
        shared_code = set(fp_a.code_lines) & set(fp_b.code_lines)
        shared_code_count = len(shared_code)
        shared_files = list(get_dirs(fp_a.files_touched) & get_dirs(fp_b.files_touched))

        # Require at least one hard signal to emit a result:
        # - Shared issue ref (any count), OR
        # - ≥8 overlapping code lines (filters coincidental matches caused by
        #   shared TypeScript boilerplate, common config patterns, or small PRs
        #   that happen to share a handful of structural lines)
        # File/directory/domain overlap alone is not sufficient — too many hot
        # shared files (community.md, CHANGELOG.md, src/agents/) cause noise.
        if not shared_issues and shared_code_count < 8:
            continue

        # Suppress cross-component boilerplate: two PRs each concentrated in a
        # different depth-≥2 subtree (different plugin/integration/package)
        # that share no issue ref are almost always scaffold FPs, not real
        # duplicates.  A shared issue ref bypasses this — competing
        # implementations of the same feature request are genuine duplicates
        # even when they live in different component directories.
        if not shared_issues and _is_concentrated_cross_component(fp_a, fp_b):
            continue

        if shared_issues:
            reason = f"Both fix #{', #'.join(shared_issues[:2])}"
        elif shared_code_count >= 3:
            sample = list(shared_code)[:1][0]
            if len(sample) > 40:
                sample = sample[:40] + "..."
            reason = f"Same code: {shared_code_count} lines overlap"
        elif shared_files:
            reason = f"Same files: {', '.join(f.split('/')[-1] for f in shared_files[:2])}"
        elif shared_domains:
            reason = f"Same domain: {', '.join(shared_domains[:2])}"
        else:
            reason = f"Similar ({sim:.0%})"

        duplicates.append(Duplicate(
            pr_a=pr_a_id,
            pr_b=pr_b_id,
            similarity=sim,
            shared_issues=shared_issues,
            shared_files=shared_files,
            shared_domains=shared_domains,
            shared_code_lines=shared_code_count,
            reason=reason,
        ))

    duplicates.sort(key=lambda d: d.similarity, reverse=True)
    return duplicates


