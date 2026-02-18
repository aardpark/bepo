"""PR fingerprinting and duplicate detection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Fingerprint:
    """Fingerprint of a PR's changes."""
    pr_id: str
    files_touched: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    issue_refs: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def similarity(self, other: "Fingerprint") -> float:
        """Compute similarity score between two fingerprints.

        Weights:
        - Issue ref overlap (10.0): Same issue = definite duplicate
        - File path overlap (5.0): Same files = likely related
        - Domain overlap (3.0): Same feature area
        - Imports (1.0): Similar dependencies
        """
        scores = []
        weights = []

        # Issue reference overlap - strongest signal
        if self.issue_refs and other.issue_refs:
            r1, r2 = set(self.issue_refs), set(other.issue_refs)
            if r1 & r2:
                scores.append(1.0)
                weights.append(10.0)

        # File path similarity (directory level)
        if self.files_touched or other.files_touched:
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
                weights.append(5.0)

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


@dataclass
class Duplicate:
    """A pair of similar PRs."""
    pr_a: str
    pr_b: str
    similarity: float
    shared_issues: list[str]
    shared_files: list[str]
    shared_domains: list[str]
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
    """Extract issue references (#123)."""
    return list(set(m.group(1) for m in re.finditer(r'#(\d+)', text)))


def _extract_domains(files: list[str], code: str) -> list[str]:
    """Extract domains from file paths and code."""
    domains = set()
    text = ' '.join(files).lower() + ' ' + code.lower()
    for keyword, domain in DOMAINS.items():
        if keyword in text:
            domains.add(domain)
    return list(domains)


def _extract_imports(diff: str) -> list[str]:
    """Extract imports from diff."""
    imports = []
    for line in diff.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            line = line[1:]
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
        files_touched=files,
        domains=_extract_domains(files, new_code),
        issue_refs=_extract_issues(f"{title} {body} {diff}"),
        imports=_extract_imports(diff),
    )


def find_duplicates(
    fingerprints: list[Fingerprint],
    threshold: float = 0.4,
) -> list[Duplicate]:
    """Find duplicate PRs from a list of fingerprints."""
    duplicates = []

    for i, fp_a in enumerate(fingerprints):
        for j, fp_b in enumerate(fingerprints):
            if i >= j:
                continue

            sim = fp_a.similarity(fp_b)
            if sim < threshold:
                continue

            shared_issues = list(set(fp_a.issue_refs) & set(fp_b.issue_refs))
            shared_domains = list(set(fp_a.domains) & set(fp_b.domains))

            def get_dirs(files):
                return set('/'.join(f.split('/')[:-1]) for f in files if '/' in f)
            shared_files = list(get_dirs(fp_a.files_touched) & get_dirs(fp_b.files_touched))

            # Determine reason
            if shared_issues:
                reason = f"Both fix #{', #'.join(shared_issues[:2])}"
            elif shared_files:
                reason = f"Same files: {', '.join(f.split('/')[-1] for f in shared_files[:2])}"
            elif shared_domains:
                reason = f"Same domain: {', '.join(shared_domains[:2])}"
            else:
                reason = f"Similar ({sim:.0%})"

            duplicates.append(Duplicate(
                pr_a=fp_a.pr_id,
                pr_b=fp_b.pr_id,
                similarity=sim,
                shared_issues=shared_issues,
                shared_files=shared_files,
                shared_domains=shared_domains,
                reason=reason,
            ))

    duplicates.sort(key=lambda d: d.similarity, reverse=True)
    return duplicates
