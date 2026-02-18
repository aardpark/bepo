"""bepo - PR duplicate detection for GitHub repos.

Detects duplicate and related pull requests using static analysis.
No ML, no embeddings, just smart diff parsing.
"""

__version__ = "0.6.0"

from .fingerprint import (
    fingerprint_pr,
    find_duplicates,
    Fingerprint,
    Duplicate,
)

__all__ = [
    "fingerprint_pr",
    "find_duplicates",
    "Fingerprint",
    "Duplicate",
]
