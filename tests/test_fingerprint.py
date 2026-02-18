"""Tests for bepo fingerprinting."""
from bepo import fingerprint_pr, find_duplicates


def test_fingerprint_extracts_files():
    diff = """+++ b/src/auth/login.ts
- old code
+ new code
+++ b/src/auth/logout.ts
+ more code
"""
    fp = fingerprint_pr("#1", diff)
    assert "src/auth/login.ts" in fp.files_touched
    assert "src/auth/logout.ts" in fp.files_touched


def test_fingerprint_extracts_issues():
    fp = fingerprint_pr("#1", "", title="Fix bug", body="Fixes #123 and #456")
    assert "123" in fp.issue_refs
    assert "456" in fp.issue_refs


def test_fingerprint_extracts_domains():
    diff = "+++ b/src/telegram/handler.ts\n+ code"
    fp = fingerprint_pr("#1", diff)
    assert "messaging" in fp.domains


def test_similar_prs_detected():
    diff1 = "+++ b/src/auth/login.ts\n+ code"
    diff2 = "+++ b/src/auth/login.ts\n+ other code"

    fp1 = fingerprint_pr("#1", diff1, body="Fixes #100")
    fp2 = fingerprint_pr("#2", diff2, body="Fixes #100")

    dups = find_duplicates([fp1, fp2], threshold=0.3)
    assert len(dups) == 1
    assert dups[0].similarity > 0.5
    assert "100" in dups[0].shared_issues


def test_different_prs_not_flagged():
    diff1 = "+++ b/src/auth/login.ts\n+ const user = await authenticate(token)"
    diff2 = "+++ b/src/payments/stripe.ts\n+ const charge = await stripe.charges.create(amount)"

    fp1 = fingerprint_pr("#1", diff1)
    fp2 = fingerprint_pr("#2", diff2)

    dups = find_duplicates([fp1, fp2], threshold=0.5)
    assert len(dups) == 0


def test_same_code_detected():
    """PRs with identical code changes should be flagged."""
    diff1 = """+++ b/src/config.ts
+ startupGraceMs = 5000
+ retryCount = 3
+ timeout = 30000
"""
    diff2 = """+++ b/src/config.ts
+ startupGraceMs = 5000
+ retryCount = 3
+ timeout = 30000
"""
    fp1 = fingerprint_pr("#1", diff1)
    fp2 = fingerprint_pr("#2", diff2)

    dups = find_duplicates([fp1, fp2], threshold=0.3)
    assert len(dups) == 1
    assert dups[0].shared_code_lines >= 3
    assert "code" in dups[0].reason.lower() or dups[0].similarity > 0.8
