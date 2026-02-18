"""Tests for bepo fingerprinting."""
import pytest
from bepo import fingerprint_pr, find_duplicates, Fingerprint


class TestFingerprinting:
    """Test PR fingerprinting."""

    def test_fingerprint_extracts_files(self):
        diff = """+++ b/src/auth.ts
+ export function login() {}
+++ b/src/utils.ts
+ export function helper() {}"""
        fp = fingerprint_pr("#1", diff)
        assert "src/auth.ts" in fp.files_touched
        assert "src/utils.ts" in fp.files_touched

    def test_fingerprint_extracts_issue_refs(self):
        fp = fingerprint_pr("#1", "+++ b/fix.ts", title="Fix bug", body="Fixes #123 and #456")
        assert "123" in fp.issue_refs
        assert "456" in fp.issue_refs

    def test_fingerprint_extracts_domains(self):
        diff = "+++ b/src/auth/login.ts\n+ const token = getAuthToken()"
        fp = fingerprint_pr("#1", diff)
        assert "auth" in fp.domains

    def test_fingerprint_extracts_code_lines(self):
        diff = "+++ b/src/app.ts\n+ const value = computeExpensiveThing()\n+ return value"
        fp = fingerprint_pr("#1", diff)
        assert "const value = computeExpensiveThing()" in fp.code_lines


class TestSimilarity:
    """Test similarity scoring."""

    def test_same_issue_ref_high_similarity(self):
        fp1 = fingerprint_pr("#1", "+++ b/a.ts", body="Fixes #100")
        fp2 = fingerprint_pr("#2", "+++ b/b.ts", body="Fixes #100")
        assert fp1.similarity(fp2) > 0.5

    def test_same_files_increases_similarity(self):
        fp1 = fingerprint_pr("#1", "+++ b/src/auth.ts\n+ code1")
        fp2 = fingerprint_pr("#2", "+++ b/src/auth.ts\n+ code2")
        fp3 = fingerprint_pr("#3", "+++ b/src/other.ts\n+ code3")
        assert fp1.similarity(fp2) > fp1.similarity(fp3)

    def test_identical_code_high_similarity(self):
        diff = "+++ b/src/fix.ts\n+ const fix = applySpecificFix()\n+ return fix"
        fp1 = fingerprint_pr("#1", diff)
        fp2 = fingerprint_pr("#2", diff)
        assert fp1.similarity(fp2) > 0.8


class TestFindDuplicates:
    """Test duplicate detection."""

    def test_finds_duplicates_above_threshold(self):
        diff = "+++ b/src/fix.ts\n+ const fix = applySpecificFix()"
        fp1 = fingerprint_pr("#1", diff, body="Fixes #100")
        fp2 = fingerprint_pr("#2", diff, body="Fixes #100")
        dups = find_duplicates([fp1, fp2], threshold=0.4)
        assert len(dups) == 1
        assert dups[0].pr_a == "#1"
        assert dups[0].pr_b == "#2"

    def test_no_duplicates_below_threshold(self):
        fp1 = fingerprint_pr("#1", "+++ b/a.ts\n+ unique code one")
        fp2 = fingerprint_pr("#2", "+++ b/b.ts\n+ completely different code")
        dups = find_duplicates([fp1, fp2], threshold=0.4)
        assert len(dups) == 0

    def test_idf_weights_rare_lines_higher(self):
        # Common line in many PRs
        common = "+++ b/a.ts\n+ return null"
        # Rare line in few PRs
        rare = "+++ b/b.ts\n+ startupGraceMs = 5000"

        fps = [
            fingerprint_pr("#1", common + "\n" + rare),
            fingerprint_pr("#2", rare),  # shares rare
            fingerprint_pr("#3", common),  # shares common
            fingerprint_pr("#4", common),
            fingerprint_pr("#5", common),
        ]

        dups = find_duplicates(fps, threshold=0.1)

        # Find similarity between #1 and #2 (share rare line)
        sim_rare = next(d.similarity for d in dups if {d.pr_a, d.pr_b} == {"#1", "#2"})
        # Find similarity between #1 and #3 (share common line)
        sim_common = next(d.similarity for d in dups if {d.pr_a, d.pr_b} == {"#1", "#3"})

        # Rare line overlap should score higher than common line overlap
        assert sim_rare > sim_common
