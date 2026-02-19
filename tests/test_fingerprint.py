"""Tests for bepo fingerprinting."""
import inspect
import pytest
from bepo import fingerprint_pr, find_duplicates, Fingerprint
from bepo.fingerprint import find_duplicates as _find_duplicates, _dominant_prefix, _dominant_component_token


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
        # Use 8 lines each so pairs pass the >=8 code-line minimum
        rare_lines  = "\n".join(f"+ startupGraceMs_{i} = {5000 + i}" for i in range(8))
        common_lines = "\n".join(f"+ logEvent_{i}('boot')" for i in range(8))

        fps = [
            fingerprint_pr("#1", f"+++ b/a.ts\n{common_lines}\n{rare_lines}"),
            fingerprint_pr("#2", f"+++ b/b.ts\n{rare_lines}"),   # shares rare
            fingerprint_pr("#3", f"+++ b/c.ts\n{common_lines}"),  # shares common
            fingerprint_pr("#4", f"+++ b/d.ts\n{common_lines}"),
            fingerprint_pr("#5", f"+++ b/e.ts\n{common_lines}"),
        ]

        dups = find_duplicates(fps, threshold=0.1)

        # Find similarity between #1 and #2 (share rare lines)
        sim_rare = next(d.similarity for d in dups if {d.pr_a, d.pr_b} == {"#1", "#2"})
        # Find similarity between #1 and #3 (share common lines)
        sim_common = next(d.similarity for d in dups if {d.pr_a, d.pr_b} == {"#1", "#3"})

        # Rare line overlap should score higher than common line overlap
        assert sim_rare > sim_common


class TestCalibration:
    """Regression tests that encode verified calibration decisions.

    Each test pins a specific tuning choice with a minimal concrete example.
    If a test breaks after a scoring or filter change, verify intentionality
    against real PR data before updating it.

    Ground truth verified against openclaw/openclaw --since 30d (Feb 2026):
      Genuine:     #20295↔#20355 (same gateway guard, shared #20245)
                   #20419↔#20441 (same webchat markdown fix, shared #20410)
                   #19595↔#19624 (identical titles, shared #19574)
      Not genuine: community.md concurrent additions (different plugins, same file)
                   boilerplate code overlap (1-7 shared TS lines)
                   sub-100 issue refs in PR body templates (#1, #2, #42)
                   domain-only pairs (both touch config/ but unrelated changes)
    """

    def test_default_threshold_is_0_65(self):
        """Domain-only similarity caps at ~0.62; threshold must sit above it.

        Two PRs in the same domain with no code/file/issue overlap score
        ~0.50-0.62. The default threshold of 0.65 keeps them out.
        If this breaks, re-validate the domain-only score ceiling first.
        """
        sig = inspect.signature(_find_duplicates)
        assert sig.parameters['threshold'].default == 0.65

    def test_sub_100_issue_refs_not_extracted(self):
        """#1, #2, #42, #88 are PR template noise, not real issue refs.

        PR bodies routinely contain 'step #1', 'closes #2', 'resolves #42'.
        These inflate similarity via the 10.0 issue-ref weight and caused
        50 false-positive pairs before the fix.
        """
        fp = fingerprint_pr("#pr", "", body="step #1, see #2, fixes #42, refs #88")
        assert fp.issue_refs == [], f"expected no refs, got {fp.issue_refs}"

    def test_3_digit_issue_refs_are_captured(self):
        """Real issue numbers (#100+) must still be extracted and matched."""
        fp = fingerprint_pr("#pr", "", body="Fixes #20245 and relates to #19302")
        assert "20245" in fp.issue_refs
        assert "19302" in fp.issue_refs

    def test_shared_file_without_code_not_flagged(self):
        """Two PRs adding different content to the same file are not duplicates.

        Models the community.md pattern: concurrent plugin additions both
        touch docs/plugins/community.md but write completely different lines.
        Before the hard-signal filter this scored sim=1.00 and was flagged.
        """
        diff_a = "+++ b/docs/plugins/community.md\n+ - [Plugin Alpha](https://alpha.example) - does X"
        diff_b = "+++ b/docs/plugins/community.md\n+ - [Plugin Beta](https://beta.example) - does Y"
        fp1 = fingerprint_pr("#1", diff_a)
        fp2 = fingerprint_pr("#2", diff_b)
        assert len(find_duplicates([fp1, fp2])) == 0

    def test_4_shared_code_lines_without_issue_ref_not_flagged(self):
        """1-4 shared lines are too often coincidental TypeScript boilerplate.

        Common patterns like config.get(), error handling, or import structure
        create false matches. The minimum was raised to 8 after auditing real
        PR pairs — 'add MiniMax TTS' vs 'suppress raw API error details'
        (1 shared line) and similar structural coincidences with 5-7 shared lines.
        """
        shared = "\n".join(f"+ cfg_{i} = config.get('key{i}')" for i in range(4))
        fp1 = fingerprint_pr("#1", f"+++ b/src/feature-a.ts\n{shared}\n+ uniqueA = doA()")
        fp2 = fingerprint_pr("#2", f"+++ b/src/feature-b.ts\n{shared}\n+ uniqueB = doB()")
        assert len(find_duplicates([fp1, fp2])) == 0

    def test_7_shared_code_lines_without_issue_ref_not_flagged(self):
        """5-7 shared lines are still too often structural coincidence.

        The minimum was raised from 5 to 8 after auditing pairs like
        #19876↔#20445 (SerpAPI vs Firecrawl config, 5 shared lines) —
        different features that happen to share a handful of config lines.
        """
        shared = "\n".join(f"+ cfg_{i} = config.get('key{i}')" for i in range(7))
        fp1 = fingerprint_pr("#1", f"+++ b/src/feature-a.ts\n{shared}\n+ uniqueA = doA()")
        fp2 = fingerprint_pr("#2", f"+++ b/src/feature-b.ts\n{shared}\n+ uniqueB = doB()")
        assert len(find_duplicates([fp1, fp2])) == 0

    def test_8_shared_code_lines_without_issue_ref_is_flagged(self):
        """≥8 shared meaningful lines without an issue ref should surface."""
        shared = "\n".join(f"+ const result_{i} = computeSpecific_{i}(input)" for i in range(8))
        diff = f"+++ b/src/fix.ts\n{shared}"
        fp1 = fingerprint_pr("#1", diff)
        fp2 = fingerprint_pr("#2", diff)
        dups = find_duplicates([fp1, fp2])
        assert len(dups) == 1
        assert dups[0].shared_code_lines >= 8

    def test_issue_ref_pair_found_regardless_of_code_overlap(self):
        """Shared issue ref alone is sufficient — models #20295↔#20355.

        These two PRs fix the same gateway guard, reference the same issue,
        but their diffs write it slightly differently (0 shared code lines).
        """
        fp1 = fingerprint_pr("#20295", "+++ b/src/gateway/config.ts\n+ if (!commands.restart) throw new GuardError()",
                              body="Fixes #20245")
        fp2 = fingerprint_pr("#20355", "+++ b/src/gateway/config.ts\n+ if (!commands.restart) { throw new GuardError() }",
                              body="Fixes #20245")
        dups = find_duplicates([fp1, fp2])
        assert len(dups) == 1
        assert "20245" in dups[0].shared_issues

    def test_cross_domain_issue_ref_not_flagged(self):
        """Issue refs shared by PRs in different domains are parent/tracking refs.

        Models the #14447 pattern: a cron PR and a WhatsApp PR both reference
        the same old umbrella issue.  The config-domain and messaging-domain
        overlap is zero, so the shared ref is a tracking issue, not a duplicate
        signal.  Before cross-domain stripping this pair scored ~0.73 and was
        falsely flagged.
        """
        diff_a = "+++ b/src/config/limits.ts\n+ export const MAX_RETRIES = 3"
        diff_b = "+++ b/src/telegram/handler.ts\n+ const msg = formatTelegramResponse()"
        fp1 = fingerprint_pr("#1", diff_a, body="Fixes #20999")
        fp2 = fingerprint_pr("#2", diff_b, body="Fixes #20999")
        assert len(find_duplicates([fp1, fp2])) == 0

    def test_same_domain_issue_ref_still_flagged(self):
        """Issue refs shared by PRs in the same domain must still surface.

        Both PRs touch gateway/config files (config domain).  The shared ref
        stays in play because the domain coverage spans only one bucket.
        Models #20295↔#20355 (gateway restart guard).
        """
        fp1 = fingerprint_pr(
            "#20295", "+++ b/src/gateway/config.ts\n+ if (!commands.restart) throw new GuardError()",
            body="Fixes #20245",
        )
        fp2 = fingerprint_pr(
            "#20355", "+++ b/src/gateway/config.ts\n+ if (!commands.restart) { throw new GuardError() }",
            body="Fixes #20245",
        )
        dups = find_duplicates([fp1, fp2])
        assert len(dups) == 1
        assert "20245" in dups[0].shared_issues

    def test_catchall_pair_with_code_overlap_keeps_issue_ref(self):
        """Catchall PRs (no detectable domain) sharing code AND a ref must surface.

        Models the elevatedDefault cluster (#19595↔#19624): both PRs fix the
        same bug in tools code (no domain keywords in path), share an issue ref
        (#19574) and have overlapping code.  Without this, removing the '_none'
        sentinel would lose the genuine pair.
        """
        shared = "\n".join(f"+ const fix_{i} = applyElevatedFix_{i}(tools)" for i in range(4))
        diff = f"+++ b/src/tools/elevated.ts\n{shared}"
        fp1 = fingerprint_pr("#1", diff, body="Fixes #19574")
        fp2 = fingerprint_pr("#2", diff, body="Fixes #19574")
        dups = find_duplicates([fp1, fp2])
        assert len(dups) == 1
        assert "19574" in dups[0].shared_issues

    def test_catchall_pair_without_code_overlap_strips_issue_ref(self):
        """Catchall PRs sharing only a parent ref (no code, no domain) must not fire.

        Models the #19367 cluster: build-script PR and logger-binding PR both
        reference the same old umbrella issue but make completely different
        changes in different parts of the codebase.  With only the '_none'
        sentinel previously, all three pairwise combos in this cluster were
        falsely flagged (3 FP pairs).
        """
        diff_a = "+++ b/src/build/bundle.ts\n" + "\n".join(
            f"+ bundleStep_{i}(output)" for i in range(8)
        )
        diff_b = "+++ b/src/plugins/logger.ts\n" + "\n".join(
            f"+ logMethod_{i} = logMethod_{i}.bind(this)" for i in range(8)
        )
        fp1 = fingerprint_pr("#1", diff_a, body="Fixes #19367")
        fp2 = fingerprint_pr("#2", diff_b, body="Fixes #19367")
        assert len(find_duplicates([fp1, fp2])) == 0

    def test_connector_pr_does_not_keep_ref_for_unrelated_pair(self):
        """A third 'connector' PR must not keep a ref alive for a cross-domain pair.

        When ref #X is shared by [A (scheduling), B (scheduling), C (messaging)],
        the A↔B pair legitimately keeps #X (same domain).  But B↔C is cross-domain
        and must NOT get credit for #X just because A↔B kept the ref globally.
        The old global _ref_is_hot approach failed here; per-pair filtering fixes it.
        """
        # A and B share domain (scheduling via "cron" in path) and fix #X
        diff_a = "+++ b/src/cron/scheduler.ts\n+ scheduleNextJob(target)"
        diff_b = "+++ b/src/cron/runner.ts\n+ runScheduledJob(target)"
        # C is messaging domain (telegram) and also references #X
        diff_c = "+++ b/src/telegram/handler.ts\n+ const msg = handleIncoming(update)"
        fp_a = fingerprint_pr("#A", diff_a, body="Fixes #99999")
        fp_b = fingerprint_pr("#B", diff_b, body="Fixes #99999")
        fp_c = fingerprint_pr("#C", diff_c, body="Fixes #99999")

        dups = find_duplicates([fp_a, fp_b, fp_c])
        pairs = [{d.pr_a, d.pr_b} for d in dups]
        # A↔B is fine (same scheduling domain)
        # B↔C and A↔C must NOT be flagged (cross-domain: scheduling vs messaging)
        assert {"#B", "#C"} not in pairs, "cross-domain pair B↔C must not be flagged"
        assert {"#A", "#C"} not in pairs, "cross-domain pair A↔C must not be flagged"

    def test_mixed_domain_overlap_not_flagged(self):
        """PRs sharing only a generic domain (config) amid diverse domain sets
        must not be flagged as duplicates.

        Models #19767↔#19804: cron validation PR (scheduling + config + api)
        vs WhatsApp history PR (messaging + config + auth + media) both
        referencing an old umbrella issue.  Before Jaccard gating, the shared
        'config' domain counted as same-domain and kept the parent ref alive.

        Jaccard = 1 shared / 7 total ≈ 0.14 < 0.25 threshold → cross-domain.
        """
        diff_a = (
            "+++ b/src/cron/scheduler.ts\n+ scheduleNextJob(target)\n"
            "+++ b/src/config/cron-types.ts\n+ export type CronConfig = {}\n"
            "+++ b/src/cron/webhook-url.ts\n+ const url = buildWebhookUrl(base)"
        )
        diff_b = (
            "+++ b/extensions/telegram/channel.ts\n+ sendTelegramMessage()\n"
            "+++ b/src/config/telegram-types.ts\n+ export type TGConfig = {}\n"
            "+++ b/src/web/session.ts\n+ const session = getActiveSession()\n"
            "+++ b/src/web/inbound/download-media.ts\n+ downloadMedia(url)"
        )
        fp1 = fingerprint_pr("#1", diff_a, body="Fixes #14447")
        fp2 = fingerprint_pr("#2", diff_b, body="Fixes #14447")
        assert len(find_duplicates([fp1, fp2])) == 0

    def test_domain_only_pair_not_flagged(self):  # noqa: E301 (keep adjacent)
        """Same domain but no code or issue overlap must not be flagged.

        Before threshold was raised to 0.65, pairs like 'add config option A'
        and 'add config option B' both landing in the config bucket scored
        ~0.50-0.60 and produced 204 false-positive 'Same domain: config' pairs.
        """
        diff_a = "+++ b/src/config/limits.ts\n+ export const MAX_RETRIES = 3\n+ export const TIMEOUT = 5000"
        diff_b = "+++ b/src/config/flags.ts\n+ export const DEBUG_MODE = false\n+ export const LOG_LEVEL = 'info'"
        fp1 = fingerprint_pr("#1", diff_a)
        fp2 = fingerprint_pr("#2", diff_b)
        assert len(find_duplicates([fp1, fp2])) == 0


class TestDominantPrefix:
    """Unit tests for the _dominant_prefix helper."""

    def test_concentrated_returns_deepest_prefix(self):
        files = ['a/b/c.py', 'a/b/d.py', 'a/b/e.py']
        assert _dominant_prefix(files) == 'a/b'

    def test_deeper_prefix_preferred(self):
        files = ['a/b/c/x.py', 'a/b/c/y.py', 'a/b/c/z.py']
        assert _dominant_prefix(files) == 'a/b/c'

    def test_spread_returns_none(self):
        # Each subdirectory covers only 1/3 of files — below 80%
        files = ['a/b/c.py', 'a/c/d.py', 'a/d/e.py']
        assert _dominant_prefix(files) is None

    def test_shallow_files_only_returns_none(self):
        # Files at depth 1 produce no depth-≥2 prefix candidates
        files = ['a/x.py', 'a/y.py']
        assert _dominant_prefix(files) is None

    def test_empty_returns_none(self):
        assert _dominant_prefix([]) is None

    def test_exactly_at_threshold(self):
        # 4 out of 5 = 80%, exactly at threshold
        files = ['a/b/c.py', 'a/b/d.py', 'a/b/e.py', 'a/b/f.py', 'a/x/g.py']
        assert _dominant_prefix(files) == 'a/b'

    def test_just_under_threshold(self):
        # 3 out of 4 = 75%, just under 80%
        files = ['a/b/c.py', 'a/b/d.py', 'a/b/e.py', 'a/x/g.py']
        assert _dominant_prefix(files) is None

    def test_root_only_file_ignored(self):
        # 'README.md' has no slash — no depth-≥2 prefix possible
        files = ['README.md', 'setup.py']
        assert _dominant_prefix(files) is None


class TestDominantComponentToken:
    """Unit tests for _dominant_component_token (token-based concentration)."""

    def test_single_directory_returns_token(self):
        # All files in one component directory
        files = ['ha/components/imou/__init__.py', 'ha/components/imou/config.py']
        assert _dominant_component_token(files) == 'imou'

    def test_src_test_mirror_same_token(self):
        # Source and test files share component name at same depth
        files = [
            'ha/components/imou/__init__.py',
            'ha/components/imou/config.py',
            'tests/components/imou/test_init.py',
        ]
        assert _dominant_component_token(files) == 'imou'

    def test_spread_across_components_returns_none(self):
        # 4 files across completely different depth-2 namespaces — no token
        # at any depth covers ≥60% of files
        files = [
            'frontend/auth/login.py',
            'backend/api/handler.py',
            'tests/unit/helpers.py',
            'docs/guide/setup.md',
        ]
        assert _dominant_component_token(files) is None

    def test_shallow_files_only_returns_none(self):
        # depth-1 files produce no depth-≥2 tokens
        files = ['CODEOWNERS', 'requirements.txt']
        assert _dominant_component_token(files) is None

    def test_prefers_deeper_token(self):
        # 'components' dominates at depth 2, 'imou' dominates at depth 3
        # Should return the deeper one
        files = [
            'ha/components/imou/__init__.py',
            'ha/components/imou/config.py',
            'ha/components/imou/sensor.py',
        ]
        assert _dominant_component_token(files) == 'imou'


class TestComponentIsolation:
    """Tests for the cross-component boilerplate FP filter.

    Feature: Component Isolation
    Rationale: Plugin/integration monorepos share scaffold code across
    components (config_flow, coordinator, manifest templates).  Two new
    integrations using the same scaffold score high similarity but are not
    real duplicates.
    Invariants:
      I-ISOLATE: Two PRs concentrated in different component tokens with no
                 shared issue ref must not be flagged.
      I-ISSUE-OVERRIDE: A shared issue ref always overrides isolation — competing
                        implementations of the same request are genuine duplicates.
      I-SAME-COMPONENT: PRs with the same dominant token are compared normally.
      I-NO-TOKEN: A PR with no dominant component token (truly spread) is not
                  considered cross-component.
    """

    def test_cross_component_boilerplate_not_flagged(self):
        """Two new integrations with identical scaffold code are not duplicates.

        Models the HA pattern: imou and eltako both ship the same config_flow /
        coordinator template but are unrelated integrations.

        Verifies I-ISOLATE.
        """
        scaffold = "\n".join(
            f"+ scaffold_{i} = setup_integration_{i}(config)" for i in range(10)
        )
        diff_a = f"+++ b/homeassistant/components/imou/__init__.py\n{scaffold}"
        diff_b = f"+++ b/homeassistant/components/eltako/__init__.py\n{scaffold}"
        fp1 = fingerprint_pr("#1", diff_a, title="Add imou integration")
        fp2 = fingerprint_pr("#2", diff_b, title="Add eltako integration")
        assert len(find_duplicates([fp1, fp2])) == 0

    def test_cross_component_with_shared_issue_ref_still_flagged(self):
        """Competing implementations for the same issue are real duplicates.

        Two PRs in different component directories both fixing the same GitHub
        issue are competing implementations — the shared ref overrides isolation.

        Verifies I-ISSUE-OVERRIDE.
        """
        scaffold = "\n".join(
            f"+ scaffold_{i} = setup_integration_{i}(config)" for i in range(10)
        )
        diff_a = f"+++ b/plugins/infrared_entity/__init__.py\n{scaffold}"
        diff_b = f"+++ b/plugins/lg_infrared/__init__.py\n{scaffold}"
        fp1 = fingerprint_pr("#1", diff_a, body="Fixes #162346")
        fp2 = fingerprint_pr("#2", diff_b, body="Fixes #162346")
        dups = find_duplicates([fp1, fp2])
        assert len(dups) == 1
        assert "162346" in dups[0].shared_issues

    def test_same_component_multi_pr_not_suppressed(self):
        """Two PRs touching the same component token are compared normally.

        Adding sensor + binary_sensor to the same integration shares code by
        design — same dominant token 'compit', so cross-component filter does
        not fire.

        Verifies I-SAME-COMPONENT.
        """
        shared = "\n".join(
            f"+ sensor_line_{i} = setup_sensor_{i}(config)" for i in range(10)
        )
        diff_a = f"+++ b/homeassistant/components/compit/sensor.py\n{shared}"
        diff_b = f"+++ b/homeassistant/components/compit/binary_sensor.py\n{shared}"
        fp1 = fingerprint_pr("#1", diff_a)
        fp2 = fingerprint_pr("#2", diff_b)
        dups = find_duplicates([fp1, fp2])
        assert len(dups) == 1

    def test_no_dominant_token_pr_not_considered_cross_component(self):
        """A PR with no dominant token is not considered cross-component.

        A PR touching files spread across many unrelated directories has no
        dominant token at any depth.  _is_concentrated_cross_component must
        return False for it, so the pair is not suppressed by this filter.

        Verifies I-NO-TOKEN.
        """
        from bepo.fingerprint import _is_concentrated_cross_component

        # 5 files across truly diverse paths — no token at any depth ≥2 covers ≥60%
        diff_a = "\n".join(
            f"+++ b/{path}/fix.py\n+ fix_{path.replace('/', '_')}()"
            for path in ['frontend/auth', 'backend/api', 'tests/e2e', 'mobile/screens', 'desktop/views']
        )
        diff_b = "+++ b/homeassistant/components/eltako/fix.py\n+ fix_eltako()"
        fp1 = fingerprint_pr("#1", diff_a)  # no dominant token → None
        fp2 = fingerprint_pr("#2", diff_b)  # dominant token → 'eltako'
        assert not _is_concentrated_cross_component(fp1, fp2)

    def test_src_test_mirror_handled_correctly(self):
        """PRs with source and test files for the same component are identified.

        Models the HA pattern: a new integration PR touches both
        homeassistant/components/X/ and tests/components/X/.  Despite having
        different full-path prefixes, both paths share the component name 'X'
        at depth 3, so _dominant_component_token returns 'X'.
        Two such PRs for DIFFERENT integrations should be cross-component.

        Verifies I-ISOLATE for repos with src/test mirroring.
        """
        scaffold = "\n".join(
            f"+ scaffold_{i} = setup_integration_{i}(config)" for i in range(10)
        )
        # imou PR: source + test files
        diff_a = (
            f"+++ b/homeassistant/components/imou/__init__.py\n{scaffold}\n"
            f"+++ b/tests/components/imou/test_init.py\n{scaffold}"
        )
        # eltako PR: source + test files
        diff_b = (
            f"+++ b/homeassistant/components/eltako/__init__.py\n{scaffold}\n"
            f"+++ b/tests/components/eltako/test_init.py\n{scaffold}"
        )
        fp1 = fingerprint_pr("#1", diff_a)
        fp2 = fingerprint_pr("#2", diff_b)
        assert len(find_duplicates([fp1, fp2])) == 0

    def test_generic_plugin_monorepo_pattern(self):
        """Component isolation works for any plugin layout, not just HA.

        Models a generic flat plugin monorepo (e.g., integrations/X/) where
        two plugins share boilerplate but are otherwise unrelated.
        """
        scaffold = "\n".join(
            f"+ plugin_setup_{i}(registry, config)" for i in range(10)
        )
        diff_a = f"+++ b/integrations/stripe/plugin.py\n{scaffold}"
        diff_b = f"+++ b/integrations/paypal/plugin.py\n{scaffold}"
        fp1 = fingerprint_pr("#1", diff_a)
        fp2 = fingerprint_pr("#2", diff_b)
        assert len(find_duplicates([fp1, fp2])) == 0
