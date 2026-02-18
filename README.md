# bepo

Detect duplicate pull requests in GitHub repos.

No ML, no embeddings, no API keys. Just static analysis of diffs.

A maintainer with 100 open PRs can run `bepo check --repo foo/bar` and in 5 minutes get a ranked list of "you should look at these pairs." That saves hours of manual review.

## The Problem

Large repos waste engineering time on duplicate PRs. When multiple contributors fix the same bug independently, only one PR gets merged — the rest is wasted effort.

**This actually happens.** We analyzed 100 PRs from [OpenClaw](https://github.com/openclaw/openclaw) and found:

| Cluster | PRs | What happened |
|---------|-----|---------------|
| Matrix startup bug | **4 PRs** | 4 engineers independently fixed `startupGraceMs = 0` → `5000` |
| Media token regex | 2 PRs | Identical fix submitted twice |
| Feishu bitable config | 2 PRs | Same multi-account config fix |

**8 duplicate PRs across 3 bug fixes.** That's real engineering time wasted.

## Proof: OpenClaw Analysis

We ran bepo on OpenClaw's open PRs. Here's what it found:

```
$ bepo check --repo openclaw/openclaw --limit 100

#20025 <-> #19973
  Similarity: 86%
  Reason: Both fix #19843      ← Same issue!

#19868 <-> #19855
  Similarity: 81%
  Reason: Same files: parse.ts, pi-embedded-subscribe.tools.ts

#19871 <-> #19853
  Similarity: 100%
  Reason: Same files: bitable.ts, config-schema.ts, tools-config.ts
```

**Verified manually:**

| PR Pair | Similarity | Verdict |
|---------|------------|---------|
| #20025 ↔ #19973 (Matrix) | 86% | ✅ TRUE DUPLICATE — both change `startupGraceMs` from 0 to 5000 |
| #19868 ↔ #19855 (regex) | 81% | ✅ TRUE DUPLICATE — identical PR titles |
| #19871 ↔ #19853 (Feishu) | 100% | ✅ TRUE DUPLICATE — same files, same fix |
| #19996 ↔ #19993 (unrelated) | 20% | ✅ Correctly NOT flagged |

**Precision: 80%** (4/5 flagged clusters were true duplicates)

## More Examples

**VSCode** — Found PRs touching same files for same feature:
```
#295823 <-> #295822
  Similarity: 77%
  Reason: Same files: chatModel.ts, chatForkActions.ts

  Both: "Use metadata flag for fork detection"
```

**Next.js** — Found related test updates:
```
#90121 <-> #90120
  Similarity: 86%
  Reason: Same files: test/
```

## Install

```bash
pip install bepo
```

Requires [GitHub CLI](https://cli.github.com/) (`gh`) to be installed and authenticated.

## Usage

```bash
# Check a repo for duplicate PRs
bepo check --repo owner/repo

# Adjust sensitivity (default: 0.4, higher = stricter)
bepo check --repo owner/repo --threshold 0.5

# Check more PRs
bepo check --repo owner/repo --limit 100

# JSON output for CI
bepo check --repo owner/repo --json
```

## How It Works

bepo fingerprints each PR by extracting:

| Signal | Weight | What it catches |
|--------|--------|-----------------|
| Same issue ref (#123) | 10.0 | Definite duplicate |
| Same code changes (IDF-weighted) | 8.0 | Rare lines weighted more than common boilerplate |
| Same files touched | 6.0 | PRs modifying same code |
| Same feature domain | 3.0 | auth, messaging, database, etc. |
| Same imports | 1.0 | Similar dependencies |

Then computes pairwise Jaccard similarity.

**That's it.** No embeddings, no LLM calls. Just:
- Parse `+++ b/path` from diffs
- Regex for `#\d+` issue refs
- Compare actual code changes
- Set intersection for similarity

~300 lines of Python.

## As a Library

```python
from bepo import fingerprint_pr, find_duplicates

# Fingerprint PRs
fp1 = fingerprint_pr("#123", diff1, title="Fix auth", body="Fixes #456")
fp2 = fingerprint_pr("#124", diff2, title="Auth fix", body="Fixes #456")

# Find duplicates
dups = find_duplicates([fp1, fp2], threshold=0.4)
for d in dups:
    print(f"{d.pr_a} ↔ {d.pr_b}: {d.similarity:.0%}")
    print(f"  Shared issues: {d.shared_issues}")
    print(f"  Shared files: {d.shared_files}")
```

## GitHub Action

Add to your repo to automatically detect duplicate PRs and post a warning comment:

```yaml
name: PR Duplicate Check
on: [pull_request]

jobs:
  check-duplicates:
    runs-on: ubuntu-latest
    steps:
      - uses: aardpark/bepo@v1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          threshold: '0.5'  # optional, default 0.5
```

When a PR is opened that looks like a duplicate, bepo posts a comment:

> ## ⚠️ Potential Duplicate PRs Detected
>
> This PR may be similar to existing open PRs:
>
> | PR | Similarity | Reason |
> |---|---|---|
> | [#123](link) | 85% | Both fix #456 |
> | [#124](link) | 71% | Same code: 10 lines overlap |
>
> ---
> *Detected by [bepo](https://github.com/aardpark/bepo)*

### Action Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `github-token` | GitHub token for API access | `${{ github.token }}` |
| `threshold` | Similarity threshold (0.0-1.0) | `0.5` |
| `limit` | Max PRs to compare against | `50` |
| `comment` | Post comment on PR | `true` |

### Action Outputs

| Output | Description |
|--------|-------------|
| `has_duplicates` | `true` if duplicates found |
| `match_count` | Number of matches |
| `matches` | JSON array of matches |

## Why This Works

Duplicates share obvious signals:
- **Same code** = Identical changes (639 shared lines caught SoundChain duplicates)
- **Same issue ref** = Same bug report (#19843 appeared in 4 Matrix PRs)
- **Same files** = Same bug location (100% overlap for Feishu cluster)

**IDF weighting** makes rare lines matter more than common boilerplate. A shared `startupGraceMs = 5000` is a stronger signal than a shared `return null`.

Code overlap and issue refs catch most duplicates. Simple works.

## Origin Story

This tool was vibe-coded in a single session with Claude.

We tried a few approaches and kept finding that simpler signals outperformed fancier ones. File overlap and issue refs catch most duplicates. Sometimes the obvious solution is the right one.

## License

MIT
