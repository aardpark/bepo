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

### 30-day window (recommended)

```
$ bepo check --repo openclaw/openclaw --since 30d

Found 52 potential duplicates:

#20472 <-> #20491
  Similarity: 100%
  Reason: Both fix #20468      ← Same Nextcloud Talk restart bug, two fixes

#19595 <-> #19624
  Similarity: 90%
  Reason: Both fix #19574      ← Identical PR titles, same elevatedDefault bug

#20419 <-> #20441
  Similarity: 83%
  Reason: Both fix #20410      ← Same WebChat markdown rendering fix

#19865 <-> #19945
  Similarity: 97%
  Reason: Same code: 100 lines overlap   ← Two embedding provider PRs duplicating core logic

#19770 <-> #20317
  Similarity: 87%
  Reason: Same code: 9 lines overlap     ← Same "hide tool calls" UI toggle, added twice
```

**Precision: ~88%** — verified by manual classification of all 52 pairs.

The remaining ~12% are concurrent PRs touching the same structural code — two provider integrations sharing schema boilerplate, two locale additions hitting the same type. The kind of overlap a reviewer would want to know about.

### Full backlog (3,000 PRs)

```
$ bepo check --repo openclaw/openclaw --limit 3000

Analyzed 3000 PRs in 9.8s

Found 1022 potential duplicates:

#17518 <-> #17653
  Similarity: 100%
  Reason: Both fix #17499      ← Identical browser dialog fix submitted twice

#12936 <-> #19050
  Similarity: 80%
  Reason: Same code: 10 lines overlap   ← Same Telegram thread_id fix, one is literally "v2"

#15512 <-> #18994
  Similarity: 67%
  Reason: Same code: 10 lines overlap   ← Both normalize Brave search language codes

#14182 <-> #15051
  Similarity: 75%
  Reason: Same code: 2403 lines overlap ← Two Zulip implementations duplicating core logic
```

Precision by similarity band, verified by manual sampling:

| Band | Pairs | Precision | Notes |
|------|-------|-----------|-------|
| 100% | 221 | ~75% | High code overlap but tiny sets — watch for short boilerplate |
| 80–90% | 318 | ~95% | Best signal — issue refs + code together |
| 70–79% | 283 | ~90% | Strong structural duplicates |
| 65–69% | 200 | ~70% | Noisier; raise `--threshold 0.75` to cut this band |
| **Overall** | **1022** | **~84%** | |

For large backlogs, `--threshold 0.75` drops to ~550 pairs at ~92% precision. `--since 30d` gives 52 actionable pairs at ~88% — the recommended default.

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

# Check recent PRs (recommended — avoids stale noise)
bepo check --repo owner/repo --since 30d

# Adjust sensitivity (default: 0.65, higher = stricter)
bepo check --repo owner/repo --threshold 0.7

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

**Cross-component filtering** suppresses boilerplate FPs in integration/plugin monorepos (e.g. Home Assistant, VSCode extensions). Two unrelated integrations sharing scaffold code (`config_flow.py`, `manifest.json`) are filtered out when each PR is concentrated in a different component subtree. Pairs sharing a GitHub issue ref always bypass this filter.

~2,000 lines of Python.

## As a Library

```python
from bepo import fingerprint_pr, find_duplicates

# Fingerprint PRs
fp1 = fingerprint_pr("#123", diff1, title="Fix auth", body="Fixes #456")
fp2 = fingerprint_pr("#124", diff2, title="Auth fix", body="Fixes #456")

# Find duplicates
dups = find_duplicates([fp1, fp2], threshold=0.65)
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
          threshold: '0.65'  # optional, default 0.65
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
| `threshold` | Similarity threshold (0.0-1.0) | `0.65` |
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
