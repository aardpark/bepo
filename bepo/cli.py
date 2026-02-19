"""CLI for bepo."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from .fingerprint import fingerprint_pr, find_duplicates


# ANSI color codes
class Colors:
    BOLD = '\033[1m'
    DIM = '\033[2m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        cls.BOLD = cls.DIM = cls.GREEN = cls.YELLOW = ''
        cls.RED = cls.CYAN = cls.RESET = ''


def check_gh_installed():
    """Check if GitHub CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ['gh', '--version'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"{Colors.RED}Error: GitHub CLI (gh) is not working properly.{Colors.RESET}", file=sys.stderr)
            print("Install it from: https://cli.github.com/", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print(f"{Colors.RED}Error: GitHub CLI (gh) is not installed.{Colors.RESET}", file=sys.stderr)
        print("Install it from: https://cli.github.com/", file=sys.stderr)
        sys.exit(1)


def get_cache_dir() -> Path:
    """Get cache directory, creating if needed."""
    cache_dir = Path.home() / '.cache' / 'bepo'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(repo: str, pr_num: int) -> str:
    """Generate cache key for a PR diff."""
    return hashlib.md5(f"{repo}:{pr_num}".encode()).hexdigest()[:12]


def load_cached_diff(repo: str, pr_num: int) -> str | None:
    """Load diff from cache if exists and fresh (< 1 hour)."""
    cache_file = get_cache_dir() / f"{get_cache_key(repo, pr_num)}.diff"
    if cache_file.exists():
        # Check if cache is fresh (< 1 hour)
        age = time.time() - cache_file.stat().st_mtime
        if age < 3600:
            return cache_file.read_text()
    return None


def save_cached_diff(repo: str, pr_num: int, diff: str):
    """Save diff to cache."""
    cache_file = get_cache_dir() / f"{get_cache_key(repo, pr_num)}.diff"
    cache_file.write_text(diff)


def get_commit_cache_key(repo: str, sha: str) -> str:
    """Generate cache key for a commit diff."""
    return hashlib.md5(f"commit:{repo}:{sha}".encode()).hexdigest()[:12]


def load_cached_commit_diff(repo: str, sha: str) -> str | None:
    """Load commit diff from cache. No TTL — commits are immutable."""
    cache_file = get_cache_dir() / f"{get_commit_cache_key(repo, sha)}.diff"
    if cache_file.exists():
        return cache_file.read_text()
    return None


def save_cached_commit_diff(repo: str, sha: str, diff: str):
    """Save commit diff to cache."""
    cache_file = get_cache_dir() / f"{get_commit_cache_key(repo, sha)}.diff"
    cache_file.write_text(diff)


def _parse_since(value: str) -> str:
    """Parse --since value (e.g. '30d') into an ISO date string (YYYY-MM-DD)."""
    if not value.endswith('d') or not value[:-1].isdigit():
        print(
            f"{Colors.RED}Error: invalid --since format {value!r}."
            " Expected Nd (e.g. 30d).{Colors.RESET}",
            file=sys.stderr,
        )
        sys.exit(1)
    days = int(value[:-1])
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    return cutoff.strftime('%Y-%m-%d')


def fetch_prs(repo: str, limit: int = 50, since_date: str | None = None) -> list[dict]:
    """Fetch open PRs from GitHub."""
    cmd = [
        'gh', 'pr', 'list', '--repo', repo, '--state', 'open',
        '--limit', str(limit), '--json', 'number,title,body',
    ]
    if since_date:
        cmd.extend(['--search', f'created:>{since_date}'])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{Colors.RED}Error fetching PRs: {result.stderr}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def fetch_commit_diff(repo: str, sha: str) -> str:
    """Fetch unified diff for a commit via GitHub API."""
    result = subprocess.run(
        ['gh', 'api', f'repos/{repo}/commits/{sha}',
         '-H', 'Accept: application/vnd.github.diff'],
        capture_output=True, text=True, errors='replace'
    )
    return result.stdout if result.returncode == 0 else ""


def fetch_commit_info(repo: str, sha: str) -> dict | None:
    """Fetch commit metadata (sha + message) via GitHub API."""
    result = subprocess.run(
        ['gh', 'api', f'repos/{repo}/commits/{sha}',
         '--jq', '{sha: .sha, message: .commit.message}'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def fetch_recent_commits(repo: str, limit: int = 100, since_date: str | None = None) -> list[dict]:
    """Fetch recent commits from the repo's default branch.

    Returns list of dicts with 'sha' and 'message' keys.
    """
    params = f'per_page={min(limit, 100)}'
    if since_date:
        params += f'&since={since_date}T00:00:00Z'
    result = subprocess.run(
        ['gh', 'api', f'repos/{repo}/commits?{params}',
         '--jq', '[.[] | {sha: .sha, message: .commit.message}]'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)


def fetch_pr_info(repo: str, pr_num: int) -> dict | None:
    """Fetch info for a single PR."""
    result = subprocess.run(
        ['gh', 'pr', 'view', str(pr_num), '--repo', repo,
         '--json', 'number,title,body'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def fetch_diff(repo: str, pr_num: int, use_cache: bool = True) -> str:
    """Fetch diff for a PR, using cache if available."""
    if use_cache:
        cached = load_cached_diff(repo, pr_num)
        if cached is not None:
            return cached

    result = subprocess.run(
        ['gh', 'pr', 'diff', str(pr_num), '--repo', repo],
        capture_output=True, text=True, errors='replace'
    )
    diff = result.stdout if result.returncode == 0 else ""

    if diff and use_cache:
        save_cached_diff(repo, pr_num, diff)

    return diff


def print_progress(current: int, total: int, pr_num: int, cached: bool = False, label: str | None = None):
    """Print progress indicator."""
    bar_width = 20
    filled = int(bar_width * current / total)
    bar = '█' * filled + '░' * (bar_width - filled)
    cache_indicator = f" {Colors.DIM}(cached){Colors.RESET}" if cached else ""
    id_str = label if label is not None else f"PR#{pr_num}"
    print(f"\r  {Colors.DIM}[{bar}]{Colors.RESET} {current}/{total} {id_str}{cache_indicator}    ",
          end='', file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='bepo - detect duplicate PRs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  bepo check --repo owner/repo
  bepo check --repo owner/repo --threshold 0.5
  bepo check-pr --repo owner/repo --pr 123
  bepo check --repo owner/repo --json
        '''
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # check command - compare all PRs
    check = subparsers.add_parser('check', help='Check repo for duplicate PRs')
    check.add_argument('--repo', '-r', required=True, help='GitHub repo (owner/repo)')
    check.add_argument('--threshold', '-t', type=float, default=0.65,
                       help='Similarity threshold (default: 0.65)')
    check.add_argument('--limit', '-l', type=int, default=None,
                       help='Max PRs to check (default: 50, or 500 when --since is used)')
    check.add_argument('--since', default=None,
                       help='Only compare PRs opened in last N days (e.g. --since 30d)')
    check.add_argument('--verbose', '-v', action='store_true',
                       help='Print bucket sizes and comparison details')
    check.add_argument('--json', action='store_true', help='Output JSON')
    check.add_argument('--no-cache', action='store_true', help='Disable diff caching')
    check.add_argument('--no-color', action='store_true', help='Disable colored output')

    # check-pr command - compare single PR against others
    check_pr = subparsers.add_parser('check-pr', help='Check if a PR duplicates others')
    check_pr.add_argument('--repo', '-r', required=True, help='GitHub repo (owner/repo)')
    check_pr.add_argument('--pr', '-p', type=int, required=True, help='PR number to check')
    check_pr.add_argument('--threshold', '-t', type=float, default=0.65,
                          help='Similarity threshold (default: 0.65)')
    check_pr.add_argument('--limit', '-l', type=int, default=50,
                          help='Max other PRs to compare against (default: 50)')
    check_pr.add_argument('--json', action='store_true', help='Output JSON')
    check_pr.add_argument('--no-cache', action='store_true', help='Disable diff caching')
    check_pr.add_argument('--no-color', action='store_true', help='Disable colored output')

    # check-commit command - compare a single commit against open PRs and recent commits
    check_commit = subparsers.add_parser(
        'check-commit', help='Check if a commit duplicates open PRs or recent commits'
    )
    check_commit.add_argument('--repo', '-r', required=True, help='GitHub repo (owner/repo)')
    check_commit.add_argument('--commit', '-c', required=True, help='Commit SHA (full or short)')
    check_commit.add_argument('--threshold', '-t', type=float, default=0.65,
                              help='Similarity threshold (default: 0.65)')
    check_commit.add_argument('--limit', '-l', type=int, default=50,
                              help='Max open PRs to compare against (default: 50)')
    check_commit.add_argument('--since', default='30d',
                              help='Compare against commits from last N days (default: 30d)')
    check_commit.add_argument('--commit-limit', type=int, default=100,
                              help='Max recent commits to compare against (default: 100)')
    check_commit.add_argument('--json', action='store_true', help='Output JSON')
    check_commit.add_argument('--no-color', action='store_true', help='Disable colored output')

    # clear-cache command
    subparsers.add_parser('clear-cache', help='Clear cached diffs')

    args = parser.parse_args()

    if args.command == 'clear-cache':
        run_clear_cache()
    elif args.command == 'check':
        if args.no_color or not sys.stderr.isatty():
            Colors.disable()
        run_check(args)
    elif args.command == 'check-pr':
        if args.no_color or not sys.stderr.isatty():
            Colors.disable()
        run_check_pr(args)
    elif args.command == 'check-commit':
        if args.no_color or not sys.stderr.isatty():
            Colors.disable()
        run_check_commit(args)


def run_clear_cache():
    """Clear the diff cache."""
    cache_dir = get_cache_dir()
    count = 0
    for f in cache_dir.glob('*.diff'):
        f.unlink()
        count += 1
    print(f"Cleared {count} cached diffs.")


def run_check(args):
    """Run duplicate check on a repo."""
    check_gh_installed()
    use_cache = not args.no_cache

    since_date = _parse_since(args.since) if args.since else None
    # When --since is used without an explicit --limit, fetch up to 500 PRs
    # so the time filter — not the count cap — is the binding constraint.
    limit = args.limit if args.limit is not None else (500 if args.since else 50)

    print(f"{Colors.BOLD}Fetching PRs from {args.repo}...{Colors.RESET}", file=sys.stderr)
    prs = fetch_prs(args.repo, limit, since_date=since_date)
    print(f"Found {Colors.CYAN}{len(prs)}{Colors.RESET} open PRs\n", file=sys.stderr)

    # Fingerprint each PR
    fingerprints = []
    cache_hits = 0
    start_time = time.time()

    for i, pr in enumerate(prs, 1):
        pr_num = pr['number']
        cached = load_cached_diff(args.repo, pr_num) is not None if use_cache else False
        if cached:
            cache_hits += 1
        print_progress(i, len(prs), pr_num, cached)

        diff = fetch_diff(args.repo, pr_num, use_cache=use_cache)
        if diff:
            fp = fingerprint_pr(
                f"#{pr_num}",
                diff,
                title=pr.get('title', ''),
                body=pr.get('body', '') or '',
            )
            fingerprints.append(fp)

    elapsed = time.time() - start_time
    print(f"\r{' ' * 60}\r", end='', file=sys.stderr)  # Clear progress line

    if use_cache and cache_hits > 0:
        print(f"{Colors.DIM}Analyzed {len(prs)} PRs in {elapsed:.1f}s ({cache_hits} cached){Colors.RESET}\n",
              file=sys.stderr)
    else:
        print(f"{Colors.DIM}Analyzed {len(prs)} PRs in {elapsed:.1f}s{Colors.RESET}\n", file=sys.stderr)

    # Find duplicates
    if args.verbose:
        print(f"{Colors.DIM}Bucket sizes:{Colors.RESET}", file=sys.stderr)
    duplicates = find_duplicates(
        fingerprints, threshold=args.threshold, verbose=args.verbose
    )

    if args.json:
        output = [
            {
                'pr_a': d.pr_a,
                'pr_b': d.pr_b,
                'similarity': round(d.similarity, 3),
                'shared_issues': d.shared_issues,
                'shared_files': d.shared_files,
                'shared_code_lines': d.shared_code_lines,
                'reason': d.reason,
            }
            for d in duplicates
        ]
        print(json.dumps(output, indent=2))
    else:
        if not duplicates:
            print(f"{Colors.GREEN}No duplicates found.{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}Found {len(duplicates)} potential duplicates:{Colors.RESET}\n")
            for d in duplicates:
                # Color code by similarity
                if d.similarity >= 0.8:
                    sim_color = Colors.RED
                elif d.similarity >= 0.6:
                    sim_color = Colors.YELLOW
                else:
                    sim_color = Colors.RESET

                print(f"{Colors.BOLD}{d.pr_a} <-> {d.pr_b}{Colors.RESET}")
                print(f"  {sim_color}Similarity: {d.similarity:.0%}{Colors.RESET}")
                print(f"  {Colors.DIM}Reason: {d.reason}{Colors.RESET}")
                print()


def run_check_pr(args):
    """Check a single PR against all other open PRs."""
    check_gh_installed()
    use_cache = not args.no_cache
    target_pr = args.pr

    # Fetch target PR info
    print(f"{Colors.BOLD}Fetching PR #{target_pr}...{Colors.RESET}", file=sys.stderr)
    pr_info = fetch_pr_info(args.repo, target_pr)
    if not pr_info:
        print(f"{Colors.RED}Error: Could not fetch PR #{target_pr}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    target_diff = fetch_diff(args.repo, target_pr, use_cache=use_cache)
    if not target_diff:
        print(f"{Colors.RED}Error: Could not fetch diff for PR #{target_pr}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    target_fp = fingerprint_pr(
        f"#{target_pr}",
        target_diff,
        title=pr_info.get('title', ''),
        body=pr_info.get('body', '') or '',
    )

    # Fetch other open PRs
    print(f"{Colors.BOLD}Fetching other open PRs...{Colors.RESET}", file=sys.stderr)
    prs = fetch_prs(args.repo, args.limit)
    # Exclude the target PR
    prs = [p for p in prs if p['number'] != target_pr]
    print(f"Comparing against {Colors.CYAN}{len(prs)}{Colors.RESET} other PRs\n", file=sys.stderr)

    # Fingerprint other PRs
    other_fps = []
    cache_hits = 0
    start_time = time.time()

    for i, pr in enumerate(prs, 1):
        pr_num = pr['number']
        cached = load_cached_diff(args.repo, pr_num) is not None if use_cache else False
        if cached:
            cache_hits += 1
        print_progress(i, len(prs), pr_num, cached)

        diff = fetch_diff(args.repo, pr_num, use_cache=use_cache)
        if diff:
            fp = fingerprint_pr(
                f"#{pr_num}",
                diff,
                title=pr.get('title', ''),
                body=pr.get('body', '') or '',
            )
            other_fps.append(fp)

    elapsed = time.time() - start_time
    print(f"\r{' ' * 60}\r", end='', file=sys.stderr)

    if use_cache and cache_hits > 0:
        print(f"{Colors.DIM}Analyzed in {elapsed:.1f}s ({cache_hits} cached){Colors.RESET}\n", file=sys.stderr)
    else:
        print(f"{Colors.DIM}Analyzed in {elapsed:.1f}s{Colors.RESET}\n", file=sys.stderr)

    # Compare target against all others using the same quality logic as `bepo check`
    target_id = f"#{target_pr}"
    all_fps = [target_fp] + other_fps
    duplicates = find_duplicates(all_fps, threshold=args.threshold)
    relevant = [d for d in duplicates if d.pr_a == target_id or d.pr_b == target_id]
    relevant.sort(key=lambda d: d.similarity, reverse=True)

    matches = [
        {
            'match': d.pr_b if d.pr_a == target_id else d.pr_a,
            'similarity': d.similarity,
            'shared_issues': d.shared_issues,
            'shared_code_lines': d.shared_code_lines,
            'reason': d.reason,
        }
        for d in relevant
    ]

    if args.json:
        print(json.dumps({'target': f'#{target_pr}', 'matches': matches}, indent=2))
    else:
        if not matches:
            print(f"{Colors.GREEN}PR #{target_pr} has no similar PRs.{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}PR #{target_pr} is similar to:{Colors.RESET}\n")
            for m in matches:
                if m['similarity'] >= 0.8:
                    sim_color = Colors.RED
                elif m['similarity'] >= 0.6:
                    sim_color = Colors.YELLOW
                else:
                    sim_color = Colors.RESET

                print(f"  {Colors.BOLD}{m['match']}{Colors.RESET} - {sim_color}{m['similarity']:.0%}{Colors.RESET}")
                print(f"    {Colors.DIM}{m['reason']}{Colors.RESET}")
                print()


def run_check_commit(args):
    """Check a single commit against open PRs and recent commits."""
    check_gh_installed()

    sha = args.commit
    print(f"{Colors.BOLD}Fetching commit {sha}...{Colors.RESET}", file=sys.stderr)
    info = fetch_commit_info(args.repo, sha)
    if not info:
        print(f"{Colors.RED}Error: Could not fetch commit {sha}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    diff = fetch_commit_diff(args.repo, sha)
    if not diff:
        print(f"{Colors.RED}Error: Could not fetch diff for commit {sha}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    full_sha = info['sha']
    commit_id = f"@{full_sha[:8]}"
    message = info.get('message', '')
    title, _, body = message.partition('\n')

    commit_fp = fingerprint_pr(commit_id, diff, title=title.strip(), body=body.strip())

    other_fps = []
    start_time = time.time()
    cache_hits = 0

    # Fetch and fingerprint open PRs
    print(f"{Colors.BOLD}Fetching open PRs...{Colors.RESET}", file=sys.stderr)
    prs = fetch_prs(args.repo, args.limit)
    print(f"  {Colors.CYAN}{len(prs)}{Colors.RESET} open PRs", file=sys.stderr)

    for i, pr in enumerate(prs, 1):
        pr_num = pr['number']
        cached = load_cached_diff(args.repo, pr_num) is not None
        if cached:
            cache_hits += 1
        print_progress(i, len(prs), pr_num, cached)
        pr_diff = fetch_diff(args.repo, pr_num, use_cache=True)
        if pr_diff:
            other_fps.append(fingerprint_pr(
                f"#{pr_num}",
                pr_diff,
                title=pr.get('title', ''),
                body=pr.get('body', '') or '',
            ))
    print(f"\r{' ' * 60}\r", end='', file=sys.stderr)

    # Fetch and fingerprint recent commits
    since_date = _parse_since(args.since) if args.since else None
    print(f"{Colors.BOLD}Fetching recent commits ({args.since})...{Colors.RESET}", file=sys.stderr)
    recent = fetch_recent_commits(args.repo, limit=args.commit_limit, since_date=since_date)
    # Exclude the target commit itself
    recent = [c for c in recent if not c['sha'].startswith(full_sha[:8])
              and not full_sha.startswith(c['sha'][:8])]
    print(f"  {Colors.CYAN}{len(recent)}{Colors.RESET} recent commits", file=sys.stderr)

    commit_cache_hits = 0
    for i, c in enumerate(recent, 1):
        short = c['sha'][:8]
        cached_cdiff = load_cached_commit_diff(args.repo, c['sha'])
        if cached_cdiff is not None:
            commit_cache_hits += 1
        print_progress(i, len(recent), 0, cached=cached_cdiff is not None, label=f"@{short}")
        c_diff = cached_cdiff or fetch_commit_diff(args.repo, c['sha'])
        if c_diff:
            if cached_cdiff is None:
                save_cached_commit_diff(args.repo, c['sha'], c_diff)
            c_msg = c.get('message', '')
            c_title, _, c_body = c_msg.partition('\n')
            other_fps.append(fingerprint_pr(
                f"@{short}",
                c_diff,
                title=c_title.strip(),
                body=c_body.strip(),
            ))
    print(f"\r{' ' * 60}\r", end='', file=sys.stderr)

    elapsed = time.time() - start_time
    total_cached = cache_hits + commit_cache_hits
    suffix = f" ({total_cached} cached)" if total_cached > 0 else ""
    print(f"{Colors.DIM}Analyzed in {elapsed:.1f}s{suffix}{Colors.RESET}\n", file=sys.stderr)

    all_fps = [commit_fp] + other_fps
    duplicates = find_duplicates(all_fps, threshold=args.threshold)
    matches_raw = [d for d in duplicates if d.pr_a == commit_id or d.pr_b == commit_id]
    matches_raw.sort(key=lambda d: d.similarity, reverse=True)

    matches = [
        {
            'match': d.pr_b if d.pr_a == commit_id else d.pr_a,
            'similarity': d.similarity,
            'shared_issues': d.shared_issues,
            'shared_code_lines': d.shared_code_lines,
            'reason': d.reason,
        }
        for d in matches_raw
    ]

    if args.json:
        print(json.dumps({'commit': commit_id, 'matches': matches}, indent=2))
    else:
        if not matches:
            print(f"{Colors.GREEN}Commit {commit_id} has no similar PRs or commits.{Colors.RESET}")
        else:
            print(f"{Colors.BOLD}Commit {commit_id} is similar to:{Colors.RESET}\n")
            for m in matches:
                if m['similarity'] >= 0.8:
                    sim_color = Colors.RED
                elif m['similarity'] >= 0.6:
                    sim_color = Colors.YELLOW
                else:
                    sim_color = Colors.RESET

                print(f"  {Colors.BOLD}{m['match']}{Colors.RESET} - {sim_color}{m['similarity']:.0%}{Colors.RESET}")
                print(f"    {Colors.DIM}{m['reason']}{Colors.RESET}")
                print()


if __name__ == '__main__':
    main()
