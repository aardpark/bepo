"""CLI for bepo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def fetch_prs(repo: str, limit: int = 50) -> list[dict]:
    """Fetch open PRs from GitHub."""
    result = subprocess.run(
        ['gh', 'pr', 'list', '--repo', repo, '--state', 'open',
         '--limit', str(limit), '--json', 'number,title,body'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"{Colors.RED}Error fetching PRs: {result.stderr}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)
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
        capture_output=True, text=True
    )
    diff = result.stdout if result.returncode == 0 else ""

    if diff and use_cache:
        save_cached_diff(repo, pr_num, diff)

    return diff


def print_progress(current: int, total: int, pr_num: int, cached: bool = False):
    """Print progress indicator."""
    bar_width = 20
    filled = int(bar_width * current / total)
    bar = '█' * filled + '░' * (bar_width - filled)
    cache_indicator = f" {Colors.DIM}(cached){Colors.RESET}" if cached else ""
    print(f"\r  {Colors.DIM}[{bar}]{Colors.RESET} {current}/{total} PR#{pr_num}{cache_indicator}    ",
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
    check.add_argument('--threshold', '-t', type=float, default=0.4,
                       help='Similarity threshold (default: 0.4)')
    check.add_argument('--limit', '-l', type=int, default=50,
                       help='Max PRs to check (default: 50)')
    check.add_argument('--json', action='store_true', help='Output JSON')
    check.add_argument('--no-cache', action='store_true', help='Disable diff caching')
    check.add_argument('--no-color', action='store_true', help='Disable colored output')

    # check-pr command - compare single PR against others
    check_pr = subparsers.add_parser('check-pr', help='Check if a PR duplicates others')
    check_pr.add_argument('--repo', '-r', required=True, help='GitHub repo (owner/repo)')
    check_pr.add_argument('--pr', '-p', type=int, required=True, help='PR number to check')
    check_pr.add_argument('--threshold', '-t', type=float, default=0.4,
                          help='Similarity threshold (default: 0.4)')
    check_pr.add_argument('--limit', '-l', type=int, default=50,
                          help='Max other PRs to compare against (default: 50)')
    check_pr.add_argument('--json', action='store_true', help='Output JSON')
    check_pr.add_argument('--no-cache', action='store_true', help='Disable diff caching')
    check_pr.add_argument('--no-color', action='store_true', help='Disable colored output')

    # clear-cache command
    clear = subparsers.add_parser('clear-cache', help='Clear cached diffs')

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
    use_cache = not args.no_cache

    print(f"{Colors.BOLD}Fetching PRs from {args.repo}...{Colors.RESET}", file=sys.stderr)
    prs = fetch_prs(args.repo, args.limit)
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
    duplicates = find_duplicates(fingerprints, threshold=args.threshold)

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

    # Compare target against all others
    matches = []
    for other_fp in other_fps:
        sim = target_fp.similarity(other_fp)
        if sim >= args.threshold:
            shared_issues = list(set(target_fp.issue_refs) & set(other_fp.issue_refs))
            shared_code = set(target_fp.code_lines) & set(other_fp.code_lines)

            if shared_issues:
                reason = f"Both fix #{', #'.join(shared_issues[:2])}"
            elif len(shared_code) >= 3:
                reason = f"Same code: {len(shared_code)} lines overlap"
            else:
                reason = f"Similar files/domains"

            matches.append({
                'pr': other_fp.pr_id,
                'similarity': sim,
                'shared_issues': shared_issues,
                'shared_code_lines': len(shared_code),
                'reason': reason,
            })

    matches.sort(key=lambda x: x['similarity'], reverse=True)

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

                print(f"  {Colors.BOLD}{m['pr']}{Colors.RESET} - {sim_color}{m['similarity']:.0%}{Colors.RESET}")
                print(f"    {Colors.DIM}{m['reason']}{Colors.RESET}")
                print()


if __name__ == '__main__':
    main()
