"""CLI for bepo."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from .fingerprint import fingerprint_pr, find_duplicates, Fingerprint


def fetch_prs(repo: str, limit: int = 50) -> list[dict]:
    """Fetch open PRs from GitHub."""
    result = subprocess.run(
        ['gh', 'pr', 'list', '--repo', repo, '--state', 'open',
         '--limit', str(limit), '--json', 'number,title,body'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error fetching PRs: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def fetch_diff(repo: str, pr_num: int) -> str:
    """Fetch diff for a PR."""
    result = subprocess.run(
        ['gh', 'pr', 'diff', str(pr_num), '--repo', repo],
        capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else ""


def main():
    parser = argparse.ArgumentParser(
        description='bepo - detect duplicate PRs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  bepo check --repo owner/repo
  bepo check --repo owner/repo --threshold 0.5
  bepo check --repo owner/repo --json
        '''
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # check command
    check = subparsers.add_parser('check', help='Check repo for duplicate PRs')
    check.add_argument('--repo', '-r', required=True, help='GitHub repo (owner/repo)')
    check.add_argument('--threshold', '-t', type=float, default=0.4,
                       help='Similarity threshold (default: 0.4)')
    check.add_argument('--limit', '-l', type=int, default=50,
                       help='Max PRs to check (default: 50)')
    check.add_argument('--json', action='store_true', help='Output JSON')

    args = parser.parse_args()

    if args.command == 'check':
        run_check(args)


def run_check(args):
    """Run duplicate check on a repo."""
    print(f"Fetching PRs from {args.repo}...", file=sys.stderr)
    prs = fetch_prs(args.repo, args.limit)
    print(f"Found {len(prs)} open PRs", file=sys.stderr)

    # Fingerprint each PR
    fingerprints = []
    for pr in prs:
        pr_num = pr['number']
        print(f"  Analyzing PR#{pr_num}...", file=sys.stderr)
        diff = fetch_diff(args.repo, pr_num)
        if diff:
            fp = fingerprint_pr(
                f"#{pr_num}",
                diff,
                title=pr.get('title', ''),
                body=pr.get('body', '') or '',
            )
            fingerprints.append(fp)

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
                'reason': d.reason,
            }
            for d in duplicates
        ]
        print(json.dumps(output, indent=2))
    else:
        if not duplicates:
            print("\nNo duplicates found.")
        else:
            print(f"\nFound {len(duplicates)} potential duplicates:\n")
            for d in duplicates:
                print(f"{d.pr_a} <-> {d.pr_b}")
                print(f"  Similarity: {d.similarity:.0%}")
                print(f"  Reason: {d.reason}")
                print()


if __name__ == '__main__':
    main()
