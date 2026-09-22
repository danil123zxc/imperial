#!/usr/bin/env bash
set -euo pipefail

dry_run=0
verifier_approved=0

usage() {
  printf 'Usage: %s --verifier-approved [--dry-run]\n' "$0"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '%s\n' "$*"
}

cleanup_created_pr_worktree() {
  if [[ "$repo_root" == "$canonical_workspace_root" ]]; then
    info "Primary worktree retained: $repo_root"
    return
  fi

  if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
    printf 'warning: linked worktree is not clean and was retained: %s\n' "$repo_root" >&2
    return
  fi

  cd "$canonical_workspace_root"
  if git worktree remove "$repo_root"; then
    git worktree prune
    info "Removed clean linked worktree: $repo_root"
  else
    printf 'warning: linked worktree cleanup failed and must be completed manually: %s\n' "$repo_root" >&2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verifier-approved)
      verifier_approved=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$verifier_approved" -eq 1 ]] || die "independent verifier APPROVE is required; pass --verifier-approved only after receiving it"

command -v git >/dev/null 2>&1 || die "git is required"

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die "run this inside a Git worktree"
cd "$repo_root"

branch=$(git symbolic-ref --quiet --short HEAD) || die "detached HEAD cannot be published"
case "$branch" in
  main|dev)
    die "protected branch '$branch' cannot be published directly"
    ;;
  codex/*)
    ;;
  *)
    die "branch '$branch' must use the codex/* prefix"
    ;;
esac

git remote get-url origin >/dev/null 2>&1 || die "origin remote is required"
[[ -z "$(git status --porcelain)" ]] || die "worktree must be clean before publishing"

info "Fetching origin..."
git fetch --no-tags origin --prune

base_branch="main"
base_ref="origin/main"
git rev-parse --verify --quiet "$base_ref^{commit}" >/dev/null || die "missing base ref $base_ref"
git merge-base --is-ancestor "$base_ref" HEAD || die "branch is stale or diverged; rebase it onto current $base_ref in a fresh worktree"

ahead_count=$(git rev-list --count "$base_ref..HEAD")
[[ "$ahead_count" -gt 0 ]] || die "there are no commits to publish"

changed_files=$(git diff --name-only "$base_ref...HEAD")
[[ -n "$changed_files" ]] || die "there are no changed files to publish"

file_count=$(printf '%s\n' "$changed_files" | sed '/^$/d' | wc -l | tr -d ' ')
approval_required=""
denied_path=""

while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  case "$path" in
    .env|.env.*|.DS_Store|.imperial_rag/*|documents/*|secrets/*|*/secrets/*|credentials/*|*/credentials/*|*_key*|*_secret*|auth.sqlite3|*/auth.sqlite3|chat_history.sqlite3|*/chat_history.sqlite3|eval_outputs/*|*/eval_outputs/*|phoenix/*|*/phoenix/*)
      denied_path="$path"
      break
      ;;
  esac
  case "$path" in
    compose.yaml|Dockerfile|uv.lock|.github/workflows/*|pyproject.toml|evals/questions.jsonl|evals/russian_judge_calibration.jsonl|scripts/ingest.py|scripts/run_*eval*.py|src/imperial_rag/observability/*|src/imperial_rag/app/*|src/imperial_rag/ingestion/*|src/imperial_rag/retrieval/*|src/imperial_rag/answering/*)
      approval_required="$approval_required $path"
      ;;
  esac
done <<< "$changed_files"

[[ -z "$denied_path" ]] || die "denylisted path changed: $denied_path"

if [[ "$file_count" -gt 10 ]]; then
  approval_required="$approval_required more-than-10-files"
fi

if [[ -n "$approval_required" && "${IMPERIAL_AGENT_APPROVED_HIGH_RISK:-0}" != "1" ]]; then
  die "separate high-risk approval is required for:$approval_required (set IMPERIAL_AGENT_APPROVED_HIGH_RISK=1 only after approval)"
fi

info "Running repository quality gate..."
common_git_dir=$(git rev-parse --git-common-dir)
if [[ "$common_git_dir" = /* ]]; then
  canonical_workspace_root=$(cd "$common_git_dir/.." && pwd -P)
else
  canonical_workspace_root=$(cd "$repo_root/$common_git_dir/.." && pwd -P)
fi
IMPERIAL_RAG_WORKSPACE_ROOT="$canonical_workspace_root" ./scripts/check.sh
git diff --check "$base_ref...HEAD"

if [[ "$dry_run" -eq 1 ]]; then
  info "Dry run successful: $ahead_count commit(s), $file_count changed file(s), branch $branch."
  exit 0
fi

command -v gh >/dev/null 2>&1 || die "GitHub CLI is required for publishing"
gh auth status --hostname github.com >/dev/null 2>&1 || die "GitHub CLI is not authenticated"

closed_pr_info=$(gh pr list --head "$branch" --state closed --limit 1 --json number,state,url --jq 'if length == 0 then "" else "\(.[0].number)\t\(.[0].state)\t\(.[0].url)" end')
open_pr_info=$(gh pr list --head "$branch" --state open --limit 1 --json number,isDraft,url --jq 'if length == 0 then "" else "\(.[0].number)\t\(.[0].isDraft)\t\(.[0].url)" end')

if [[ -n "$closed_pr_info" ]]; then
  die "branch already has a closed or merged PR: $closed_pr_info; create a fresh codex/* branch"
fi

open_pr_number=""
if [[ -n "$open_pr_info" ]]; then
  IFS=$'\t' read -r open_pr_number open_pr_is_draft open_pr_url <<< "$open_pr_info"
  [[ "$open_pr_is_draft" == "true" ]] || die "open PR #$open_pr_number is ready for review; a human must authorize further publishing"
fi

title=$(git log -1 --format=%s)
body=$(printf '%s\n' \
  '## Summary' \
  "- Automated assisted-publish update from \`$branch\`." \
  '- Review the commit history and base diff for implementation details.' \
  '' \
  '## Verification' \
  '- `./scripts/check.sh`' \
  "- \`git diff --check $base_ref...HEAD\`" \
  '- Independent verifier: `APPROVE` required by repository policy.' \
  '' \
  '## Scope and safety' \
  "- Changed files: $file_count" \
  '- Denylist scan passed.' \
  '- Auto-merge remains disabled.')

info "Pushing $branch..."
git push -u origin "$branch"

if [[ -n "$open_pr_number" ]]; then
  gh pr edit "$open_pr_number" --title "$title" --body "$body" >/dev/null
  pr_url=$(gh pr view "$open_pr_number" --json url --jq .url)
  info "Updated draft PR: $pr_url"
else
  pr_url=$(gh pr create --draft --base "$base_branch" --head "$branch" --title "$title" --body "$body")
  info "Created draft PR: $pr_url"
  cleanup_created_pr_worktree
fi
