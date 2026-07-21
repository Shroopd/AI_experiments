---
tags:
  - llm
---

# LLM Git Safety Guidelines

## Safe Operations (No History Modification)

These operations preserve git history and are generally safe:

| Command | Use Case |
|---------|----------|
| `git add` | Stage files for commit |
| `git commit` | Create new commit |
| `git push` | Push to remote (normal, not --force) |
| `git pull` | Fetch and merge from remote |
| `git checkout` / `git switch` | Switch branches |
| `git merge` | Merge branches (normal merge) |
| `git branch` | Create new branches |
| `git status`, `git log`, `git diff` | Inspect repository |

## Dangerous Operations (History Modification)

These operations rewrite or destroy history and require explicit confirmation:

| Command | Risk |
|---------|------|
| `git push --force` / `--force-with-lease` | Overwrites remote history |
| `git reset --hard` | Discards commits and changes |
| `git rebase` (interactive or otherwise) | Rewrites commit history |
| `git branch -D` / `--delete` | Deletes branches |
| `git commit --amend` | Changes existing commit |
| `git filter-branch` / `filter-repo` | Rewrites entire history |

## Communication Patterns

**To allow safe operations:**
> "Commit and push only, no history modification"
> "Safe operations only: add, commit, push, pull, checkout, merge"

**To restrict all operations:**
> "Read-only" or "No changes"
> "Explain only"
> "Show me the commands, don't run them"

**When git operations are involved:**
> "git read-only"
> "Analyze the state"

## Best Practices

1. **Always verify branch tracking** before push/pull operations
2. **Check current branch** before committing
3. **Review diff** before committing
4. **Confirm force operations** explicitly
5. **Backup before destructive operations**

## LLM Interaction Guidelines

When working with an LLM on git tasks:

1. Specify **exactly** what operations are allowed
2. Use **explicit safety phrases** for read-only mode
3. Request **command review** before execution
4. Confirm **branch state** before any push
5. Never allow **force operations** without explicit confirmation