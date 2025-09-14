# Git Terminal Quick Reference

## Check branch & status
```bash
git branch                # shows all branches, * = current branch
git status                # shows current branch + staged/unstaged files
```

## Updating your branch (with rebase)
```bash
git fetch origin          # update info about remote branches
git pull                  # with pull.rebase=true, this rebases by default
```

## Adding & committing
```bash
git add <file>            # stage specific file
git add .                 # stage all changes
git commit -m "feat: message"       # quick commit
git commit -F msg.txt               # use message from file
```

## Pushing
```bash
git push                  # push current branch
git push -u origin <branch>   # first time pushing a new branch
```

## Rebasing manually (if push rejected)
```bash
git fetch origin
git rebase origin/<branch>     # replay local commits on top of remote
# resolve conflicts if any, then:
git add <fixed-files>
git rebase --continue
git push
```

## Undo (careful)
```bash
git restore <file>             # discard unstaged changes in file
git restore --staged <file>    # unstage a file, keep changes
git reset --hard HEAD~1        # drop last commit & changes (dangerous)
```

## Logs & diffs
```bash
git log --oneline --graph --decorate --all   # nice commit overview
git diff                                    # unstaged changes
git diff --staged                           # staged changes
```

## Stash (set work aside)
```bash
git stash
git stash list
git stash pop
```
