# Git Essentials Cheat Sheet (Denis)

## Daily basics
```bash
git status                 # what changed?
git add .                  # stage all changes (new + modified)
git cm "feat: message"     # commit with your alias (commit -m)
git push                   # push current branch
git pull                   # pull latest for current branch
```

## First-time setup in a folder
```bash
git init
git remote add origin https://github.com/denisbuckley/chatgpt_igc.git
git branch -M main
git push -u origin main
```

## Branch workflow
```bash
git checkout -b dev                          # create dev from current branch
git push -u origin dev                       # publish dev

git checkout -b feature/my-experiment        # start feature branch
git push -u origin feature/my-experiment     # publish feature branch

git checkout dev
git merge feature/my-experiment              # bring feature into dev
git push origin dev

git checkout main
git merge dev                                # promote dev → main
git push origin main
```

## Clean up branches
```bash
git branch -d feature/my-experiment          # delete local
git push origin --delete feature/my-experiment  # delete remote
```

## See history & changes
```bash
git log --oneline --graph --decorate --all   # nice overview
git diff                                     # unstaged changes
git diff --staged                            # staged changes
```

## Undo (safe-ish)
```bash
git restore <file>           # discard unstaged changes to file
git restore --staged <file>  # unstage a file (keep changes)
git reset --hard HEAD~1      # ⚠️ drop last commit & changes (careful)
```

## Stash (put work aside temporarily)
```bash
git stash            # save dirty state
git stash list
git stash pop        # re-apply and remove from stash
```

## Global quality-of-life (already set)
```bash
git config --global alias.cm "commit -m"
git config --global alias.cma "commit -am"
git config --global commit.template ~/.gitmessage.txt
```
