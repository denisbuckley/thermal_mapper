# Git Workflow Cheat Sheet

## Branch Strategy
- **main** → stable, tested code
- **dev** → integration branch
- **feature/*** → experimental branches

---

## Daily Workflow

### 1. Sync main & dev
```bash
git checkout main
git pull origin main

git checkout dev
git pull origin dev
```

### 2. Create a feature branch
```bash
git checkout dev
git checkout -b feature/my-new-experiment
```

### 3. Work, commit, push
```bash
git add .
git commit -m "Describe change"
git push -u origin feature/my-new-experiment
```

### 4. Merge feature → dev
```bash
git checkout dev
git pull origin dev
git merge feature/my-new-experiment
git push origin dev
```

### 5. Merge dev → main (when stable)
```bash
git checkout main
git pull origin main
git merge dev
git push origin main
```

### 6. Clean up (optional)
```bash
git branch -d feature/my-new-experiment
git push origin --delete feature/my-new-experiment
```

---

## Notes
- Use `feature/<short-description>` names for clarity.
- Always test on `dev` before merging into `main`.
- Keep commits small and descriptive.
