# Commit Message Convention

## Format
Each commit message should follow this format:

```
[type]: [short summary]

[optional longer description / context]
```

### Types
- **feat**: New feature (e.g. `feat: add thermal clustering`)
- **fix**: Bug fix (e.g. `fix: handle empty IGC parse`)
- **refactor**: Code change that improves structure without changing behavior
- **perf**: Performance improvement
- **docs**: Documentation changes
- **test**: Add or update tests
- **chore**: Maintenance tasks (build, config, cleanup)

### Rules
1. **Summary line ≤ 72 characters**
2. Use **present tense** (“add” not “added”)
3. Be specific about what changed and why
4. Use the optional body for more detail if needed

---

## Examples
```
feat: detect full 360° circles with tuned duration
fix: convert time column to datetime before diff
refactor: split circle detection into helper function
docs: add git workflow cheat sheet
```
