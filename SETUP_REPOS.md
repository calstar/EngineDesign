# Repository Setup Guide

This workspace contains two separate projects that should be in separate repositories:

1. **Engine Design** - Pintle injector liquid rocket engine pipeline
2. **Parachute Dynamics** - Multi-body parachute recovery simulation system

## Current Setup

Both remotes are configured:
- `engine-design` → `git@github.com:KushMahajan/EngineDesign.git`
- `parachute-dynamics` → `git@github.com:KushMahajan/Parachute-Dynamics.git`

## Repository Contents

### Engine Design Repository Should Contain:
- `pintle_models/` - Core engine models
- `pintle_pipeline/` - Pipeline infrastructure
- `examples/pintle_engine/` - Engine examples and configs
- `README.md` - Engine design documentation
- `requirements.txt` - Dependencies

### Parachute Dynamics Repository Should Contain:
- `parachute/` - Parachute simulation engine
- `examples/parachute/` - Parachute examples and configs
- `README_PARACHUTE.md` - Parachute documentation
- `BUGFIXES_SUMMARY.md` - Bug fixes documentation
- `INFLATION_ISSUE_SUMMARY.md` - Inflation troubleshooting
- `requirements.txt` - Dependencies (shared with engine)

## How to Push to Each Repository

### Push Engine Design Code:
```bash
# Create a branch with only engine files
git checkout -b engine-only
# Or push current branch to engine-design remote
git push engine-design pintle-only
```

### Push Parachute Dynamics Code:
```bash
# Push to parachute-dynamics remote
git push parachute-dynamics pintle-only
```

## Recommended Workflow

1. **For Engine Design work:**
   - Work on `pintle_models/`, `pintle_pipeline/`, `examples/pintle_engine/`
   - Commit and push to `engine-design` remote

2. **For Parachute Dynamics work:**
   - Work on `parachute/`, `examples/parachute/`
   - Commit and push to `parachute-dynamics` remote

3. **Shared files:**
   - `requirements.txt` - Should be in both repos (or separate)
   - `.gitignore` - Should be configured for each repo's needs

## Next Steps

1. Decide if you want separate branches for each project
2. Or create completely separate local repositories
3. Update `.gitignore` to properly exclude/include files for each repo

