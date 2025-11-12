# Repository Separation Guide

## Current Situation

You have two separate projects in one local repository:
1. **Engine Design** (Pintle Engine Pipeline) - 103 files
2. **Parachute Dynamics** - 35 files

Both remotes are now configured:
- `engine-design` → `git@github.com:KushMahajan/EngineDesign.git`
- `parachute-dynamics` → `git@github.com:KushMahajan/Parachute-Dynamics.git`

## Problem

The current branch `pintle-only` contains BOTH projects, and we've been pushing everything to both repos. We need to separate them properly.

## Solution Options

### Option 1: Keep Both in Same Local Repo (Current Setup)

**Pros:** Easy to work on both projects
**Cons:** Need to be careful about what gets pushed where

**How to use:**
```bash
# Push engine code to engine-design repo
git push engine-design pintle-only

# Push parachute code to parachute-dynamics repo  
git push parachute-dynamics pintle-only
```

**Note:** Both repos will have all files, but that's okay if you want to keep them together locally.

### Option 2: Create Separate Branches (Recommended)

Create separate branches for each project:

```bash
# Create engine-only branch
git checkout -b engine-only
# Remove parachute files from this branch
git rm -r parachute/ examples/parachute/ README_PARACHUTE.md BUGFIXES_SUMMARY.md INFLATION_ISSUE_SUMMARY.md
git commit -m "Remove parachute files for engine-only branch"

# Create parachute-only branch
git checkout -b parachute-only
git checkout pintle-only -- parachute/ examples/parachute/ README_PARACHUTE.md BUGFIXES_SUMMARY.md INFLATION_ISSUE_SUMMARY.md
# Remove engine files
git rm -r pintle_models/ pintle_pipeline/ examples/pintle_engine/ README.md
git commit -m "Remove engine files for parachute-only branch"

# Push each branch to its respective repo
git push engine-design engine-only
git push parachute-dynamics parachute-only
```

### Option 3: Separate Local Repositories (Cleanest)

Clone the repos separately:

```bash
# For Engine Design
cd ..
git clone git@github.com:KushMahajan/EngineDesign.git EngineDesign
cd EngineDesign
# Copy engine files here
# Commit and push

# For Parachute Dynamics  
cd ..
git clone git@github.com:KushMahajan/Parachute-Dynamics.git Parachute-Dynamics
cd Parachute-Dynamics
# Copy parachute files here
# Commit and push
```

## Recommended Approach

I recommend **Option 2** (separate branches) because:
- Keeps both projects in one local repo (convenient)
- Clean separation in remote repos
- Easy to switch between projects
- Each repo only has relevant files

## Next Steps

1. Decide which option you prefer
2. If Option 2, I can help create the branches and push them
3. Update `.gitignore` files for each repo (templates provided: `.gitignore.engine` and `.gitignore.parachute`)

## Current Status

- ✅ Both remotes configured
- ✅ Parachute code pushed to `parachute-dynamics` repo
- ⚠️ Engine code needs to be pushed to `engine-design` repo (or separated)
- ⚠️ `.gitignore` needs to be updated for proper separation

