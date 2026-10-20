# Janelia Deploy Notes

This file documents how this Janelia fork of Neuroglancer, used to build an internal deployment, is maintained.

## Branch model

- `master` — kept in sync with upstream `google/neuroglancer`, no Janelia-specific commits. This lets us pull upstream changes without conflicts.
- feature branches — Janelia-specific changes. Can be intended for an upstream PR or Janelia-only use.
- `deploy` — long-lived integration branch that accumulates `master` syncs and feature branch merges. This is what gets built for internal deployment.

## Workflow

### For developers:

**New feature:**
1. Develop a feature branch off `master` in this fork.
2. Open a PR against `deploy` in this fork.
3. Open a PR against upstream `google/neuroglancer` if applicable.
4. Add your feature branch to the tracking table on the wiki page for the Janelia Neuroglancer internal deployment.

### For maintainer(s):

Recommended: enable `git rerere` to reuse conflict resolutions across repeated merges of the same feature branch into successive deploy branches.
```
git config rerere.enabled true
```
**New internal Neuroglancer deploy:**

1. `git fetch origin`
2. For each branch in the tracking table whose upstream PR has been merged, check whether `deploy` is missing any commits from the branch. Upstream squash-merges PRs from this fork's branches, so the branch tip has the same content as the squash commit on `master`; bringing `deploy` up to the tip first makes the `master` merge a no-op for that feature.
   ```
   git log --oneline deploy..origin/<branch>
   ```
   Empty output means `deploy` already has the final version. Otherwise merge the branch tip into `deploy` now:
   ```
   git checkout deploy
   git merge origin/<branch>
   ```
3. Sync this fork's `master` with upstream. Fast-forward only, so no Janelia commits can land on `master`:
   ```
   git remote add upstream https://github.com/google/neuroglancer.git   # once
   git fetch upstream
   git checkout master
   git merge --ff-only upstream/master
   git push origin master
   ```
4. Merge `master` into `deploy`. Resolve conflicts favoring `master`'s (upstream) side. If `git rerere` auto-applied a resolution, check `git rerere diff` before trusting it, since it may have been recorded against an older version of the same conflict.
5. Merge in any new or updated feature branches.
6. Clean up: delete feature branches and their corresponding rows in wiki tracking table whose upstream PR has landed and was handled in step 2.
   ```
   git branch -d <branch>
   git push origin --delete <branch>
   ```
7. Build the new deploy by running `npm run build` from the internally hosted Neuroglancer directory (see wiki for location details).
8. Record the new updated `deploy` branch's commit and date in the wiki table.
