# Push commands for this repo

Two flows: pick whichever matches your local setup.
Replace `OWNER` with your GitHub username or org.

> **Warning -- never commit these:**
> `model_payload/`, any `*.pth`, any `*.tar.gz` (or `model.tar.gz`).
> The bundled `.gitignore` already blocks them, but double-check
> `git status` before the first commit. The `checkpoint_best.pth`
> (819 MB) and `model.tar.gz` (1.5 GB) both exceed GitHub's 100 MB
> hard limit and will fail the push if accidentally staged.

---

## Flow A -- Plain git CLI from scratch

Use this when you have already created the empty repo on github.com
(see `REPO_FORM.md`).

```bash
cd /workspace/submission/v0/github_repo
git init -b main
git add .
git -c user.email=you@example.com -c user.name=you commit -m "Initial commit: PENGWIN 2026 Task 1 V0 (ABBC bw=10)"
git remote add origin git@github.com:OWNER/pengwin2026-task1-abbc.git
git push -u origin main
git tag v0.2.0
git push origin v0.2.0
```

Notes:
- Replace `you@example.com` / `you` with your real identity, or set up
  `git config --global user.email/user.name` once and drop the `-c`
  overrides.
- The HTTPS remote works too:
  `https://github.com/OWNER/pengwin2026-task1-abbc.git`.
- Tagging `v0.2.0` is what Grand Challenge "Link to GitHub" picks up
  for the container build.

---

## Flow B -- GitHub CLI (one-shot, no manual repo creation)

Use this if you have `gh` installed and authenticated. It creates the
repo, pushes the initial commit, and sets the remote in one step.

```bash
cd /workspace/submission/v0/github_repo
git init -b main
git add .
git -c user.email=you@example.com -c user.name=you commit -m "Initial commit: PENGWIN 2026 Task 1 V0 (ABBC bw=10)"
gh repo create OWNER/pengwin2026-task1-abbc --public --source=. --remote=origin --push
git tag v0.2.0
git push origin v0.2.0
```

If you prefer `gh` to also commit, drop the manual `git add` / `git commit`
lines -- `gh repo create --push` will still push whatever is staged. The
explicit commit is safer because it gives you a sanity-check point before
anything reaches the remote.

---

## Sanity check before the first push

```bash
cd /workspace/submission/v0/github_repo
git status
du -sh .
find . -size +50M -type f -not -path "./.git/*"
```

Expected:
- `git status` lists `Dockerfile`, `requirements.txt`, `README.md`,
  `LICENSE`, `.gitignore`, `inference/`, `code_task1/`, `scripts/`,
  `docs/`, `REPO_FORM.md`, `PUSH_COMMANDS.md` and nothing else.
- Total size well under 100 MB.
- `find ... -size +50M` returns empty -- no large blobs leaked.
