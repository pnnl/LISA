# Git LFS Guide For LISA Collaborators

This repository uses Git LFS (Git Large File Storage) for large checkpoint artifacts under `checkpoints/**`.

Tracked checkpoint formats in this repo:

- `checkpoints/**/*.bin`
- `checkpoints/**/*.pt`
- `checkpoints/**/*.pth`
- `checkpoints/**/*.ckpt`
- `checkpoints/**/*.safetensors`
- `checkpoints/**/*.onnx`

## What Git LFS Is

Git LFS stores large files outside normal Git history and keeps a small pointer file in the repository instead.

Why this matters:

- GitHub rejects regular Git blobs over 100 MB.
- Large binaries make repository history slow to clone and push.
- LFS keeps the Git history manageable while still letting collaborators pull the real files when needed.

In practice:

- Git tracks a lightweight text pointer.
- Git LFS downloads the real file content when you clone, pull, or explicitly fetch LFS objects.

## One-Time Setup

Install Git LFS on your machine first.

Project page:

- https://git-lfs.com/

After installation, run:

```bash
git lfs install
```

You only need to run `git lfs install` once per machine.

## Fresh Clone

If you are cloning the repository for the first time:

```bash
git lfs install
git clone https://github.com/pnnl/LISA.git
cd LISA
git lfs pull
```

Notes:

- `git clone` should automatically fetch LFS content in most setups.
- `git lfs pull` is included here to make the checkout explicit and predictable.

If you need the internal remote as well:

```bash
git remote add origin-internal https://tanuki.pnnl.gov/aces-wind/lisa.git
git fetch origin-internal
```

## Existing Clone After The Branch Rewrite

`develop` and `add-lisa-aies-paper-materials` were rewritten to move checkpoint artifacts into Git LFS. If you already had a local clone from before that rewrite, resync your local branches before doing more work.

First install and enable LFS if needed:

```bash
git lfs install
```

Then refresh remote refs:

```bash
git fetch --all --prune
git lfs fetch --all
```

To update local `develop` to match the rewritten remote branch exactly:

```bash
git checkout develop
git reset --hard origin/develop
git lfs pull
```

If you also work on the GitHub feature branch:

```bash
git checkout add-lisa-aies-paper-materials
git reset --hard upstream/add-lisa-aies-paper-materials
git lfs pull
```

If you had unpushed local commits before the rewrite, do not run the reset commands until you have backed up your work with a temporary branch or patch.

## Daily Usage

Most of the time, Git LFS should feel like normal Git.

Typical workflow:

```bash
git pull
git add <files>
git commit -m "Your change"
git push
```

If you add a new checkpoint file under `checkpoints/**` with one of the tracked extensions, Git LFS should handle it automatically.

You can verify which files are LFS-managed with:

```bash
git lfs ls-files
```

## Adding New Large Files

For this repository, checkpoint artifacts should stay under `checkpoints/**` and use one of the tracked extensions listed above.

Recommended:

- Keep training outputs and model weights under `checkpoints/`.
- Use tracked checkpoint extensions for large model artifacts.
- Avoid committing large derived data outside the tracked paths.

Avoid:

- Committing files over 100 MB outside Git LFS.
- Renaming large checkpoint files to untracked extensions.
- Checking in dataset copies unless the repo owners explicitly want them versioned.

## Useful Commands

Show tracked LFS rules:

```bash
git lfs track
```

Show LFS-managed files in the current checkout:

```bash
git lfs ls-files
```

Download LFS objects for the current checkout:

```bash
git lfs pull
```

Fetch all LFS objects referenced by fetched history:

```bash
git lfs fetch --all
```

Check the working tree state:

```bash
git status
```

## Troubleshooting

If a checkout leaves you with small text pointer files instead of real model files:

```bash
git lfs pull
```

If Git says `git: 'lfs' is not a git command`, install Git LFS and run:

```bash
git lfs install
```

If a push fails because of a large file, check whether it lives outside the tracked `checkpoints/**` patterns.

Helpful checks:

```bash
git lfs track
git lfs ls-files
git status
```

## Branches And Remotes

Current primary branches:

- `origin/develop`
- `upstream/develop`
- `upstream/add-lisa-aies-paper-materials`

Backup branches created during the LFS migration:

- `origin/develop-bak-20260713`
- `upstream/develop.bak.20260713`

## Summary

For most collaborators, the important steps are:

1. Install Git LFS.
2. Run `git lfs install` once.
3. Clone the repo normally.
4. Run `git lfs pull` if checkpoint files are needed.
5. Keep new large checkpoint artifacts under `checkpoints/**` using the tracked checkpoint extensions.
