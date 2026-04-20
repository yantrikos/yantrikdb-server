# PyPI Trusted Publisher setup — yantrikdb-client

One-time configuration so `.github/workflows/publish-sdk.yml` can
publish the SDK to PyPI via OIDC. No long-lived tokens stored anywhere.

## Why this and not API tokens

PyPI supports **Trusted Publishers** via OpenID Connect: the PyPI
publish endpoint trusts short-lived identity tokens issued by
GitHub Actions for a specific workflow in a specific repo. Tokens
expire in minutes and are scoped to one job. Compromising a repo
secret can't leak publish access because no secret exists.

## One-time setup

Do these steps once as the PyPI account owner. You need:

- a PyPI account with permissions to create/manage the
  `yantrikdb-client` project
- admin access to the `yantrikos/yantrikdb-server` GitHub repo

### 1. PyPI: add a pending (or existing) Trusted Publisher

**If `yantrikdb-client` is not yet registered on PyPI** (first-ever
release):

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI project name**: `yantrikdb-client`
   - **Owner**: `yantrikos`
   - **Repository name**: `yantrikdb-server`
   - **Workflow name**: `publish-sdk.yml`
   - **Environment name**: `pypi`
4. Click "Add"

**If the project already exists**:

1. Go to https://pypi.org/manage/project/yantrikdb-client/settings/publishing/
2. Click "Add a new publisher"
3. Fill in the same four fields as above
4. Click "Add"

### 2. TestPyPI (optional but recommended)

Mirror the same setup on https://test.pypi.org so you can dry-run via
`workflow_dispatch` before a real release.

1. Go to https://test.pypi.org/manage/account/publishing/
2. "Add a new pending publisher" with:
   - **PyPI project name**: `yantrikdb-client`
   - **Owner**: `yantrikos`
   - **Repository name**: `yantrikdb-server`
   - **Workflow name**: `publish-sdk.yml`
   - **Environment name**: `testpypi`

### 3. GitHub: create the `pypi` environment

1. Go to https://github.com/yantrikos/yantrikdb-server/settings/environments
2. Click "New environment"
3. Name: `pypi`
4. Recommended protections:
   - **Required reviewers**: add yourself (or a small trusted group).
     This means every real publish needs a human click — prevents a
     malicious tag push from auto-publishing.
   - **Deployment branches and tags**: restrict to tags matching `sdk-v*`

5. Save.

Repeat for `testpypi` if you set up TestPyPI above (you can skip the
required-reviewer rule on testpypi since it's not user-facing).

## How to release

Local:

```bash
# 1. bump version in sdk/python/pyproject.toml
# 2. commit
git commit -am "sdk: vX.Y.Z release"

# 3. tag with sdk-v prefix
git tag -a sdk-vX.Y.Z -m "release notes..."

# 4. push tag
git push server-origin sdk-vX.Y.Z
```

Pushing the tag triggers the workflow. If the `pypi` environment has
required reviewers, approve the job on the GitHub Actions run to
actually publish. Otherwise it publishes immediately.

## Dry run via TestPyPI

Go to the repo's Actions tab → "Publish Python SDK to PyPI" → Run workflow.
Select `testpypi` as target. TestPyPI receives the build without
affecting the real index.

## Troubleshooting

**"pending publisher" didn't convert to an active one**: The first
publish converts a pending publisher to a real one. If the workflow
errored before calling `gh-action-pypi-publish`, the pending publisher
is still pending. Just re-run the workflow (workflow_dispatch is
fine).

**"trusted publisher not configured"** during a run: the Owner /
Repository / Workflow / Environment must all match *exactly*. Check
for typos, especially in the environment name.

**Version already exists**: PyPI doesn't allow re-uploading the same
version. Bump the version, re-commit, re-tag.

## Security notes

- The workflow uses `id-token: write` only in the publish jobs, not the
  build job. Narrowing permissions is the point.
- The `environment: pypi` rule means the publish job cannot run from
  forks or branches that don't match the environment's branch/tag
  filter.
- No secrets named `PYPI_API_TOKEN` exist in the repo. If someone
  tries to add one, that's a smell — we don't need it.
