# GitLab CI

The repository lives at `gitlab.com/experiential-labs-group/experiential`.
`.gitlab-ci.yml` holds only the pipeline-wide settings and includes every file
in `.gitlab/ci/`. Each file there owns one pipeline area and is named after it;
`python-package` is the only area. `.gitlab/ci/_templates.yml` holds the shared
hidden keys: pinned tool versions and install snippets. The gate and
gateway-latency workflows under `.github/workflows/` have no runner.

## Pipelines and triggers

`workflow:rules` runs merge request pipelines for branches with an open merge
request and branch pipelines otherwise, and admits tag pushes and manual
(`web`, `api`) pipelines. Because one pipeline can come from any of these
sources, **every job has `rules:`** that name the sources it runs for. A job
without rules would run in every pipeline.

`python-package` runs for two sources only:

| Runs on | Rule |
| --- | --- |
| Protected release tags | `$CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+(rc\d+)?$/ && $CI_COMMIT_REF_PROTECTED == "true" && $CI_PIPELINE_SOURCE == "push"` |
| Manual runs | `$CI_PIPELINE_SOURCE =~ /^(web\|api)$/ && $RUN_WORKFLOW == "python-package"` |

Merge requests and pushes to `main` run nothing. Every job uses a GitLab-hosted
runner, which bills the group's compute minutes, and macOS minutes cost six
times the small Linux rate.

A manual run sets `RUN_WORKFLOW=python-package` and the area's inputs in the
**Run pipeline** form:

| Variable | Choices | Use |
| --- | --- | --- |
| `PYTHON_PACKAGE_SOURCE` | `latest-release` (default), `selected-ref` | Rebuild the latest release's tag, or the ref the pipeline runs on |
| `PYTHON_PACKAGE_PUBLISH` | `false` (default), `true` | Run the release checks and publish to PyPI |

## Jobs

- **Images.** Linux and macOS jobs set `image:`. The hosted Windows runner
  cannot: it runs a fixed Windows Server 2022 VM with PowerShell.
- **Tools.** uv and maturin come from `UV_VERSION` and `MATURIN_VERSION` in
  `.tools`. Rust follows the stable channel through rustup.
- **Ordering.** Jobs order with `needs:`; a job without dependencies sets
  `needs: []` so it starts at once. Values pass between jobs through
  `artifacts:reports:dotenv`, and files through `artifacts:`.

## Runners

| Runner tag | Used by | GitLab plan |
| --- | --- | --- |
| `saas-linux-small-amd64` (untagged default) | The linux-x86_64 wheel, the `experiential` build, the Capture smoke, and the publish job | Free |
| `saas-linux-small-arm64` | The linux-aarch64 wheel | Free |
| `saas-windows-medium-amd64` (beta) | The windows-x86_64 wheel | Free |
| `saas-macos-medium-m1` (beta) | The macos-arm64 and macos-x86_64 wheels | Premium or Ultimate |

A job whose runner the plan does not include stays pending, and the publish
job waits for every wheel. A self-hosted runner registered with the same tag
serves it instead.

## Releases

A release is a tag `v<version>`, where `<version>` is the `experiential`
version in `pyproject.toml` (for example `v0.7.179` or `v0.7.94rc1`). `v*` tags
must be protected (**Settings > Repository > Protected tags**, Maintainers
only): a tag that is not protected starts no jobs. Creating a GitLab release
with a new tag, or pushing the tag, starts a tag pipeline that builds the
`experiential` sdist and wheel and the `exp-gateway-native` sdist and abi3
wheels for linux-x86_64, linux-aarch64, macos-x86_64, macos-arm64, and
windows-x86_64, smoke-tests them, and publishes both distributions to PyPI.

A manual `python-package` run on `main` with `PYTHON_PACKAGE_SOURCE=latest-release`
and `PYTHON_PACKAGE_PUBLISH=true` rebuilds and publishes the tag of the latest
GitLab release. Files whose exact names are already on PyPI are skipped.

While `.github/deferred-native-release.toml` exists, release tags and
publishing runs fail before they publish.

Publishing uses PyPI trusted publishing. The `python-package-publish` job runs
in the `pypi` environment and requests a GitLab ID token with audience `pypi`
as `PYPI_ID_TOKEN`; `uv publish` exchanges it for a short-lived PyPI token.
Both PyPI projects, `experiential` and `exp-gateway-native`, trust this
publisher:

| Field | Value |
| --- | --- |
| Namespace | `experiential-labs-group` |
| Project name | `experiential` |
| Top-level pipeline file path | `.gitlab-ci.yml` |
| Environment name | `pypi` |

## Variables and settings

The pipelines read no secret CI/CD variables. Jobs authenticate with
`CI_JOB_TOKEN` and with the PyPI ID token.

| Setting | Use |
| --- | --- |
| **Settings > Repository > Protected tags**: `v*`, Maintainers | Release tag pipelines run and publish only for protected tags |
| **Settings > CI/CD > Variables > Minimum role to use pipeline variables**: Maintainer | Manual runs set `RUN_WORKFLOW` and the `PYTHON_PACKAGE_*` inputs |
| Environment `pypi` | `python-package-publish` deploys to it |
