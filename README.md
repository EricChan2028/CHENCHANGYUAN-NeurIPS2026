# CPU container deployment

This directory prepares a free, public GitHub Actions build path for the Eastern Eagles
T2 and T3 candidates. It does not upload to CodaBench or contain team authentication.
Linux container execution has not been validated on the current Windows machine: it has
no Docker daemon or installed WSL distribution. A successful future workflow run is
required before treating an image as deployment-ready.

## Prepare the source artifact

Run from the campaign directory using Python 3.13:

```powershell
.\.venv\Scripts\python.exe tools\prepare_public_source.py --out deployment\artifacts\eastern-eagles-public-source-20261013-v3.tar.gz
tar -tzf deployment\artifacts\eastern-eagles-public-source-20261013-v3.tar.gz
```

The exporter reads only the exact paths in `public-source-files.json`. It refuses missing
files, links, hard links, binary files, oversized sources and recognizable credential or
submission-proof material. It never traverses the secret store, private packages,
environments, output directories or official task data. The archive has fixed ordering,
timestamps and permissions, plus a per-file SHA-256 manifest. Identical source bytes yield
identical archive bytes. It refuses to overwrite an earlier artifact; choose a new name
after any source change. Review the whitelist and archive before publication. Pattern
scanning is an extra guard, not a guarantee that arbitrary source text is safe to publish.

Extract the archive into a new, empty directory for the selected **public** GitHub repository:
[EricChan2028/CHENCHANGYUAN-NeurIPS2026](https://github.com/EricChan2028/CHENCHANGYUAN-NeurIPS2026).
The archive installs `cpu-build.yml` at `.github/workflows/cpu-build.yml`; preserve that
hidden directory when uploading. Do not publish the campaign workspace itself. Publishing
the reviewed sources to `main` intentionally triggers the build, tests and image publication.

## Build and check

The workflow runs on pushes to `main` that change `t2/**`, `t3/**`, `deployment/**` or
`.github/workflows/cpu-build.yml`. Each passing image is then published. A manual test run
is also available under **Actions → CPU container build and public smoke → Run workflow**;
leave `publish` false to test without publishing. A job-level guard skips every job before
runner allocation if the repository is private. A skipped job is not a successful container test.

The workflow uses a standard `ubuntu-24.04` CPU runner, one track at a time. It builds with
the explicit `t2/` or `t3/` Docker context and inspects the resulting Linux/amd64 image for
the required interface label and the absence of Docker `VOLUME` declarations. Public
validator repositories are fetched at pinned commits outside the image. Only input files
are mounted into candidate containers; reference outputs remain on the validator host.

Containers run without networking or GPU access, with a read-only root filesystem,
4 CPUs, 8 GiB memory, a 64 MiB temporary filesystem and a 300-second check timeout.
T2 runs the actual `forecast` command on both the official daily Treasury-curve exemplar
and a monthly macro unit, then the official gates-only scorer without fabricated future
labels. Both original input cards and manifests are validated before staging. T3 runs `simulate` twice on
the public STP scenario, checks stable trace and ledger hashes, and runs `simulate-batch`
on the public three-scenario batch followed by official developer gates. These checks
cover a few public examples; they establish neither private-test performance nor rank.

Before the T2 container starts, the unchanged shared toolkit validates `card.toml` and
the manifest of the original pinned public unit, including its declared file hashes and
coverage. The reduced input mount is not used for this check. An invalid card, bad hash,
missing file or unsupported integrity check fails the job rather than becoming a skipped
success. T3's official smoke command also performs these common checks.

## Publish the tested images

Review the exported source before pushing it to `main`, or before a manual run with
`publish` true. Each passing image is pushed as
`ghcr.io/OWNER/eastern-eagles-t2:sha-COMMIT` or the T3 equivalent. The built image includes
OCI source and revision labels so GitHub can associate the package with this repository.
Only the automatically issued `GITHUB_TOKEN` is used (`contents: read`, `packages: write`);
there is no personal token, Team Key, paid model API, cloud credential or OIDC setup.
The tested image ID must equal the pushed image ID before its digest is recorded.

**A newly published GHCR package may be private.** After first publication, check each
package's settings and set its visibility to **Public**. The source label links the package
to the repository; it does not establish anonymous access. If anonymous verification failed
because the package was private, use **Re-run failed jobs** after changing visibility.
This reruns the same source commit's build and offline checks before repeating anonymous
verification. Successful publication alone is not a usable competition image.

The final verification obtains an anonymous registry token, hashes the exact manifest,
configuration and every downloaded layer, checks architecture/label/volumes again, and
performs a Docker pull with an empty credential configuration. Only after that passes
does the job summary show the verified immutable image reference. Keep that digest
pullable through official results publication. Use the private, local submission tool
to generate the CodaBench descriptor/proof ZIP; never add that ZIP to this repository.

Standard GitHub-hosted runners are free for public repositories, and GHCR container
storage/bandwidth is currently free. The workflow uploads no Actions artifacts or caches.
Recheck these policies before running if GitHub changes its terms:
[runner policy](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job),
[container billing](https://docs.github.com/en/billing/concepts/product-billing/github-packages),
[package visibility](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

The included third-party license notices must remain with their respective files. T2
installs its official dependencies during the image build; T3 retains the official adapter
license and ABIDES BSD-3 notice. No public practice panels, corpora, scenarios, expected
traces, model weights or precomputed answers are included in the source archive or images.
