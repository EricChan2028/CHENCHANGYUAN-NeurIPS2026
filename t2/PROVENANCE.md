## Executive summary (read this first)

The participant implementation is original CPU statistical code. It imports official
task interfaces and scoring utilities without changing them. No pretrained weights,
paid service, external inference API, hard-coded historical outcomes, or data files
are included in the image. Dependencies keep their own licenses.

- Toolkit: `Agenthon-2026/Agenthon2026-public`, tag `v2.6.0`.
- Track source: `Agenthon-2026/track2-forecasting-public`, commit `a2c93e7b66b8c6a6c429fda81d6bc7873b7c5dec`.
- Public panel and corpus licenses: official repository `DATA-LICENSE.md` and per-card manifests.
- Runtime only reads the task's supplied numeric panels and optional dated corpus.
- Experiments import official T2 scoring functions; no scoring formula is copied.
- The optional text uncertainty overlay is off by default because its gain has not been validated.
- Windows runtime and output checks are local evidence. Linux image build, offline/read-only
  container checks and official input integrity gates remain necessary before submission.
