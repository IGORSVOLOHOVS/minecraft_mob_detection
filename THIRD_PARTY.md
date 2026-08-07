# Third-party code in this repository

Most of the directory tree here is not this project's work. It is
[mmdetection](https://github.com/open-mmlab/mmdetection) by OpenMMLab, vendored
so the detector could be trained and run without a separate checkout.

That code is under the **Apache License 2.0**, and section 4 of that licence
requires a copy of it to travel with any redistribution. `LICENSES/Apache-2.0-
OpenMMLab.txt` is that copy, taken verbatim from mmdetection, including its
`Copyright 2018-2023 OpenMMLab. All rights reserved.` line. mmdetection carries
no NOTICE file, so there is none to reproduce.

## What belongs to whom

| path | whose | terms |
| --- | --- | --- |
| `tools/` | OpenMMLab (mmdetection) | Apache-2.0 |
| `demo/` | OpenMMLab (mmdetection) | Apache-2.0 |
| `configs/` | OpenMMLab (mmdetection) | Apache-2.0 |
| `mobtools/` | this project | MIT, see `LICENSE` |
| `tests/` | this project | MIT |
| `scripts/` | this project | MIT |
| `utils.py`, `notebook.ipynb` | this project | MIT |

Fifty-three files under `tools/` and `demo/` still carry their OpenMMLab
copyright header; they were not modified and the headers were not touched.

## Why this file exists

Until 2026-08-07 the only licence in the tree was this project's MIT, at the
root, where it read as a claim over everything below it - including code that is
not ours and is under a licence with different obligations. MIT permits
sublicensing; Apache-2.0 requires attribution and a copy of its own terms. One
file at the root cannot be true of both halves of the tree.

Nothing about the code changed. What changed is that the repository now says
which half is which.
