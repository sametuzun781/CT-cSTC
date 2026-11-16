# Third-party source and notices

The root MIT License applies to ICORE-owned portions only. The components below
remain separately licensed. Preserve their original license texts and notices
when distributing them with ICORE; the root license does not replace them.

ICORE includes reviewed source copies so the native builds do not need to download
solver dependencies. The archives are unchanged; only their outer filenames are
made readable. Exact version identifiers and automatic integrity checks are internal
to the build helpers, not user settings.

| Component | Purpose | License | Retained license / notice |
|---|---|---|---|
| BLASFEO | Packed small/medium matrix kernels | BSD-2-Clause | `icore/vendor/blasfeo/BLASFEO_LICENSE.txt` |
| HPIPM | Comparison solver | BSD-2-Clause | `icore/vendor/hpipm/LICENSE.txt` |
| QOCO | Comparison solver | BSD-3-Clause | `icore/vendor/qoco/LICENSE` |
| QDLDL in the QOCO archive | Sparse factorization | Apache-2.0 | `icore/vendor/qoco/QDLDL_LICENSE` |
| QDLDL in ICORE's chain path | Augmented sparse recovery | Apache-2.0 | `icore/src/chain_sparse/qdldl/LICENSE` |
| AMD / SuiteSparse_config | Sparse ordering support | BSD-3-Clause | `icore/src/chain_sparse/amd/AMD-License.txt`, `SuiteSparse-config-License.txt`, and `LICENSE-NOTICE.txt` |
| Retained QOCO adaptation notices | Chain sparse support provenance | BSD-3-Clause | `icore/src/chain_sparse/QOCO-License.txt` |

`icore/src/chain_sparse/MODIFICATIONS.txt` documents retained local adaptations.
`icore/vendor/blasfeo/patches/` contains the existing reviewed numerical repairs.
The original notices inside the source archives remain intact. This document is a
navigation aid, not a replacement for those licenses. In particular, Apache-2.0
redistribution conditions for QDLDL remain applicable, including its license,
relevant notices, and notices of modified files. BSD component notices and any
applicable non-endorsement conditions also remain in force.
