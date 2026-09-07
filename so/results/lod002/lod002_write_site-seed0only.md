# LOD-001 — the detection limit of an accessibility audit

Seeds [0], 16 pods per seed, delta = 0.3, modes ['alias', 'direct']. Worst seed reported in every summary row; a limit is 'worst' when it is the LEAST sensitive, and an undetected seed dominates.

Pre-registration: `docs/novelty/lod001-preregister.md`. Every bar in it was written before this file existed.


## Validity rows (worst seed)

| row | observed | bar | |
|---|---|---|---|
| `V1 alias/answer/a1` | 0.9643 | >= 0.8 | PASS |
| `V2 alias/final/raw a1-never` | 0.8973 | >= 0.3 | PASS |
| `V3a alias/answer a0-never` | 0.0000 | <= 0.05 | PASS |
| `V3b |alias/final/raw a0-never|` | 0.0089 | <= 0.1 | PASS |
| `V4a gate at chord 0` | 0.9985 | >= 0.9 | PASS |
| `V4b gate at chord 1.0` | 0.0154 | <= 0.1 | PASS |

## Detection limits — alias mode, worst seed

A cell is the smallest rung whose separation from the never-written control reaches 0.3. `> 1` means no rung cleared it: the audit could not have seen the memory with all of it present.

| site | jspace | random | pca | unembed | raw | answer |
|---|---|---|---|---|---|---|
| `site5` | **> 1** (max -0.013) | **> 1** (max -0.022) | **> 1** (max +0.004) | **> 1** (max -0.009) | **> 1** (max -0.036) | 0.375 |
| `site6` | **> 1** (max -0.027) | **> 1** (max -0.018) | **> 1** (max -0.027) | **> 1** (max -0.022) | **> 1** (max -0.018) | 0.375 |
| `site7` | 0.375 | 0.5 | 0.25 | 0.25 | 0.375 | 0.375 |
| `site8` | 0.375 | 0.5 | 0.25 | 0.25 | 0.375 | 0.375 |
| `site9` | 0.375 | 0.5 | 0.25 | 0.375 | 0.375 | 0.375 |
| `site10` | 0.375 | 0.5 | 0.25 | 0.375 | 0.25 | 0.375 |
| `final` | 0.375 | 0.75 | 0.75 | 0.375 | 0.25 | 0.375 |

Store-side ladder (marker chord; a SMALLER chord is a LARGER residue, so the limit is the LARGEST chord still detected and `< 0` means none was):

| site | jspace | random | pca | unembed | raw | answer |
|---|---|---|---|---|---|---|
| `site5` | **> 0** (max -0.013) | **> 0** (max -0.022) | **> 0** (max +0.004) | **> 0** (max +0.000) | **> 0** (max -0.036) | 0.7 |
| `site6` | **> 0** (max -0.018) | **> 0** (max -0.018) | **> 0** (max -0.022) | **> 0** (max -0.013) | **> 0** (max -0.018) | 0.7 |
| `site7` | 0.825 | 0.8 | 0.75 | 0.8 | 0.8 | 0.7 |
| `site8` | 0.825 | 0.8 | 0.75 | 0.8 | 0.8 | 0.7 |
| `site9` | 0.825 | 0.8 | 0.8 | 0.8 | 0.8 | 0.7 |
| `site10` | 0.825 | 0.825 | 0.8 | 0.825 | 0.8 | 0.7 |
| `final` | 0.75 | 0.8 | 0.7 | 0.75 | 0.8 | 0.7 |

### The curves — alias, `jspace` and the answer, per seed

| seed | site | a=0 | a=0.03125 | a=0.0625 | a=0.125 | a=0.1875 | a=0.25 | a=0.375 | a=0.5 | a=0.75 | a=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `site5` | -0.152 | -0.147 | -0.147 | -0.143 | -0.134 | -0.116 | -0.098 | -0.085 | -0.031 | -0.013 |
| 0 | `site6` | -0.522 | -0.513 | -0.513 | -0.496 | -0.496 | -0.464 | -0.397 | -0.290 | -0.165 | -0.027 |
| 0 | `site7` | +0.018 | +0.022 | +0.022 | +0.031 | +0.076 | +0.134 | +0.402 | +0.701 | +0.906 | +0.920 |
| 0 | `site8` | +0.022 | +0.022 | +0.022 | +0.018 | +0.045 | +0.071 | +0.339 | +0.683 | +0.902 | +0.920 |
| 0 | `site9` | +0.022 | +0.013 | +0.013 | +0.018 | +0.049 | +0.152 | +0.464 | +0.768 | +0.893 | +0.915 |
| 0 | `site10` | +0.022 | +0.022 | +0.027 | +0.027 | +0.089 | +0.170 | +0.536 | +0.830 | +0.902 | +0.920 |
| 0 | `final` | +0.013 | +0.022 | +0.022 | +0.009 | +0.022 | +0.067 | +0.335 | +0.661 | +0.821 | +0.821 |
| 0 | answer | +0.000 | +0.000 | +0.000 | +0.000 | +0.004 | +0.036 | +0.366 | +0.737 | +0.942 | +0.964 |

## Detection limits — direct mode, worst seed

A cell is the smallest rung whose separation from the never-written control reaches 0.3. `> 1` means no rung cleared it: the audit could not have seen the memory with all of it present.

| site | jspace | random | pca | unembed | raw | answer |
|---|---|---|---|---|---|---|
| `site5` | **> 1** (max +0.018) | **> 1** (max -0.009) | **> 1** (max +0.009) | **> 1** (max -0.009) | **> 1** (max -0.018) | 0.375 |
| `site6` | **> 1** (max -0.018) | **> 1** (max -0.018) | **> 1** (max +0.000) | **> 1** (max -0.027) | **> 1** (max +0.000) | 0.375 |
| `site7` | 0.375 | 0.375 | 0.25 | 0.375 | 0.25 | 0.375 |
| `site8` | 0.5 | 0.5 | 0.25 | 0.375 | 0.25 | 0.375 |
| `site9` | 0.375 | 0.5 | 0.375 | 0.375 | 0.25 | 0.375 |
| `site10` | 0.375 | 0.5 | 0.375 | 0.375 | 0.25 | 0.375 |
| `final` | 0.375 | 0.5 | 0.5 | 0.375 | 0.25 | 0.375 |

Store-side ladder (marker chord; a SMALLER chord is a LARGER residue, so the limit is the LARGEST chord still detected and `< 0` means none was):

| site | jspace | random | pca | unembed | raw | answer |
|---|---|---|---|---|---|---|
| `site5` | **> 0** (max +0.018) | **> 0** (max +0.000) | **> 0** (max +0.000) | **> 0** (max +0.009) | **> 0** (max +0.009) | 0.75 |
| `site6` | **> 0** (max +0.000) | **> 0** (max -0.018) | **> 0** (max +0.009) | **> 0** (max +0.000) | **> 0** (max +0.000) | 0.75 |
| `site7` | 0.825 | 0.8 | 0.825 | 0.825 | 0.8 | 0.75 |
| `site8` | 0.825 | 0.8 | 0.825 | 0.825 | 0.825 | 0.75 |
| `site9` | 0.825 | 0.8 | 0.825 | 0.825 | 0.825 | 0.75 |
| `site10` | 0.85 | 0.825 | 0.825 | 0.85 | 0.825 | 0.75 |
| `final` | 0.8 | 0.8 | 0.75 | 0.8 | 0.825 | 0.75 |

### The curves — direct, `jspace` and the answer, per seed

| seed | site | a=0 | a=0.03125 | a=0.0625 | a=0.125 | a=0.1875 | a=0.25 | a=0.375 | a=0.5 | a=0.75 | a=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `site5` | -0.277 | -0.277 | -0.277 | -0.277 | -0.277 | -0.277 | -0.277 | -0.259 | -0.143 | +0.018 |
| 0 | `site6` | -0.777 | -0.768 | -0.768 | -0.741 | -0.714 | -0.670 | -0.571 | -0.438 | -0.143 | -0.018 |
| 0 | `site7` | +0.000 | +0.000 | +0.000 | +0.000 | +0.018 | +0.071 | +0.348 | +0.786 | +0.902 | +0.902 |
| 0 | `site8` | -0.009 | -0.018 | -0.018 | -0.018 | -0.018 | +0.000 | +0.188 | +0.696 | +0.893 | +0.893 |
| 0 | `site9` | -0.018 | -0.018 | -0.018 | -0.018 | +0.009 | +0.018 | +0.312 | +0.804 | +0.875 | +0.875 |
| 0 | `site10` | +0.027 | +0.027 | +0.027 | +0.027 | +0.054 | +0.134 | +0.500 | +0.875 | +0.920 | +0.920 |
| 0 | `final` | +0.009 | +0.009 | +0.009 | +0.009 | +0.009 | +0.018 | +0.357 | +0.688 | +0.795 | +0.786 |
| 0 | answer | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.027 | +0.321 | +0.732 | +0.938 | +0.946 |

## Saturation: the probe's absolute accuracy, not the difference (worst seed)

| mode | site | family | never | a=0 | a=1 | chance |
|---|---|---|---|---|---|---|
| alias | `site5` | `jspace` | 0.286 | 0.134 | 0.272 | 0.062 |
| alias | `site5` | `random` | 0.268 | 0.125 | 0.246 | 0.062 |
| alias | `site5` | `pca` | 0.272 | 0.080 | 0.277 | 0.062 |
| alias | `site5` | `unembed` | 0.277 | 0.152 | 0.268 | 0.062 |
| alias | `site5` | `raw` | 0.696 | 0.223 | 0.661 | 0.062 |
| alias | `site6` | `jspace` | 0.696 | 0.174 | 0.670 | 0.062 |
| alias | `site6` | `random` | 0.451 | 0.098 | 0.433 | 0.062 |
| alias | `site6` | `pca` | 0.567 | 0.223 | 0.540 | 0.062 |
| alias | `site6` | `unembed` | 0.527 | 0.107 | 0.504 | 0.062 |
| alias | `site6` | `raw` | 0.946 | 0.223 | 0.929 | 0.062 |
| alias | `site7` | `jspace` | 0.049 | 0.067 | 0.969 | 0.062 |
| alias | `site7` | `random` | 0.103 | 0.076 | 0.978 | 0.062 |
| alias | `site7` | `pca` | 0.116 | 0.103 | 0.964 | 0.062 |
| alias | `site7` | `unembed` | 0.049 | 0.076 | 0.969 | 0.062 |
| alias | `site7` | `raw` | 0.103 | 0.080 | 0.969 | 0.062 |
| alias | `site8` | `jspace` | 0.049 | 0.071 | 0.969 | 0.062 |
| alias | `site8` | `random` | 0.103 | 0.062 | 0.969 | 0.062 |
| alias | `site8` | `pca` | 0.121 | 0.103 | 0.969 | 0.062 |
| alias | `site8` | `unembed` | 0.040 | 0.062 | 0.969 | 0.062 |
| alias | `site8` | `raw` | 0.107 | 0.076 | 0.969 | 0.062 |
| alias | `site9` | `jspace` | 0.054 | 0.076 | 0.969 | 0.062 |
| alias | `site9` | `random` | 0.089 | 0.062 | 0.964 | 0.062 |
| alias | `site9` | `pca` | 0.094 | 0.121 | 0.969 | 0.062 |
| alias | `site9` | `unembed` | 0.049 | 0.071 | 0.969 | 0.062 |
| alias | `site9` | `raw` | 0.094 | 0.089 | 0.969 | 0.062 |
| alias | `site10` | `jspace` | 0.049 | 0.071 | 0.969 | 0.062 |
| alias | `site10` | `random` | 0.071 | 0.058 | 0.969 | 0.062 |
| alias | `site10` | `pca` | 0.080 | 0.080 | 0.969 | 0.062 |
| alias | `site10` | `unembed` | 0.049 | 0.071 | 0.969 | 0.062 |
| alias | `site10` | `raw` | 0.080 | 0.080 | 0.969 | 0.062 |
| alias | `final` | `jspace` | 0.058 | 0.071 | 0.879 | 0.062 |
| alias | `final` | `random` | 0.080 | 0.058 | 0.911 | 0.062 |
| alias | `final` | `pca` | 0.112 | 0.058 | 0.897 | 0.062 |
| alias | `final` | `unembed` | 0.058 | 0.071 | 0.879 | 0.062 |
| alias | `final` | `raw` | 0.071 | 0.080 | 0.969 | 0.062 |
| direct | `site5` | `jspace` | 0.420 | 0.143 | 0.438 | 0.062 |
| direct | `site5` | `random` | 0.357 | 0.196 | 0.348 | 0.062 |
| direct | `site5` | `pca` | 0.411 | 0.268 | 0.402 | 0.062 |
| direct | `site5` | `unembed` | 0.420 | 0.125 | 0.411 | 0.062 |
| direct | `site5` | `raw` | 0.750 | 0.143 | 0.732 | 0.062 |
| direct | `site6` | `jspace` | 0.893 | 0.116 | 0.875 | 0.062 |
| direct | `site6` | `random` | 0.741 | 0.152 | 0.723 | 0.062 |
| direct | `site6` | `pca` | 0.884 | 0.366 | 0.884 | 0.062 |
| direct | `site6` | `unembed` | 0.812 | 0.223 | 0.786 | 0.062 |
| direct | `site6` | `raw` | 1.000 | 0.393 | 1.000 | 0.062 |
| direct | `site7` | `jspace` | 0.062 | 0.062 | 0.964 | 0.062 |
| direct | `site7` | `random` | 0.125 | 0.071 | 0.946 | 0.062 |
| direct | `site7` | `pca` | 0.152 | 0.170 | 0.964 | 0.062 |
| direct | `site7` | `unembed` | 0.045 | 0.116 | 0.973 | 0.062 |
| direct | `site7` | `raw` | 0.214 | 0.089 | 0.964 | 0.062 |
| direct | `site8` | `jspace` | 0.080 | 0.071 | 0.973 | 0.062 |
| direct | `site8` | `random` | 0.152 | 0.071 | 0.955 | 0.062 |
| direct | `site8` | `pca` | 0.161 | 0.196 | 0.964 | 0.062 |
| direct | `site8` | `unembed` | 0.054 | 0.071 | 0.973 | 0.062 |
| direct | `site8` | `raw` | 0.196 | 0.143 | 0.964 | 0.062 |
| direct | `site9` | `jspace` | 0.098 | 0.080 | 0.973 | 0.062 |
| direct | `site9` | `random` | 0.143 | 0.071 | 0.955 | 0.062 |
| direct | `site9` | `pca` | 0.152 | 0.170 | 0.964 | 0.062 |
| direct | `site9` | `unembed` | 0.062 | 0.080 | 0.973 | 0.062 |
| direct | `site9` | `raw` | 0.152 | 0.152 | 0.964 | 0.062 |
| direct | `site10` | `jspace` | 0.054 | 0.080 | 0.973 | 0.062 |
| direct | `site10` | `random` | 0.107 | 0.080 | 0.955 | 0.062 |
| direct | `site10` | `pca` | 0.098 | 0.134 | 0.973 | 0.062 |
| direct | `site10` | `unembed` | 0.054 | 0.080 | 0.973 | 0.062 |
| direct | `site10` | `raw` | 0.134 | 0.143 | 0.964 | 0.062 |
| direct | `final` | `jspace` | 0.098 | 0.107 | 0.884 | 0.062 |
| direct | `final` | `random` | 0.080 | 0.116 | 0.875 | 0.062 |
| direct | `final` | `pca` | 0.080 | 0.045 | 0.911 | 0.062 |
| direct | `final` | `unembed` | 0.098 | 0.107 | 0.884 | 0.062 |
| direct | `final` | `raw` | 0.152 | 0.134 | 0.964 | 0.062 |

## Mediators (worst seed), the rows WSC-001 reports beside every probe number

| mode | site | shred_moves | never_moves |
|---|---|---|---|
| alias | `site5` | 3.745 | 6.714 |
| alias | `site6` | 4.213 | 15.799 |
| alias | `site7` | 27.945 | 33.688 |
| alias | `site8` | 28.916 | 34.614 |
| alias | `site9` | 33.204 | 38.194 |
| alias | `site10` | 52.935 | 45.771 |
| alias | `final` | 90.025 | 47.127 |
| alias | `write5` | 3.745 | - |
| alias | `write7` | 27.496 | - |
| direct | `site5` | 1.501 | 2.077 |
| direct | `site6` | 2.099 | 2.672 |
| direct | `site7` | 26.430 | 22.723 |
| direct | `site8` | 27.486 | 22.291 |
| direct | `site9` | 31.423 | 27.092 |
| direct | `site10` | 52.671 | 32.176 |
| direct | `final` | 82.311 | 24.131 |
| direct | `write5` | 1.501 | - |
| direct | `write7` | 26.195 | - |

## The store-side ladder, in the quantity the payload sees

| requested chord | achieved chord | gate |
|---|---|---|
| 0 | 0.193 | 0.9985 |
| 0.5 | 0.524 | 0.9698 |
| 0.7 | 0.709 | 0.7165 |
| 0.75 | 0.748 | 0.5895 |
| 0.8 | 0.802 | 0.3831 |
| 0.825 | 0.841 | 0.2485 |
| 0.85 | 0.880 | 0.1399 |
| 0.9 | 0.906 | 0.0921 |
| 1 | 1.004 | 0.0154 |

Run in 2339s.
