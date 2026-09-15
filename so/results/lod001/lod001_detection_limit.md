# LOD-001 — the detection limit of an accessibility audit

Seeds [0, 1, 2], 16 pods per seed, delta = 0.3, modes ['alias', 'direct']. Worst seed reported in every summary row; a limit is 'worst' when it is the LEAST sensitive, and an undetected seed dominates.

Pre-registration: `docs/novelty/lod001-preregister.md`. Every bar in it was written before this file existed.


## Validity rows (worst seed)

| row | observed | bar | |
|---|---|---|---|
| `V1 alias/answer/a1` | 0.9688 | >= 0.8 | PASS |
| `V2 alias/final/raw a1-never` | 0.8125 | >= 0.3 | PASS |
| `V3a alias/answer a0-never` | 0.0179 | <= 0.05 | PASS |
| `V3b |alias/final/raw a0-never|` | 0.0625 | <= 0.1 | PASS |
| `V4a gate at chord 0` | 0.9988 | >= 0.9 | PASS |
| `V4b gate at chord 1.0` | 0.0101 | <= 0.1 | PASS |

## Detection limits — alias mode, worst seed

A cell is the smallest rung whose separation from the never-written control reaches 0.3. `> 1` means no rung cleared it: the audit could not have seen the memory with all of it present.

| site | jspace | random | pca | unembed | raw | answer |
|---|---|---|---|---|---|---|
| `site8` | **> 1** (max +0.022) | **> 1** (max +0.009) | **> 1** (max -0.004) | **> 1** (max +0.004) | **> 1** (max +0.004) | 0.375 |
| `site9` | **> 1** (max -0.004) | **> 1** (max +0.004) | **> 1** (max -0.004) | **> 1** (max -0.004) | **> 1** (max -0.004) | 0.375 |
| `site10` | 0.375 | 0.375 | 0.25 | 0.375 | 0.375 | 0.375 |
| `final` | 0.375 | 0.5 | 0.375 | 0.375 | 0.1875 | 0.375 |

Store-side ladder (marker chord; a SMALLER chord is a LARGER residue, so the limit is the LARGEST chord still detected and `< 0` means none was):

| site | jspace | random | pca | unembed | raw | answer |
|---|---|---|---|---|---|---|
| `site8` | **> 0** (max +0.004) | **> 0** (max +0.009) | **> 0** (max -0.004) | **> 0** (max +0.004) | **> 0** (max +0.004) | 0.5 |
| `site9` | **> 0** (max -0.004) | **> 0** (max +0.009) | **> 0** (max +0.000) | **> 0** (max +0.000) | **> 0** (max +0.000) | 0.5 |
| `site10` | 0.75 | 0.7 | 0.7 | 0.75 | 0.75 | 0.5 |
| `final` | 0.75 | 0.7 | 0.7 | 0.75 | 0.75 | 0.5 |

### The curves — alias, `jspace` and the answer, per seed

| seed | site | a=0 | a=0.03125 | a=0.0625 | a=0.125 | a=0.1875 | a=0.25 | a=0.375 | a=0.5 | a=0.75 | a=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `site8` | -0.080 | -0.080 | -0.067 | -0.045 | -0.027 | -0.027 | +0.000 | +0.004 | +0.022 | +0.004 |
| 0 | `site9` | -0.554 | -0.558 | -0.549 | -0.513 | -0.482 | -0.437 | -0.344 | -0.228 | -0.121 | -0.036 |
| 0 | `site10` | -0.013 | -0.013 | -0.004 | +0.018 | +0.045 | +0.094 | +0.379 | +0.728 | +0.906 | +0.902 |
| 0 | `final` | +0.054 | +0.049 | +0.049 | +0.080 | +0.134 | +0.246 | +0.562 | +0.821 | +0.946 | +0.946 |
| 0 | answer | +0.000 | +0.000 | +0.000 | +0.036 | +0.098 | +0.210 | +0.661 | +0.839 | +0.942 | +0.969 |
| 1 | `site8` | -0.089 | -0.085 | -0.080 | -0.076 | -0.054 | -0.054 | -0.054 | -0.036 | -0.009 | -0.004 |
| 1 | `site9` | -0.616 | -0.616 | -0.607 | -0.603 | -0.540 | -0.504 | -0.371 | -0.219 | -0.058 | -0.013 |
| 1 | `site10` | -0.045 | -0.036 | -0.018 | +0.018 | +0.058 | +0.152 | +0.487 | +0.719 | +0.866 | +0.884 |
| 1 | `final` | -0.004 | +0.000 | +0.004 | +0.036 | +0.103 | +0.223 | +0.411 | +0.625 | +0.884 | +0.915 |
| 1 | answer | +0.018 | +0.022 | +0.027 | +0.040 | +0.098 | +0.308 | +0.701 | +0.866 | +0.960 | +0.978 |
| 2 | `site8` | -0.107 | -0.094 | -0.094 | -0.089 | -0.067 | -0.071 | -0.062 | -0.027 | -0.009 | -0.013 |
| 2 | `site9` | -0.656 | -0.656 | -0.656 | -0.621 | -0.571 | -0.509 | -0.344 | -0.196 | -0.058 | -0.004 |
| 2 | `site10` | -0.027 | -0.022 | -0.022 | +0.000 | +0.058 | +0.246 | +0.728 | +0.862 | +0.915 | +0.915 |
| 2 | `final` | +0.013 | +0.018 | +0.022 | +0.058 | +0.125 | +0.312 | +0.701 | +0.884 | +0.929 | +0.929 |
| 2 | answer | +0.000 | +0.000 | +0.000 | +0.022 | +0.094 | +0.317 | +0.728 | +0.884 | +0.978 | +0.982 |

## Detection limits — direct mode, worst seed

A cell is the smallest rung whose separation from the never-written control reaches 0.3. `> 1` means no rung cleared it: the audit could not have seen the memory with all of it present.

| site | jspace | random | pca | unembed | raw | answer |
|---|---|---|---|---|---|---|
| `site8` | **> 1** (max +0.027) | **> 1** (max +0.054) | **> 1** (max +0.009) | **> 1** (max +0.027) | **> 1** (max +0.000) | 0.375 |
| `site9` | **> 1** (max +0.000) | **> 1** (max +0.000) | **> 1** (max +0.009) | **> 1** (max +0.000) | **> 1** (max +0.000) | 0.375 |
| `site10` | 0.375 | 0.375 | 0.375 | 0.375 | 0.375 | 0.375 |
| `final` | 0.375 | 0.5 | 0.375 | 0.375 | 0.1875 | 0.375 |

Store-side ladder (marker chord; a SMALLER chord is a LARGER residue, so the limit is the LARGEST chord still detected and `< 0` means none was):

| site | jspace | random | pca | unembed | raw | answer |
|---|---|---|---|---|---|---|
| `site8` | **> 0** (max +0.009) | **> 0** (max +0.009) | **> 0** (max +0.000) | **> 0** (max +0.000) | **> 0** (max +0.000) | 0.5 |
| `site9` | **> 0** (max +0.000) | **> 0** (max +0.000) | **> 0** (max +0.000) | **> 0** (max +0.000) | **> 0** (max +0.000) | 0.5 |
| `site10` | 0.75 | 0.7 | 0.75 | 0.75 | 0.7 | 0.5 |
| `final` | 0.75 | 0.7 | 0.75 | 0.75 | 0.75 | 0.5 |

### The curves — direct, `jspace` and the answer, per seed

| seed | site | a=0 | a=0.03125 | a=0.0625 | a=0.125 | a=0.1875 | a=0.25 | a=0.375 | a=0.5 | a=0.75 | a=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `site8` | -0.205 | -0.196 | -0.188 | -0.196 | -0.205 | -0.196 | -0.152 | -0.152 | -0.071 | +0.009 |
| 0 | `site9` | -0.741 | -0.741 | -0.723 | -0.679 | -0.652 | -0.580 | -0.411 | -0.241 | -0.018 | +0.000 |
| 0 | `site10` | -0.036 | -0.036 | -0.036 | -0.027 | +0.009 | +0.036 | +0.312 | +0.705 | +0.893 | +0.893 |
| 0 | `final` | -0.009 | -0.009 | +0.009 | +0.036 | +0.089 | +0.205 | +0.625 | +0.848 | +0.920 | +0.929 |
| 0 | answer | +0.000 | +0.000 | +0.000 | +0.000 | +0.027 | +0.125 | +0.536 | +0.848 | +0.964 | +0.973 |
| 1 | `site8` | -0.009 | -0.018 | -0.018 | -0.009 | +0.000 | -0.009 | +0.009 | +0.027 | +0.009 | +0.000 |
| 1 | `site9` | -0.741 | -0.741 | -0.714 | -0.688 | -0.607 | -0.509 | -0.259 | -0.143 | -0.054 | +0.000 |
| 1 | `site10` | -0.098 | -0.098 | -0.089 | -0.080 | -0.045 | +0.036 | +0.446 | +0.714 | +0.812 | +0.821 |
| 1 | `final` | -0.045 | -0.009 | -0.009 | +0.000 | +0.054 | +0.188 | +0.482 | +0.705 | +0.857 | +0.866 |
| 1 | answer | -0.009 | -0.009 | -0.009 | +0.000 | +0.036 | +0.205 | +0.643 | +0.857 | +0.955 | +0.973 |
| 2 | `site8` | -0.152 | -0.152 | -0.152 | -0.152 | -0.134 | -0.080 | -0.054 | -0.062 | -0.027 | +0.000 |
| 2 | `site9` | -0.777 | -0.759 | -0.759 | -0.723 | -0.696 | -0.670 | -0.384 | -0.170 | -0.062 | +0.000 |
| 2 | `site10` | -0.036 | -0.036 | -0.036 | -0.036 | -0.009 | +0.036 | +0.491 | +0.821 | +0.893 | +0.893 |
| 2 | `final` | -0.045 | -0.036 | -0.045 | -0.018 | +0.027 | +0.125 | +0.616 | +0.812 | +0.866 | +0.866 |
| 2 | answer | +0.000 | +0.000 | +0.000 | +0.018 | +0.080 | +0.223 | +0.607 | +0.839 | +0.991 | +0.991 |

## Saturation: the probe's absolute accuracy, not the difference (worst seed)

| mode | site | family | never | a=0 | a=1 | chance |
|---|---|---|---|---|---|---|
| alias | `site8` | `jspace` | 0.214 | 0.134 | 0.219 | 0.062 |
| alias | `site8` | `random` | 0.210 | 0.094 | 0.205 | 0.062 |
| alias | `site8` | `pca` | 0.179 | 0.129 | 0.161 | 0.062 |
| alias | `site8` | `unembed` | 0.241 | 0.156 | 0.241 | 0.062 |
| alias | `site8` | `raw` | 0.536 | 0.152 | 0.540 | 0.062 |
| alias | `site9` | `jspace` | 0.701 | 0.147 | 0.665 | 0.062 |
| alias | `site9` | `random` | 0.612 | 0.098 | 0.598 | 0.062 |
| alias | `site9` | `pca` | 0.812 | 0.165 | 0.804 | 0.062 |
| alias | `site9` | `unembed` | 0.728 | 0.147 | 0.710 | 0.062 |
| alias | `site9` | `raw` | 0.996 | 0.174 | 0.987 | 0.062 |
| alias | `site10` | `jspace` | 0.076 | 0.054 | 0.978 | 0.062 |
| alias | `site10` | `random` | 0.058 | 0.054 | 0.978 | 0.062 |
| alias | `site10` | `pca` | 0.125 | 0.062 | 0.978 | 0.062 |
| alias | `site10` | `unembed` | 0.076 | 0.054 | 0.978 | 0.062 |
| alias | `site10` | `raw` | 0.129 | 0.076 | 0.982 | 0.062 |
| alias | `final` | `jspace` | 0.040 | 0.054 | 0.973 | 0.062 |
| alias | `final` | `random` | 0.045 | 0.067 | 0.969 | 0.062 |
| alias | `final` | `pca` | 0.054 | 0.062 | 0.969 | 0.062 |
| alias | `final` | `unembed` | 0.040 | 0.054 | 0.973 | 0.062 |
| alias | `final` | `raw` | 0.129 | 0.067 | 0.987 | 0.062 |
| direct | `site8` | `jspace` | 0.330 | 0.179 | 0.330 | 0.062 |
| direct | `site8` | `random` | 0.286 | 0.134 | 0.286 | 0.062 |
| direct | `site8` | `pca` | 0.214 | 0.134 | 0.214 | 0.062 |
| direct | `site8` | `unembed` | 0.286 | 0.196 | 0.286 | 0.062 |
| direct | `site8` | `raw` | 0.679 | 0.312 | 0.679 | 0.062 |
| direct | `site9` | `jspace` | 0.893 | 0.152 | 0.893 | 0.062 |
| direct | `site9` | `random` | 0.893 | 0.080 | 0.893 | 0.062 |
| direct | `site9` | `pca` | 0.991 | 0.348 | 0.991 | 0.062 |
| direct | `site9` | `unembed` | 0.911 | 0.161 | 0.911 | 0.062 |
| direct | `site9` | `raw` | 1.000 | 0.393 | 1.000 | 0.062 |
| direct | `site10` | `jspace` | 0.098 | 0.062 | 0.991 | 0.062 |
| direct | `site10` | `random` | 0.116 | 0.062 | 0.991 | 0.062 |
| direct | `site10` | `pca` | 0.232 | 0.080 | 0.991 | 0.062 |
| direct | `site10` | `unembed` | 0.098 | 0.062 | 0.991 | 0.062 |
| direct | `site10` | `raw` | 0.223 | 0.062 | 0.991 | 0.062 |
| direct | `final` | `jspace` | 0.054 | 0.045 | 0.973 | 0.062 |
| direct | `final` | `random` | 0.071 | 0.045 | 0.973 | 0.062 |
| direct | `final` | `pca` | 0.107 | 0.054 | 0.973 | 0.062 |
| direct | `final` | `unembed` | 0.054 | 0.045 | 0.973 | 0.062 |
| direct | `final` | `raw` | 0.232 | 0.161 | 0.991 | 0.062 |

## Mediators (worst seed), the rows WSC-001 reports beside every probe number

| mode | site | shred_moves | never_moves |
|---|---|---|---|
| alias | `site8` | 3.746 | 4.750 |
| alias | `site9` | 4.135 | 7.460 |
| alias | `site10` | 76.517 | 64.497 |
| alias | `final` | 31.280 | 31.638 |
| alias | `write8` | 3.746 | - |
| alias | `write10` | 76.865 | - |
| direct | `site8` | 0.467 | 0.390 |
| direct | `site9` | 1.238 | 0.690 |
| direct | `site10` | 79.142 | 59.972 |
| direct | `final` | 33.590 | 30.599 |
| direct | `write8` | 0.467 | - |
| direct | `write10` | 80.328 | - |

## The store-side ladder, in the quantity the payload sees

| requested chord | achieved chord | gate |
|---|---|---|
| 0 | 0.186 | 0.9988 |
| 0.5 | 0.520 | 0.9714 |
| 0.7 | 0.715 | 0.6762 |
| 0.75 | 0.763 | 0.4721 |
| 0.8 | 0.809 | 0.3026 |
| 0.825 | 0.834 | 0.2178 |
| 0.85 | 0.870 | 0.1242 |
| 0.9 | 0.909 | 0.0676 |
| 1 | 1.007 | 0.0100 |

Run in 369s.
