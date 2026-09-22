# Tests

Function-based unit tests using MATLAB's `matlab.unittest` framework. They
cover the pure, deterministic utility functions of the toolbox (transforms,
kernels, statistics helpers, tensor operations) plus statistical checks for
the stochastic samplers.

## Running

From the repository root:

```matlab
% Run the whole suite
results = runtests("tests");

% Or via the build tool (also runs code-issues check + clean)
buildtool test
```

In CI the suite runs automatically through `buildtool` (see
`.github/workflows/matlab.yml`), since `plan("test") = TestTask` discovers
every `*Test.m` file in this folder.

## Path handling

Each test's `setup` calls `addProjectRootToPath`, which **prepends** the
repository root to the MATLAB path (and restores it on teardown). This makes
the suite runnable from any working directory and ensures the toolbox's own
functions take precedence over any shadowing copies elsewhere on the path —
e.g. MATLAB's built-in `normalize`/`acf`, or a sibling checkout such as
`impala_matlab_snl`. Without this, `runtests('tests')` from outside the repo
resolves those names to the wrong function.

## Coverage map

| Test file              | Function(s) under test                         |
|------------------------|------------------------------------------------|
| `normalizeTest.m`      | `normalize`, `unnormalize`                     |
| `probitTest.m`         | `probit`, `invprobit`                          |
| `cor2covTest.m`        | `cor2cov`                                      |
| `kernelTest.m`         | `ldig_kern`, `ldhc_kern`                        |
| `tranUnifTest.m`       | `tran_unif`                                    |
| `cfBoundsTest.m`       | `cf_bounds`                                    |
| `statsUtilTest.m`      | `iqrange`, `plims`, `acf`, `autocorrelation`   |
| `essTest.m`            | `ESS`, `mcmcgr`                                |
| `tensorproductTest.m`  | `tensorproduct` (+ core/initialize)            |
| `cov3dPcmTest.m`       | `cov_3d_pcm`                                   |
| `cholSampleTest.m`     | `chol_sample`                                  |

Stochastic tests (`essTest`, `cholSampleTest`, parts of `statsUtilTest`) seed
the RNG with `rng(...)` and assert against loose tolerances so they are
deterministic and stable in CI.
