# The science and mathematics behind burstkin

This document explains what the code computes and why each method is trusted.
Time is measured in mRNA (or protein) lifetimes throughout: the decay rate is 1,
so every other rate means "events per lifetime".

## 1. The biological question

Most mammalian genes are not transcribed at a steady trickle. The promoter
switches between an inactive state and an active state, and while active it
fires off transcripts in a burst. Two numbers summarise this behaviour:

* **burst frequency**: how often a promoter turns on (bursts per mRNA lifetime);
* **burst size**: how many transcripts one burst produces on average.

Regulation can act on either. Enhancers tend to change burst frequency, while
core-promoter elements tend to change burst size (Larsson et al. 2019). Snapshot
data (one count per cell, many cells) cannot show bursts directly, but the
*shape* of the count distribution carries their signature. Recovering the
kinetics from that shape is an inverse problem with real pitfalls: noise from
molecule capture, parameters the data cannot pin down, and models whose
distributions have no closed form. This package solves the problem end to end
and checks each step against an independent method.

## 2. The telegraph model

```
G_off  --k_on-->  G_on            promoter activation
G_on   --k_off--> G_off           promoter inactivation
G_on   --k_syn--> G_on + M        transcription (only while active)
M      --1----->  0               mRNA decay
```

The mean burst size is `k_syn / k_off`: while active, the promoter makes
transcripts at rate `k_syn` for an exponentially distributed time with mean
`1 / k_off`. The burst frequency is `k_on`.

## 3. The chemical master equation

For a network with stoichiometric change vectors `nu_j` and propensities
`a_j(x)`, the probability `p(x, t)` of copy-number vector `x` obeys

```
dp(x)/dt = sum_j [ a_j(x - nu_j) p(x - nu_j) - a_j(x) p(x) ].
```

Propensities follow stochastic mass action: `a_j(x) = c_j * prod_i C(x_i, r_ij)`,
where `C` is the binomial coefficient, i.e. the number of distinct combinations
of reactant molecules (`network.py`). For `P + P -> 0` this gives
`c x (x - 1) / 2`, not `c x^2`. The difference matters at low copy numbers, and
the tests check it explicitly.

The master equation is linear but infinite-dimensional. The package solves it in
three independent ways.

### 3a. Exact simulation (Gillespie's direct method, `ssa.py`)

Waiting times between events are exponential with rate `a_0 = sum_j a_j(x)`, and
the next reaction is `j` with probability `a_j / a_0`. Every cell is an
independent realisation; the implementation advances all unfinished cells by one
event per loop iteration, so the numpy work is shared across cells. Simulation
is exact, but its histograms carry sampling noise. The validation therefore uses
chi-square goodness-of-fit tests rather than tolerances.

### 3b. Finite state projection (`fsp.py`)

Keep only the states reachable from the initial state inside a box of copy-number
bounds (Munsky & Khammash 2006). The reachable set is found by breadth-first
search over transitions with positive propensity. Species covered by a
conservation law need no explicit bound, because a small linear program finds one:
minimise `l . x0` over non-negative `l` with `l^T S = 0` and `l_i = 1`.

The truncated generator `A` is linear in the rate constants, so it is assembled
as `A = sum_j c_j A_j` from per-reaction patterns computed once. That makes
repeated evaluation inside an optimiser cheap.

There are two boundary treatments:

* **Sink.** Probability that would leave the box is discarded. The truncated
  solution is then a lower bound on the true probabilities state by state, and
  the lost mass `epsilon = 1 - sum p` is exactly the L1 error. This is the FSP
  theorem, and `tests/test_fsp.py` checks it against the exact birth-death
  solution.
* **Reflect.** Transitions out of the box are dropped. The generator stays
  conservative, so a stationary distribution exists. Its quality is reported as
  the probability on states adjacent to the boundary.

**Stationary solve.** `A p = 0` has a one-dimensional null space for an
irreducible chain. The balance equation of one reference state is redundant;
dropping it and fixing that state's unnormalised probability to one gives a
non-singular sparse system. If the reference state is astronomically improbable
(for example `P = 0` when the mean is 300), the anchored solution spans hundreds
of orders of magnitude. The solver detects this, solves the scale-free variant
(normalisation in place of one balance equation) to locate the bulk, and
re-anchors on the most probable state. This was a real bug, found by the
system-size test, and it now has a regression test.

**Transient solve.** There are three methods, all accurate to about 1e-14:

* *Uniformization* (Jensen's method). With `Lambda >= max_i |A_ii|`, the matrix
  `M = I + A / Lambda` is non-negative and
  `exp(A t) = sum_k Poisson(k; Lambda t) M^k`. All terms are positive, so there is
  no cancellation. One sequence of matrix-vector products serves every output
  time, and the Poisson weights are applied in one matrix product. Cost grows
  with `Lambda * t`.
* *Dense scaling-and-squaring* matrix exponential. Cost grows with
  `n_states^3`, but hardly with the rates, so it wins for small, stiff systems.
* *scipy's `expm_multiply`* (Al-Mohy & Higham 2011), kept as a third opinion.

`method="auto"` picks between the first two from an operation count.

### 3c. The closed form (`analytic.py`)

At steady state the promoter spends a random fraction of recent time active, and
conditional on that history the mRNA count is Poisson. Peccoud & Ycart (1995)
showed the stationary law is a Poisson distribution with mean `k_syn * p`, mixed
over `p ~ Beta(k_on, k_off)`:

```
P(n) = integral_0^1 Poisson(n; k_syn p) Beta(p; k_on, k_off) dp
     = k_syn^n / n! * Gamma(k_on + n) Gamma(k_on + k_off) / (Gamma(k_on) Gamma(k_on + k_off + n))
       * 1F1(k_on + n; k_on + k_off + n; -k_syn).
```

The confluent hypergeometric function at a negative argument is an alternating
series whose terms grow like `k_syn^k / k!` before cancelling. For
`k_syn = 1000` that wipes out all precision. Kummer's transformation
`1F1(a; b; -z) = e^{-z} 1F1(b - a; b; z)` turns it into a series of positive terms
`t_k = (a)_k / (b)_k * z^k / k!` with `a = k_off`, `b = k_on + k_off + n`,
`z = k_syn`, which is summed in log space.

**When to stop summing.** The term ratio
`r_k = (a + k) z / ((b + k)(k + 1))` is unimodal in `k`. Its derivative's sign
is that of `-k^2 - 2ak + (b - a - ab)`, a downward parabola with its vertex at
`k = -a < 0`. Once the ratio is falling and below one, every later ratio is
smaller, so the tail is bounded by a geometric series
`t_K r_K / (1 - r_K)`. Summation stops when that bound is below 1e-16 of the
partial sum. The same argument bounds the terms skipped *below* a window
starting at `k_lo`: every ratio there is at least `min(r_0, r_{k_lo - 1})`.
The largest term sits at the positive root of
`k^2 + (b + 1 - z) k + (b - a z) = 0`, which sets the window.

**Speed.** In a likelihood every cell shares `k_on` and `k_off`; only `n_i` and
`z_i = k_syn * s_i` differ. The log of each term splits into a part shared by
all cells, a table lookup indexed by `n_i + k`, and `k log z_i`. The per-cell
work becomes a gather, an exponential and a matrix product instead of a log per
term (`telegraph_loglik_terms`). Derivatives come from the same sums:
`d log t_k / da = psi(a + k) - psi(a)`, `d log t_k / db = -(psi(b + k) - psi(b))`,
`d log t_k / dz = k / z`.

**Precision.** The cancellation in `-k_syn + log 1F1(...; k_syn)` limits absolute
accuracy in log-probability to about `10 k_syn` machine epsilons, about 4e-12 at
`k_syn = 2000`. Against 40-digit mpmath the series agrees to about 2e-15 over six
decades of every argument.

### 3d. Moments and their inversion

The factorial moments are `E[n (n-1) ... (n-k+1)] = k_syn^k E[p^k]`, and
`E[p^k] = prod_{j<k} (k_on + j) / (k_on + k_off + j)`. With `b_k` the ratio of
consecutive factorial moments, `b_k = k_syn (k_on + k - 1) / (k_on + k_off + k - 1)`.
Three such equations invert in closed form:

```
k_on + k_off = 2 (b_2 - b_3) / (b_1 - 2 b_2 + b_3)
k_syn        = (k_on + k_off)(b_2 - b_1) + b_2
k_on         = b_1 (k_on + k_off) / k_syn
```

This is a cheap, noisy estimator (third moments are heavy-tailed), used only as
a starting point for maximum likelihood.

### 3e. Limits

* **Bursty limit** (`k_off -> infinity` at fixed burst size `b`). `Beta(k_on, k_off)`
  tends to `Gamma(k_on, 1 / k_off)`, so the count is Poisson-Gamma, i.e. negative
  binomial with size `k_on` and mean `k_on b` (Shahrezaei & Swain 2008). The error
  falls as `1 / k_off` (tested).
* **Constitutive limit** (promoter almost always on). The count tends to
  Poisson(`k_syn`).

## 4. Molecule capture in single-cell sequencing

Each transcript in cell `i` is captured independently with probability `s_i`
(typically 5-30%, and varying between cells). A Poisson(`lambda`) count thinned
binomially is Poisson(`s lambda`), so the observed count is again Poisson-Beta
with `k_syn` replaced by `k_syn s_i`. The likelihood stays exact; this is checked
by thinning the FSP distribution with explicit binomial sums.

A consequence: multiplying every `s_i` by `c` and dividing `k_syn` by `c` leaves
the likelihood unchanged. Count data alone cannot fix the absolute capture
efficiency, so absolute burst sizes inherit whatever mean efficiency is assumed.
Burst frequency and `k_off` do not. The validation checks that fitting with
halved efficiencies returns exactly doubled `k_syn` and unchanged switching
rates. In practice relative efficiencies come from library sizes
(`scdata.estimate_capture`) and the mean is an assumption the user states.

## 5. The linear noise approximation (`lna.py`)

Write copy numbers as the deterministic solution plus Gaussian fluctuations
(van Kampen's system-size expansion to first order). The mean follows the rate
equations `d phi / dt = S a(phi)`, and the covariance follows

```
d Sigma / dt = J Sigma + Sigma J^T + S diag(a(phi)) S^T,
```

where `J` is the Jacobian of the drift. At steady state this is a Lyapunov
equation.

**Conservation laws.** A fixed number of gene copies makes `J` singular, so
the Lyapunov equation has no unique solution in the full space. Fluctuations can
only move within the column space of `S`, so the equation is solved in an
orthonormal basis `W` of that space, `(W^T J W) Sigma_r + ... = 0`, and
`Sigma = W Sigma_r W^T`. This is exact, not a regularisation. The fixed point
itself is found by integrating the rate equations and then running Newton's
method on the drift projected onto `W`, plus the conservation laws.

**When it is exact, and when it is not.** For networks of zeroth- and
first-order reactions the moment equations close, and the LNA mean and covariance
are exact, at every time. The telegraph model is such a network, and the tests
confirm agreement with the master equation to 1e-12. For `P + P -> 0` with
system size `Omega`, the LNA predicts mean `Omega` and Fano factor 3/4. The FSP
shows the exact mean is `Omega + 1/8 + O(1/Omega)` and the variance
`3 Omega / 4 + 1/16 + ...`, so the LNA's relative error falls as `1 / Omega`. The
measured log-log slope is -1.003.

## 6. Inference of burst kinetics (`inference.py`)

**Likelihood.** The log-likelihood of one gene's counts is
`sum_i log P(n_i; k_on, k_off, k_syn s_i)`. It is optimised over the logs of the
three rates with L-BFGS-B, using the exact gradient. Starting points come from
the method of moments, a negative-binomial guess and a small grid. Each start
gets a short exploratory run, and the two best are refined to full precision.

**Profile likelihood.** For a quantity `psi` (a rate, or the burst size), the
profile is `PL(psi) = max over the other parameters of L`. By Wilks' theorem the
set `{psi : 2 (log L_max - log PL(psi)) < chi2_1(0.95) = 3.84}` is an approximate
95% confidence interval. The profile is walked outward from the optimum with
growing steps and warm starts, and each crossing is located by root finding. A
coverage study (200 simulated data sets of 500 cells) measures the actual
coverage of these intervals: 0.95 for burst frequency and 0.96 for burst size.

**Identifiability.** If an interval runs into a parameter bound without
crossing the threshold, the data cannot bound that parameter: it is *practically
non-identifiable* (Raue et al. 2009). The telegraph model has a characteristic
case. When active periods are short (`k_off` large), the counts approach a
negative binomial that depends only on `k_on` and `k_syn / k_off`. Burst
frequency and burst size remain identifiable, but `k_off` and `k_syn` separately
do not. Genes are classified as `full`, `burst` (only frequency and size) or
`none`.

**A subtlety the code handles.** With box constraints, the profile of `k_off`
can close artificially. As `k_off` grows at fixed burst size, `k_syn` must grow
too, and once `k_syn` hits its upper bound the likelihood falls for reasons that
have nothing to do with the data. In one test case the `k_off` interval "closed"
at 571 for exactly this reason. The pipeline therefore profiles both `k_off` and
`k_syn`: in the bursty regime at least one of the two always runs open.

**Model comparison.** Poisson (constitutive), negative binomial (bursty limit)
and telegraph models are compared by BIC. The models are nested at parameter
boundaries (`1/r -> 0`, `1/k_off -> 0`), so likelihood-ratio tests use the
50:50 mixture of chi-square(0) and chi-square(1) as the null (Self & Liang 1987).
For the telegraph-vs-NB test this is an approximation, and it is labelled as one.

## 7. Models without a closed form: negative autoregulation (`fsp_inference.py`)

A gene whose protein represses its own promoter:

```
G   --rho_u-->  G + P        free promoter, full output
G_b --rho_b-->  G_b + P      bound promoter, leaky output
G + P --sigma_b--> G_b       repression (bimolecular: nonlinear)
G_b --sigma_u--> G + P       unbinding
P   --1-->      0
```

The binding step makes the propensities nonlinear, so neither the closed form
nor the LNA is exact. Time-course snapshot data (protein counts in independent
cells at several times after induction, as from smFISH or flow cytometry) have
likelihood `prod_t prod_i p_t(n_i)`, with `p_t` from the FSP transient solver. With
the sink boundary the computed likelihood is a rigorous lower bound on the exact
one, and the gap is at most the reported truncation error. Standard errors come
from the observed Fisher information, a finite-difference Hessian of the
negative log-likelihood in log-parameters.

With 1,000 cells at each of six times, all four rates are recovered with the
truth inside each 95% interval. The leak rate `rho_b` is the hardest. Its
interval spans a factor of 2.3, against 1.07 for `rho_u`, because low-level
leaky expression barely changes protein snapshots. The interval widens quickly
as cells per time point are reduced. That is a statement about the experiment
design, not a numerical failure.

## 8. What is validated, and how

`burstkin validate` runs 15 quick cross-checks (about 15 s) and, with `--full`,
the coverage study and the feedback-model recovery (about 8 minutes). Every method is compared to
an *independent* one: series to 40-digit arithmetic and to quadrature, closed
form to master equation, master equation to exact transient solutions,
simulation to both by chi-square tests, the LNA to exact moments and to its
predicted error order, and the inference to known truth.

## 9. Limitations

* One gene at a time: genes are assumed independent, and extrinsic noise shared
  across genes (cell size, cell cycle) is absorbed only through capture
  efficiency.
* The telegraph model has two promoter states. Real promoters can have more,
  which typically shows up as a poor fit rather than a wrong estimate.
* Steady state is assumed for single-cell snapshots of the telegraph model.
  Time-course inference (section 7) relaxes this, at higher cost.
* Absolute burst sizes depend on the assumed mean capture efficiency (section 4).

## References

* Al-Mohy AH, Higham NJ (2011). Computing the action of the matrix exponential. *SIAM J Sci Comput* 33:488-511.
* Gillespie DT (1977). Exact stochastic simulation of coupled chemical reactions. *J Phys Chem* 81:2340-2361.
* Kim JK, Marioni JC (2013). Inferring the kinetics of stochastic gene expression from single-cell RNA-sequencing data. *Genome Biol* 14:R7.
* Larsson AJM et al. (2019). Genomic encoding of transcriptional burst kinetics. *Nature* 565:251-254.
* Munsky B, Khammash M (2006). The finite state projection algorithm for the solution of the chemical master equation. *J Chem Phys* 124:044104.
* Munsky B, Trinh B, Khammash M (2009). Listening to the noise: random fluctuations reveal gene network parameters. *Mol Syst Biol* 5:318.
* Peccoud J, Ycart B (1995). Markovian modeling of gene-product synthesis. *Theor Popul Biol* 48:222-234.
* Raj A, Peskin CS, Tranchina D, Vargas DY, Tyagi S (2006). Stochastic mRNA synthesis in mammalian cells. *PLoS Biol* 4:e309.
* Raue A et al. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 25:1923-1929.
* Self SG, Liang KY (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *J Am Stat Assoc* 82:605-610.
* Shahrezaei V, Swain PS (2008). Analytical distributions for stochastic gene expression. *PNAS* 105:17256-17261.
* van Kampen NG (2007). *Stochastic Processes in Physics and Chemistry*, 3rd ed. Elsevier.
