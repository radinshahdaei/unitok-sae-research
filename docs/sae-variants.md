# SAE families and training dynamics

Primary-source review and toolkit implementation, 4 November 2026. These are
candidate measurement instruments for UniTok, not proven improvements to its
features or robustness. No full paper training recipe or benchmark is reproduced.
The same encode/decode/linear-decoder interface supports all eight families.

## What is implemented and what changes

| `SAEConfig.variant` | Mechanism | Comparison it makes possible |
| --- | --- | --- |
| `relu` | Nonnegative ReLU codes plus L1 | Conventional shrinkage/sparsity baseline; L1=0 is an unregularized control |
| `topk` | Largest k preactivations, then ReLU | Per-vector competition, at most k positives |
| `batchtopk` | Training-batch budget over all vectors; calibrated fixed threshold at eval | Variable allocation across patches/images versus fixed per-vector budgets |
| `jumprelu` | Learned positive thresholds, exact hard forward, KDE threshold surrogates and L0 loss | Learned sparsity without penalizing active magnitudes via L1 |
| `gated` | Hard feature gate; separately parameterized magnitude with shared encoder directions | Separate feature selection from amplitude estimation |
| `matryoshka` | BatchTopK codes with reconstruction losses at fixed dictionary prefixes | Nested representations and sensitivity to feature absorption/fragmentation |
| `multitopk` | Loss at k plus weighted reconstruction at a larger k in the same dictionary | Sensitivity to the training sparsity budget and progressive recovery |
| `abstopk` | Select largest absolute preactivations and preserve their signs | Signed directions versus nonnegative feature coordinates |

JumpReLU, Matryoshka and Gated are the first comparisons I would prioritize;
AbsTopK is a useful additional test of the nonnegativity assumption, and Multi-
TopK tests budget dependence without introducing a new decoder architecture.
These priorities are our judgment about experimental distinctions, not a ranking
established in UniTok.

## Versioned sources and implementation audit

**ReLU and Top-K.** [Gao et al., arXiv:2406.04093v1](https://arxiv.org/html/2406.04093v1),
§§2.2–2.4 and Appendix A, give the baseline, Top-K and dead-latent machinery.
Their §5.3.1 supplies the Multi-TopK example `L(k) + L(4k)/8`. We implement those
two budgets, with an explicit dictionary-width cap, and expose k overrides.
Our loss uses elementwise mean squared error; coefficients cannot be copied
unchanged from a mean squared-vector-norm loss. No EMA or geometric-median
initialization is included automatically.

**BatchTopK.** [Bussmann, Leask and Nanda, arXiv:2412.06410v1](https://arxiv.org/html/2412.06410v1),
method/training-inference description. Our existing implementation pools all
input vectors during training and explicitly calibrates a global threshold on a
separate panel. This is not the reference training EMA. A threshold does not
guarantee a fixed inference L0; ties and small positive counts matter.

**JumpReLU.** [Rajamanoharan et al., arXiv:2407.14435v3](https://arxiv.org/html/2407.14435v3),
§3 Eqs. (8)–(12), Appendix I and corrected Appendix J. `activations.py` implements
the rectangular-window threshold pseudo-derivatives, ordinary masked input
derivatives, and log-positive threshold parameters. L0 is computed from the
rectified preactivations as in Eq. (10), so threshold gradients include values on
both sides of the jump; it is not computed from already-zeroed codes. The
reference trainer inspected below instead uses post-jump codes and a target-L0
penalty. We implement the paper's direct L0 coefficient. v1/v2 pseudocode had
bugs corrected in later versions. The implementation uses explicit surrogate gradients;
finite-difference gradcheck at the discontinuity is not an appropriate test.
The paper's normalization, schedules and optimizer settings are not silently
transferred to raw UniTok activations.

**Gated.** [Rajamanoharan et al., arXiv:2404.16014v2](https://arxiv.org/html/2404.16014v2),
§3.2 and Appendix G. The gate uses the existing encoder bias. Magnitudes share
encoder directions, with learned exponential per-feature scales and their own
bias. Sparsity penalizes rectified gate preactivations. An auxiliary gate-based
reconstruction detaches the decoder weight and output-bias contribution, giving
selection a reconstruction signal. Tests verify that auxiliary gradients do not
reach decoder weights or magnitude-only parameters. Resampling and RI-L1 are not
included; this is the original gated objective with our documented MSE scaling.

**Matryoshka BatchTopK.** [Bussmann et al., arXiv:2503.17547v1](https://arxiv.org/html/2503.17547v1),
§§3.1–3.2, Appendix B and §G.2. One globally selected code vector supplies fixed
ordered prefix dictionaries; prefixes are not sorted separately for each image.
We average prefix MSEs, matching the inspected trainer (the illustrative paper
listing sums). `decode_prefix` exposes the corresponding partial reconstruction.
The optional stop-gradient flag detaches each accumulated prefix before the next
group, enabling the stated training-dynamics ablation. No hierarchy or improved
disentanglement is assumed to emerge in UniTok.

There is direct vision precedent: [Pach et al., arXiv:2504.02821v3](https://arxiv.org/html/2504.02821v3),
§§3.1 and 4.1–4.2, study BatchTopK and Matryoshka BatchTopK on visual encoders.
Distinguish this from [Zaigrajew et al., arXiv:2502.20578v1](https://arxiv.org/html/2502.20578v1),
§3.2: that CLIP MSAE trains multiple top-k granularities, including the full
representation. Sharing the word Matryoshka does not make these identical
objectives. Our `matryoshka` is the fixed-prefix BatchTopK formulation above;
`multitopk` is Gao's two-budget loss, not a claimed reproduction of CLIP MSAE.

**AbsTopK.** [Zhu et al., arXiv:2510.00404v1](https://arxiv.org/html/2510.00404v1),
§2.3, defines signed largest-magnitude selection. We implement that per-vector
operator with the shared toolkit centering/bias/decoder conventions. The inspected
official repository also exposes a batch-absolute-k implementation; we do not
label our per-vector budget as that batch recipe. Its reported results are in
LLMs, not evidence for UniTok. Metrics count all nonzeros; signed editing is
explicit, the attached SAE independently enforces its sign contract, and maps can
show positive/negative coefficients separately.

The source inspection also checked `dictionary_learning` at revision
`60ec6bf5264944d64a4ca271f45a29ebfb9d4946`, files
`trainers/{jumprelu,gated_anneal,matryoshka_batch_top_k,p_anneal,top_k}.py`, and the
AbsTopK repository at `7d0658a20f4be8687ad049ddaba139a10e80ed8e`. These were read as
reference evidence; the toolkit does not import their trainers or vendor their
source. These reference implementations are not runtime dependencies.

## Training dynamics are separate controls

`SparsitySchedule` is an explicit callable for `fit(schedule=..., start_step=...)`:

- **Sparsity warmup:** ramp the L1/Lp or JumpReLU L0 coefficient; auxiliary
  reconstruction losses are not warmed automatically.
- **k curriculum:** start with a larger training budget and anneal to the declared
  inference k; batch/global selection semantics stay unchanged.
- **Smoothed Lp annealing:** move p from 1 toward a chosen value in (0,1] for
  ReLU/Gated. The penalty is `sum((abs(z)+1e-6)^p - 1e-6^p)`, averaged over vectors;
  Gated applies it to gate activations. Smoothing avoids undefined zero gradients.
  This is an explicitly simplified control inspired by [Karvonen et al.,
  arXiv:2408.00113v2, §4.3](https://arxiv.org/html/2408.00113v2). It omits their
  adaptive coefficient matching and resampling, so it is not an exact p-annealing
  reproduction. Save settings and the optimizer-update cursor for continuation.
- **AuxK:** opt-in for nonnegative Top-K families. Dead-window counters count
  input vectors at batch resolution and advance after successful optimizer steps.
  Inactive dead features reconstruct the detached main residual using a bias-free
  decoder. It uses raw MSE, not the reference trainer's variance normalization.
  All-negative candidate preactivations can still get zero revival gradients;
  there is no guarantee of revival and no automatic neuron resampling.

These controls can change the learned dictionary even with the same activation
function. Save their settings, initializer, optimizer, data ordering, seeds and
realized L0. Changing p, loss scale or input normalization changes what the
regularization coefficient means. No schedule is enabled implicitly.

## Code examples

```python
from unitok_sae import SAEConfig, SparseAutoencoder, SparsitySchedule
from unitok_sae.training import fit

jump = SparseAutoencoder(SAEConfig(
    1024, 8192, variant="jumprelu", l0_coefficient=1e-4,
    initial_threshold=0.1, bandwidth=0.2,
))  # Illustrative values, not calibrated UniTok hyperparameters.

nested = SparseAutoencoder(SAEConfig(
    1024, 8192, variant="matryoshka", k=64,
    prefixes=(512, 1536, 3584, 8192), aux_coefficient=1/32,
))

schedule = SparsitySchedule(warmup_steps=1000, k_start=256, k_end=64, k_steps=1000)
# history, optimizer = fit(nested, discovery_batches, max_steps=..., schedule=schedule)
# nested.calibrate_threshold(calibration_panel)  # after weights stop changing
# nested.eval()
```

The CLI accepts all eight names. `train --sae-options FILE.json` supplies family-
specific SAEConfig fields, e.g. `{"prefixes":[512,1536,3584,8192]}` or
`{"l0_coefficient":0.0001,"initial_threshold":0.1,"bandwidth":0.2}`.
`--schedule FILE.json` supplies SparsitySchedule fields; checkpoint provenance
stores the resolved settings. These options do not start new jobs automatically.

## What else is worth checking later

[SAEBench, ICML 2025](https://proceedings.mlr.press/v267/karvonen25a.html), compares
families using multiple endpoints; its results motivate separate fidelity,
interpretability and task-effect measurements, not choosing the smallest MSE.
For UniTok, compare measured L0, activation fidelity, classification and native-
decoder fidelity, code switching, feature stability and controlled edit effects.
Keep data, initialization, budget and seeds controlled; dictionary IDs do not
align across families. This is comparison guidance, not a prebuilt experiment.

Two further directions are worth a separate implementation decision:

- [End-to-end SAE training, arXiv:2405.12241v2, §§2.3–2.4](https://arxiv.org/html/2405.12241v2):
  use downstream behavior in the loss rather than only local MSE. Particularly
  relevant to classification/reconstruction dissociation, but adapting the loss
  and quantizer gradients requires its own validity checks. Not implemented here.
- [Switch SAEs, arXiv:2410.08201v2, §3](https://arxiv.org/html/2410.08201v2):
  expert routing changes compute allocation. More useful if wide-dictionary
  compute becomes limiting; routing/load-balancing introduces another variable.
  Not implemented here. Crosscoders/transcoders likewise change the reconstructed
  target or number of activation streams and remain separate from this drop-in
  single-layer SAE comparison.

These are toolkit implementations, not reproductions of full paper training
recipes or benchmark results. See the [usage guide](guide.md) for setup.
