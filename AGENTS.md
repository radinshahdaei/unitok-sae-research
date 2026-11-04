# Working in this repository

This is a research workspace for UniTok and sparse autoencoders. Keep it easy to
read and change. Prefer an existing toolkit function and a short experiment over
new infrastructure. Do not add CI, release workflows, validator suites or layers
of configuration unless explicitly requested. Keep documentation near the code;
update the guide only when a usage contract changes.

## Research context

The immediate goal is to discover groups of SAE features using individual and
shared targeted pixel attacks, then test classification and native-image
reconstruction control. Attack-induced changes nominate candidates; fixed group
interventions and held-out steering establish their usefulness. Interpretability
characterizes groups without gating discovery. Broader I/S/C comparisons and a
standard-versus-robust model diff are later questions.

When the sibling `unitok-sae-private` is available, consult its
`docs/causal-feature-groups.md` and `RESEARCH_STATE.md` for detailed protocols and
recorded results. This repository works independently of that private workspace;
do not copy private reports or assets into it.

## Practical rules

- Read the relevant module and recipe before changing behavior. Preserve the
  toolkit's composable APIs; avoid a mandatory experiment pipeline.
- Public images are float32 RGB BCHW in [0,1]. Native reconstructions remain
  raw model-scale tensors. Measure token counts and dimensions for the checkpoint.
- Feature edits preserve the SAE residual by default. Measure reconstruction
  replacement separately. Feature IDs are specific to a dictionary and layer.
- Evaluate attacks through the native hard forward; record any surrogate
  gradients. Fit shared deltas on discovery images and freeze them for evaluation.
- Keep discovery, calibration and evaluation images separate. Compare groups
  against no-op and appropriate random/dense controls. Clean donor restoration
  is an oracle diagnostic, not a deployable steering rule or defense.
- Distinguish implemented code, paper claims, hypotheses and measured findings.
  Save the seed, model/dictionary identity, input IDs, settings and denominators
  with an experiment. Do not infer semantics from an attractive example alone.
- Keep weights, data, credentials, local configs and outputs ignored. Do not
  download assets, launch long runs or publish changes without task authorization.
- Check a change with the smallest relevant example or manual computation.
  Explain what ran and what it establishes; add a focused test only when useful.
  Development regression tests remain in the private workspace and Git history.
