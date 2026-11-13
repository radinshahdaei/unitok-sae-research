# Short experiments that answer a question

Use the toolkit to resolve one uncertainty at a time. A useful experiment states
what evidence would support an idea, what would weaken it, and which test would
follow either result. Implement the smallest missing function and check its
numerics before running a model. A completed script is a software milestone;
its measured outputs determine the research decision.

## Before a run

Write a short note beside its config containing the hypothesis, dataset/split and
checkpoint/dictionary identities, primary outcome, useful-effect threshold,
controls, condition count, compute estimate and a finite job limit. Use existing
assets and explicit user-authorized resources. A continuing campaign still needs
finite individual jobs, resource accounting and compliance with its host's rules.
Profile new operations on a tiny input first; batch conditions and cache states
where their equivalence to the native forward is verified.

Split source images before extracting tokens. Fit selectors on discovery data,
tune declared choices on calibration, and reserve fresh final images. An opened
final split is development evidence for a subsequently changed method. Repeated
attack seeds and targets on one image are repeated measurements, not new images.

Check the classifier as well as the model: record text prompts and class order,
test prompt/token collisions and duplicate or near-duplicate prototypes, and
query prototypes against the complete class set. Preserve old classifier versions
and regenerate relevant baselines when prompts change. Keep logit scaling fixed
and explicit within a comparison.

## Separate three questions about a feature group

| Experiment | Allowed evaluation information | What it tests |
| --- | --- | --- |
| Image-specific oracle search | This image's clean/attacked states and outcome-dependent support | Whether the declared search finds an effective group for that image |
| Discovery-frozen support | Fixed feature IDs, with this image's paired donor values | Whether the feature IDs transfer across images |
| Donor-free rule | Frozen procedure using the current input and requested task | Whether the edit works without obtaining an evaluation attack donor |

Record the query budget for every search. Failure of a finite heuristic is not
proof that no compact group exists. A frozen mask with per-image attacked values
remains an oracle intervention. Donor-free rule amplitudes and norm caps must not
depend on an evaluation donor. Steering a clean input is distinct from defending
against attacks on the edited system.

## Read the outcome before expanding

Use continuous class margins alongside prediction flips. Inspect realized edit
norms, token changes and both classification and reconstruction on identical
eligible subsets. A threshold crossing can make reversal easier than replay;
conditional repair alone cannot rank two discovery methods. A pixel distance or
plausible decoded image does not establish semantic control.

Keep native no-op, SAE replacement, whole-state, all-code and residual controls.
Compare selected groups with appropriate random features, dense directions,
PCA and neuron coordinates. Match spatial support and search capacity, and report
actual norms: equal caps do not imply equal realized edits. Shrink larger edits
for a direct norm comparison instead of amplifying near-zero controls. Distinguish
feature-ID count, edited tokens and effective activation-space rank.

When a variant fails, choose one next test that distinguishes bad selection,
poor transfer, unsuitable amplitudes or a downstream threshold. Avoid expanding
the same failed grid without a new prediction. When an early signal appears,
freeze the procedure and evaluate untouched images before promoting it to a
confirmed result. Report paired uncertainty, all denominators and class/pair
concentration; a tie with a simpler basis limits claims about the SAE itself.

## Leave a compact record

Save seeds, input IDs/hashes, code/config/asset identities, job IDs, elapsed compute,
query counts, numerical outcomes and failures. Keep a small fixed visual panel;
label additional selected examples as diagnostics. Each update should say what
was tested, the result, its limits, resource cost and the next decision. Keep
private protocols, reports and generated evidence out of public toolkit history.

Available building blocks are in [the guide](guide.md) and
[toolkit recipes](../examples/toolkit_recipes.py). Selection algorithms and fitted
steering rules discussed here are experiment designs, not extra implemented APIs.
