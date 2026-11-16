# Using the toolkit

## Setup

Use Python 3.12 or 3.13 and matching PyTorch/torchvision builds. From the root:

```sh
python -m pip install -e '.[model]'
bash setup_source.sh
```

The script fetches `roronoazoero/unitok-sae` at
`3a22e697e8c6586e8c270808458f17b24d8a575f` into `external/unitok-sae`, then applies
the included timm compatibility patch. It downloads no weights or data.
The loader checks this source revision and patch.

Copy `configs/model.example.json` to `configs/model.local.json`. Obtain the
intended checkpoint separately and set its path, SHA-256, state-dictionary
branch and architecture fields. The example describes the previously checked
FoundationVision/unitok_tokenizer asset, not every UniTok variant. Paths in model
configs resolve from the working directory; image paths from their image
manifest; shard paths from their shard manifest.

```python
import json
from unitok_sae.loading import load_unitok, load_pretrained_sae

with open("configs/model.local.json") as f:
    config = json.load(f)
session, tokenize, audit = load_unitok(config, device="cpu")

# Original-author TopKSAE format; replace these asset fields.
sae, sae_audit = load_pretrained_sae(
    {"path": "/path/to/sae.pt", "sha256": "VERIFIED_HASH", "provider": "SOURCE"},
    implementation_repo=config["implementation_repo"], block=15,
    block_provenance="INDEPENDENT_BLOCK_EVIDENCE_IF_OMITTED_IN_CHECKPOINT",
    device="cpu",
)
```

Toolkit-trained checkpoints instead use `training.load_sae(path)`, returning
`(sae, checkpoint)`. No original SAE weights or verified robust UniTok weights
are supplied. On Colab retain its working CUDA torch/torchvision installation.

## Inspect and edit

```python
import torch
from PIL import Image
from unitok_sae import SAEAttachment, edit_features
from unitok_sae.model import preprocess_image

# Supply image_path, device and feature_ids for your checkpoint/dictionary.
with Image.open(image_path) as image:
    images = preprocess_image(image, image_size=session.image_size)[None].to(device)
with torch.inference_mode():
    native = session.forward(images, layers=[15])
    baseline = session.forward(images, attachments={15: SAEAttachment(sae)})
    edited = session.forward(images, attachments={15: SAEAttachment(
        sae, edit=lambda z: edit_features(z, feature_ids, operation="set", value=0),
    )})
    codes = baseline.layers[15].codes
    decoded = session.decode_tokens(native.indices)
```

Public images are float32 RGB `[B,3,H,W]` in [0,1] at the checkpoint resolution;
no ImageNet mean/std normalization is applied. Activations are `[B,T,D]`, codes
`[B,T,F]`, token indices `[B,codebooks,T]`. Measure token counts rather than
assuming a grid. Reconstructions remain raw native-scale tensors, approximately
[-1,1]; clamp only for display. `decode=False` skips image decoding.

The default edit is `H_edit = H + decoder(Z_edit - Z)`, preserving the recipient's
SAE residual. `mode="replacement"` discards it and can change outputs even
without an edit. Set/add/scale edits support token masks; signed AbsTopK edits
require `allow_signed=True`. Later attached layers see earlier edits.
Models and SAEs must be in eval mode. BatchTopK/Matryoshka need threshold
calibration. Avoid simultaneous forwards on the same model: hooks are temporary
model-global state. Feature IDs are specific to a dictionary and layer.

Classification uses normalized text prototypes in explicit class order:

```python
prototypes = session.text_prototypes(class_names, tokenize,
    templates=("a photo of a {}",))
logits = session.logits(images, prototypes)
```

Audit the supplied prototypes and their exact prompts before an experiment:

```python
from unitok_sae.classification import audit_classifier

prompts = [f"a photo of a {name}" for name in class_names]
audit = audit_classifier(prototypes, prompts=prompts, tokens=tokenize(prompts))
```

The audit reports duplicate prompts, tokenizer collisions, exact/near-duplicate
prototype rows and prototype self-query failures. It does not rewrite prompts.
Save the class order, tokenizer identity and templates alongside the audit;
a later class with an identical prototype cannot win a first-index argmax tie.
Self-query agreement does not measure image accuracy.

Use all 1000 classes for ImageNet. `top_features`, `feature_map` and the
`visualization` module help inspect codes and decoded edits. A plausible image
or heatmap does not validate a feature's semantics or explain a prediction.

## Spatial activation steering

A fixed spatial template can edit native activations without requiring SAE
encoding or nonnegative codes:

```python
from unitok_sae.steering import svd_span, project_edit

# decoder_columns is [D,F]; neutral_edit is a discovery-fitted [T,D] tensor.
basis, span_info = svd_span(decoder_columns[:, feature_ids])
edit, coefficients = project_edit(neutral_edit, basis)
result = session.forward(images, activation_edits={block: lambda h: h + edit})
```

Direct callbacks receive a protected copy of `[B,T,D]` and must preserve its
shape, dtype, device and finite values. They retain gradients and can be passed
to `forward`, `pre_quantization` or `logits`. An activation callback and an SAE
attachment cannot share the same block. Later layers see earlier edits.

`steering.mean_feature_edit` averages decoded discovery code changes over images
while retaining token positions. `mean_preserving_span` keeps a mean direction
plus up to q-1 centered principal components. `rotated_dictionary` preserves
column norms and their Gram matrix. `svd_span` keeps actual rank rather than
completing deficient columns with arbitrary directions. Record its tolerance.
A span with q directions allows T*q spatial coefficients, and re-encoding an
edited state can change SAE coordinates outside the selected support.

See [steering recipes](../examples/steering_recipes.py) for donor-free templates
and matched fixed-span controls. Frozen IDs with per-evaluation-image donor
values remain an oracle intervention; a mean template requires discovery donors
only. Match selection budgets separately before comparing learned dictionaries.

## Bounded support search and coefficient fitting

`control.select_support(dictionary, proposals, evaluate, ...)` evaluates explicit
candidate supports using a caller-supplied hard-forward score. Set finite
`max_groups`, `max_image_evaluations` and `images_per_group`; the latter must
include every image forward inside a candidate test, including magnitude probes.
The function charges the declared cost and cannot instrument hidden work inside
a callback. Duplicate proposals count toward the proposal limit. Reuse the same
candidate-generation procedure and resources across dictionaries.

`control.fit_span_edit` fits spatial coefficients and edit magnitude inside one
frozen orthonormal basis. Supply a neutral initialization, a differentiable
`objective(edit)` returning `(task_loss, per_image_mse)`, and a separate native
hard-forward `measure(edit)` returning `(success_bool, per_image_mse)`. Set the
step limit, norm bound and distortion cap explicitly. The common added penalty is
`penalty_weight * mean(relu(mse / cap - 1)**2)`. The caller defines the task
loss, reconstruction reference and raw units identically for every geometry.

The fitter checks the start and declared step intervals, retaining a measured
iterate feasible on every fitting image when one exists. Otherwise it returns
the least-violating measured edit with `feasible=False`. Neither a soft penalty
nor fitting-set feasibility ensures feasibility on new images. Inspect its
trace and evaluate held-out distortion as an outcome. Magnitude is projected
after updates to retain its gradient at the maximum norm.

The `fit_template` function in [steering recipes](../examples/steering_recipes.py)
connects these callbacks to UniTok's native head, differentiable reconstruction
penalty and hard-token decoding. Freeze model parameters and keep discovery,
calibration and final images separate. It is a small fitting recipe, not the
full experiment controller, a reproduced benchmark or a defense.

## Activation data and SAE training

`data.extract_activations(session, batches, destination, layers=..., provenance=...)`
saves hashed CPU shards. Each batch supplies `images`, stable `sample_ids`,
source-file `sha256` values and a `split` (discovery/calibration/evaluation).
Choose splits by source image before extracting tokens.
`data.ActivationDataset(manifest, layer=..., split=...)` lazily exposes token rows.

`training.fit` accepts a dictionary, vector batches, `max_steps` and learning
rate. It stops at the step cap or iterable exhaustion; it does not restart
epochs. `training.train_step` supports custom loops. See
[toolkit recipes](../examples/toolkit_recipes.py) and [SAE methods](sae-variants.md)
for training composition, all eight families, schedules and paper differences.

Calibrate BatchTopK/Matryoshka thresholds on separate calibration data before
evaluation. `training.fidelity` measures activation reconstruction, not downstream
behavior. `save_sae` stores configuration, block, provenance and optional optimizer
state; resuming still requires an explicit step cursor and data order.

## Attacks

Attacks need autograd: do not wrap them in `inference_mode()`. Pixel inputs and
Linf budgets use [0,1] units, e.g. `4/255`. Classification follows native hard
quantization with its straight-through backward approximation.

```python
from unitok_sae.attacks import PGDConfig, pgd, apply_delta

logits_fn = lambda x: session.logits(x, prototypes)
result = pgd(logits_fn, discovery_images, target_labels, PGDConfig(
    targeted=True, shared=True, epsilon=4/255, step_size=1/255,
    steps=20, restarts=1, seed=42,
))
held_out_attacked = apply_delta(held_out_images, result.delta)
```

`shared=True` fits one persistent delta and selects one common iterate for the
discovery batch. `shared=False` optimizes individual perturbations. Freeze shared
deltas before evaluation. Batching individual attacks does not make them shared.

For AutoAttack, Foolbox and tokenizer APGD (or install `.[model,attacks]` for both extras):

```sh
python -m pip install -e '.[attacks]'
```

`external_attacks.UniTokClassifier` wraps a session, prototypes and optional SAE
attachments. `run_autoattack` exposes official evaluation profiles;
`run_foolbox` exposes FGSM, PGD, CW, DeepFool and HopSkipJump. See
[external attack recipes](../examples/external_attack_recipes.py) and function
docstrings for arguments. Fixed A→B targets use custom PGD or supported Foolbox
methods; AutoAttack's internal target searches seek any misclassification.
Reduced AutoAttack budgets require `version="custom"`. Query caps raise on
exhaustion. Forward chunking requires batch-independent edits. External backends
were checked on CPU/CUDA, not MPS.

[Tokenizer objectives](tokenizer-paper-methods.md) cover pre-quantization APGD
divergence, target-image matching, frozen-reference loss and token-change counts.
An embedding-distance improvement alone does not establish downstream failure.

## Running experiments

The [short experiment loop](experiment-loop.md) describes falsifiable pilot
decisions, oracle versus donor-free edits, controls and confirmation on fresh data.

The recipes are independent functions with explicit inputs. The CLI exposes
`extract`, `train`, `inspect` and `attack`; use `python -m unitok_sae COMMAND --help`.
Outputs require new directories. Keep local configs, checkpoints, data and runs
ignored. Start with a few images and inspect no-op behavior, gradients, edit
norms and both output branches before scaling. Record seeds, asset identities,
input IDs and attack settings with results. Clean donor replay is an oracle
check; reusable steering needs a frozen rule evaluated on unseen inputs.
