# Tokenizer embedding attacks and consistency loss

This extension makes the reusable objectives from Bhagwatkar et al.,
**On the Adversarial Robustness of Discrete Image Tokenizers**,
[arXiv:2602.18252v1, 20 February 2026](https://arxiv.org/html/2602.18252v1),
accessible through the toolkit. It adds methods and independent examples,
not a prescribed experiment or a full adversarial-training pipeline.

## What was actually available

Checked 4 November 2026: the authors' [project page](https://robust-tokenizers.github.io/)
labels links Code and Weights, but both resolve to the arXiv paper. The public
`robust-tokenizers` organization exposes the project website repository; our
search did not identify a verified official implementation or robust checkpoint.
This is a bounded availability check, not proof that no code exists elsewhere.
No new tokenizer checkpoint was downloaded or substituted.

The paper uses **APGD**, whose official implementation is already installed from
[AutoAttack revision a39220048b3c9f2cca9a4d3a54604793c68eca7e](https://github.com/fra31/auto-attack/tree/a39220048b3c9f2cca9a4d3a54604793c68eca7e).
We reuse its optimizer and add our own objective adapters. Do not call this an
import of the authors' tokenizer code, a complete recipe reproduction, or a
reproduction of their benchmark results.

Install the optional attack backends with `python -m pip install -e '.[attacks]'`.
No additional dependency is introduced.

## Methods and exact scope

| API | Purpose and provenance |
| --- | --- |
| `session.pre_quantization(images, attachments=...)` | Native encoder plus quant_proj, returning differentiable `[B,T,D]` quantizer-input vectors. Codebooks, classifier and image decoder do not execute. Uses the same temporary SAE hooks as the full forward. |
| `UniTokEmbedding(session, attachments=...)` | Registers model and SAEs as an eval-mode nn.Module feature extractor for APGD. |
| `attack_embeddings(..., mode="diverge")` | Maximize summed squared clean/attacked embedding distance, §3.1 Eq.(1). By default the reference is the same encoder on clean images. |
| `attack_embeddings(..., mode="match", reference_features=...)` | Minimize distance to a target image's embeddings, §4.2 and Appendix B.1. This does not directly target a class label or caption. |
| `embedding_consistency_loss(student, frozen_clean_reference)` | Batch mean of token/channel-summed squared distance, §3.2 Eq.(2) outer loss. Reference is detached. Inner adversary must use the same original-teacher reference. |
| `token_change_counts(clean_indices, attacked_indices)` | Hard-token diagnostics inspired by Appendix B.2: separate changed codebook entries and changed positions, with measured denominators. No numeric token-ID distance or semantic-success claim. |

The objective is

```text
D(x, r) = sum over token i and channel j of (h(x)[i,j] - r[i,j])**2
```

`h` is the **actual quantizer input**, not an arbitrary middle layer or pooled
text-aligned embedding. Distances are not normalized by token count or channel
width; compare their scale carefully across architectures. Pixel budgets are
always raw `[0,1]` units. The paper evaluates Linf pixel constraints; the L2 pixel
constraint is our explicit extension, using upstream APGD's L2 projection.

The optimizer uses random starts: at the exact clean point Eq.(1)'s distance and
input gradient are zero. For every source, APGD selects the best loss across all
iterations/restarts; we also retain the clean image as a candidate. Divergence
therefore cannot decrease its measured distance; matching cannot increase it.
These inequalities alone do not establish a strong attack. A shared universal
perturbation is not fitted here.

### How upstream APGD accepts this objective

At the pinned revision, APGD does not accept an arbitrary loss callback. Our
small local subclass replaces its `dlr_loss` method with the **exact signed
squared-distance score**. The `loss="dlr"` string is only the internal dispatch
slot: no DLR ratio, cross-entropy, softmax or classification surrogate is used.
A two-column score carrier meets APGD's output interface; it is never interpreted
as real classifier logits or reported as class predictions.

`perturb(..., best_loss=True)` keeps the complete paired batch in all restarts,
without classification early stopping or subsetting that would misalign target
rows. We reuse upstream initialization, momentum, projection, adaptive step size
and oscillation checks without copying or patching its source. The exact backend
commit is checked at call time because this extension uses an internal loss hook.
This is an embedding APGD adaptation, **not** a standard AutoAttack evaluation.

## Usage

```python
import torch
from unitok_sae.tokenizer_attacks import (
    EmbeddingAttackConfig, UniTokEmbedding, attack_embeddings,
)

# session is already loaded; source/target are float32 RGB BCHW [0,1].
extractor = UniTokEmbedding(session)
config = EmbeddingAttackConfig(epsilon=4/255, steps=100, restarts=1, seed=42)

# Label-free per-image distortion:
result = attack_embeddings(extractor, source, config, max_evaluations=1000)

# Target-image matching; a single target can be expanded to the whole source batch.
with torch.no_grad():
    target_features = extractor(target).detach()
matched = attack_embeddings(
    extractor, source, config, mode="match",
    reference_features=target_features, max_evaluations=1000,
)
```

See [independent recipes](../examples/tokenizer_attack_recipes.py) for SAE-edited
encoders, target matching, a frozen-original-teacher inner attack/outer loss, and
hard-token diagnostics. The teacher example requires a separate frozen original
model; it does not silently choose trainable student parameters. Freezing a
model's weights does not mean detaching its input gradients.

Results retain adversarial images, exact reference tensors, squared distances
before/after, actual pixel norms, backend identity, objective/configuration and
extractor evaluation counts. The optional hard cap includes clean/final feature
checks and raises `QueryBudgetExceeded`. Parameter flags, gradient buffers and
caller RNG states are restored after attack success or failure. Save checkpoint
hashes, source/target IDs, preprocessing and the result tensors/settings yourself.
No classification success, reconstruction fidelity or robust accuracy is inferred
from an embedding-distance improvement.

The full native forward remains available to check the actual hard tokens,
classification and image reconstruction afterward. The pre-quantization objective
needs no STE; downstream classification attacks still traverse hard quantization
with its existing STE. A readable decoded image does not establish correct
classification; §5/Appendix B.1 illustrate this dissociation qualitatively.

These are toolkit implementations, not reproductions of full paper training
recipes or benchmark results. See the [usage guide](guide.md) for setup.
