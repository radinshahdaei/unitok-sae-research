"""Independent, executable Python recipes; nothing runs on import.

These functions illustrate composition, not a prescribed research pipeline.
Read their inputs and return values, then reuse the underlying toolkit methods
directly. No function downloads assets or infers a checkpoint, class vocabulary,
feature interpretation, data split or compute budget.
"""

import torch

from unitok_sae import SAEAttachment, SAEConfig, SparseAutoencoder, edit_features
from unitok_sae.attacks import PGDConfig, apply_delta, fgsm, pgd
from unitok_sae.data import ActivationDataset
from unitok_sae.features import feature_map, top_features
from unitok_sae.loading import load_pretrained_sae, load_unitok
from unitok_sae.training import fidelity, fit, save_sae
from unitok_sae.visualization import feature_overlay, reconstruction_grid


def inspect_and_edit(config, images, *, feature_ids, value, grid_shape, device="cpu"):
    """Return native/edited model states, feature rankings, heatmap and image grid.

    config: explicit verified asset/model config, including block identity.
    images: already preprocessed BCHW float32 RGB [0,1], on device.
    grid_shape: validated hook grid, e.g. (16,16) for the measured local setup.
    value: absolute nonnegative value to set at every token for selected features.

    This is a user-chosen intervention, with no semantic or causal-group claim.
    """
    session, _, _ = load_unitok(config, device=device)
    sae, _ = load_pretrained_sae(
        config["sae_checkpoint"],
        implementation_repo=config["implementation_repo"],
        block=config["block"],
        block_provenance=config.get("block_provenance"),
        device=device,
    )
    attachment = SAEAttachment(
        sae, lambda z: edit_features(z, feature_ids, value=value)
    )
    with torch.inference_mode():
        native = session.forward(
            images, attachments={config["block"]: SAEAttachment(sae)}
        )
        edited = session.forward(images, attachments={config["block"]: attachment})
        codes = native.layers[config["block"]].codes
        spatial = feature_map(codes, feature_ids[0], grid_shape=grid_shape)
        overlay = feature_overlay(images[0], spatial[0])
        grid = reconstruction_grid(images, native.reconstruction, edited.reconstruction)
    return native, edited, top_features(codes), overlay, grid


def train_from_activations(
    manifest,
    *,
    layer,
    variant,
    hidden_dim,
    k,
    max_steps,
    output,
    seed,
    device="cpu",
    l1_coefficient=0.0,
):
    """Train a bounded baseline on discovery tokens, calibrate/evaluate separately.

    Uses a single finite pass; fewer than max_steps are possible. Real training
    needs a planned data order, epochs, resource cap and held-out fidelity study.
    BatchTopK calibration uses a bounded 128-token calibration panel here; this
    convenience example makes no claim that this is statistically sufficient.
    """
    torch.manual_seed(seed)
    discovery = ActivationDataset(manifest, layer=layer, split="discovery")
    calibration = ActivationDataset(manifest, layer=layer, split="calibration")
    sae = SparseAutoencoder(
        SAEConfig(discovery.input_dim, hidden_dim, variant, k, l1_coefficient)
    ).to(device)
    loader = torch.utils.data.DataLoader(discovery, batch_size=256, shuffle=False)
    history, optimizer = fit(sae, (x.to(device) for x in loader), max_steps=max_steps)
    panel = torch.stack([calibration[i] for i in range(min(128, len(calibration)))]).to(
        device
    )
    if variant == "batchtopk":
        sae.calibrate_threshold(panel)
    scores = fidelity(sae, panel)
    save_sae(
        output,
        sae,
        block=layer,
        steps=len(history),
        optimizer=optimizer,
        provenance={
            "activation_manifest": str(manifest),
            "seed": seed,
            "purpose": "bounded recipe",
        },
    )
    return sae.eval(), history, scores


def attack_examples(
    session, images, true_labels, text_prototypes, *, target_class, epsilon, steps
):
    """Return FGSM, individual targeted PGD and a shared targeted discovery delta.

    For ImageNet, text_prototypes must contain all 1000 class rows in recorded
    order. Supply discovery images only when fitting the shared attack. Its
    returned delta can be applied to held-out images using apply_delta without
    new optimization. No candidate group selection happens here.
    """

    def logits(x):
        return session.logits(x, text_prototypes)

    targets = torch.full_like(true_labels, target_class)
    one_step = fgsm(logits, images, true_labels, epsilon=epsilon)
    individual = pgd(
        logits,
        images,
        targets,
        PGDConfig(
            epsilon=epsilon,
            step_size=epsilon / 4,
            steps=steps,
            targeted=True,
            objective="margin",
        ),
    )
    shared = pgd(
        logits,
        images,
        targets,
        PGDConfig(
            epsilon=epsilon,
            step_size=epsilon / 4,
            steps=steps,
            targeted=True,
            shared=True,
        ),
    )
    torch.testing.assert_close(shared.images, apply_delta(images, shared.delta))
    return one_step, individual, shared
