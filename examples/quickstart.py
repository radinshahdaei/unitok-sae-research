"""CPU-only five-step SAE example on synthetic vectors; no assets or downloads.

Run: python examples/quickstart.py --output runs/first-sae
Uses a new output directory. Scores describe only this synthetic wiring check.
"""

import argparse
import json
from pathlib import Path

import torch

from unitok_sae import SAEConfig, SparseAutoencoder, edit_features, top_features
from unitok_sae.training import fidelity, fit, load_sae, save_sae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    discovery = torch.randn(5, 16, 16)
    held_out = torch.randn(1, 16, 16)
    sae = SparseAutoencoder(SAEConfig(input_dim=16, hidden_dim=64, variant="topk", k=4))
    history, optimizer = fit(sae, iter(discovery), max_steps=5, learning_rate=1e-3)
    sae.eval()
    with torch.no_grad():
        codes = sae.encode(held_out)[0]
        selected = top_features(codes, count=1).indices[0].item()
        changed = edit_features(codes, [selected], operation="set", value=0)
        assert torch.isfinite(sae.decode(changed)).all()
        scores = fidelity(sae, held_out)
    checkpoint = args.output / "sae.pt"
    save_sae(
        checkpoint,
        sae,
        block=0,
        steps=len(history),
        optimizer=optimizer,
        provenance={
            "purpose": "synthetic quickstart, no UniTok layer claim",
            "seed": 42,
        },
    )
    reloaded, _ = load_sae(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(reloaded.encode(held_out)[0], codes)
    report = {
        "scope": "synthetic wiring only",
        "steps": len(history),
        "selected_feature": selected,
        "fidelity": scores,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
