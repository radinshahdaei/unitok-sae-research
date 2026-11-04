# UniTok SAE research

Tools for studying and steering UniTok through sparse autoencoder features.
Capture encoder activations, train dictionaries, edit groups of features, and
measure changes in classification and native image reconstruction.

The current research question is whether individual and shared pixel attacks
reveal feature groups that provide repeatable control on unseen inputs.
The toolkit supports these experiments; useful causal groups and reusable
steering have yet to be demonstrated. The broader question is how feature
interpretability, adversarial stability and causal importance relate.

## Getting started

Use Python 3.12 or 3.13. Install matching PyTorch and torchvision builds for your
machine, then:

```sh
python -m pip install -e .
python examples/quickstart.py --output runs/first-sae
```

The quickstart trains and edits a tiny SAE on synthetic vectors, without model
weights. For UniTok setup and practical usage, see [the guide](docs/guide.md).

## Where things live

- `unitok_sae/`: model access, feature edits, SAE families/training, activation
  datasets, attacks and visualization. Function docstrings describe the APIs.
- `examples/`: small Python recipes to adapt to your own images and checkpoints.
  Start with `toolkit_recipes.py`; the other recipes cover SAE configurations,
  external attacks and tokenizer objectives.
- `configs/`: model and image-manifest templates; local copies stay ignored.
- `docs/`: [usage](docs/guide.md), [SAE methods](docs/sae-variants.md) and
  [tokenizer objectives](docs/tokenizer-paper-methods.md).
- [AGENTS.md](AGENTS.md): working context for an agent continuing the research.

Weights, data, source checkouts and generated runs stay outside Git. This is a
research workspace: use the functions directly and keep experiments small until
their computation paths and outputs make sense. Earlier software checks covered
synthetic models and bounded real-weight runs; no full scientific benchmark is
provided. Method implementations are not reproductions of published results.
