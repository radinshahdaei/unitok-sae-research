# UniTok SAE research

Composable UniTok layer access, sparse autoencoder training, residual-preserving
feature edits, activation data, visualization and individual/shared pixel PGD.

Use Python 3.12 or 3.13 with matching PyTorch/torchvision builds.

```sh
python -m pip install -e .
python examples/quickstart.py --output runs/first-sae
```

For native model access install `.[model]`, run `bash setup_source.sh`, and supply
your own checkpoint using `configs/model.example.json`. See module docstrings
and `examples/toolkit_recipes.py` for the explicit APIs. Weights and data are
separate assets; the synthetic example does not establish scientific results.
