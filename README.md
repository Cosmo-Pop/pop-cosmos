# pop-cosmos

This is a repository containing utilities for working with the pop-cosmos galaxy population model. 

The pop-cosmos model was first introduced and trained by [Alsing et al. (2024)](https://arxiv.org/abs/2402.00935). The use of this population model as a prior in SED fitting is described in [Thorp et al. (2024)](https://arxiv.org/abs/2406.19437). 

The code and models in this repository are based on the updates described in Thorp et al. (2025). If you make use of this code, please cite all of these papers.

The documentation is in the `docs` directory and the docstrings within the code. The `pop_cosmos` module contains the code, and the `trained_models` directory contains binary files with the trained models.

# Installation
To install the code, please clone this repo:
```bash
  git clone https://github.com/Cosmo-Pop/pop-cosmos
```
Then move into the top level directory and run:
```bash
  pip install .
```
This will obtain any dependencies and will install the code, which can then be imported in Python by doing:
```python
import pop_cosmos
```
To install `pop_cosmos` without updating the dependencies:
```bash
pip install poetry
poetry install --no-update
```
Alternatively (Recommended):
```bash
pip install --upgrade-strategy only-if-needed .
```

# Usage
See the `demo` directory for example notebooks.
