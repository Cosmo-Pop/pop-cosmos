# pop-cosmos
[![Static Badge](https://img.shields.io/badge/arXiv-2402.00935-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2402.00935)
[![Static Badge](https://img.shields.io/badge/arXiv-2406.19437-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2406.19437)
[![Static Badge](https://img.shields.io/badge/arXiv-2506.12122-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2506.12122)

This is a repository containing utilities for working with the pop-cosmos galaxy population model. 

The pop-cosmos model was first introduced and trained by [Alsing et al. (2024)](https://arxiv.org/abs/2402.00935). The use of this population model as a prior in SED fitting is described in [Thorp et al. (2024)](https://arxiv.org/abs/2406.19437). 

The code and models in this repository are based on the updates described in [Thorp et al. (2025)](https://arxiv.org/abs/2506.12122). If you make use of this code, please cite all of these papers.

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

# Documentation
To check the documentation for the code, please check the `docs` directory.

# Additional Data
Pregenerated pop-cosmos mock galaxy catalogs: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15622325.svg)](https://doi.org/10.5281/zenodo.15622325)

Posteriors over SPS parameters for individual COSMOS2020 galaxies: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15623082.svg)](https://doi.org/10.5281/zenodo.15623082)

