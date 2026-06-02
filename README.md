# pop-cosmos
[![Static Badge](https://img.shields.io/badge/arXiv-2402.00935-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2402.00935)
[![Static Badge](https://img.shields.io/badge/arXiv-2406.19437-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2406.19437)
[![Static Badge](https://img.shields.io/badge/arXiv-2506.12122-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2506.12122)
[![Static Badge](https://img.shields.io/badge/arXiv-2509.20430-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2509.20430)
[![Static Badge](https://img.shields.io/badge/arXiv-2602.03930-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2602.03930)
[![Static Badge](https://img.shields.io/badge/arXiv-2602.03935-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2602.03935)

This is a repository containing utilities for working with the pop-cosmos galaxy population model. 

The pop-cosmos model was first introduced and trained by Alsing et al. ([2024](https://ui.adsabs.harvard.edu/abs/2024ApJS..274...12A/abstract)). The use of this population model as a prior in SED fitting is described in Thorp et al. ([2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...975..145T/abstract)).

The code and models in this repository are based on the updates described in Thorp et al. ([2025](https://ui.adsabs.harvard.edu/abs/2025ApJ...993..240T/abstract)). If you make use of this code, please cite all of these papers. If you use the Speculator photometry emulators included in this repository, please also cite Alsing et al. ([2020](https://ui.adsabs.harvard.edu/abs/2020ApJS..249....5A/abstract)). If you use the rest-frame $NUVrJ$ photometry emulators, please cite Deger et al. ([2026](https://ui.adsabs.harvard.edu/abs/2026MNRAS.549ag764D/abstract)). If you use the KiDS or DECaLS photometry emulators, please cite Halder et al. ([2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260203930H/abstract)) and Leistedt et al. ([2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260203935L/abstract)).

The documentation is in the `docs` directory and the docstrings within the code. The `pop_cosmos` module contains the code, and the `trained_models` directory contains binary files with the trained models.

## Installation
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

## Usage
See the `demo` directory for example notebooks.

## Documentation
To check the documentation for the code, please check the `docs` directory.

## Additional Data
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.15622324-%231682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.15622324)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.13627488-%231682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.13627488)

You can find the pre-generated COSMOS-like mock galaxy catalogs described in Thorp et al. ([2025](https://ui.adsabs.harvard.edu/abs/2025ApJ...993..240T/abstract)) and Deger et al. ([2026](https://ui.adsabs.harvard.edu/abs/2026MNRAS.549ag764D/abstract)) on Zenodo. The most up-to-date versions of these will always be linked from DOI:[10.5281/zenodo.15622324](https://doi.org/10.5281/zenodo.15622324). Our most up-to-date SPS parameter posteriors for COSMOS2020 will always be linked from DOI:[10.5281/zenodo.13627488](https://doi.org/10.5281/zenodo.13627488). The Thorp et al. ([2025](https://ui.adsabs.harvard.edu/abs/2025ApJ...993..240T/abstract)) results correspond to `v2+` of the Zenodo record. The Thorp et al. ([2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...975..145T/abstract)) results are preserved in `v1`.

## Related Software
[![Static Badge](https://img.shields.io/badge/GitHub-justinalsing%2Faffine-%23181717?logo=GitHub&logoColor=white)](https://github.com/justinalsing/affine)
[![Static Badge](https://img.shields.io/badge/GitHub-stevet40%2Fhist__contour-%23181717?logo=GitHub&logoColor=white)](https://github.com/stevet40/hist_contour)
[![Static Badge](https://img.shields.io/badge/GitHub-stevet40%2Fquantile__utilities-%23181717?logo=GitHub&logoColor=white)](https://github.com/stevet40/quantile_utilities)

The GitHub links above contain some related software packages that may be useful. The package `affine` is an optional dependency if you want to run MCMC under the `pop_cosmos` prior. The `hist_contour` and `quantile_utilities` repos contain some auxilliary scripts for plotting.
