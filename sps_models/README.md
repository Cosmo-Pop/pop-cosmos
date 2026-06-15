# pop-cosmos/sps_models

This directory contains the configuration file(s) needed to reproduce our standard SPS model parametrization using the Prospector and FSPS libraries.

## Installation
Our current set of papers and models are based on the version of the core FSPS library tagged as v3.2. You can obtain a matching version by first cloning the repo, and then checking out this tag:
```bash
git clone https://github.com/cconroy20/fsps
cd fsps
git checkout v3.2
```
To build this version, you'll need to follow the instructions in `fsps/doc/INSTALL`. Before running `make`, you'll need to set the `$SPS_HOME` environment variable to point to your clone of `fsps`:
```bash
export SPS_HOME=/PATH/TO/fsps/
```
It is recommended to add this to your `.bashrc`, `.zshrc`, or similar. 

Once you've successfully compiled FSPS, you should install `python-fsps`, `prospector`, and `sedpy` as follows:
```bash
pip install fsps==0.4.1
pip install astro-prospector
pip install astro-sedpy
```
The version `fsps==0.4.1` should be compatible with v3.2 of the core FSPS library.

If you use these packages in your work, please cite them accordingly -- BibTeX entries are included in our `CITATION.md`.

## Usage
With the dependencies installed, our specific SPS parametrization is defined by the `*_params.py` files that are included in this `sps_models` directory. These are [*parameter files*](https://prospect.readthedocs.io/en/stable/usage.html) in Prospector parlance. 

Currently we provide one parameter file, corresponding to the SPS parametrization introduced in Alsing et al. ([2024](https://ui.adsabs.harvard.edu/abs/2024ApJS..274...12A/abstract)). This is a slight variation on the Prospector-$\alpha$ model developed by Leja et al. ([2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...837..170L/abstract), [2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...854...62L/abstract), [2019a](https://ui.adsabs.harvard.edu/abs/2019ApJ...876....3L/abstract), [2019b](https://ui.adsabs.harvard.edu/abs/2019ApJ...877..140L/abstract)).

We provide two utilities for working with our parameter files in Python:
```python
from pop_cosmos.utils import initialise_fsps, generate_fsps_spec
```
The first of these can be used to load in a parameter file as a module, and to initialise the Prospector observation dictionary, and SPS model objects:
```python
pfile, obs, sps, mod = initialise_fsps(pro_name='prospector_alpha_plus_params',
                                       pro_path='PATH/TO/sps_models/',
                                       fsps_path='PATH/TO/fsps/'
                                      )
```
The MCMC catalogue demo (`demo/mcmc_catalogue_tutorial.ipynb`) shows how to post-process some MCMC chains to generate FSPS model spectra.

## References
- J. Alsing et al. (2024). ApJS 274, 12. [arXiv:2402.00935](https://arxiv.org/abs/2402.00935)
- J. Leja et al. (2017).  ApJ 837, 170.  [arXiv:1609.09073](https://arxiv.org/abs/1609.09073)
- J. Leja et al. (2018).  ApJ 854, 62.   [arXiv:1709.04469](https://arxiv.org/abs/1709.04469)
- J. Leja et al. (2019a). ApJ 876, 3.    [arXiv:1811.03637](https://arxiv.org/abs/1811.03637)
- J. Leja et al. (2019b). ApJ 877, 140.  [arXiv:1812.05608](https://arxiv.org/abs/1812.05608)