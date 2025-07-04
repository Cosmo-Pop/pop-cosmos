# trained_models

This directory contains a variety of binaries contained trained models, and the components thereof.

The files included are as follows:
  - `catalogueModelA24.pt`: Alsing et al. (2024) pop-cosmos model
  - `catalogueModelT25.pt`: Thorp et al. (2025) pop-cosmos model
  - `noiseModelA24.pt`: Alsing et al. (2024) uncertainty model (MDN)
  - `noiseModelT25.pt`: Thorp et al. (2025) uncertainty model (diffusion)
  - `populationModelA24.pt`: Alsing et al. (2024) population model (diffusion)
  - `populationModelT25.pt`: Thorp et al. (2025) population model (diffusion)
  - `stellar_mass_emulator.pt`: Emulator for stellar mass remaining
  - `stellar_mass_parameter_shift-scale.pkl`: Parameter shift/scale for mass emulator
  - `emline_speculator_models/`: Speculator model for emission lines
  - `emline_bandpass_models/`: Generalized GMM parameters for emission line bandpass models
    - `COSMOS`: Filter set from COSMOS2020; two-component GGMM
    - `LSST`: Filter set from [LSST](https://github.com/lsst/throughputs) (baseline v1.9); six-component GGMM
  - `photulator_models/`: Photometry emulators
    - `COSMOS`: Filter set from COSMOS2020
    - `LSST`: Filter set from [LSST](https://github.com/lsst/throughputs) (baseline v1.9)
