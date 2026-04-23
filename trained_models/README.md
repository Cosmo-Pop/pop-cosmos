# trained_models

This directory contains a variety of binaries containing trained models, and the components thereof.

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
    - `Blast/`: Filter set from [Blast](https://blast.scimma.org); seven-component GGMM
    - `COSMOS/`: Filter set from COSMOS2020; two-component GGMM
    - `DECaLS/`: Filter set from DECaLS + _WISE_; seven-component GGMM
    - `KiDS/`: Filter set from KiDS; three-component GGMM
    - `LSST/`: Filter set from [LSST](https://github.com/lsst/throughputs) (baseline v1.9); six-component GGMM
    - `Roman/`: Filter set from [Roman](https://roman.gsfc.nasa.gov/science/WFI_technical.html) (average over 18 SCAs); five-component GGMM
  - `photulator_models/`: Photometry emulators
    - `Blast/`: Filter set from [Blast](https://blast.scimma.org); $z_\text{max}=1.5$
    - `COSMOS/`: Filter set from COSMOS2020; $z_\text{max}=6.0$
      - `restframe_models/`: Emulators for rest-frame absolute magnitudes in $NUVrJ$; $z_\text{max}=6.0$ 
    - `DECaLS/`: Filter set from DECaLS + _WISE_; $z_\text{max}=4.5$
    - `KiDS/`: Filter set from KiDS; $z_\text{max}=4.5$; OmegaCAM bands included with (`*_atm`) & without atmospheric transmission
    - `LSST/`: Filter set from [LSST](https://github.com/lsst/throughputs) (baseline v1.9); $z_\text{max}=6.0$
    - `Roman/`: Filter set from [Roman](https://roman.gsfc.nasa.gov/science/WFI_technical.html) (average over 18 SCAs); $z_\text{max}=6.0$
