<!-- markdownlint-disable -->

<a href="../../docs/pop_cosmos/catalogue#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pop_cosmos.catalogue`

**Global Variables**
---------------
- **COSMOS_FLUX_SOFTENING**

---

<a href="../../docs/pop_cosmos/catalogue/NoiseModel#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NoiseModel`

Class that generates noise and adds it to the mock fluxes.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `flowfusion.diffusion.PopulationModelDiffusionConditional` | Trained uncertainty model |
| `magnitude_min` | `torch.Tensor` | Minimum allowable input magnitudes. Used with `magnitude_max` to clamp the inputs |
| `magnitude_max` | `torch.Tensor` | Maximum allowable input magnitudes |
| `log_error_min` | `torch.Tensor` | Minimum allowable flux error. Used with `log_error_max` to clamp the outputs. Units are natural logarithm of maggies |
| `log_error_max` | `torch.Tensor` | Maximum allowable flux error |
| `error_floor` | `torch.Tensor` | Fractional error floor to be added in quadrature |
| `zp_star` | `torch.Tensor` | Zero-point correction assumed in training |
| `n_bands` | `int` | Number of photometric passbands |
| `f_b` | `torch.Tensor` | Flux softening parameter in nanomaggies |
| `nine_log_ten` | `torch.Tensor` | Conversion factor, 9*ln(10) |

<a href="../../docs/pop_cosmos/catalogue/__init__#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model,
    log_error_min=None,
    log_error_max=None,
    magnitude_min=None,
    magnitude_max=None,
    n_bands=26,
    noise_floor=0.0,
    zp_star=1.0,
    f_b=None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `flowfusion.diffusion.PopulationModelDiffusionConditional` | - | Diffusion model |
| `log_error_min` | `torch.Tensor`, optional | `None` | Minimum allowable flux error (as natural log of maggies). Default is `None` (`-inf`) |
| `log_error_max` | `torch.Tensor`, optional | `None` | Maximum allowable flux error. Default is `None` (`inf`) |
| `magnitude_min` | `torch.Tensor`, optional | `None` | Minimum allowable input asinh magnitudes. Default is `None` (`-inf`) |
| `magnitude_max` | `torch.Tensor`, optional | `None` | Maximum allowable input asinh magnitudes. Default is `None` (`inf`) |
| `n_bands` | `int`, optional | `26` | Number of photometric bands |
| `noise_floor` | `torch.Tensor`, optional | `0.0` | Fractional noise floor to be added in quadrature. Default is 0.0 in all bands |
| `zp_star` | `torch.Tensor`, optional | `1.0` | Fractional zero-point correction assumed in training. Default is 1.0 in all bands |
| `f_b` | `torch.Tensor`, optional | `None` | Flux softening parameter in nanomaggies |

---

<a href="../../docs/pop_cosmos/catalogue/noise_realization#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `noise_realization`

```python
noise_realization(
    n_noise,
    n_sigma,
    fluxes,
    asinh_magnitudes,
    zero_points=None,
    emission_line_errors=None
)
```

Generate a noise realisation from the model.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_noise` | `torch.Tensor` | - | Base random draws to be transformed into flux uncertainties |
| `n_sigma` | `torch.Tensor` | - | Base random draws to be transformed into flux errors |
| `fluxes` | `torch.Tensor` | - | True model fluxes in nanomaggies (without zero-point correction) |
| `asinh_magnitudes` | `torch.Tensor` | - | Zero-point corrected asinh magnitudes |
| `zero_points` | `torch.Tensor`, optional | `None` | Fractional zero-point corrections |
| `emission_line_errors` | `torch.Tensor`, optional | `None` | Uncertainties in emission line strength in nanomaggies |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `noisy_fluxes` | `torch.Tensor` | Noisy, zero-point corrected `fluxes` in nanomaggies |
| `flux_sigmas` | `torch.Tensor` | Total flux uncertainties (incl. em. lines) in nanomaggies |

---

<a href="../../docs/pop_cosmos/catalogue/CatalogueGenerator#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CatalogueGenerator`

Generates mock catalogues and SPS parameters according to a trained diffusion model.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `param_names` | `list of str` | List of parameter names in the order they are handled |
| `n_params` | `int` | Number of SPS parameters |
| `power` | `torch.Tensor` | Power to raise parameters to for Speculator |
| `lower` | `torch.Tensor` | Lower limits for parameters |
| `upper` | `torch.Tensor` | Upper limits for parameters |
| `range` | `torch.Tensor` | Parameter ranges (used for prior transform) |
| `f_b` | `torch.Tensor` | Flux softening parameter |
| `noise_model` | `pop_cosmos.catalogue.NoiseModel` | Uncertainty model |
| `population_model` | `flowfusion.diffusion.PopulationModelDiffusion` | Population model |
| `zero_points` | `torch.Tensor` | Fractional zero point corrections |
| `emline_offsets` | `torch.Tensor` | Fractional emission line strength corrections |
| `ln_emline_scatters` | `torch.Tensor` | Fractional emission line strength standard deviations |
| `flux_emulator` | `speculator.PhotulatorModelStack` | Photometry emulators |
| `emission_line_flux_emulator` | `pop_cosmos.emlines.EmLineEmulator` | Emission line emulator |
| `noise_base` | `torch.distributions.Distribution` | Base distribution for the noise |
| `n_bands` | `int` | Number of photometric bands |

<a href="../../docs/pop_cosmos/catalogue/__init__#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    noise_model: NoiseModel,
    population_model: PopulationModelDiffusion,
    calibration_parameters,
    flux_emulator: PhotulatorModelStack,
    emission_line_flux_emulator: EmLineEmulator,
    f_b=array([0.0010796 , 0.00308505, 0.0050811 , 0.00087933, 0.00291287,
       0.00405923, 0.00262367, 0.00498927, 0.00333445, 0.00114648,
       0.00526336, 0.00399871, 0.00662031, 0.00471849, 0.00616338,
       0.00114765, 0.00429656, 0.0062214 , 0.0018218 , 0.00411816,
       0.00823126, 0.00890925, 0.010834  , 0.00792853, 0.00322935,
       0.0034519 ]),
    device='cuda',
    zmax=6.0
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_model` | `pop_cosmos.catalogue.NoiseModel` | - | Uncertainty model to use |
| `population_model` | `flowfusion.diffusion.PopulationModelDiffusion` | - | Population model to use |
| `calibration_parameters` | `tuple of torch.Tensor` | - | Tensors containing the fractional zero-point corrections, emission line strength corrections, and emission line scatter parameters |
| `flux_emulator` | `speculator.PhotulatorModelStack` | - | Photometry emulator stack |
| `emission_line_flux_emulator` | `pop_cosmos.emlines.EmLineEmulator` | - | Emission line emulator object |
| `f_b` | `torch.Tensor`, optional | `COSMOS_FLUX_SOFTENING` | Flux softening parameter for each band |
| `device` | `str` or `torch.device`, optional | `'cuda'` | Device for the model to live on |
| `zmax` | `float`, optional | `6.0` | Maximum redshift assumed by the model |

---

<a href="../../docs/pop_cosmos/catalogue/emulator_parameter_transform#L375"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `emulator_parameter_transform`

```python
emulator_parameter_transform(theta)
```

Transform the parameter array for the emulators.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `theta` | `torch.Tensor` | SPS parameters to transform |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `theta_transformed` | `torch.Tensor` | Parameter tensor with the dust parameter square-rooted |

---

<a href="../../docs/pop_cosmos/catalogue/forward#L431"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(base_samples_noise, base_samples_sigma, base_samples_phi)
```

Generates mock catalogue without applying selection.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_samples_noise` | `torch.Tensor` | Base samples for the error model |
| `base_samples_sigma` | `torch.Tensor` | Base samples for the uncertainty model |
| `base_samples_phi` | `torch.Tensor` | Base samples for the population model |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `noisy_fluxes` | `torch.Tensor` | Noisy model fluxes in maggies |
| `noisy_magnitudes` | `torch.Tensor` | Noisy model magnitudes (logarithmic, AB system) |
| `noisy_asinh_magnitudes` | `torch.Tensor` | Noisy model magnitudes (asinh, AB system) |
| `flux_sigmas` | `torch.Tensor` | Flux uncertainties in maggies |
| `theta_samples` | `torch.Tensor` | SPS parameter array |
| `model_fluxes` | `torch.Tensor` | Zero-point-corrected noiseless model fluxes |

---

<a href="../../docs/pop_cosmos/catalogue/generate_base_samples#L334"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate_base_samples`

```python
generate_base_samples(nsamples)
```

Creates base draws for the catalogue model.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `nsamples` | `int` | Number of samples to generate |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `base_samples_noise` | `torch.Tensor` | Base samples for the error model |
| `base_samples_sigma` | `torch.Tensor` | Base samples for the uncertainty model |
| `base_samples_phi` | `torch.Tensor` | Base samples for the population model |

---

<a href="../../docs/pop_cosmos/catalogue/phi2theta#L357"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `phi2theta`

```python
phi2theta(phi)
```

Converts latent parameters to SPS parameters.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `phi` | `torch.Tensor` | Input parameters in latent space |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `theta` | `torch.Tensor` | SPS parameters |

---

<a href="../../docs/pop_cosmos/catalogue/selection_cut#L391"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `selection_cut`

```python
selection_cut(
    noisy_fluxes,
    noisy_magnitudes,
    noisy_flux_sigmas,
    magnitude_limits,
    flux_cut=False,
    snr_cut=False
)
```

Apply a selection cut to mock catalogue.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noisy_fluxes` | `torch.Tensor` | - | Tensor of noisy fluxes in maggies |
| `noisy_magnitudes` | `torch.Tensor` | - | Tensor of noisy (asinh) magnitudes |
| `noisy_flux_sigmas` | `torch.Tensor` | - | Tensor of flux errors in maggies |
| `magnitude_limits` | `torch.Tensor` | - | Tensor of magnitude limits to apply |
| `flux_cut` | `bool`, optional | `False` | If `True`, requires flux to be positive |
| `snr_cut` | `bool`, optional | `False` | If `True`, requires SNR>0 in all bands |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `selection` | `torch.Tensor` | Boolean tensor with same shape as the inputs, indicating whether a galaxy passes/fails selection |

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._