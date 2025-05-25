<!-- markdownlint-disable -->

<a href="../pop_cosmos/emlines.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `emlines`

---

<a href="../pop_cosmos/emlines.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_lines_model`

```python
load_lines_model(
    dir_spsmodels,
    n_parameters,
    filternames,
    device,
    dir_linegmms,
    n_hidden=[128, 128, 128]
)
```

Load a Speculator-based emission line emulator.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dir_spsmodels` | `str` | - | Path to a trained Speculator model |
| `n_parameters` | `int` | - | Number of SPS parameters |
| `filternames` | `list of str` | - | List of filter names |
| `device` | `str` or `torch.device` | - | Device (`'cpu'`) or (`'cuda'`) for the model |
| `dir_linegmms` | `str` | - | Path to a directory containing bandpass models |
| `n_hidden` | `list of int`, optional | `[128, 128, 128]` | Number of hidden units per layer |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `n_lines` | `int` | Number of emission lines loaded |
| `speculator_emlines` | `speculator.Speculator` | Speculator model |
| `line_lambdas_selected` | `numpy.array` | List of emission line wavelengths |
| `line_idx_selected` | `numpy.array` | List of line indexes in the loaded line list |
| `gengmm_amps` | `numpy.array` | Generalised Gaussian amplitudes for bandpass models |
| `line_names` | `numpy.array` | List of emission line names |
| `gengmm_locs` | `numpy.array` | Generalised Gaussian locations for bandpass models |
| `gengmm_sigs` | `numpy.array` | Generalised Gaussian scales for bandpass models |
| `gengmm_betas` | `numpy.array` | Generalised Gaussian shapes for bandpass models |

### See Also

- `EmLineEmulator` : Emission line emulator class

---

<a href="../pop_cosmos/emlines.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gennorm_pdf`

```python
gennorm_pdf(x, beta, sigma, mean)
```

Torch implementation of the generalized normal distribution PDF.

Based on https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Inputs to evaluate the PDF at |
| `beta` | `torch.Tensor` | Shape parameter |
| `sigma` | `torch.Tensor` | Scale parameter |
| `mean` | `torch.Tensor` | Location parameter |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `p` | `torch.Tensor` | Probability density evaluated at `x` |

---

<a href="../pop_cosmos/emlines.py#L279"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EmLineEmulator`

Emission line emulator class.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `filternames` | `list of str` | List of filter names |
| `n_parameters` | `int` | Number of SPS parameters |
| `n_hidden` | `list of int` | Number of hidden units per layer |
| `device` | `str` or `torch.device` | Device the model is on |
| `saved_model_dir` | `str` | Path the Speculator model was taken from |
| `gengmm_dir` | `str` | Path the bandpass models were taken from |
| `speculator_emlines` | `speculator.Speculator` | Speculator model |
| `line_idx_selected` | `torch.Tensor` | Tensor of line indices used |
| `gengnn_amps` | `torch.Tensor` | Tensor of generalised normal amplitudes |
| `gengnn_locs` | `torch.Tensor` | Tensor of generalised normal locations |
| `gengnn_sigs` | `torch.Tensor` | Tensor of generalised normal scales |
| `gengnn_betas` | `torch.Tensor` | Tensor of generalised normal shapes |
| `line_lambdas_selected` | `torch.Tensor` | Tensor of line wavelengths used |

<a href="../pop_cosmos/emlines.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    saved_model_dir,
    gengmm_dir,
    device,
    filternames,
    n_parameters,
    n_hidden=[128, 128, 128]
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `saved_model_dir` | `str` | - | Path to the Speculator model |
| `gengmm_dir` | `str` | - | Path to bandpass models |
| `device` | `str` or `torch.device` | - | Device to place the model on |
| `filternames` | `list of str` | - | List of bandpass names |
| `n_parameters` | `int` | - | Number of SPS parameters |
| `n_hidden` | `list of int`, optional | `[128, 128, 128]` | Number of hidden units per network layer |

---

<a href="../pop_cosmos/emlines.py#L385"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_bandpasses`

```python
apply_bandpasses(
    filternames,
    em_line_strengths_,
    theta,
    line_lambdas,
    gengnn_locs,
    gengnn_sigs,
    gengnn_betas,
    gengnn_amps
)
```

Applies bandpass models to a list of lines.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `filternames` | `list of str` | List of filternames to compute fluxes in |
| `em_line_strengths_` | `torch.Tensor` | Line strengths in solar luminosity per unit mass formed |
| `theta` | `torch.Tensor` | SPS parameters |
| `line_lambdas` | `torch.Tensor` | List of line wavelengths |
| `gengnn_locs` | `torch.Tensor` | Bandpass model locations |
| `gengnn_sigs` | `torch.Tensor` | Bandpass model scales |
| `gengnn_betas` | `torch.Tensor` | Bandpass model shapes |
| `gengnn_amps` | `torch.Tensor` | Bandpass model amplitudes |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `em_line_phot_` | `torch.Tensor` | Flux contributions of emission lines in nanomaggies |

---

<a href="../pop_cosmos/emlines.py#L464"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(theta)
```

Computes emission line fluxes given SPS parameters.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `theta` | `torch.Tensor` | Tensor of SPS parameters |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `emline_bandpass_phot` | `torch.Tensor` | Emission line flux contributions to each bandpass. Units of nanomaggies (AB system) |

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._