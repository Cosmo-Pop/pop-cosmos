<!-- markdownlint-disable -->

<a href="../pop_cosmos/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils.py`

---

<a href="../pop_cosmos/utils.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_derived_quantities`

```python
compute_derived_quantities(thetas)
```

PyTorch routine for generating useful derived parameters from a tensor of SPS parameters.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `thetas` | `torch.Tensor` | Sixteen-column tensor containing the base SPS parameters for some model galaxies |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `log10M_formed` | `torch.Tensor` | Base 10 logarithm of stellar mass formed. Units of solar masses |
| `mw_age` | `torch.Tensor` | Mass weighted age. Units of Gyr |
| `log10SFR` | `torch.Tensor` | Base 10 logarithm of star formation rate. Units of solar masses per year, averaged over the past 100 Myr |
| `log10sSFR` | `torch.Tensor` | Base 10 logarithm of specific star formation rate. Units of solar masses per year per unit solar mass formed |

### See Also

- `compute_mass_remaining` : Routine for correcting for mass loss
- `mass_weighted_age` : Underlying routine for computing mass weighted age
- `specific_star_formation_rate` : Underlying routine for computing sSFR and SFR
- `catalogue.CatalogueGenerator` : Class that generates the `thetas` used as input

---

<a href="../pop_cosmos/utils.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_mass_remaining`

```python
compute_mass_remaining(
    log10M_formed,
    log10sSFR,
    thetas,
    theta_shift,
    theta_scale,
    mass_fraction_emulator
)
```

PyTorch routine for computing a fraction of stellar mass remaining using a tensor of SPS parameters and an emulator for FSPS/Prospector.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `log10M_formed` | `torch.Tensor` | Base 10 logarithm of stellar mass formed. Units of solar masses |
| `log10sSFR` | `torch.Tensor` | Base 10 logarithm of specific star formation rate. Units of solar masses per year per unit solar mass formed |
| `thetas` | `torch.Tensor` | Sixteen-column tensor containing the base SPS parameters for some model galaxies |
| `theta_shift` | `torch.Tensor` | Shift to be applied to parameters before entering the emulator |
| `theta_scale` | `torch.Tensor` | Scale to be applied to parameters before entering the emulator |
| `mass_fraction_emulator` | `torch.nn.Sequential` | Emulator for the mass remaining fraction |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `log10M` | `torch.Tensor` | Base 10 logarithm of stellar mass remaining. Units of solar masses |
| `log10sSFR` | `torch.Tensor` | Base 10 logarithm of specific starformation rate. Units of solar masses per year per unit solar mass remaining |
| `Mfrac` | `torch.Tensor` | Fraction of stellar mass remaining |

---

<a href="../pop_cosmos/utils.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mass_weighted_age`

```python
mass_weighted_age(logsfr_ratios, z)
```

PyTorch routine for converting SFR ratios and redshift into a mass-weighted age.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `logsfr_ratios` | `torch.Tensor` | Six-column tensor containing the logarithm of the SFR ratios for a sample of model galaxies |
| `z` | `torch.Tensor` | One-column tensor containing the redshifts of the model galaxies |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `mw_age` | `torch.Tensor` | Mass weighted age. Units of Gyr |

---

<a href="../pop_cosmos/utils.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `specific_star_formation_rate`

```python
specific_star_formation_rate(logsfr_ratios, z)
```

PyTorch routine for converting SFR ratios and redshift into sSFR.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `logsfr_ratios` | `torch.Tensor` | Six-column tensor containing the logarithm of the SFR ratios for a sample of model galaxies |
| `z` | `torch.Tensor` | One-column tensor containing the redshifts of the model galaxies |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `log10sSFR` | `torch.Tensor` | Base 10 logarithm of the specific star formation rate (sSFR). Units of 1/yr |

### Notes

The star formation rate is averaged over the last 100 Myr of a galaxy's life. This does not include a correction for mass loss. The quantity returned has the definition SFR/M_form, i.e. SFR per unit solar mass formed.

---

<a href="../pop_cosmos/utils.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `chs_two_segment`

```python
chs_two_segment(x, x0, x1, x2, y0, y1, y2, s0, s1, s2)
```

Generic two-segment Cubic hermite spline.

Defined by knot positions (x0, x1, x2), knot values (y0, y1, y2), and slopes (s0, s1, s2).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `np.array` | Coordinates to evaluate curve at |
| `x0` | `float` | Position of first knot |
| `x1` | `float` | Position of second knot |
| `x2` | `float` | Position of third knot |
| `y0` | `float` | Curve at first knot |
| `y1` | `float` | Curve at second knot |
| `y2` | `float` | Curve at third knot |
| `s0` | `float` | Slope at first knot |
| `s1` | `float` | Slope at second knot |
| `s2` | `float` | Slope at third knot |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `y` | `np.array` | Curve evaluated at `x` |

---

<a href="../pop_cosmos/utils.py#L242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mass_limit_rolling`

```python
mass_limit_rolling(
    z,
    z1=0.7948,
    z2=6.0,
    M0=9.7108,
    M1=11.6535,
    M2=11.3251,
    dMdz0=6.4555
)
```

Rolling upper mass limit based on the 99.5th percentile of Ch.1<26 model draws.

Fits the percentile (computed as a rolling function of z in redshift window of size 0.5), using a two segment cubic hermite spline with an asymptote at z>4.0.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `z` | `np.array` | - | Redshift array to compute the mass limit at |
| `z1` | `float`, optional | `0.7948` | Peak redshift (redshift with the largest limiting mass) |
| `z2` | `float`, optional | `6.0` | Asymptote redshift (transition to a constant mass limit) |
| `M0` | `float`, optional | `9.7108` | Intercept mass (mass limit at z = 0) |
| `M1` | `float`, optional | `11.6535` | Peak mass (mass limit at z = `z1`) |
| `M2` | `float`, optional | `11.3251` | Asymptote mass (mass limit at z >= `z2`) |
| `dMdz0` | `float`, optional | `6.4555` | Intercept slope (dM_lim/dz at z = 0) |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `M` | `np.array` | Limiting mass remaining, log10(M_lim(z) / M_sun), evaluated at `M` |

---

<a href="../pop_cosmos/utils.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mass_completeness_rolling`

```python
mass_completeness_rolling(
    z,
    z1=0.7404,
    M0=6.5184,
    M1=8.5429,
    M2=9.6966,
    dMdz0=6.9873,
    dMdz1=0.5549,
    dMdz2=0.2956
)
```

Rolling completeness limit.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `z` | `np.array` | - | Redshift array to compute the mass limit at |
| `z1` | `float`, optional | `0.7404` | Central spline knot in redshift |
| `M0` | `float`, optional | `6.5184` | Completeness limit at z = 0 |
| `M1` | `float`, optional | `8.5429` | Completeness limit at z = `z1` |
| `M2` | `float`, optional | `9.6966` | Completeness limit at z = 6 |
| `dMdz0` | `float`, optional | `6.9873` | Slope at z = 0 |
| `dMdz1` | `float`, optional | `0.5549` | Slope at z = `z1` |
| `dMdz2` | `float`, optional | `0.2956` | Slope at z = 6 |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `M` | `np.array` | Limiting mass remaining, log10(M_lim(z) / M_sun), evaluated at `M` |

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._