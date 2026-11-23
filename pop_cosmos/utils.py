import os
import sys
import importlib
import torch
import numpy as np
from astropy.cosmology import Planck18
import astropy.units as u
from astropy.io import ascii
from .constants import PLANCK18_H0, PLANCK18_OMEGAM, PLANCK18_TH, SPEED_OF_LIGHT
from dataclasses import dataclass

def adachi_kasai_phi(x):
    """
    Pade' approximant formula from Adachi & Kasai (2012).

    Parameters
    ----------
    x : torch.Tensor
        Input parameter.

    Returns
    -------
    phi : torch.Tensor
        Formula evaluated at `x`.
    """
    return (1.0 + 1.320*x + 0.4415*x**2. + 0.02656*x**3.)/(1.0 + 1.392*x + 0.5121*x**2. + 0.03944*x**3.)

def comoving_distance(z, H0=PLANCK18_H0, OmegaM=PLANCK18_OMEGAM):
    """
    PyTorch routine for computing comoving distance in flat LCDM.

    Based on the approximation from Adachi & Kasai (2012).

    Parameters
    ----------
    z : torch.Tensor
        Redshifts to evaluate for.
    H0 : float, optional
        Hubble constant in km/s/Mpc. Defaults to Planck 2018.
    OmegaM : float, optional
        Matter density. Defaults to Planck 2018.

    Returns
    -------
    Dc : torch.Tensor
        Comoving distance in Mpc.
    """
    x0 = (1. - OmegaM)/OmegaM
    xz = x0/(1. + z)**3.
    phi0 = adachi_kasai_phi(x0)
    phiz = adachi_kasai_phi(xz)
    DH = SPEED_OF_LIGHT/H0
    return  2.0 * DH * (phi0 - phiz/torch.sqrt(1. + z)) / np.sqrt(OmegaM)

def comoving_volume_element(z, H0=PLANCK18_H0, OmegaM=PLANCK18_OMEGAM):
    """
    PyTorch routine for computing comoving volume element.

    Parameters
    ----------
    z : torch.Tensor
        Redshifts to evaluate for.
    H0 : float, optional
        Hubble constant in km/s/Mpc. Defaults to Planck 2018.
    OmegaM : float, optional
        Matter density. Defaults to Planck 2018.

    Returns
    -------
    dVc : torch.Tensor
        Comoving volume element in Mpc^3/steradian.

    See Also
    --------
    `comoving_distance` : Approximate distance integral.
    """
    Dc = comoving_distance(z, H0, OmegaM)
    DH = SPEED_OF_LIGHT/H0
    Ez = torch.sqrt(OmegaM*(1. + z)**3. + (1. - OmegaM))
    return DH*Dc*Dc/Ez

def distance_modulus(z, H0=PLANCK18_H0, OmegaM=PLANCK18_OMEGAM):
    """
    PyTorch routine for computing distance modulus.

    Parameters
    ----------
    z : torch.Tensor
        Redshifts to evaluate for.
    H0 : float, optional
        Hubble constant in km/s/Mpc. Defaults to Planck 2018.
    OmegaM : float, optional
        Matter density. Defaults to Planck 2018.

    Returns
    -------
    mu : torch.Tensor
        Distance modulus in magnitudes.

    See Also
    --------
    `comoving_distance` : Approximate distance integral.
    """
    DL = (1. + z)*comoving_distance(z, H0, OmegaM)
    return 5.0*torch.log10(DL) + 25.0 

def age_of_universe(z, tH=PLANCK18_TH, OmegaM=PLANCK18_OMEGAM):
    """
    PyTorch routine for computing age of Universe in flat LCDM.

    Equation 13.20 in Peebles (1993).

    Parameters
    ----------
    z : torch.Tensor
        Redshifts to evaluate for.
    tH : float, optional
        Hubble time in Gyr. Defaults to Planck 2018.
    OmegaM : float, optional
        Matter density. Defaults to Planck 2018.

    Returns
    -------
    t : torch.Tensor
        Age of the Universe evaluated at `z`.
    """
    OmegaL = 1. - OmegaM
    t = tH * 2.0 * torch.asinh(np.sqrt(OmegaL/OmegaM)*(1.+z)**-1.5) / (3.0*np.sqrt(OmegaL))
    return t


def compute_derived_quantities(thetas, use_astropy=False, return_bins=False):
    """
    PyTorch routine for generating useful derived parameters from
    a tensor of SPS parameters.

    Parameters
    ----------
    thetas : torch.Tensor
        Sixteen-column tensor containing the base SPS parameters for some
        model galaxies.
    use_astropy : bool, optional
        If `True`, computes cosmological quantities using AstroPy.
        Default is `False`, which uses an approximation in PyTorch.
    return_bins : bool, optional
        If `True`, returns fraction of mass formed, and SFR in solar masses,
        per year, for each bin of the star formation history (SFH). 
        See `specific_star_formation_rate`. Default is `False`.

    Returns
    -------
    log10M_formed : torch.Tensor
        Base 10 logarithm of stellar mass formed. Units of solar masses.
    mw_age : torch.Tensor
        Mass weighted age. Units of Gyr.
    log10SFR : torch.Tensor
        Base 10 logarithm of star formation rate.
        Units of solar masses per year, averaged over the past 100 Myr.
    log10sSFR : torch.Tensor
        Base 10 logarithm of specific star formation rate.
        Units of solar masses per year per unit solar mass formed.
    M_frac_bins : torch.Tensor, optional
        Fraction of total mass formed per bin of the 7-bin SFH.
        Only returned if `return_bins` is `True`.
    log10SFR_bins : torch.Tensor, optional
        Average SFR in each bin of the 7-bin SFH. Units of solar masses
        per year. Only returned if `return_bins` is `True`.
    t_edge_bins : torch.Tensor, optional
        Edges of SFH bins in lookback time, with units of Gyr.
        Only returned if `return_bins` is `True`.
    
    See Also
    --------
    `compute_mass_remaining` : Routine for correcting for mass loss.
    `mass_weighted_age` : Underlying routine for computing mass weighted age.
    `specific_star_formation_rate` : Underlying routine for computing sSFR and SFR.
    `catalogue.CatalogueGenerator` : Class that generates the `thetas` used as input.
    """
    if use_astropy:
        z = thetas[:,-1].detach().cpu().numpy()
        log10M_formed = -0.4*(thetas[:,0] - torch.from_numpy(Planck18.distmod(z).value))
        mw_age = mass_weighted_age(thetas[:,2:8], z, use_astropy=True)
        if return_bins:
            log10sSFR, M_frac_bins, t_edge_bins = specific_star_formation_rate(thetas[:,2:8], z, use_astropy=True, return_bins=True)
        else:
            log10sSFR = specific_star_formation_rate(thetas[:,2:8], z, use_astropy=True)
    else:
        log10M_formed = -0.4*(thetas[:,0] - distance_modulus(thetas[:,-1]))
        mw_age = mass_weighted_age(thetas[:,2:8], thetas[:,-1])
        if return_bins:
            log10sSFR, M_frac_bins, t_edge_bins = specific_star_formation_rate(thetas[:,2:8], thetas[:,-1], return_bins=True)
        else:
            log10sSFR = specific_star_formation_rate(thetas[:,2:8], thetas[:,-1])

    if return_bins:
        # SFH bin widths in Gyr
        t_width_bins = t_edge_bins[:,1:] - t_edge_bins[:,:-1]
        # SFR per bin in solar masses per year
        log10SFH = log10M_formed.unsqueeze(-1) + torch.log10(M_frac_bins) - torch.log10(t_width_bins) - 9.0
        return (log10M_formed, mw_age, log10sSFR + log10M_formed, log10sSFR, 
                M_frac_bins, log10SFH, t_edge_bins)
    else:
        return log10M_formed, mw_age, log10sSFR + log10M_formed, log10sSFR

def compute_mass_remaining(
        log10M_formed, 
        log10sSFR, 
        thetas, 
        theta_shift, 
        theta_scale, 
        mass_fraction_emulator
    ):
    """
    PyTorch routine for computing a fraction of stellar mass remaining
    using a tensor of SPS parameters and an emulator for FSPS/Prospector.

    Parameters
    ----------
    log10M_formed : torch.Tensor
        Base 10 logarithm of stellar mass formed. Units of solar masses.
    log10sSFR : torch.Tensor
        Base 10 logarithm of specific star formation rate.
        Units of solar masses per year per unit solar mass formed.
    thetas : torch.Tensor
        Sixteen-column tensor containing the base SPS parameters for some
        model galaxies.
    theta_shift : torch.Tensor
        Shift to be applied to parameters before entering the emulator.
    theta_scale : torch.Tensor
        Scale to be applied to parameters before entering the emulator.
    mass_fraction_emulator : torch.nn.Sequential
        Emulator for the mass remaining fraction.

    Returns
    -------
    log10M : torch.Tensor
        Base 10 logarithm of stellar mass remaining. Units of solar masses.
    log10sSFR : torch.Tensor
        Base 10 logarithm of specific starformation rate.
        Units of solar masses per year per unit solar mass remaining.
    M_frac_remain : torch.Tensor
        Fraction of stellar mass remaining.
    """

    M_frac_remain = mass_fraction_emulator((thetas - theta_shift) / theta_scale)[:,0]
    log10M = log10M_formed + torch.log10(M_frac_remain)
    log10sSFR = log10sSFR - torch.log10(M_frac_remain)

    return log10M, log10sSFR, M_frac_remain

def mass_weighted_age(logsfr_ratios, z, use_astropy=False):
    """
    PyTorch routine for converting SFR ratios and redshift into 
    a mass-weighted age.

    Parameters
    ----------
    logsfr_ratios : torch.Tensor
        Six-column tensor containing the logarithm of the SFR ratios
        for a sample of model galaxies.
    z : torch.Tensor
        One-column tensor containing the redshifts of the model galaxies.
    use_astropy : bool, optional
        If `True`, computes cosmological quantities using AstroPy.
        Default is `False`, which uses an approximation in PyTorch.

    Returns
    -------
    mw_age : torch.Tensor
        Mass weighted age. Units of Gyr.
    """
    # number of samples and bins
    n_samples = logsfr_ratios.shape[0]
    n_bins = logsfr_ratios.shape[1] + 1
    
    # age of the universe at the given redshift
    if use_astropy:
        tuniv = torch.from_numpy(Planck18.age(z).value).unsqueeze(-1) #Gyr
    else:
        tuniv = age_of_universe(z).unsqueeze(-1) # Gyr
    
    # stellar age time grid
    # this first line does a tensorized logspace, since torch doesn't provide one
    log_age_inner_edges = np.log10(0.1)*torch.ones(n_samples, n_bins-2) + torch.arange(0, n_bins-2, 1)*(torch.log10(0.85*tuniv) - np.log10(0.1))/(n_bins-3.0) # logspaced to 0.85 tuniv
    age_edges = torch.hstack([torch.tensor([[0.0, 0.03]])*torch.ones(n_samples,2),
        10**log_age_inner_edges, tuniv]) 
    
    # bin widths
    bin_widths = (age_edges[:,1:] - age_edges[:,:-1])
    
    # mean age per bin (integrals over the bin)                      
    mean_age_per_bin = (age_edges[:,1:]**2 - age_edges[:,:-1]**2) / 2.0
    
    # star formation rate in each bin
    sfr = torch.ones(n_samples, n_bins)
    sfr[:,1:] = 10**torch.cumsum(-logsfr_ratios, dim=1)
    
    # normalize the sfr
    normalized_sfr = sfr / torch.sum(sfr*bin_widths, dim=1).unsqueeze(1)
    
    # compute mass weighted age
    mw_age = torch.sum(normalized_sfr * mean_age_per_bin, dim=1)
    
    return mw_age

def specific_star_formation_rate(logsfr_ratios, z, use_astropy=False, return_bins=False):
    """
    PyTorch routine for converting SFR ratios and redshift into sSFR.

    Parameters
    ----------
    logsfr_ratios : torch.Tensor
        Six-column tensor containing the logarithm of the SFR ratios
        for a sample of model galaxies.
    z : torch.Tensor
        One-column tensor containing the redshifts of the model galaxies.
    use_astropy : bool, optional
        If `True`, computes cosmological quantities using AstroPy.
        Default is `False`, which uses an approximation in PyTorch.
    return_bins : bool, optional
        If `True`, returns fraction of mass formed, and bin edges in Gyr,
        for each bin of the star formation history (SFH). Can be used to
        compute the SFR in each bin, when combined with the total mass
        formed. See `compute_derived_quantities`. Default is `False`.

    Returns
    -------
    log10sSFR : torch.Tensor
        Base 10 logarithm of the specific star formation rate (sSFR).
        Units of 1/yr. Averaged over the past 100 Myr.
    M_frac_bins : torch.Tensor, optional
        Fraction of total mass formed per bin of the 7-bin SFH.
        Only returned if `return_bins` is `True`.
    t_edge_bins : torch.Tensor, optional
        Edges of SFH bins in lookback time, with units of Gyr.
        Only returned if `return_bins` is `True`.
        
    Notes
    -----
    The sSFR is averaged over the last 100 Myr of a galaxy's life.
    This does not include a correction for mass loss. The quantity
    returned has the definition SFR/M_form, i.e. SFR per unit solar mass formed.
    """
    # number of samples and bins
    n_samples = logsfr_ratios.shape[0]
    n_bins = logsfr_ratios.shape[1] + 1
    
    # age of the universe at the given redshift
    if use_astropy:
        tuniv = torch.from_numpy(Planck18.age(z).value).unsqueeze(-1) #Gyr
    else:
        tuniv = age_of_universe(z).unsqueeze(-1) # Gyr
    
    # stellar age time grid
    # this first line does a tensorized logspace, since torch doesn't provide one
    log_age_inner_edges = np.log10(0.1)*torch.ones(n_samples, n_bins-2) + torch.arange(0, n_bins-2, 1)*(torch.log10(0.85*tuniv) - np.log10(0.1))/(n_bins-3.0) # logspaced to 0.85 tuniv
    age_edges = torch.hstack([torch.tensor([[0.0, 0.03]])*torch.ones(n_samples,2),
        10**log_age_inner_edges, tuniv])
    
    # bin widths in Gyr
    bin_widths = (age_edges[:,1:] - age_edges[:,:-1])
    
    # star formation rate in each bin
    sfr = torch.ones(n_samples, n_bins)
    sfr[:,1:] = 10**torch.cumsum(-logsfr_ratios, dim=1)
    
    # normalize the sfr
    normalized_sfr = sfr / torch.sum(sfr*bin_widths, dim=1).unsqueeze(1)
    
    # fractional mass formed per bin
    mass_formed = normalized_sfr * bin_widths
    
    # specific star formation rate
    log10sSFR = torch.clip(torch.log10((mass_formed[:,0] + mass_formed[:,1] + 1e-20) / 0.1) - 9.0, -18, 0)
    
    if return_bins:
        return log10sSFR, mass_formed, age_edges
    else:
        return log10sSFR

def chs_two_segment(x, x0, x1, x2, y0, y1, y2, s0, s1, s2):
    """
    Generic two-segment Cubic hermite spline.
    
    Defined by knot positions (x0, x1, x2), knot values (y0, y1, y2), and slopes (s0, s1, s2).

    Parameters
    ----------
    x : np.array
        Coordinates to evaluate curve at.
    x0 : float
        Position of first knot.
    x1 : float
        Position of second knot.
    x2 : float, optional
        Position of third knot.
    y0 : float
        Curve at first knot.
    y1 : float
        Curve at second knot.
    y2 : float, optional
        Curve at third knot.
    s0 : float
        Slope at first knot.
    s1 : float
        Slope at second knot.
    s2 : float, optional
        Slope at third knot.
        
    Returns
    -------
    y : np.array
        Curve evaluated at `x`.
    """
    # sort things as being above/below the central knot
    mask1 = x < x1
    mask2 = x >= x1
    # empty array for function values
    y = np.zeros_like(x)
    # t values
    t1 = (x[mask1] - x0)/(x1 - x0)
    t2 = (x[mask2] - x1)/(x2 - x1)
    # evaluate
    y[mask2] = y1*(2.0*t2**3 - 3.0*t2**2 + 1.0) + y2*(-2.0*t2**3 + 3.0*t2**2) + (t2**3 - 2.0*t2**2 + t2)*(x2 - x1)*s1 + (t2**3 - t2**2)*(x2 - x1)*s2
    y[mask1] = y0*(2.0*t1**3 - 3.0*t1**2 + 1.0) + y1*(-2.0*t1**3 + 3.0*t1**2) + (t1**3 - 2.0*t1**2 + t1)*(x1 - x0)*s0 + (t1**3 - t1**2)*(x1 - x0)*s1
    return y

def mass_limit_rolling(z, z1=0.7948, z2=6.0, M0=9.7108, M1=11.6535, M2=11.3251, dMdz0=6.4555):
    """
    Rolling upper mass limit based on the 99.5th percentile of Ch.1<26 model draws.
    
    Fits the percentile (computed as a rolling function of z in 
    redshift window of size 0.5), using a two segment cubic hermite spline
    with an asymptote at z>4.0.
    
    Parameters
    ----------
    z : np.array
        Redshift array to compute the mass limit at.
    z1 : float, optional
        Peak redshift (redshift with the largest limiting mass).
    z2 : float, optional
        Asymptote redshift (transition to a constant mass limit).
    M0 : float, optional
        Intercept mass (mass limit at z = 0).
    M1 : float, optional
        Peak mass (mass limit at z = `z1`).
    M2 : float, optional
        Asymptote mass (mass limit at z >= `z2`).
    dMdz0 : float, optional
        Intercept slope (dM_lim/dz at z = 0).
        
    Returns
    -------
    M : np.array
        Limiting mass remaining, log10(M_lim(z) / M_sun), evaluated at `M`.
    """
    
    M = chs_two_segment(z, 0.0, z1, z2, M0, M1, M2, dMdz0, 0.0, 0.0)
    return M

def mass_completeness_rolling(z, z1=0.7404, M0=6.5184, M1=8.5429, M2=9.6966, 
                              dMdz0=6.9873, dMdz1=0.5549, dMdz2=0.2956):
    """
    Rolling completeness limit.

    Parameters
    ----------
    z : np.array
        Redshift array to compute the mass limit at.
    z1 : float, optional
        Central spline knot in redshift.
    M0 : float, optional
        Completeness limit at z = 0.
    M1 : float, optional
        Completeness limit at z = `z1`.
    M2 : float, optional
        Completeness limit at z = 6.
    dMdz0 : float, optional
        Slope  at z = 0.
    dMdz1 : float, optional
        Slope  at z = `z1`.
    dMdz2 : float, optional
        Slope  at z = 6.
        
    Returns
    -------
    M : np.array
        Limiting mass remaining, log10(M_lim(z) / M_sun), evaluated at `M`.
    """
    M = chs_two_segment(z, 0.0, z1, 6.0, M0, M1, M2, dMdz0, dMdz1, dMdz2)
    return M

def initialise_fsps(
        pro_name='prospector_alpha_plus_params',
        pro_path='../sps_models/',
        fsps_path=None,
    ):
    """
    Setup Prospector/FSPS.

    Parameters
    ----------
    pro_name : str, optional
        Name of the Prospector params .py file.
    pro_path : str, optional
        Path to the Prospector params .py file.
    fsps_path : str, optional
        Path to the top-level of your own FSPS clone (i.e. your 'SPS_HOME').

    Returns
    -------
    pfile : module
        Module generated from the Prospector params file.
    obs : dict
        Empty Prospector-style observation dictionary returned by `pfile.load_obs`.
    sps : prospect.sources.SSPBasis
        SPS basis returned by `pfile.load_sps`.
    mod : prospect.models.sedmodel.SedModel
        Prospector SED model object returned by `pfile.load_model`.
    """
    # check for SPS_HOME
    if 'SPS_HOME' not in os.environ and fsps_path is None:
        raise RuntimeError('SPS_HOME environment variable unset and fsps_path not provided!')
    if fsps_path is not None:
        os.environ['SPS_HOME'] = fsps_path
    # import python-fsps if it seems like SPS_HOME is set
    import fsps

    # try to import the prospector param file
    _spec = importlib.util.spec_from_file_location(pro_name, pro_path + pro_name + '.py')
    pfile = importlib.util.module_from_spec(_spec)
    sys.modules['prospector_alpha_params'] = pfile
    _spec.loader.exec_module(pfile)

    # load from param file
    obs = pfile.load_obs(**pfile.run_params)
    sps = pfile.load_sps(**pfile.run_params)
    mod = pfile.load_model(**pfile.run_params)

    return pfile, obs, sps, mod


def generate_fsps_spec(
        thetas, 
        pfile, 
        obs, 
        sps, 
        mod,
        wavegrid=None,
        unit='Mgy',
        emlines=True,
        neb_continuum=True,
        progress=True
    ):
    """
    Generates spectra using Prospector/FSPS.

    Parameters
    ----------
    thetas : np.array
        Sixteen-column array containing the SPS parameters.
    pfile : module
        Module generated from the Prospector params file.
    obs : dict
        Empty Prospector-style observation dictionary returned by `pfile.load_obs`.
    sps : prospect.sources.SSPBasis
        SPS basis returned by `pfile.load_sps`.
    mod : prospect.models.sedmodel.SedModel
        Prospector SED model object returned by `pfile.load_model`.
    wavegrid : np.array, optional
        If provided, interpolates all spectra onto this wavelength grid
        in observer frame. Otherwise uses the default FSPS rest-frame
        wavelength grid.
    unit : str, optional
        Units to generate SEDs in. One of `('Mgy', 'nMgy', 'Jy', 'muJy')`.
    emlines : bool, optional
        If `False`, nebular emission lines are excluded from the spectra.
        Default is `True`.
    neb_continuum : bool, optional
        If `False`, the nebular continuum is excluded from the spectra.
        If both `emlines` and `neb_continuum` are `False`, nebular emission
        is switched off entirely (`add_neb_emission=False`). Default is `True`.
    progress : bool, optional
        If `True` shows a progress bar using `tqdm`.

    Returns
    -------
    wl : np.array
        Rest-frame FSPS wavelength grid.
    waves : np.array
        Observer-frame wavelength grid (per galaxy if `wavegrid=None`).
    spectra : np.array
        Observer-frame spectra in units of `unit`.

    See Also
    --------
    `initialise_fsps` : Sets up FSPS and generates `pfile`, `obs`, `sps`, `mod`.
    """
    # unit converson
    if unit == 'Mgy':
        unit_factor = 1.
    elif unit == 'nMgy':
        unit_factor = 1e9
    elif unit == 'muJy':
        unit_factor = 3.631e9
    elif unit == 'Jy':
        unit_factor = 3.631e3
    else:
        raise ValueError("Unrecognised units. Use: 'nMgy', 'Mgy', 'muJy', 'Jy'")

    # loop over parameter samples
    if progress:
        import tqdm
        loop = tqdm.trange(len(thetas))
    else:
        loop = range(len(thetas))
    waves   = [] # for observer-frame wavelengths
    spectra = [] # for observer-frame spectra
    for i in loop:
        # update parameters
        pfile.run_params['zred'] = thetas[i,-1]
        mod = pfile.load_model(**pfile.run_params)
        mod.params['zred'] = thetas[i,-1]
        mod.params['nebemlineinspec'] = np.array([emlines])
        mod.params['add_neb_continuum'] = np.array([neb_continuum])
        mod.params['add_neb_emission'] = np.array([emlines or neb_continuum])
        # generate spectrum
        spec, _, _ = mod.mean_model(thetas[i,0:-1], sps=sps, obs=obs) # spectrum
        # generate rest-frame wavelength grid if not done already
        if i == 0:
            wl, _, _ = sps.get_galaxy_spectrum() # rest-frame wavelength
        # interpolate spectrum onto grid if requested
        if wavegrid is not None:
            spec_obs = np.interp(wavegrid, wl*(1. + thetas[i,-1]), spec)
            spectra.append(spec_obs)
        else:
            waves.append(wl*(1. + thetas[i,-1]))
            spectra.append(spec)
    # apply unit conversion
    spectra = np.array(spectra)*unit_factor
    if wavegrid is not None:
        waves = wavegrid
    else:
        waves = np.array(waves)

    return wl, waves, spectra


#################################################
####### CLASSES FOR DEALING WITH MCMC CHAINS ####
#################################################

# hardcoded params here follow MCMC order:
TAU2_INDEX = 8
AGN_SLICE = slice(11, 13)

# units
LSUN = 3.846e33   # erg/s [cgs units]
JY_CGS = 1e-23    # erg/s/cm^2/Hz

#paths
PRO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sps_models', '')
PRO_NAME = 'prospector_alpha_plus_params'
SPECULATOR_RANGES = ['400_1000', '1000_4000', '4000_11000', '11000_30000', '30000_100000']


@dataclass
class Spectrum:
    """
    Spectra for every chain sample of one galaxy on an observed-frame grid.

    Attributes
    ----------
    wl : np.ndarray
        Observed-frame wavelength grid in Angstroms, shape ``(n_wave,)``.
    flux : np.ndarray
        Flux in nanomaggies for each drawn sample, shape ``(n_draw, n_wave)``.
    label : str, optional
        Name of the model that generated the spectra. Default is ``''``.
    """
    wl: np.ndarray      # observed-frame wavelength [AA], (n_wave,)
    flux: np.ndarray    # flux [nMgy], (n_draw, n_wave)
    label: str = ''

    def percentiles(self, q=(16, 50, 84)):
        """
        Percentiles of the flux over the chain samples at each wavelength.

        Parameters
        ----------
        q : sequence of float, optional
            Percentiles to compute, between 0 and 100.
            Default is ``(16, 50, 84)``.

        Returns
        -------
        flux_q : np.ndarray
            Flux percentiles in nanomaggies, shape ``(len(q), n_wave)``.
        """
        return np.percentile(self.flux, q, axis=0)

    def summary(self):
        """
        Median spectrum with its 68% interval.

        Returns
        -------
        wl : np.ndarray
            Observed-frame wavelength grid in Angstroms.
        lo : np.ndarray
            16th percentile of the flux in nanomaggies.
        med : np.ndarray
            Median flux in nanomaggies.
        hi : np.ndarray
            84th percentile of the flux in nanomaggies.

        See Also
        --------
        `Spectrum.percentiles` : Underlying percentile calculation.
        """
        lo, med, hi = self.percentiles()
        return self.wl, lo, med, hi


class EmceeChain:
    """
    emcee chains from one run, with shape (walker, step, galaxy, parameter).

    The FSPS and Speculator models are loaded the first time they are
    needed and then shared between all `Galaxy` objects from this chain.

    Attributes
    ----------
    chain : np.ndarray
        Full chain, shape ``(n_walkers, n_steps, n_galaxies, n_params)``.
    chainsFlat : np.ndarray
        Chain with walkers and steps merged, shape
        ``(n_walkers*n_steps, n_galaxies, n_params)``.
    photometry : astropy.table.Table or None
        Photometry table, one row per galaxy, or ``None`` if not given.
    proPath : str
        Path to the directory holding the Prospector params .py file.
    proName : str
        Name of the Prospector params .py file.
    speculatorDir : str
        Directory holding the Speculator models.
    speculatorNoEmDir : str or None
        Directory holding the Speculator models trained without emission lines.
    speculatorRanges : list of str
        Wavelength ranges of the Speculator models, as used in their file names.
    """

    def __init__(self, path, photometryPath=None, key=None,
                 proPath=PRO_PATH, proName=PRO_NAME,
                 speculatorDir='', speculatorNoEmDir=None,
                 speculatorRanges=SPECULATOR_RANGES):
        """
        Parameters
        ----------
        path : str
            Path to a .npy or .npz file holding the chain, with shape
            ``(n_walkers, n_steps, n_galaxies, n_params)``.
        photometryPath : str, optional
            Path to a photometry table readable by `astropy.io.ascii`, with
            one row per galaxy in chain order. Default is ``None``.
        key : str, optional
            Name of the array to use if `path` is a .npz file.
            Default is ``None``, which uses the first array in the file.
        proPath : str, optional
            Path to the directory holding the Prospector params .py file.
            Default is the package's ``sps_models`` directory.
        proName : str, optional
            Name of the Prospector params .py file.
            Default is ``'prospector_alpha_plus_params'``.
        speculatorDir : str, optional
            Directory holding the Speculator models, saved as
            ``speculator_<range>.pt``. Default is ``''``.
        speculatorNoEmDir : str, optional
            Directory holding the Speculator models trained without emission
            lines. Needed only for the no-emission-line Speculator spectra.
            Default is ``None``.
        speculatorRanges : sequence of str, optional
            Wavelength ranges of the Speculator models, as used in their file
            names. Default is `SPECULATOR_RANGES`.
        """
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data[key if key is not None else data.files[0]]
        self.chain = data
        self.chainsFlat = data.reshape(-1, data.shape[2], data.shape[3])
        self.photometry = ascii.read(photometryPath) if photometryPath else None
        self.proPath = proPath
        self.proName = proName
        self.speculatorDir = speculatorDir
        self.speculatorNoEmDir = speculatorNoEmDir
        self.speculatorRanges = list(speculatorRanges)
        self._fsps = None
        self._speculatorModels = None
        self._speculatorNoEmModels = None

    @property
    def nGalaxies(self):
        """
        Number of galaxies in the chain.

        Returns
        -------
        n_galaxies : int
            Number of galaxies.
        """
        return self.chainsFlat.shape[1]

    def galaxy(self, index):
        """
        Flattened chain of one galaxy.

        Parameters
        ----------
        index : int
            Position of the galaxy in the chain.

        Returns
        -------
        galaxy : Galaxy
            Samples for that galaxy, linked back to this chain.
        """
        return Galaxy(self.chainsFlat[:, index], index, self)

    def __getitem__(self, index):
        """
        Flattened chain of one galaxy. Same as `EmceeChain.galaxy`.

        Parameters
        ----------
        index : int
            Position of the galaxy in the chain.

        Returns
        -------
        galaxy : Galaxy
            Samples for that galaxy, linked back to this chain.
        """
        return self.galaxy(index)

    def __len__(self):
        """
        Number of galaxies in the chain.

        Returns
        -------
        n_galaxies : int
            Number of galaxies.
        """
        return self.nGalaxies

    def summarise(self, q=(16, 50, 84)):
        """
        Per-galaxy, per-parameter median and asymmetric errors.

        Parameters
        ----------
        q : sequence of float, optional
            Lower, middle and upper percentiles, between 0 and 100.
            Default is ``(16, 50, 84)``.

        Returns
        -------
        summary : np.ndarray
            Array of shape ``(3, n_galaxies, n_params)``. The first axis is
            (median, error-, error+), with error- = median - ``q[0]`` percentile
            and error+ = ``q[2]`` percentile - median (both positive).
        """
        lo, med, hi = np.percentile(self.chainsFlat, q, axis=0)
        return np.stack([med, med - lo, hi - med])

    # expensive model setup loaded once and shared between galaxy classes
    @property
    def fsps(self):
        """
        Prospector/FSPS setup, loaded on first use.

        Returns
        -------
        fsps : tuple
            ``(pfile, obs, sps, mod)`` as returned by `initialise_fsps`.

        See Also
        --------
        `initialise_fsps` : Sets up FSPS and generates `pfile`, `obs`, `sps`, `mod`.
        """
        if self._fsps is None:
            self._fsps = initialise_fsps(pro_path=self.proPath, pro_name=self.proName)
        return self._fsps

    @property
    def speculatorModels(self):
        """
        Speculator models, loaded from `speculatorDir` on first use.

        Returns
        -------
        models : list
            One Speculator model per entry in `speculatorRanges`.
        """
        if self._speculatorModels is None:
            self._speculatorModels = [
                torch.load(os.path.join(self.speculatorDir, f'speculator_{r}.pt'),
                           map_location='cpu', weights_only=False)
                for r in self.speculatorRanges
            ]
        return self._speculatorModels

    @property
    def speculatorNoEmModels(self):
        """
        Speculator models trained on spectra without emission lines,
        loaded from `speculatorNoEmDir` on first use.

        Raises a `RuntimeError` if `speculatorNoEmDir` was not set.

        Returns
        -------
        models : list
            One Speculator model per entry in `speculatorRanges`.
        """
        if self._speculatorNoEmModels is None:
            if self.speculatorNoEmDir is None:
                raise RuntimeError('speculatorNoEmDir not set; pass it to EmceeChain to use the no-emline models.')
            self._speculatorNoEmModels = [
                torch.load(os.path.join(self.speculatorNoEmDir, f'speculator_{r}.pt'),
                           map_location='cpu', weights_only=False)
                for r in self.speculatorRanges
            ]
        return self._speculatorNoEmModels

class Galaxy:
    """
    Flattened chain of a single object, with shape (n_samples, n_params).

    Parameters follow the MCMC order, with the normalisation N first
    and redshift last.

    Attributes
    ----------
    samples : np.ndarray
        Chain samples, shape ``(n_samples, n_params)``.
    index : int
        Position of the galaxy in the parent chain.
    parent : EmceeChain or None
        Chain this galaxy came from. Needed for photometry and spectra.
    """

    def __init__(self, samples, index, parent=None):
        """
        Parameters
        ----------
        samples : array_like
            Chain samples, shape ``(n_samples, n_params)``.
        index : int
            Position of the galaxy in the parent chain.
        parent : EmceeChain, optional
            Chain this galaxy came from, used for the photometry and the
            shared model setup. Default is ``None``.
        """
        self.samples = np.asarray(samples)
        self.index = index
        self.parent = parent

    @property
    def z(self):
        """
        Redshift of each chain sample.

        Returns
        -------
        z : np.ndarray
            Redshifts, shape ``(n_samples,)``.
        """
        return self.samples[:, -1]

    @property
    def photometry(self):
        """
        Row of the parent chain's photometry table for this galaxy.

        Returns
        -------
        photometry : astropy.table.Row or None
            Photometry for this galaxy, or ``None`` if there is no parent
            chain or it has no photometry.
        """
        if self.parent is None or self.parent.photometry is None:
            return None
        return self.parent.photometry[self.index]

    # ---- parameter transforms where each returns a transformed copy ----

    def convertNToMass(self, theta=None):
        """
        Convert the normalisation N to log10 stellar mass.

        Parameters
        ----------
        theta : np.ndarray, optional
            Parameters to transform, shape ``(n, n_params)``.
            Default is ``None``, which uses `samples`.

        Returns
        -------
        theta : np.ndarray
            Copy of the parameters with the first column replaced by the
            base 10 logarithm of stellar mass formed, in solar masses.
        """
        theta = (self.samples if theta is None else theta).copy()
        theta[:, 0] = (theta[:, 0] - Planck18.distmod(theta[:, -1]).value) / (-2.5)
        return theta

    def sqrtTau2(self, theta=None):
        """
        Take the square root of the dust parameter tau2, as the Speculator
        models expect.

        Parameters
        ----------
        theta : np.ndarray, optional
            Parameters to transform, shape ``(n, n_params)``.
            Default is ``None``, which uses `samples`.

        Returns
        -------
        theta : np.ndarray
            Copy of the parameters with the tau2 column square-rooted.
        """
        theta = (self.samples if theta is None else theta).copy()
        theta[:, TAU2_INDEX] = np.sqrt(theta[:, TAU2_INDEX])
        return theta

    def expAGNParams(self, theta=None):
        """
        Exponentiate the two AGN parameters, which the chain stores as
        natural logarithms.

        Parameters
        ----------
        theta : np.ndarray, optional
            Parameters to transform, shape ``(n, n_params)``.
            Default is ``None``, which uses `samples`.

        Returns
        -------
        theta : np.ndarray
            Copy of the parameters with the AGN columns exponentiated.
        """
        theta = (self.samples if theta is None else theta).copy()
        theta[:, AGN_SLICE] = np.exp(theta[:, AGN_SLICE])
        return theta

    @property
    def logMass(self):
        """
        Base 10 logarithm of stellar mass formed for each chain sample.

        Returns
        -------
        log_mass : np.ndarray
            Log stellar mass in solar masses, shape ``(n_samples,)``.

        See Also
        --------
        `Galaxy.convertNToMass` : Underlying conversion from N.
        """
        return self.convertNToMass()[:, 0]

    def draw(self, nDraw=None, seed=None):
        """
        Indices of a random subset of samples, drawn without replacement.

        Parameters
        ----------
        nDraw : int, optional
            Number of samples to draw. Default is ``None``, which returns
            all samples (as does any `nDraw` at least the number of samples).
        seed : int, optional
            Seed for `numpy.random.default_rng`. Default is ``None``.

        Returns
        -------
        idx : np.ndarray
            Indices into `samples`.
        """
        n = len(self.samples)
        if nDraw is None or nDraw >= n:
            return np.arange(n)
        return np.random.default_rng(seed).choice(n, nDraw, replace=False)

    # ---- spectrum generation ----

    def _fspsFlux(self, idx, wavegrid, emlines=True, nebContinuum=True):
        """
        Observed-frame wavelength and FSPS flux for the chosen samples.

        Parameters
        ----------
        idx : np.ndarray
            Indices of the samples to use.
        wavegrid : np.ndarray
            Observed-frame wavelength grid in Angstroms.
        emlines : bool, optional
            If `False`, nebular emission lines are excluded. Default is `True`.
        nebContinuum : bool, optional
            If `False`, the nebular continuum is excluded. Default is `True`.

        Returns
        -------
        wl : np.ndarray
            Observed-frame wavelength grid in Angstroms.
        flux : np.ndarray
            Flux in nanomaggies, shape ``(len(idx), n_wave)``.

        See Also
        --------
        `generate_fsps_spec` : Underlying FSPS routine.
        """
        if self.parent is None:
            raise RuntimeError('Galaxy has no parent EmceeChain; spectrum generation needs one for the model setup.')

        theta = self.expAGNParams(self.convertNToMass())[idx]
        pfile, obs, sps, mod = self.parent.fsps
        _, obsWl, obsSpec = generate_fsps_spec(theta, pfile, obs, sps, mod, wavegrid=wavegrid, unit='nMgy',
                                               emlines=emlines, neb_continuum=nebContinuum)
        return np.asarray(obsWl), np.asarray(obsSpec)

    def generateFspsSpec(self, returnGrid=False, nDraw=None, seed=None,
                         wavegrid=np.arange(2500, 100000, 1)):
        """
        FSPS spectrum for each chain sample, in nanomaggies.

        Parameters
        ----------
        returnGrid : bool, optional
            If `True`, returns the spectrum of every drawn sample. Default is
            `False`, which returns the median and 68% interval.
        nDraw : int, optional
            Number of samples to draw. Default is ``None`` (all samples).
        seed : int, optional
            Seed for the random draw. Default is ``None``.
        wavegrid : np.ndarray, optional
            Observed-frame wavelength grid in Angstroms.
            Default is 2500 to 99999 in steps of 1.

        Returns
        -------
        spec : Spectrum or tuple of np.ndarray
            `Spectrum` if `returnGrid` is `True`, otherwise
            (wavelength, bottom, median, top) from `Spectrum.summary`.
        """
        spec = Spectrum(*self._fspsFlux(self.draw(nDraw, seed), wavegrid), label='fsps')
        return spec if returnGrid else spec.summary()

    def generateFspsNoEmSpec(self, returnGrid=False, nDraw=None, seed=None,
                             wavegrid=np.arange(2500, 100000, 1), nebContinuum=True):
        """
        FSPS spectrum without emission lines for each chain sample, in nanomaggies.

        Parameters
        ----------
        returnGrid : bool, optional
            If `True`, returns the spectrum of every drawn sample. Default is
            `False`, which returns the median and 68% interval.
        nDraw : int, optional
            Number of samples to draw. Default is ``None`` (all samples).
        seed : int, optional
            Seed for the random draw. Default is ``None``.
        wavegrid : np.ndarray, optional
            Observed-frame wavelength grid in Angstroms.
            Default is 2500 to 99999 in steps of 1.
        nebContinuum : bool, optional
            If `False`, the nebular continuum is also removed
            (``add_neb_emission=False``). Default is `True`.

        Returns
        -------
        spec : Spectrum or tuple of np.ndarray
            `Spectrum` if `returnGrid` is `True`, otherwise
            (wavelength, bottom, median, top) from `Spectrum.summary`.
        """
        spec = Spectrum(*self._fspsFlux(self.draw(nDraw, seed), wavegrid, emlines=False,
                                        nebContinuum=nebContinuum),
                        label='fsps_no_em' if nebContinuum else 'fsps_no_neb')
        return spec if returnGrid else spec.summary()

    def generateFspsEmlineSpec(self, returnGrid=False, nDraw=None, seed=None,
                               wavegrid=np.arange(2500, 100000, 1), includeNebContinuum=False):
        """
        Emission-line-only FSPS spectrum for each chain sample, in nanomaggies.

        Residual between the full and no-emission-line FSPS spectra, computed
        per sample before taking percentiles.

        Parameters
        ----------
        returnGrid : bool, optional
            If `True`, returns the spectrum of every drawn sample. Default is
            `False`, which returns the median and 68% interval.
        nDraw : int, optional
            Number of samples to draw. Default is ``None`` (all samples).
        seed : int, optional
            Seed for the random draw. Default is ``None``.
        wavegrid : np.ndarray, optional
            Observed-frame wavelength grid in Angstroms.
            Default is 2500 to 99999 in steps of 1.
        includeNebContinuum : bool, optional
            If `True`, the residual also contains the nebular continuum
            (the baseline has ``add_neb_emission=False``). Default is `False`.

        Returns
        -------
        spec : Spectrum or tuple of np.ndarray
            `Spectrum` if `returnGrid` is `True`, otherwise
            (wavelength, bottom, median, top) from `Spectrum.summary`.
        """
        idx = self.draw(nDraw, seed)
        waveGrid, fluxFull = self._fspsFlux(idx, wavegrid)
        _, fluxNoEm = self._fspsFlux(idx, wavegrid, emlines=False, nebContinuum=not includeNebContinuum)

        spec = Spectrum(waveGrid, fluxFull - fluxNoEm,
                        label='fsps_neb' if includeNebContinuum else 'fsps_emlines')
        return spec if returnGrid else spec.summary()

    def generateSpeculatorSpec(self, returnGrid=False, nDraw=None, seed=None, nWave=5174):
        """
        Speculator spectrum for each chain sample, in nanomaggies.

        Parameters
        ----------
        returnGrid : bool, optional
            If `True`, returns the spectrum of every drawn sample. Default is
            `False`, which returns the median and 68% interval.
        nDraw : int, optional
            Number of samples to draw. Default is ``None`` (all samples).
        seed : int, optional
            Seed for the random draw. Default is ``None``.
        nWave : int, optional
            Number of points on the log-spaced observed-frame wavelength grid,
            which covers the range shared by all drawn samples. Default is 5174.

        Returns
        -------
        spec : Spectrum or tuple of np.ndarray
            `Spectrum` if `returnGrid` is `True`, otherwise
            (wavelength, bottom, median, top) from `Spectrum.summary`.
        """
        if self.parent is None:
            raise RuntimeError('Galaxy has no parent EmceeChain; spectrum generation needs one for the model setup.')
        models = self.parent.speculatorModels

        idx = self.draw(nDraw, seed)
        theta = torch.tensor(self.sqrtTau2()[idx], dtype=torch.float32)
        wavelengths = np.concatenate([m.wavelengths for m in models])

        with torch.no_grad():
            logSpec = torch.cat([m.log_spectrum(theta[:, 1:]) for m in models], dim=1).numpy()

        # rest-frame L_nu [Lsun/Hz/Msun] to observed-frame flux [nMgy]
        z = self.z[idx][:, None]
        mass = 10**self.logMass[idx][:, None]
        lnu = 10**logSpec * LSUN * mass
        dl = Planck18.luminosity_distance(z).to(u.cm).value
        fnu = (1 + z) * lnu / (4 * np.pi * dl**2)
        nMgy = fnu / (3631 * JY_CGS) * 1e9
        waveObs = wavelengths[None, :] * (1 + z)

        # interpolation step
        waveGrid = np.geomspace(waveObs[:, 0].max(), waveObs[:, -1].min(), nWave)
        fluxGrid = np.array([np.interp(waveGrid, wo, f) for wo, f in zip(waveObs, nMgy)])

        spec = Spectrum(waveGrid, fluxGrid, label='speculator')
        return spec if returnGrid else spec.summary()

    def generateSpeculatorNoEmSpec(self, returnGrid=False, nDraw=None, seed=None, nWave=5174):
        """
        Speculator spectrum without emission lines for each chain sample,
        in nanomaggies. Needs `speculatorNoEmDir` set on the parent chain.

        Parameters
        ----------
        returnGrid : bool, optional
            If `True`, returns the spectrum of every drawn sample. Default is
            `False`, which returns the median and 68% interval.
        nDraw : int, optional
            Number of samples to draw. Default is ``None`` (all samples).
        seed : int, optional
            Seed for the random draw. Default is ``None``.
        nWave : int, optional
            Number of points on the log-spaced observed-frame wavelength grid,
            which covers the range shared by all drawn samples. Default is 5174.

        Returns
        -------
        spec : Spectrum or tuple of np.ndarray
            `Spectrum` if `returnGrid` is `True`, otherwise
            (wavelength, bottom, median, top) from `Spectrum.summary`.
        """
        if self.parent is None:
            raise RuntimeError('Galaxy has no parent EmceeChain; spectrum generation needs one for the model setup.')
        models = self.parent.speculatorNoEmModels

        idx = self.draw(nDraw, seed)
        theta = torch.tensor(self.sqrtTau2()[idx], dtype=torch.float32)
        wavelengths = np.concatenate([m.wavelengths for m in models])

        with torch.no_grad():
            logSpec = torch.cat([m.log_spectrum(theta[:, 1:]) for m in models], dim=1).numpy()

        # rest-frame L_nu [Lsun/Hz/Msun] to observed-frame flux [nMgy]
        z = self.z[idx][:, None]
        mass = 10**self.logMass[idx][:, None]
        lnu = 10**logSpec * LSUN * mass
        dl = Planck18.luminosity_distance(z).to(u.cm).value
        fnu = (1 + z) * lnu / (4 * np.pi * dl**2)
        nMgy = fnu / (3631 * JY_CGS) * 1e9
        waveObs = wavelengths[None, :] * (1 + z)

        # interpolation step
        waveGrid = np.geomspace(waveObs[:, 0].max(), waveObs[:, -1].min(), nWave)
        fluxGrid = np.array([np.interp(waveGrid, wo, f) for wo, f in zip(waveObs, nMgy)])

        spec = Spectrum(waveGrid, fluxGrid, label='speculator_no_em')
        return spec if returnGrid else spec.summary()

    def generateSpeculatorEmlineSpec(self, returnGrid=False, nDraw=None, seed=None, nWave=5174):
        """
        Emission-line-only Speculator spectrum for each chain sample, in nanomaggies.

        Residual between the full and no-emission-line Speculator spectra,
        computed per sample before taking percentiles. Needs `speculatorNoEmDir`
        set on the parent chain.

        Parameters
        ----------
        returnGrid : bool, optional
            If `True`, returns the spectrum of every drawn sample. Default is
            `False`, which returns the median and 68% interval.
        nDraw : int, optional
            Number of samples to draw. Default is ``None`` (all samples).
        seed : int, optional
            Seed for the random draw. Default is ``None``.
        nWave : int, optional
            Number of points on the log-spaced observed-frame wavelength grid,
            which covers the range shared by both models and all drawn samples.
            Default is 5174.

        Returns
        -------
        spec : Spectrum or tuple of np.ndarray
            `Spectrum` if `returnGrid` is `True`, otherwise
            (wavelength, bottom, median, top) from `Spectrum.summary`.
        """
        if self.parent is None:
            raise RuntimeError('Galaxy has no parent EmceeChain; spectrum generation needs one for the model setup.')

        idx = self.draw(nDraw, seed)
        theta = torch.tensor(self.sqrtTau2()[idx], dtype=torch.float32)
        z = self.z[idx][:, None]
        mass = 10**self.logMass[idx][:, None]
        dl = Planck18.luminosity_distance(z).to(u.cm).value

        # observed-frame wavelengths and flux [nMgy] for the full and no-emline models
        waveObs, nMgy = [], []
        for models in (self.parent.speculatorModels, self.parent.speculatorNoEmModels):
            wavelengths = np.concatenate([m.wavelengths for m in models])
            with torch.no_grad():
                logSpec = torch.cat([m.log_spectrum(theta[:, 1:]) for m in models], dim=1).numpy()
            lnu = 10**logSpec * LSUN * mass
            fnu = (1 + z) * lnu / (4 * np.pi * dl**2)
            nMgy.append(fnu / (3631 * JY_CGS) * 1e9)
            waveObs.append(wavelengths[None, :] * (1 + z))
        waveFull, waveNoEm = waveObs
        fluxFull, fluxNoEm = nMgy

        # common grid covering the overlap of both model wavelength ranges
        waveGrid = np.geomspace(max(waveFull[:, 0].max(), waveNoEm[:, 0].max()),
                                min(waveFull[:, -1].min(), waveNoEm[:, -1].min()), nWave)
        fluxGrid = (np.array([np.interp(waveGrid, wo, f) for wo, f in zip(waveFull, fluxFull)])
                    - np.array([np.interp(waveGrid, wo, f) for wo, f in zip(waveNoEm, fluxNoEm)]))

        spec = Spectrum(waveGrid, fluxGrid, label='speculator_emlines')
        return spec if returnGrid else spec.summary()
