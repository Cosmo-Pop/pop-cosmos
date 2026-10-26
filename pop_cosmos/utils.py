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
    """Spectra for every chain sample of one galaxy on observed-frame grid"""
    wl: np.ndarray      # observed-frame wavelength [AA], (n_wave,)
    flux: np.ndarray    # flux [nMgy], (n_draw, n_wave)
    label: str = ''

    def percentiles(self, q=(16, 50, 84)):
        return np.percentile(self.flux, q, axis=0)

    def summary(self):
        """Returns (lambda, bottom, median, top) 68% interval"""
        lo, med, hi = self.percentiles()
        return self.wl, lo, med, hi


class EmceeChain:
    """emcee chains from one run with shape (walker, step, galaxy, parameter)"""

    def __init__(self, path, photometryPath=None, key=None,
                 proPath=PRO_PATH, proName=PRO_NAME,
                 speculatorDir='', speculatorRanges=SPECULATOR_RANGES):
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data[key if key is not None else data.files[0]]
        self.chain = data
        self.chainsFlat = data.reshape(-1, data.shape[2], data.shape[3])
        self.photometry = ascii.read(photometryPath) if photometryPath else None
        self.proPath = proPath
        self.proName = proName
        self.speculatorDir = speculatorDir
        self.speculatorRanges = list(speculatorRanges)
        self._fsps = None
        self._speculatorModels = None

    @property
    def nGalaxies(self):
        return self.chainsFlat.shape[1]

    def galaxy(self, index):
        return Galaxy(self.chainsFlat[:, index], index, self)

    def __getitem__(self, index):
        return self.galaxy(index)

    def __len__(self):
        return self.nGalaxies

    # expensive model setup loaded once and shared between galaxy classes
    @property
    def fsps(self):
        if self._fsps is None:
            self._fsps = initialise_fsps(pro_path=self.proPath, pro_name=self.proName)
        return self._fsps

    @property
    def speculatorModels(self):
        if self._speculatorModels is None:
            self._speculatorModels = [
                torch.load(os.path.join(self.speculatorDir, f'speculator_{r}.pt'),
                           map_location='cpu', weights_only=False)
                for r in self.speculatorRanges
            ]
        return self._speculatorModels

class Galaxy:
    """flattened chain of a single object with shape (n_samples, n_params)"""

    def __init__(self, samples, index, parent=None):
        self.samples = np.asarray(samples)
        self.index = index
        self.parent = parent

    @property
    def z(self):
        return self.samples[:, -1]

    @property
    def photometry(self):
        if self.parent is None or self.parent.photometry is None:
            return None
        return self.parent.photometry[self.index]

    # ---- parameter transforms where each returns a transformed copy ----

    def convertNToMass(self, theta=None):
        """Convert the normalisation N to log10 stellar mass."""
        theta = (self.samples if theta is None else theta).copy()
        theta[:, 0] = (theta[:, 0] - Planck18.distmod(theta[:, -1]).value) / (-2.5)
        return theta

    def sqrtTau2(self, theta=None):
        theta = (self.samples if theta is None else theta).copy()
        theta[:, TAU2_INDEX] = np.sqrt(theta[:, TAU2_INDEX])
        return theta

    def expAGNParams(self, theta=None):
        theta = (self.samples if theta is None else theta).copy()
        theta[:, AGN_SLICE] = np.exp(theta[:, AGN_SLICE])
        return theta

    @property
    def logMass(self):
        return self.convertNToMass()[:, 0]

    def _requireParent(self):
        if self.parent is None:
            raise RuntimeError('Galaxy has no parent EmceeChain; spectrum generation needs one for the model setup.')
        return self.parent

    def draw(self, nDraw=None, seed=None):
        """indices of a random subset of samples (all samples if nDraw is None)"""
        n = len(self.samples)
        if nDraw is None or nDraw >= n:
            return np.arange(n)
        return np.random.default_rng(seed).choice(n, nDraw, replace=False)

    # ---- spectrum generation ----

    def generateFspsSpec(self, returnGrid=False, nDraw=None, seed=None,
                         wavegrid=np.arange(2500, 100000, 1)):
        """FSPS spectrum per chain sample [nMgy]

        Returns Spectrum if returnGrid otherwise summary spectra (wavelength, bottom, median, top).
        """
        theta = self.expAGNParams(self.convertNToMass())[self.draw(nDraw, seed)]
        pfile, obs, sps, mod = self._requireParent().fsps
        _, obsWl, obsSpec = generate_fsps_spec(theta, pfile, obs, sps, mod,
                                               wavegrid=wavegrid, unit='nMgy')
        spec = Spectrum(np.asarray(obsWl), np.asarray(obsSpec), label='fsps')
        return spec if returnGrid else spec.summary()

    def generateSpeculatorSpec(self, returnGrid=False, nDraw=None, seed=None, nWave=5174):
        """Speculator spectrum per chain sample [nMgy]

        Returns Spectrum if returnGrid otherwise summary spectra (wavelength, bottom, median, top).
        """
        models = self._requireParent().speculatorModels
        idx = self.draw(nDraw, seed)
        theta = torch.tensor(self.sqrtTau2()[idx], dtype=torch.float32)
        wavelengths = np.concatenate([m.wavelengths for m in models])

        with torch.no_grad():
            logSpec = torch.cat([m.log_spectrum(theta[:, 1:]) for m in models], dim=1).numpy()

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