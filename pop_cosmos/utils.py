import torch
import numpy as np
from astropy.cosmology import Planck18

def compute_derived_quantities(thetas):
    """
    PyTorch routine for generating useful derived parameters from
    a tensor of SPS parameters.

    Parameters
    ----------
    thetas : torch.Tensor
        Sixteen-column tensor containing the base SPS parameters for some
        model galaxies.

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

    See Also
    --------
    `compute_mass_remaining` : Routine for correcting for mass loss.
    `mass_weighted_age` : Underlying routine for computing mass weighted age.
    `specific_star_formation_rate` : Underlying routine for computing sSFR and SFR.
    `catalogue.CatalogueGenerator` : Class that generates the `thetas` used as input.
    """
    z = thetas[:,-1].detach().cpu().numpy()

    log10M_formed = -0.4*(thetas[:,0] - torch.from_numpy(Planck18.distmod(z).value))

    mw_age = mass_weighted_age(thetas[:,2:8], z)
    log10sSFR = specific_star_formation_rate(thetas[:,2:8], z)

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
    Mfrac : torch.Tensor
        Fraction of stellar mass remaining.
    """

    Mfrac = mass_fraction_emulator((thetas - theta_shift) / theta_scale)[:,0]
    log10M = log10M_formed + torch.log10(Mfrac)
    log10sSFR = log10sSFR - torch.log10(Mfrac)

    return log10M, log10sSFR, Mfrac

def mass_weighted_age(logsfr_ratios, z):
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

    Returns
    -------
    mw_age : torch.Tensor
        Mass weighted age. Units of Gyr.
    """
    # number of samples and bins
    n_samples = logsfr_ratios.shape[0]
    n_bins = logsfr_ratios.shape[1] + 1
    
    # age of the universe at the given redshift
    tuniv = torch.from_numpy(Planck18.age(z).value).unsqueeze(-1) # Gyr
    
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

def specific_star_formation_rate(logsfr_ratios, z):
    """
    PyTorch routine for converting SFR ratios and redshift into sSFR.

    Parameters
    ----------
    logsfr_ratios : torch.Tensor
        Six-column tensor containing the logarithm of the SFR ratios
        for a sample of model galaxies.
    z : torch.Tensor
        One-column tensor containing the redshifts of the model galaxies.

    Returns
    -------
    log10sSFR : torch.Tensor
        Base 10 logarithm of the specific star formation rate (sSFR).
        Units of 1/yr.
        
    Notes
    -----
    The star formation rate is averaged over the last 100 Myr of a galaxy's life.
    This does not include a correction for mass loss. The quantity
    returned has the definition SFR/M_form, i.e. SFR per unit solar mass formed.
    """
    # number of samples and bins
    n_samples = logsfr_ratios.shape[0]
    n_bins = logsfr_ratios.shape[1] + 1
    
    # age of the universe at the given redshift
    tuniv = torch.from_numpy(Planck18.age(z).value).unsqueeze(-1) # Gyr
    
    # stellar age time grid
    # this first line does a tensorized logspace, since torch doesn't provide one
    log_age_inner_edges = np.log10(0.1)*torch.ones(n_samples, n_bins-2) + torch.arange(0, n_bins-2, 1)*(torch.log10(0.85*tuniv) - np.log10(0.1))/(n_bins-3.0) # logspaced to 0.85 tuniv
    age_edges = torch.hstack([torch.tensor([[0.0, 0.03]])*torch.ones(n_samples,2),
        10**log_age_inner_edges, tuniv])
    
    # bin widths
    bin_widths = (age_edges[:,1:] - age_edges[:,:-1])
    
    # star formation rate in each bin
    sfr = torch.ones(n_samples, n_bins)
    sfr[:,1:] = 10**torch.cumsum(-logsfr_ratios, dim=1)
    
    # normalize the sfr
    normalized_sfr = sfr / torch.sum(sfr*bin_widths, dim=1).unsqueeze(1)
    
    # mass formed per bin
    mass_formed = normalized_sfr * bin_widths
    
    # star formation rate
    log10sSFR = torch.clip(torch.log10((mass_formed[:,0] + mass_formed[:,1] + 1e-20) / 0.1) - 9.0, -18, 0)

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
