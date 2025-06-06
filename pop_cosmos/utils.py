import torch
from astropy.cosmology import Planck18

def compute_derived_quantities(thetas):
    z = thetas[:,-1].detach().cpu().numpy()

    log10M_formed = -0.4*(thetas[:,0] - torch.from_numpy(Planck18.distmod(z).value))

    mw_age = mass_weighted_age(thetas[:,2:8], z)
    log10sSFR = specific_star_formation_rate(thetas[:,2:8], z)

    return log10M_formed, mw_age, log10sSFR + log10M_formed, log10sSFR

def mass_weighted_age(logsfr_ratios, z):
    # number of samples and bins
    n_samples = logsfr_ratios.shape[0]
    n_bins = logsfr_ratios.shape[1] + 1
    
    # age of the universe at the given redshift
    tuniv = torch.from_numpy(Planck18.age(z).value) # Gyr
    
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
    # number of samples and bins
    n_samples = logsfr_ratios.shape[0]
    n_bins = logsfr_ratios.shape[1] + 1
    
    # age of the universe at the given redshift
    tuniv = torch.from_numpy(Planck18.age(z).value) # Gyr
    
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
    Rolling upper mass limit based on the upper percentile of Ch.1<26 model draws.
    
    Fits the percentile (computed as a rolling function of z in 
    redshift window of size 0.5), using a two segment cubic hermite spline
    with an asymptote at z>4.0.
    
    OLD FIT WITH REGULAR z BINNING
    0.5%: z1=0.8359, z2=3.0, M0=9.9219, M1=11.6658, M2=11.5246, dMdz0=5.5384
    
    UPDATED FIT WITH n(z) BINNING
    0.5%: z1=0.7948, z2=6.0, M0=9.7108, M1=11.6535, M2=11.3251, dMdz0=6.4555
    
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
        Peak mass (mass limit at z = z1).
    M2 : float, optional
        Asymptote mass (mass limit at z >= z2).
    dMdz0 : float, optional
        Intercept slope (dM_lim/dz at z = 0).
        
    Returns
    -------
    M : np.array
        Limiting mass remaining, log10(M_lim(z) / M_sun).
    """
    
    M = chs_two_segment(z, 0.0, z1, z2, M0, M1, M2, dMdz0, 0.0, 0.0)
    return M

def mass_completeness(z, z1=0.7404, M0=6.5184, M1=8.5429, M2=9.6966, dMdz0=6.9873, dMdz1=0.5549, dMdz2=0.2956):
    """
    Rolling completeness limit.
    """
    M = chs_two_segment(z, 0.0, z1, 6.0, M0, M1, M2, dMdz0, dMdz1, dMdz2)
    return M
