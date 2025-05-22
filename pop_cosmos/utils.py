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
    sfr = torch.zeros(n_samples, n_bins)
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