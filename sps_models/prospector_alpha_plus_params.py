import numpy as np
import os
from prospect.utils.obsutils import fix_obs
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from scipy.stats import truncnorm
from sedpy.observate import load_filters

####################
# MASS-METALLICITY #
####################
# Gallazzi+05 mass-metallicity relation in stellar mass bins
# P{16, 50, 84} = {16, 50, 84}th percentile of stellar metallicity
# corrected to Chabrier IMF
gallazzi_05_massmet = [
#    mass    P50    P16    P84
    [8.87,  -0.60, -1.11, -0.00],
    [9.07,  -0.61, -1.07, -0.00],
    [9.27,  -0.65, -1.10, -0.05],
    [9.47,  -0.61, -1.03, -0.01],
    [9.68,  -0.52, -0.97,  0.05],
    [9.87,  -0.41, -0.90,  0.09],
    [10.07, -0.23, -0.80,  0.14],
    [10.27, -0.11, -0.65,  0.17],
    [10.47, -0.01, -0.41,  0.20],
    [10.68,  0.04, -0.24,  0.22],
    [10.87,  0.07, -0.14,  0.24],
    [11.07,  0.10, -0.09,  0.25],
    [11.27,  0.12, -0.06,  0.26],
    [11.47,  0.13, -0.04,  0.28],
    [11.68,  0.14, -0.03,  0.29],
    [11.87,  0.15, -0.03,  0.30]
]

#########
# UNITS #
#########

lsun = 3.846e33 # erg/s [cgs units]
pc = 3.085677581467192e18 # cm [cgs units]

lightspeed = 2.998e18 # AA/s
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2) # erg/c/cm^2 [cgs units]
jansky_mks = 1e-26 # W/m^2/Hz [mks units]

##############
# RUN_PARAMS #
##############
run_params = {'verbose':True,
              'debug': False,
              'outfile': 'output/',
              'nofork': True,
              # dynesty params
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'rwalk', # sampling method
              'nested_walks': 50, # MC walks
              'nested_nlive_batch': 200, # size of live point "batches"
              'nested_nlive_init': 200, # number of initial live points
              'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
              'nested_dlogz_init': 0.01,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              'nbins_sfh': 7,
              'sigma': 0.3,
              'df': 2,
              'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0],
              'datdir':'',
              'runname': 'td_new',
              'objname':'AEGIS_13'
              }
#######
# OBS #
#######
def load_obs(**extras):
    """
    Generates a dummy Prospector observation dictionary.

    Parameters
    ----------
    **extras : dict, optional
        Extra keyword arguments.

    Returns
    -------
    obs : dict
        Prospector-style observation dictionary filled with placeholders.
        Has keys 'filters', 'maggies', 'maggies_unc', 'phot_mask', 'wavelength', 'spectrum'.
    """
    # placeholder filter list (SEDPy's VISTA VIRCAM filter set)
    filternames = ['vista_vircam_' + n for n in ['Z','Y','J','H','Ks']]

    # fake magnitudes (zero in all bands)
    mags = np.zeros(len(filternames))
    # build dummy observation dictionary
    obs = {}
    # load the dummy filters using SEDPy
    obs['filters'] = load_filters(filternames)
    # fake fluxes in maggies (1 Mgy in all bands)
    obs['maggies'] = np.squeeze(10**(-mags/2.5))
    # fake flux uncertainties (7% in all bands)
    obs['maggies_unc'] = obs['maggies'] * 0.07
    # mask for NaNs and infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))
    # empty spectrum
    obs['wavelength'] = None
    obs['spectrum'] = None

    # ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    # returns the dictionary to be filled in
    return obs

#######
# SPS #
#######
def load_sps(**extras):
    """
    Load an SPS basis.

    Parameters
    ----------
    **extras : dict, optional
        Keyword arguments passed to NebSFH.

    Returns
    -------
    sps : NebSFH
        Instance of the `NebSFH` class.
    """
    sps = NebSFH(**extras)
    return sps

#########
# MODEL #
#########
def load_model(nbins_sfh=7, sigma=0.3, df=2, agelims=[], zred=0.0, **extras):
    """
    Build a Prospector SED model.

    Parameters
    ----------
    nbins_sfh : int, optional
        Number of SFH bins.
    sigma : float, optional
        Student-t scale parameter for continuity prior on SFR bin ratios.
    df : int, optiona;
        Student-t d.o.f. parameter for continuity prior.
    agelims : list, optional
        Edges of age bins in SFH.
    zred : float, optional
        Redshift.
    **extras : dict, optional
        Extra keyword arguments.

    Returns
    -------
    mod : prospect.models.sedmodel.SedModel
        Prospector SED model object.
    """

    # we'll need this to access specific model parameters
    n = [p['name'] for p in model_params]

    # first calculate redshift and corresponding t_universe
    # if no redshift is specified, read from file
    tuniv = WMAP9.age(zred).value*1e9

    # now construct the nonparametric SFH
    # current scheme: last bin is 15% age of the Universe, first two are 0-30, 30-100
    # remaining N-3 bins spaced equally in logarithmic space
    tbinmax = (tuniv*0.85)
    agelims = np.array(agelims[:2] + np.linspace(agelims[2],np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)])
    agebins = np.array([agelims[:-1], agelims[1:]])

    # load nvariables and agebins
    model_params[n.index('agebins')]['N'] = nbins_sfh
    model_params[n.index('agebins')]['init'] = agebins.T
    model_params[n.index('mass')]['N'] = nbins_sfh
    model_params[n.index('logsfr_ratios')]['N'] = nbins_sfh-1
    model_params[n.index('logsfr_ratios')]['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params[n.index('logsfr_ratios')]['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                      scale=np.full(nbins_sfh-1,sigma),
                                                                      df=np.full(nbins_sfh-1,df))
    # set mass-metallicity prior
    # insert redshift into model dictionary
    model_params[n.index('massmet')]['prior'] = MassMet(z_mini=-1.98, z_maxi=0.19, mass_mini=7, mass_maxi=12.5)
    model_params[n.index('zred')]['init'] = zred

    return sedmodel.SedModel(model_params)

############################
# TRANSFORMATION FUNCTIONS #
############################
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    """
    Convert birth cloud dust fraction to optical depth.

    Parameters
    ----------
    dust1_fraction : float, optional
        Ratio of dust1/dust2 optical depth (birth cloud / diffuse).
    dust1 : float, optional
        Birth cloud optical depth (unused).
    dust2 : float, optional
        Diffuse dust optical depth.
    **extras : dict, optional
        Extra keyword arguments (unused).

    Returns
    -------
    dust1 : float
        Birth cloud optical depth.
    """
    return dust1_fraction*dust2

def massmet_to_logmass(massmet=None, **extras):
    """
    Extract logmass from mass--metallicity tuple.

    Parameters
    ----------
    massmet : tuple or array-like, optional
        Tuple containing (logmass, logzsol).
    **extras : dict, optional
        Extra keyword arguments (unused).

    Returns
    -------
    logmass : float
        Base 10 logarithm of stellar mass / Msol.
    """
    return massmet[0]

def massmet_to_logzsol(massmet=None, **extras):
    """
    Extract logzsol from mass--metallicity tuple.

    Parameters
    ----------
    massmet : tuple or array-like, optional
        Tuple containing (logmass, logzsol).
    **extras : dict, optional
        Extra keyword arguments (unused).

    Returns
    -------
    logzsol : float
        Base 10 logarithm of stellar metallicity / Zsol.
    """
    return massmet[1]

def logmass_to_masses(massmet=None, logsfr_ratios=None, agebins=None, **extras):
    """
    Mass formed per SFH bin.

    Parameters
    ----------
    massmet : array-like, optional
        Array containing stellar mass and stellar metallicity.
        `mass = massmet[0]`
        `met  = massmet[1]`
    logsfr_ratios : array-like, optional
        Array containing SFR ratios between adjacent SFH bins.
    agebins : array-like, optional
        Edges of age bins in SFH.
    **extras : dict, optional
        Extra keyword arguments (unused).

    Returns
    -------
    masses : array-like
        Stellar mass formed per SFH bin.
    """
    logsfr_ratios = np.clip(logsfr_ratios, -10, 10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**np.array(logsfr_ratios)
    dt = 10**np.array(agebins[:,1])-10**np.array(agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**massmet[0]) / coeffs.sum()

    return m1 * coeffs

################
# MODEL PARAMS #
################
model_params = []

# BASIC PARAMETERS #
model_params.append({'name': 'zred', 'N': 1,
                     'isfree': False,
                     'init': 0.0,
                     'units': '',
                     'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'add_igm_absorption', 'N': 1,
                     'isfree': False,
                     'init': 1,
                     'units': None,
                     'prior_function': None,
                     'prior_args': None})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                     'isfree': False,
                     'init': True,
                     'units': None,
                     'prior_function': None,
                     'prior_args': None})

model_params.append({'name': 'pmetals', 'N': 1,
                     'isfree': False,
                     'init': -99,
                     'units': '',
                     'prior_function': None,
                     'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'massmet', 'N': 2,
                     'isfree': True,
                     'init': np.array([10,-0.5]),
                     'prior': None})

model_params.append({'name': 'logmass', 'N': 1,
                     'isfree': False,
                     'depends_on': massmet_to_logmass,
                     'init': 10.0,
                     'units': 'Msun',
                     'prior': None})

model_params.append({'name': 'logzsol', 'N': 1,
                     'isfree': False,
                     'init': -0.5,
                     'depends_on': massmet_to_logzsol,
                     'units': r'$\log (Z/Z_\odot)$',
                     'prior': None})
                        
# SFH #
model_params.append({'name': 'sfh', 'N':1,
                     'isfree': False,
                     'init': 0,
                     'units': None})

model_params.append({'name': 'mass', 'N': 1,
                     'isfree': False,
                     'depends_on': logmass_to_masses,
                     'init': 1.,
                     'units': r'M$_\odot$',})

model_params.append({'name': 'agebins', 'N': 1,
                     'isfree': False,
                     'init': [],
                     'units': 'log(yr)',
                     'prior': None})

model_params.append({'name': 'logsfr_ratios', 'N': 7,
                     'isfree': True,
                     'init': [],
                     'units': '',
                     'prior': None})

# IMF #
model_params.append({'name': 'imf_type', 'N': 1,
                     'isfree': False,
                     'init': 1, # 1 = chabrier
                     'units': None,
                     'prior': None})

# DUST ATTENUATION #
model_params.append({'name': 'dust_type', 'N': 1,
                     'isfree': False,
                     'init': 4,
                     'units': 'index',
                     'prior_function_name': None,
                     'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                     'isfree': False,
                     'depends_on': to_dust1,
                     'init': 1.0,
                     'units': '',
                     'prior': None})

model_params.append({'name': 'dust1_fraction', 'N': 1,
                     'isfree': True,
                     'init': 1.0,
                     'init_disp': 0.8,
                     'disp_floor': 0.8,
                     'units': '',
                     'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)})

model_params.append({'name': 'dust2', 'N': 1,
                     'isfree': True,
                     'init': 1.0,
                     'init_disp': 0.25,
                     'disp_floor': 0.15,
                     'units': '',
                     'prior': priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)})

model_params.append({'name': 'dust_index', 'N': 1,
                     'isfree': True,
                     'init': 0.0,
                     'init_disp': 0.25,
                     'disp_floor': 0.15,
                     'units': '',
                     'prior': priors.TopHat(mini=-1.0, maxi=0.4)})

model_params.append({'name': 'dust1_index', 'N': 1,
                     'isfree': False,
                     'init': -1.0,
                     'units': '',
                     'prior': None})

model_params.append({'name': 'dust_tesc', 'N': 1,
                     'isfree': False,
                     'init': 7.0,
                     'units': 'log(Gyr)',
                     'prior_function_name': None,
                     'prior_args': None})

# DUST EMISSION #
model_params.append({'name': 'add_dust_emission', 'N': 1,
                     'isfree': False,
                     'init': 1,
                     'units': None,
                     'prior': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                     'isfree': False,
                     'init': 0.01,
                     'init_disp': 0.2,
                     'disp_floor': 0.15,
                     'units': None,
                     'prior': priors.TopHat(mini=0.0, maxi=1.0)})

model_params.append({'name': 'duste_umin', 'N': 1,
                     'isfree': False,
                     'init': 1.0,
                     'init_disp': 5.0,
                     'disp_floor': 4.5,
                     'units': None,
                     'prior': priors.TopHat(mini=0.1, maxi=25.0)})

model_params.append({'name': 'duste_qpah', 'N': 1,
                     'isfree': False,
                     'init': 2.0,
                     'init_disp': 3.0,
                     'disp_floor': 3.0,
                     'units': 'percent',
                     'prior': priors.TopHat(mini=0.0, maxi=7.0)})

# NEBULAR EMISSION #
model_params.append({'name': 'add_neb_emission', 'N': 1,
                     'isfree': False,
                     'init': True,
                     'units': r'log Z/Z_\odot',
                     'prior': None})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                     'isfree': False,
                     'init': True,
                     'units': r'log Z/Z_\odot',
                     'prior': None})

model_params.append({'name': 'nebemlineinspec', 'N': 1,
                     'isfree': False,
                     'init': True,
                     'prior': None})

model_params.append({'name': 'gas_logz', 'N': 1,
                     'isfree': True,
                     'init': 0.0,
                     'units': r'log Z/Z_\odot',
                     'prior': priors.TopHat(mini=-2.0, maxi=0.5)})

model_params.append({'name': 'gas_logu', 'N': 1, # scale with sSFR?
                     'isfree': True,
                     'init': -1.0,
                     'units': '',
                     'prior': priors.TopHat(mini=-4.0, maxi=-1.0)})

# AGN DUSTY TORUS #
model_params.append({'name': 'add_agn_dust', 'N': 1,
                     'isfree': False,
                     'init': True,
                     'units': '',
                     'prior': None})

model_params.append({'name': 'fagn', 'N': 1,
                     'isfree': True,
                     'init': 0.01,
                     'init_disp': 0.03,
                     'disp_floor': 0.02,
                     'units': '',
                     'prior': priors.LogUniform(mini=1e-5, maxi=3.0)})

model_params.append({'name': 'agn_tau', 'N': 1,
                     'isfree': True,
                     'init': 20.0,
                     'init_disp': 5,
                     'disp_floor': 2,
                     'units': '',
                     'prior': priors.LogUniform(mini=5.0, maxi=150.0)})

# CALIBRATION #
model_params.append({'name': 'phot_jitter', 'N': 1,
                     'isfree': False,
                     'init': 0.0,
                     'init_disp': 0.5,
                     'units': 'fractional maggies (mags/1.086)',
                     'prior': priors.TopHat(mini=0.0, maxi=0.5)})

# UNITS #
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mformed'})

#######################
# CLASS REDEFINITIONS #
#######################

# MASS-METALLICITY PRIOR #
class MassMet(priors.Prior):
    """
    A Gaussian prior designed to approximate the Gallazzi+05
    stellar mass--stellar metallicity relationship.

    Inherits from `prospect.models.priors.Prior`.

    Attributes
    ----------
    params : dict
        Dictionary of parameters defining the prior.
    massmet : np.array
        Array containing the Gallazzi+05 mass metallicity relation.
    """
    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    dirpath = os.path.dirname(os.path.realpath(__file__))
    massmet = gallazzi_05_massmet

    def __len__(self):
        """ 
        Hack to work with Prospector 0.3.

        Returns
        -------
        len : int
            Fake length = 2.
        """
        return 2

    def scale(self, mass):
        """
        Scale parameter of Gaussian prior.

        Parameters
        ----------
        mass : float or array-like
            Stellar mass to evaluate scale at.

        Returns
        -------
        scale : float or array-like
            Scale at `mass` (interpolated from Gallazzi+05 68% C.I.).
        """
        upper_84 = np.interp(mass, self.massmet[:,0], self.massmet[:,3]) 
        lower_16 = np.interp(mass, self.massmet[:,0], self.massmet[:,2])
        return (upper_84-lower_16)

    def loc(self, mass):
        """
        Location parameter of Gaussian prior.

        Parameters
        ----------
        mass : float or array-like
            Stellar mass to evaluate location at.

        Returns
        -------
        loc : float or array-like
            Location at `mass` (interpolated from Gallazzi+05 median metallicity).
        """
        return np.interp(mass, self.massmet[:,0], self.massmet[:,1])

    def get_args(self, mass):
        """
        Truncation limits for stellar metallicity.

        Follows the `scipy.stats.truncnorm` convention:
        `a = (min - loc)/scale`
        `b = (max - loc)/scale`

        Parameters
        ----------
        mass : float or array-like
            Stellar mass to calculate the limits at.

        Returns
        -------
        a : float or array-like
            Lower truncation limit.
        b : float or array-like
            Upper truncation limit. 
        """
        a = (self.params['z_mini'] - self.loc(mass)) / self.scale(mass)
        b = (self.params['z_maxi'] - self.loc(mass)) / self.scale(mass)
        return [a, b]

    @property
    def range(self):
        """
        Allowable parameter ranges.

        Returns
        -------
        range : tuple of tuple
            Contains `(('mass_mini', 'mass_maxi'), ('z_mini', 'z_maxi'))`.
        """
        return ((self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi']))

    def bounds(self, **kwargs):
        """
        Update parameter bounds.

        Parameters
        ----------
        **kwargs : dict, optional
            Named properties to update. Allowable keys are
            `('mass_mini', 'mass_maxi', 'z_mini', 'z_maxi')`.

        Returns
        -------
        range : tuple of tuple
            See `range`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """
        Compute the value of the log probability density function at x.

        Parameters
        ----------
        x : array-like
            Values to evaluate the prior at. First column should be mass, 
            second metallicity, `x[...,0] = mass`, `x[...,1] = metallicity`.
        **kwargs : dict, optional
            All extra keyword arguments are used to update `self.params`.

        Returns
        -------
        lnp : array-like
            Natural log of the prior probability at `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.atleast_2d(np.zeros_like(x))
        a, b = self.get_args(x[...,0])
        p[...,1] = self.distribution.pdf(x[...,1], a, b, loc=self.loc(x[...,0]), scale=self.scale(x[...,0]))
        with np.errstate(invalid='ignore'):
            p[...,1] = np.log(p[...,1])
        return p

    def sample(self, nsample=None, **kwargs):
        """
        Draw samples from the prior distribution.

        Parameters
        ----------
        nsample : int, optional
            Number of samples to generate.
        **kwargs : dict, optional
            All extra keyword arguments are used to update `self.params`.

        Returns
        -------
        x : array-like
            Stellar masses and stellar metallicities.
            `mass = x[0]`
            `met  = x[1]`
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = np.random.uniform(low=self.params['mass_mini'],high=self.params['mass_maxi'],size=nsample)
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), scale=self.scale(mass), size=nsample)

        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        """
        Transform from U(0,1) random variable to parameter space.

        Parameters
        ----------
        x : array-like
            Values in [0,1] to transform to masses and metallicities.
            `x[0]` -> stellar mass
            `x[1]` -> stellar metallicity
        **kwargs : dict, optional
            All extra keyword arguments are used to update `self.params`.

        Returns
        -------
        x : array-like
            Stellar masses and stellar metallicities.
            `mass = x[0]`
            `met  = x[1]`
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = x[0]*(self.params['mass_maxi'] - self.params['mass_mini']) + self.params['mass_mini']
        a, b = self.get_args(mass)
        met = self.distribution.ppf(x[1], a, b, loc=self.loc(mass), scale=self.scale(mass))
        return np.array([mass,met])

# SPS BASIS #
class NebSFH(FastStepBasis):
    """
    SPS basis with nebular emission.

    Inherits from `prospect.source.FastStepBasis`.
    """
    @property
    def emline_wavelengths(self):
        """
        Emission line wavelengths.

        Returns
        -------
        emline_wavelengths : np.array
            Wavelengths of emission lines.
        """
        return self.ssp.emline_wavelengths

    @property
    def get_nebline_luminosity(self):
        """
        Emission line luminosities normalised by stellar mass formed.

        Returns
        -------
        emline_luminosity : np.array
            Emission line luminosity [Lsol/Msol]
        """
        return self.ssp.emline_luminosity/self.params['mass'].sum()

    def nebline_photometry(self, filters, z):
        """
        Emission line contribution to photometry.

        Parameters
        ----------
        filters : list of sedpy.observate.Filter
            SEDPy filter transmission curves.
        z : float
            Redshift.

        Returns
        -------
        flux : np.array
            Flux contribution from emission lines to each band in `filters`.
        """
        emlams = self.emline_wavelengths * (1+z)
        elums = self.get_nebline_luminosity # Lsun / solar mass formed
        flux = np.empty(len(filters))
        for i,filt in enumerate(filters):
            # calculate transmission at nebular emission
            trans = np.interp(emlams, filt.wavelength, filt.transmission, left=0., right=0.)
            idx = (trans > 0)
            if True in idx:
                flux[i] = (trans[idx]*emlams[idx]*elums[idx]).sum()/filt.ab_zero_counts
            else:
                flux[i] = 0.0
        return flux

    def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):
        """
        Generate a spectrum and photometry in observer frame.

        Parameters
        ----------
        outwave : array-like, optional
            Wavelengths.
        filters : list of sedpy.observate.Filter, optional
            Filters to compute photometry in.
        peraa : bool, optional
            If `True`, returns in erg/s/cm^2/AA.
            If `False` (default), returns maggies.
        **params : dict, optional
            Additional parameters.

        Returns
        -------
        spec : np.array
            Spectrum in maggies.
        phot : np.array
            Photometry in maggies (AB system).
        mfrac : float
            Fraction of stellar mass remaining.
        """

        # Spectrum in Lsun/Hz per solar mass formed, restframe
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)

        # Redshifting + Wavelength solution
        # We do it ourselves.
        a = 1 + self.params.get('zred', 0)
        af = a
        b = 0.0

        if 'wavecal_coeffs' in self.params:
            x = wave - wave.min()
            x = 2.0 * (x / x.max()) - 1.0
            c = np.insert(self.params['wavecal_coeffs'], 0, 0)
            # assume coeeficients give shifts in km/s
            b = chebval(x, c) / (lightspeed*1e-13)

        wa, sa = wave * (a + b), spectrum * af  # Observed Frame
        if outwave is None:
            outwave = wa
        
        spec_aa = lightspeed/wa**2 * sa # convert to perAA
        # Observed frame photometry, as absolute maggies
        if filters is not None:
            mags = observate.getSED(wa, spec_aa * to_cgs, filters)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0

        ### if we don't have emission lines, add them
        if (not self.params['nebemlineinspec']) and self.params['add_neb_emission']:
            phot += self.nebline_photometry(filters,a-1)*to_cgs

        # Spectral smoothing.
        do_smooth = (('sigma_smooth' in self.params) and
                     ('sigma_smooth' in self.reserved_params))
        if do_smooth:
            # We do it ourselves.
            smspec = self.smoothspec(wa, sa, self.params['sigma_smooth'],
                                     outwave=outwave, **self.params)
        elif outwave is not wa:
            # Just interpolate
            smspec = np.interp(outwave, wa, sa, left=0, right=0)
        else:
            # no interpolation necessary
            smspec = sa

        # Distance dimming and unit conversion
        zred = self.params.get('zred', 0.0)
        if (zred == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
        else:
            lumdist = WMAP9.luminosity_distance(zred).value
            dfactor = (lumdist * 1e5)**2
        if peraa:
            # spectrum will be in erg/s/cm^2/AA
            smspec *= to_cgs / dfactor * lightspeed / outwave**2
        else:
            # Spectrum will be in maggies
            smspec *= to_cgs / dfactor / 1e3 / (3631*jansky_mks)

        # Convert from absolute maggies to apparent maggies
        phot /= dfactor

        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mstar') == 'mstar'):
            # Convert from current stellar mass to mass formed
            mass /= mfrac

        return smspec * mass, phot * mass, mfrac