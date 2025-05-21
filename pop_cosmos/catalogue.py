import numpy as np
import torch
from speculator import flux2asinhmag, PhotulatorModelStack
from torch.distributions.studentT import StudentT
from flowfusion.diffusion import PopulationModelDiffusion
from .constants import COSMOS_FLUX_SOFTENING
from .emlines import EmLineEmulator

class NoiseModel(torch.nn.Module):
    """
    Class that generates noise and adds it to the mock fluxes.

    Attributes
    ----------
    model : flowfusion.diffusion.PopulationModelDiffusionConditional
        Trained uncertainty model.
    magnitude_min : torch.Tensor
        Minimum allowable input magnitudes. Used with `magnitude_max`
        to clamp the inputs.
    magnitude_max : torch.Tensor
        Maximum allowable input magnitudes.
    log_error_min : torch.Tensor
        Minimum allowable flux error. Used with `log_error_max` to
        clamp the outputs. Units are natural logarithm of maggies.
    log_error_max : torch.Tensor
        Maximum allowable flux error.
    error_floor : torch.Tensor
        Fractional error floor to be added in quadrature.
    zp_star : torch.Tensor
        Zero-point correction assumed in training.
    n_bands : int
        Number of photometric passbands.
    f_b : torch.Tensor
        Flux softening parameter in nanomaggies.
    nine_log_ten : torch.Tensor
        Conversion factor, 9*ln(10).
    """

    def __init__(
        self,
        model,
        log_error_min=None,
        log_error_max=None,
        magnitude_min=None,
        magnitude_max=None,
        n_bands=26,
        noise_floor=0.0,
        zp_star=1.0,
        f_b=None,
    ):
        """
        Parameters
        ----------
        model : flowfusion.diffusion.PopulationModelDiffusionConditional
            Diffusion model.
        log_error_min : torch.Tensor, optional
            Minimum allowable flux error (as natural log of maggies).
            Default is ``None`` (``-inf``).
        log_error_max : torch.Tensor, optional
            Maximum allowable flux error.
            Default is ``None`` (``inf``).
        magnitude_min : torch.Tensor, optional
            Minimum allowable input asinh magnitudes.
            Default is ``None`` (``-inf``).
        magnitude_max : torch.Tensor, optional
            Maximum allowable input asinh magnitudes.
            Default is ``None`` (``inf``).
        n_bands : int, optional
            Number of photometric bands.
            Default is 26.
        noise_floor : torch.Tensor, optional
            Fractional noise floor to be added in quadrature.
            Default is 0.0 in all bands.
        zp_star : torch.Tensor, optional
            Fractional zero-point correction assumed in training.
            Default is 1.0 in all bands.
        f_b : torch.Tensor, optional
            Flux softening parameter in nanomaggies.
        """

        super().__init__()

        self.model = model  # uncertainty model
        self.register_buffer(
            "magnitude_min",
            (
                -torch.inf * torch.ones(n_bands, dtype=torch.float32)
                if magnitude_min is None
                else magnitude_min
            ),
        )
        self.register_buffer(
            "magnitude_max",
            (
                torch.inf * torch.ones(n_bands, dtype=torch.float32)
                if magnitude_max is None
                else magnitude_max
            ),
        )
        self.register_buffer(
            "log_error_min",
            (
                -torch.inf * torch.ones(n_bands, dtype=torch.float32)
                if log_error_min is None
                else log_error_min
            ),
        )
        self.register_buffer(
            "log_error_max",
            (
                torch.inf * torch.ones(n_bands, dtype=torch.float32)
                if log_error_max is None
                else log_error_max
            ),
        )
        self.register_buffer("nine_log_ten", torch.tensor(9.0 * np.log(10.0)))
        self.n_bands = n_bands
        self.register_buffer(
            "error_floor", torch.tensor(noise_floor, dtype=torch.float32)
        )

        # zero point assumed in training to be corrected for (default is 1.0)
        self.register_buffer("zp_star", torch.tensor(zp_star, dtype=torch.float32))
        # f_b for asinh magnitude conversion
        if f_b is not None:
            self.register_buffer("f_b", torch.tensor(f_b, dtype=torch.float32))

    def noise_realization(
        self,
        n_noise,
        n_sigma,
        fluxes,
        asinh_magnitudes,
        zero_points=None,
        emission_line_errors=None,
    ):
        """
        Generate a noise realisation from the model.

        Parameters
        ----------
        n_noise : torch.Tensor
            Base random draws to be transformed into flux uncertainties.
        n_sigma : torch.Tensor
            Base random draws to be transformed into flux errors.
        fluxes : torch.Tensor
            True model fluxes in nanomaggies (without zero-point correction).
        asinh_magnitudes : torch.Tensor
            Zero-point corrected asinh magnitudes.
        zero_points : torch.Tensor, optional
            Fractional zero-point corrections.
        emission_line_errors : torch.Tensor, optional
            Uncertainties in emission line strength in nanomaggies.

        Returns
        -------
        noisy_fluxes : torch.Tensor
            Noisy, zero-point corrected `fluxes` in nanomaggies.
        flux_sigmas : torch.Tensor
            Total flux uncertainties (incl. em. lines) in nanomaggies.
        """ 
        flux_sigmas = torch.exp(
            self.nine_log_ten
            + torch.clamp(
                self.model(
                    n_sigma,
                    conditional=torch.clamp(
                        asinh_magnitudes, min=self.magnitude_min, max=self.magnitude_max
                    ),
                ),
                min=self.log_error_min,
                max=self.log_error_max,
            )
        )

        # divide out the zero point if it's provided
        if zero_points is not None:
            flux_sigmas = flux_sigmas / zero_points

        # add emission line errors in quadrature
        if emission_line_errors is not None:
            flux_sigmas = torch.sqrt(
                flux_sigmas**2
                + emission_line_errors**2
                + (self.error_floor * fluxes) ** 2
            )

        noise = n_noise * flux_sigmas
        noisy_fluxes = fluxes + noise

        if zero_points is not None:
            noisy_fluxes = noisy_fluxes * zero_points
            flux_sigmas = flux_sigmas * zero_points

        # returns nanomaggies
        return noisy_fluxes, flux_sigmas


class CatalogueGenerator(torch.nn.Module):
    """
    Generates mock catalogues and SPS parameters according to a trained diffusion model.

    Attributes
    ----------
    param_names : list of str
        List of parameter names in the order they are handled.
    n_params : int
        Number of SPS parameters.
    power : torch.Tensor
        Power to raise parameters to for Speculator.
    lower : torch.Tensor
        Lower limits for parameters.
    upper : torch.Tensor
        Upper limits for parameters.
    range : torch.Tensor
        Paramater ranges (used for prior transform).
    f_b : torch.Tensor
        Flux softening parameter.
    noise_model : pop_cosmos.catalogue.NoiseModel
        Uncertainty model.
    population_model : flowfusion.diffusion.PopulationModelDiffusion
        Population model.
    zero_points : torch.Tensor
        Fractional zero point corrections.
    emline_offsets : torch.Tensor
        Fractional emission line strength corrections.
    ln_emline_scatters : torch.Tensor
        Fractional emission line strength standard deviations.
    flux_emulator : speculator.PhotulatorModelStack
        Photometry emulators.
    emission_line_flux_emulator : pop_cosmos.emlines.EmLineEmulator
        Emission line emulator.
    noise_base : torch.distributions.Distribution
        Base distribution for the noise.
    n_bands : int
        Number of photometric bands.
    """

    def __init__(self, noise_model: NoiseModel, population_model: PopulationModelDiffusion, calibration_parameters, flux_emulator: PhotulatorModelStack, emission_line_flux_emulator: EmLineEmulator, f_b=COSMOS_FLUX_SOFTENING, device='cuda', zmax=6.0):
        """
        Parameters
        ----------
        noise_model : pop_cosmos.catalogue.NoiseModel
            Uncertainty model to use.
        population_model : flowfusion.diffusion.PopulationModelDiffusion
            Population model to use.
        calibration_parameters : tuple of torch.Tensor
            Tensors containing the fractional zero-point corrections,
            emission line strength corrections, and emission line
            scatter parameters.
        flux_emulator : speculator.PhotulatorModelStack
            Photometry emulator stack.
        emission_line_flux_emulator : pop_cosmos.emlines.EmLineEmulator
            Emission line emulator object.
        f_b : torch.Tensor, optional
            Flux softening parameter for each band.
            Default is `pop_cosmos.constants.COSMOS_FLUX_SOFTENING`.
        device : str or torch.device, optional
            Device for the model to live on. Default is ``'cuda'``.
        zmax : float, optional
            Maximum redshift assumed by the model. Default is 6.0.
        """
        super().__init__()

        # parameter names and limits
        self.param_names = (
            ["N", "log10Z"]
            + ["logsfr_ratio" + str(i + 1) for i in range(6)]
            + [
                "dust2",
                "dust_index",
                "dust1_fraction",
                "lnfagn",
                "lnagntau",
                "gaslog10Z",
                "gaslog10U",
                "z",
            ]
        )
        self.n_params = len(self.param_names)
        self.power = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        self.lower = torch.tensor(
            [
                7.0, # not used
                -1.98,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                0.0,
                -1.0,
                0.0,
                np.log(1e-5),
                np.log(5.0),
                -2.0,
                -4.0,
                0.0,
            ],
            dtype=torch.float32,
        ).to(device)
        self.upper = torch.tensor(
            [
                13.0, # not used
                0.19,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                4.0,
                0.4,
                2.0,
                np.log(3.0),
                np.log(150.0),
                0.5,
                -1.0,
                zmax,
            ],
            dtype=torch.float32,
        ).to(device)
        self.range = self.upper - self.lower
        self.f_b = torch.tensor(f_b, device=device, dtype=torch.float32)
        self.noise_model = noise_model
        self.population_model = population_model
        self.zero_points, self.emline_offsets, self.ln_emline_scatters = calibration_parameters
        self.flux_emulator = flux_emulator
        self.emission_line_flux_emulator = emission_line_flux_emulator 
        self.noise_base = StudentT(2) 
        self.n_bands = self.flux_emulator.n_emulators

    def generate_base_samples(self, nsamples):
        """
        Creates base draws for the catalogue model.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        base_samples_phi : torch.Tensor
            Base samples for the population model.
        base_samples_sigma : torch.Tensor
            Base samples for the uncertainty model.
        base_samples_noise : torch.Tensor
            Base samples for the error model.
        """
        base_samples_phi = torch.randn(nsamples, self.n_params)
        base_samples_sigma = torch.randn(nsamples, self.n_bands)
        base_samples_noise = self.noise_base.sample(nsamples, self.n_bands)
        return base_samples_phi, base_samples_sigma, base_samples_noise

    def phi2theta(self, phi):
        """
        Converts latent parameters to SPS parameters.

        Parameters
        ----------
        phi : torch.Tensor
            Input parameters in latent space.

        Returns
        theta : torch.Tensor
            SPS parameters.
        """
        theta = phi.clone()
        theta[:,1:] = 0.5 + 0.5*torch.erf(phi[:,1:]/np.sqrt(2.))
        theta[:,1:] = self.lower[1:] + self.range[1:]*theta[:,1:]
        return theta

    def emulator_parameter_transform(self, theta):
        """
        Transform the parameter array for the emulators.

        Parameters
        ----------
        theta : torch.Tensor
            SPS parameters to transform.

        Returns
        -------
        theta_transformed : torch.Tensor
            Parameter tensor with the dust parameter square-rooted.
        """
        return theta ** self.power

    def selection_cut(self, noisy_fluxes, noisy_magnitudes, noisy_flux_sigmas, magnitude_limits, flux_cut=False, snr_cut=False):
        """
        Apply a selection cut to mock catalogue.

        Parameters
        ----------
        noisy_fluxes : torch.Tensor
            Tensor of noisy fluxes in maggies.
        noisy_magnitudes : torch.Tensor
            Tensor of noisy (asinh) magnitudes.
        noisy_flux_sigmas : torch.Tensor
            Tensor of flux errors in maggies.
        magnitude_limits : torch.Tensor
            Tensor of magnitude limits to apply.
        flux_cut : bool, optional
            If ``True``, requires flux to be positive.
            Default is ``False``.
        snr_cut : bool, optional
            If ``True``, requires SNR>0 in all bands.
            Default is ``False``.

        Returns
        -------
        selection : torch.Tensor
            Boolean tensor with same shape as the inputs,
            indicating whether a galaxy passes/fails selection.
        """
        selection = (
            torch.all(~torch.isnan(noisy_magnitudes), dim=-1)
            * torch.all(~torch.isinf(noisy_magnitudes), dim=-1)
            * torch.all(noisy_magnitudes < self.mag_limits, dim=-1)
        )
        if flux_cut:
            selection = selection * torch.all(noisy_fluxes > 0.0, dim=-1)
        if snr_cut:
            selection = selection * torch.all(
                noisy_fluxes / noisy_flux_sigmas > 0.0, dim=-1
            )
        return selection

    def forward(self, base_samples_noise, base_samples_sigma, base_samples_phi):
        """
        Generates mock catalogue without applying selection.

        Parameters
        ----------
        base_samples_noise : torch.Tensor
            Base samples for the error model.
        base_samples_sigma : torch.Tensor
            Base samples for the uncertainty model.
        base_samples_phi : torch.Tensor
            Base samples for the population model.

        Returns
        -------
        noisy_fluxes : torch.Tensor
            Noisy model fluxes in maggies.
        noisy_magnitudes : torch.Tensor
            Noisy model magnitudes (logarithmic, AB system).
        noisy_asinh_magnitudes : torch.Tensor
            Noisy model magnitudes (asinh, AB system).
        flux_sigmas : torch.Tensor
            Flux uncertainties in maggies.
        theta_samples : torch.Tensor
            SPS parameter array.
        model_fluxes : torch.Tensor
            Zero-point-corrected noiseless model fluxes.
        """

        phi_samples = self.population_model.forward(base_samples_phi)

        # Clamp/restrict the base samples for the parameters
        phi_samples[:, 1:] = torch.nan_to_num(
            phi_samples[:, 1:], posinf=5.0, neginf=-5.0
        )
        phi_samples[:, 0] = torch.nan_to_num(
            phi_samples[:, 0], posinf=100.0, neginf=-100.0
        )

        # latent parameters -> physical parameters
        theta_samples = self.phi2theta(phi_samples)

        emission_line_flux_contributions = self.emission_line_flux_emulator(self.emulator_parameter_transform(theta_samples))  # nano maggies
        # cut base samples based on reference magnitude cut, if provided
        model_magnitudes = self.flux_emulator.magnitudes(self.emulator_parameter_transform(theta_samples)[:,1:],
            torch.unsqueeze(theta_samples[:, 0], -1),
        )  # without ZP
        model_fluxes = 10 ** ((22.5 - model_magnitudes) / 2.5) + torch.sum(
            self.emline_offsets * emission_line_flux_contributions, dim=-1
        )  # nano maggies
        model_asinh_magnitudes = flux2asinhmag(
            model_fluxes * self.zero_points, self.f_b
        )  # with ZP (for noise model's benefit)
        emission_line_error_contributions = torch.sum(
            torch.exp(self.ln_emline_scatters)
            * (1 + self.emline_offsets)
            * emission_line_flux_contributions,
            dim=-1,
        )  # nano maggies

        # magnitude conversion now handled intternal to the noise model object
        noisy_fluxes, flux_sigmas = self.noise_model.noise_realization(
            base_samples_noise,
            base_samples_sigma,
            model_fluxes,
            model_asinh_magnitudes,
            zero_points=self.zero_points,
            emission_line_errors=emission_line_error_contributions,
        )  # returns nanomaggies

        # requires flux in units of nanomaggies
        noisy_asinh_magnitudes = flux2asinhmag(noisy_fluxes, self.f_b)

        # noisy fluxes now coming as nanomaggies
        noisy_magnitudes = 22.5 - 2.5 * torch.log10(noisy_fluxes)

        return (
            noisy_fluxes * 1e-9,
            noisy_magnitudes,
            noisy_asinh_magnitudes,
            flux_sigmas * 1e-9,
            theta_samples,
            model_fluxes * self.zero_points * 1e-9,
        )
