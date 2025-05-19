import torch
import numpy as np
from speculator import Speculator

def load_lines_model(
    dir_spsmodels,
    n_parameters,
    filternames,
    device,
    dir_linegmms,
    n_hidden=[128, 128, 128],
):
    """
    Load a Speculator-based emission line emulator.

    Parameters
    ----------
    dir_spsmodels : str
        Path to a trained Speculator model.
    n_parameters : int
        Number of SPS parameters.
    filternames : list of str
        List of filter names.
    device : str or torch.device
        Device (``'cpu'``) or (``'cuda'``) for the model.
    dir_linegmms : str
        Path to a directory containing bandpass models.
    n_hidden : list of int, optional
        Number of hidden units per layer.
        Default ``[128, 128, 128]``.

    Returns
    -------
    n_lines : int
        Number of emission lines loaded.
    speculator_emlines : speculator.Speculator
        Speculator model.
    line_lambdas_selected : numpy.array
        List of emission line wavelengths.
    line_idx_selected : numpy.array
        List of line indexes in the loaded line list.
    gengmm_amps : numpy.array
        Generalised Gaussian amplitudes for bandpass models.
    line_names : numpy.array
        List of emission line names.
    gengmm_locs: numpy array
        Generalised Gaussian locations for bandpass models.
    gengmm_sigs : numpy.array
        Generalised Gaussian scales for bandpass models.
    gengmm_betas :  numpy.array
        Generalised Gaussian shapes for bandpass models.

    See Also
    --------
    `EmLineEmulator` : Emission line emulator class.
    """

    dir_emlines = dir_spsmodels + "/" 
    restore_filename = dir_emlines + "speculator-emlinesabsmags"
    PCABasis = np.load(dir_emlines + "PCABasis.npz")
    wavelengths = np.load(dir_emlines + "elams.npy")
    speculator_emlines = Speculator(
        restore=True,
        restore_filename=restore_filename,
        n_parameters=n_parameters,
        wavelengths=wavelengths,
        pca_transform_matrix=PCABasis["pca_transform_matrix"],
        parameters_shift=PCABasis["parameters_shift"],
        parameters_scale=PCABasis["parameters_scale"],
        pca_shift=PCABasis["pca_shift"],
        pca_scale=PCABasis["pca_scale"],
        log_spectrum_shift=PCABasis["log_spectrum_shift"],
        log_spectrum_scale=PCABasis["log_spectrum_scale"],
        n_hidden=n_hidden,  # network architecture
        device=device,
        optimizer=torch.optim.Adam
    )

    line_lambdas = speculator_emlines.wavelengths

    line_lambdas_selected = np.array(
        [
            17366.885,
            2326.11,
            2321.664,
            6302.046,
            4069.75,
            1215.6701,
            2669.951,
            30392.02,
            7753.19,
            37405.76,
            32969.8,
            18179.2,
            9017.8,
            2661.146,
            6313.81,
            9232.2,
            3722.75,
            2803.53,
            19450.89,
            9548.8,
            7067.138,
            6549.86,
            6732.673,
            10052.6,
            1908.73,
            6679.995,
            2796.352,
            6718.294,
            21661.178,
            40522.79,
            7137.77,
            10833.306,
            1906.68,
            10941.17,
            4472.735,
            4364.435,
            6585.27,
            12821.578,
            26258.71,
            9071.1,
            3798.987,
            10832.057,
            3889.75,
            3836.485,
            3968.59,
            5877.249,
            3890.166,
            9533.2,
            3971.198,
            18756.4,
            3727.1,
            4102.892,
            3729.86,
            3869.86,
            4341.692,
            4960.295,
            4862.71,
            6564.6,
            5008.24,
        ]
    )
    line_names = np.array(
        [
            "H I (Br-6)",
            "C II] 2326",
            "[O III] 2321",
            "[O I] 6302",
            "[S II] 4070",
            "H I (Ly-alpha)",
            "[Al II] 2670",
            "H I (Pf-5)",
            "[Ar III] 7753",
            "H I (Pf-gamma)",
            "H I (Pf-delta)",
            "H I (Br-5)",
            "H I (Pa-7)",
            "[Al II] 2660",
            "[S III] 6314",
            "H I (Pa-6)",
            "[S III] 3723",
            "Mg II 2800",
            "H I (Br-delta)",
            "H I (Pa-5)",
            "He I 7065",
            "[N II] 6549",
            "[S II] 6732",
            "H I (Pa-delta)",
            "C III]",
            "He I 6680",
            "Mg II 2800",
            "[S II] 6717",
            "H I (Br-gamma)",
            "H I (Br-alpha)",
            "[Ar III] 7138",
            "He I 10833",
            "[C III]",
            "H I (Pa-gamma)",
            "He I 4472",
            "[O III] 4364",
            "[N II] 6585",
            "H I (Pa-beta)",
            "H I (Br-beta)",
            "[S III] 9071",
            "H-8 3798",
            "He I 10829",
            "He I 3889",
            "H-7 3835",
            "[Ne III] 3968",
            "He I 5877",
            "H-6 3889",
            "[S III] 9533",
            "H-5 3970",
            "H I (Pa-alpha)",
            "[O II] 3726",
            "H-delta 4102",
            "[O II] 3729",
            "[Ne III] 3870",
            "H-gamma 4340",
            "[O III] 4960",
            "H-beta 4861",
            "H-alpha 6563",
            "[O III] 5007",
        ]
    )
    ind = line_lambdas_selected > 1e3
    ind &= line_lambdas_selected < 1e4
    line_names = line_names[ind]
    line_lambdas_selected = line_lambdas_selected[ind]

    n_lines = line_lambdas_selected.size
    line_idx_selected = []
    for l in line_lambdas_selected:
        diff = (l - line_lambdas) ** 2
        loc = np.argmin(diff)
        if diff[loc] > 1.0:
            print("Not loading line", l, "because nearest is", line_lambdas[l])
        line_idx_selected += [loc]
    line_idx_selected = np.array(line_idx_selected)
    assert np.max(np.abs(line_lambdas[line_idx_selected] - line_lambdas_selected)) < 10

    bandCoefs = np.concatenate(
        [
            np.load(dir_linegmms + "/line_gengmmcoeffs_" + filter + ".npy").astype(
                np.float32
            )[None, :]
            for filter in filternames
        ]
    )
    n = bandCoefs.shape[1] // 4
    gengnn_amps = bandCoefs[:, 0:n]
    gengnn_sigs = bandCoefs[:, n : 2 * n]
    gengnn_betas = bandCoefs[:, 2 * n : 3 * n]
    gengnn_locs = bandCoefs[:, 3 * n : 4 * n]


    return (
        n_lines,
        speculator_emlines,
        line_lambdas_selected,
        line_idx_selected,
        gengnn_amps,
        line_names,
        gengnn_locs[None, None, :, :],
        gengnn_sigs[None, None, :, :],
        gengnn_betas[None, None, :, :],
    )


def gennorm_pdf(x, beta, sigma, mean):
    """
    Torch implementation of the generalized normal distribution PDF.

    Based on https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

    Parameters
    ----------
    x : torch.Tensor
        Inputs to evaluate the PDF at.
    beta : torch.Tensor
        Shape parameter.
    sigma : torch.Tensor
        Scale parameter.
    mean : torch.Tensor
        Location parameter.

    Returns
    -------
    p : torch.Tensor
        Probability density evaluated at `x`.
    """

    return (beta / (2.0 * sigma * torch.exp(torch.special.gammaln(1.0 / beta)))
    ) * torch.exp(-1.0 * (torch.abs((x - mean) / sigma) ** beta))



class EmLineEmulator(torch.nn.Module):
    """
    Emission line emulator class.

    Attributes
    ----------
    filternames : list of str
        List of filter names.
    n_parameters : int
        Number of SPS parameters.
    n_hidden : list of int
        Number of hidden units per layer.
    device : str or torch.device
        Device the model is on.
    saved_model_dir : str
        Path the Speculator model was taken from.
    gengmm_dir : str
        Path the bandpass models were taken from.
    speculator_emlines = speculator.Speculator
        Speculator model.
    line_idx_selected = torch.Tensor
        Tensor of line indices used.
    gengnn_amps : torch.Tensor
        Tensor of generalised normal amplitudes.
    gengnn_locs : torch.Tensor
        Tensor of generalised normal locations.
    gengnn_sigs : torch.Tensor
        Tensor of generalised normal scales.
    gengnn_betas : torch.Tensor
        Tensor of generalised normal shapes.
    line_lambdas_selected : torch.Tensor
        Tensor of line wavelengths used.
    """

    def __init__(
        self,
        saved_model_dir,
        gengmm_dir,
        device,
        filternames,
        n_parameters,
        n_hidden=[128, 128, 128],
    ):
        """
        Parameters
        ----------
        saved_model_dir : str
            Path to the Speculator model.
        gengmm_dir : str
            Path to bandpass models.
        device : str or torch.device
            Device to place the model on.
        filternames : list of str
            List of banpass names.
        n_parameters : int
            Number of SPS parameters.
        n_hidden : list of int, optional
            Number of hidden units per network layer.
        """

        super().__init__()

        self.filternames = filternames
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.device = device
        self.saved_model_dir = saved_model_dir
        self.gengmm_dir = gengmm_dir

        # load the model using `load_lines_model`
        (
            n_lines,
            speculator_emlines,  # the emulator of emission line strengths from FSPS
            line_lambdas_selected,  # lambdas [AA] of selected lines
            line_idx_selected,  # indices of selected lines
            gengnn_amps,
            line_names,
            gengnn_locs,
            gengnn_sigs,
            gengnn_betas,
            # Gen Gaussian approximations of bandpasses
        ) = load_lines_model(
            self.saved_model_dir,
            self.n_parameters,
            self.filternames,
            self.device,
            self.gengmm_dir,
            self.n_hidden,
        )

        line_idx_selected = torch.Tensor(line_idx_selected)
        line_idx_selected = line_idx_selected.type(torch.int64)
        gengnn_amps = torch.Tensor(gengnn_amps)
        gengnn_locs = torch.Tensor(gengnn_locs)
        gengnn_sigs = torch.Tensor(gengnn_sigs)
        gengnn_betas = torch.Tensor(gengnn_betas)
        line_lambdas_selected = torch.Tensor(line_lambdas_selected)

        self.speculator_emlines = speculator_emlines.to(self.device)
        self.line_idx_selected = line_idx_selected.to(self.device)
        self.gengnn_amps = gengnn_amps.to(self.device)
        self.gengnn_locs = gengnn_locs.to(self.device)
        self.gengnn_sigs = gengnn_sigs.to(self.device)
        self.gengnn_betas = gengnn_betas.to(self.device)
        self.line_lambdas_selected = line_lambdas_selected.to(self.device)

    def apply_bandpasses(
        self,
        filternames,
        em_line_strengths_,
        theta,
        line_lambdas,
        gengnn_locs,
        gengnn_sigs,
        gengnn_betas,
        gengnn_amps,
    ):
        """
        Applies bandpass models to a list of lines.

        Parameters
        ----------
        filternames : list of str
            List of filternames to compute fluxes in.
        em_line_strengths_ : torch.Tensor
            Line strengths in solar luminosity per unit mass formed.
        theta : torch.Tensor
            SPS parameters.
        line_lambdas : torch.Tensor
            List of line wavelengths.
        gengnn_locs : torch.Tensor
            Bandpass model locations.
        gengnn_sigs : torch.Tensor
            Bandpass model scales.
        gengnn_betas : torch.Tensor
            Bandpass model shapes.
        gengnn_amps : torch.Tensor
            Bandpass model amplitudes.

        Returns
        -------
        em_line_phot_ : torch.Tensor
            Flux contributions of emission lines in nanomaggies.
        """

        # line locations as fct of redshift
        line_lambdas_z = (1 + theta[:, -1, None]) * line_lambdas[None, :].to(
            self.device
        )  # nobj, n_lines

        # # units
        flux_norm = (
            10 ** (9 - 0.4 * theta[..., 0]) * 3.2143882024295685e-07
        )  # nobj, nwalkers  ; number is to_cgs normalization.

        emline_phot_ = torch.empty(
            size=(theta.size(dim=0), len(filternames), line_lambdas_z.size(dim=1))
        ).to(self.device)
        # loop through lines and computing approximate flux using bandpass model
        for i in range(line_lambdas_z.size(dim=1)):

            # evaluate bandpasses
            bandpasses = gennorm_pdf(
                line_lambdas_z[:, i, None, None],
                gengnn_betas[0, :, :],
                gengnn_sigs[0, :, :],
                gengnn_locs[0, :, :],
            )

            # delta function of unit amplitude, redshifted, and evaluated through the bandpasses
            line_flux_window = (
                torch.sum(gengnn_amps[None, :, :] * bandpasses, dim=-1)  #
                * flux_norm[
                    :, None
                ]  # multiply by normalization of fluxes, in right units
                * line_lambdas_z[:, i, None]  # times lambda at z
            )  # nobj, n_bands

            emline_photometric_flux = (
                line_flux_window * em_line_strengths_[:, None, i]
            )  # nobj, nbands
            emline_phot_[:, :, i] = emline_photometric_flux

        return emline_phot_

    def forward(self, theta):
        """
        Computes emission line fluxes given SPS parameters.

        Parameters
        ----------
        theta : torch.Tensor
            Tensor of SPS parameters.

        Returns
        -------
        emline_bandpass_phot : torch.Tensor
            Emission line flux contributions to each bandpass.
            Units of nanomaggies (AB system).
        """

        # compute lines in units of solar luminosity per solar mass formed
        emlines_rffluxes = 10 ** (
            -0.4
            * (
                self.speculator_emlines.log_spectrum(theta)[
                    ..., self.line_idx_selected
                ]  # selected lines
            )
        )  # nobj, n_lines

        # push lines through bandpass models
        emline_bandpass_phot = self.apply_bandpasses(
            self.filternames,
            emlines_rffluxes,
            theta,
            self.line_lambdas_selected,
            self.gengnn_locs,
            self.gengnn_sigs,
            self.gengnn_betas,
            self.gengnn_amps,
        )

        return emline_bandpass_phot
