from typing import List, Tuple, Dict, Self
from collections.abc import Sequence
from copy import deepcopy
import inspect
from itertools import product
import json

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import erfc
import lmfit

from .model import LifetimeModel, combine_models, parse_model, dump_model
from .utils import may_be_nan


sqrt_2 = np.sqrt(2)


class LifetimeSpectrum:
    def __init__(
        self,
        spectrum: np.ndarray | List | Tuple | None = None,
        detname: str = "A",
        tcal: Tuple | List | None = None,
        name : str | None = None,
        lt_model: LifetimeModel | Dict | None = None,
        res_model: LifetimeModel | Dict | None = None,
        calibrate: bool = False,
        show_fits: bool = True,
        show: bool = False,
        verbose: bool = False,
        autocompute: bool = True,
        dtype: type | None = int,
        **kwargs
    ) -> None:

        self.name = name
        self.show_fits: bool = show_fits
        self.show: bool = show
        self.verbose: bool = verbose
        self.detname: str = detname

        self.spectrum = None if spectrum is None else np.array(spectrum, dtype=dtype)

        self.tcal: Tuple[float | int, ...] | List[float | int] | None = tcal
        self.times: None | np.ndarray = None
        if tcal is not None and self.spectrum is not None:
            self.times = tcal[0] + tcal[1] * np.arange(len(self.spectrum))

        if show and self.spectrum is not None:
            plt.figure()
            plt.title("Imported Spectrum")
            if self.times is not None:
                plt.semilogy(self.times, self.spectrum)
                plt.xlabel("Time")
            else:
                plt.semilogy(self.spectrum)
                plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        self.model = combine_models(lt_model, res_model)
        self.input_model = None if self.model is None else deepcopy(self.model)

        self.fit_result: None | lmfit.model.ModelResult = None

        self.peak_center = np.nan
        self.dpeak_center = np.nan

        self.peak_fwhm = np.nan
        self.dpeak_fwhm = np.nan

        if self.model is not None:
            self.n_l = len(self.model.lifetime_components)
            self.n_r = len(self.model.resolution_components)

            if calibrate is False:
                for res_comp in self.model.resolution_components:
                    res_comp.sigma.vary = False
                    res_comp.intensity.vary = False
                    res_comp.t0.vary = False

            self.l_vary = [lt.intensity.vary for lt in self.model.lifetime_components]
            l_vary_idcs = [i for i, vary in enumerate(self.l_vary) if vary]
            self.l_last_vary_idx = max(l_vary_idcs) if len(l_vary_idcs) > 0 else 0

            self.r_vary = [res.intensity.vary for res in self.model.resolution_components]
            r_vary_idcs = [j for j, vary in enumerate(self.r_vary) if vary]
            self.r_last_vary_idx = max(r_vary_idcs) if len(r_vary_idcs) > 0 else 0

        self.autocompute: bool = autocompute
        if (
            autocompute and
            self.spectrum is not None and
            self.times is not None and
            self.model is not None
        ):
            self.fit(**kwargs)

    def __getitem__(self, item):
        return getattr(self, item)

    def __add__(self, other: Self):
        new = LifetimeSpectrum(
            spectrum = self.spectrum + other.spectrum,
            detname = self.detname if self.detname == other.detname else "A",
            tcal = (
                None if any(t is None for t in (self.tcal, other.tcal))
                else np.average((self.tcal, other.tcal), axis=0)
            ),
            name = (
                None if any(n is None for n in (self.name, other.name))
                else " + ".join(self.name, other.name)
            ),
            show_fits = self.show_fits or other.show_fits,
            show = self.show or other.show,
            verbose = self.verbose or other.verbose,
            autocompute = self.autocompute and other.autocompute,
            dtype = None,
        )
        new.model = self.model
        return new

    def make_model(
        self,
        n: float | int,
        shift: float | int
    ) -> Tuple[lmfit.Model, lmfit.Parameters]:
        assert self.spectrum is not None
        assert self.times is not None
        assert self.model is not None
        assert len(self.model.lifetime_components) > 0
        assert len(self.model.resolution_components) > 0

        self.n_l = len(self.model.lifetime_components)
        self.n_r = len(self.model.resolution_components)

        self.l_vary = [lt.intensity.vary for lt in self.model.lifetime_components]
        l_vary_idcs = [i for i, vary in enumerate(self.l_vary) if vary]
        self.l_last_vary_idx = max(l_vary_idcs) if len(l_vary_idcs) > 0 else 0

        self.r_vary = [res.intensity.vary for res in self.model.resolution_components]
        r_vary_idcs = [j for j, vary in enumerate(self.r_vary) if vary]
        self.r_last_vary_idx = max(r_vary_idcs) if len(r_vary_idcs) > 0 else 0

        def convolved_decay(t, N, t0, background, **kwargs):
            y = np.full(t.shape, 0.0)

            l_int = [
                kwargs[f"h_{i}"] if v else kwargs[f"intensity_{i}"]
                for i, v in enumerate(self.l_vary, 1)
            ]
            l_cumul = sum(i * (not v) for i, v in zip(l_int, self.l_vary))
            for i in range(self.n_l):
                if self.l_vary[i]:
                    l_int[i] *= max(1 - l_cumul, 0)
                    l_cumul += l_int[i]

            r_int = [
                kwargs[f"res_h_{j}"] if v else kwargs[f"res_intensity_{j}"]
                for j, v in enumerate(self.r_vary, 1)
            ]
            r_cumul = sum(i * (not v) for i, v in zip(r_int, self.r_vary))
            for j in range(self.n_r):
                if self.r_vary[j]:
                    r_int[j] *= max(1 - r_cumul, 0)
                    r_cumul += r_int[j]

            for i, j in product(range(1, self.n_l+1), range(1, self.n_r+1)):
                l_tau = kwargs[f"lifetime_{i}"]
                l_i = l_int[i-1]
                r_sigma = kwargs[f"res_sigma_{j}"]
                r_i = r_int[j-1]
                r_t0 = kwargs[f"res_t0_{j}"]
                # e becomes numerically unstable for lifetimes ~ tcal[1]
                _t = t - (t0 + r_t0)
                e = np.exp(-_t / l_tau + (r_sigma/(sqrt_2*l_tau))**2)
                c = erfc(r_sigma / (l_tau * sqrt_2) - _t / (r_sigma * sqrt_2))
                y += (l_i * r_i / (2*l_tau)) * e * c

            if self.verbose and np.any(np.isnan(y)):
                print(f"{'Parameters of failed model eval':-^50}")
                print(f"{N = }")
                print(f"{t0 = }")
                print(f"{background = }")
                for i in range(1, self.n_l+1):
                    l = kwargs[f"lifetime_{i}"]
                    h = kwargs[f"h_{i}"]
                    print(f"lifetime_{i} = {l}")
                    print(f"h_{i} = {h}")
                    print(f"intensity_{i} = {l_int[i-1]}")
                for j in range(1, self.n_r+1):
                    s = kwargs[f"res_sigma_{j}"]
                    h = kwargs[f"res_h_{j}"]
                    t0 = kwargs[f"res_t0_{j}"]
                    print(f"res_sigma_{j} = {s}")
                    print(f"res_h_{j} = {h}")
                    print(f"res_t0_{j} = {t0}")
                    print(f"res_intensity_{j} = {r_int[j-1]}")
                print("-"*50)

            return N * y + background

        p = lambda x: inspect.Parameter(x, inspect.Parameter.POSITIONAL_OR_KEYWORD)

        convolved_decay.__signature__ = inspect.Signature([
            p("t"), p("N"), p(f"t0"), p(f"background"),
            *[p(f"lifetime_{i}") for i in range(1, self.n_l+1)],
            *[p(f"intensity_{i}") for i in range(1, self.n_l+1)],
            *[p(f"h_{i}") for i in range(1, self.n_l+1)],
            *[p(f"res_sigma_{j}") for j in range(1, self.n_r+1)],
            *[p(f"res_intensity_{j}") for j in range(1, self.n_r+1)],
            *[p(f"res_h_{j}") for j in range(1, self.n_r+1)],
            *[p(f"res_t0_{j}") for j in range(1, self.n_r+1)],
        ])

        model = lmfit.Model(convolved_decay)
        params = lmfit.Parameters()

        l = np.min(self.times)
        r = np.max(self.times)
        d = r - l

        params.add("N", n, min=0)
        params.add("t0", shift, min=l-d, max=r+d)

        if self.model.background_component is not None:
            param = self.model.background_component.background
            param.name = "background"
            params.add(deepcopy(param))
        else:
            params.add("background", 0, vary=False)

        #### Lifetime Components

        for i, lt in enumerate(self.model.lifetime_components, 1):
            param = lt.lifetime
            param.name = f"lifetime_{i}"
            params.add(deepcopy(param))

            param = lt.intensity
            param.name = f"intensity_{i}"
            param = deepcopy(param)
            param.vary = False
            params.add(param)

            param = lt.intensity
            param.name = f"h_{i}"
            param.min = min(max(param.min, 0), 1)
            param.max = max(min(param.max, 1), 0)
            if self.n_l == 1 or i == self.l_last_vary_idx + 1:
                param.value = 1
                param.vary = False
            params.add(deepcopy(param))

        #### Resolution Components

        for j, res in enumerate(self.model.resolution_components, 1):
            param = res.sigma
            param.name = f"res_sigma_{j}"
            params.add(deepcopy(param))

            param = res.intensity
            param.name = f"res_intensity_{j}"
            param = deepcopy(param)
            param.vary = False
            params.add(param)

            param = res.intensity
            param.name = f"res_h_{j}"
            param.min = min(max(param.min, 0), 1)
            param.max = max(min(param.max, 1), 0)
            if self.n_r == 1 or (j == self.r_last_vary_idx + 1 and self.n_r > 1) :
                param.value = 1
                param.vary = False
            params.add(deepcopy(param))

            param = res.t0
            param.name = f"res_t0_{j}"
            params.add(deepcopy(param))

        return model, params

    def fit(
        self,
        fit_channels_left_of_peak: int = 500,
        fit_channels_right_of_peak: int = 3000,
        left_fit_idx: None | int = None,
        right_fit_idx: None | int = None,
        channel_mask: Ellipsis | List[bool | int] = ...,
        **kwargs
    ) -> lmfit.model.ModelResult:
        """Fit the spectrum to determine lifetimes and intensities.

        Parameters
        ----------
        fit_channels_left_of_peak : int
            How many channels to the left of the peak to use for fitting.
            The default is 500.
        fit_channels_right_of_peak : int
            How many channels to the right of the peak to use for fitting.
            The default is 3000.
        left_fit_idx : int or None
            Channel index in the uncut spectrum that marks the beginning of the
            part to use for the fit. If provided, `fit_channels_left_of_peak` is
            ignored. If None, `fit_channels_left_of_peak` is used instead.
            The default is None.
        right_fit_idx : int or None
            Channel index in the uncut spectrum that marks the end of the
            part to use for the fit. If provided, `fit_channels_right_of_peak`
            is ignored. If None, `fit_channels_right_of_peak` is used instead.
            The default is None.
        channel_mask : Sequence of bool or Sequence of int or Ellipsis
            Boolean mask or fancy index set to determine, which parts in the
            cut spectrum to use in the fit and which parts to ignore.
            The default is Ellipsis (use all).
        """
        err = "LifetimeSpectrum is missing {}. Could not fit"
        assert self.spectrum is not None, err.format("spectrum data")
        assert self.times is not None, err.format("time calibration")
        assert self.model is not None, err.format("fit model")

        # Initial Guesses
        peak_center_guess_idx = np.argmax(self.spectrum)
        t0_idx = np.argmax(self.spectrum)
        n = np.sum(self.spectrum) * 2
        shift = self.times[t0_idx]

        peak_range = slice(max(peak_center_guess_idx-15, 0), peak_center_guess_idx+15)
        peak = self.spectrum[peak_range]
        peak_times = self.times[peak_range]

        peak_model = lmfit.models.GaussianModel()

        peak_params = peak_model.make_params()

        peak_params["amplitude"].set(value=n, min=0)
        peak_params["center"].set(value=shift)
        peak_params["sigma"].set(value=n/(self.spectrum[t0_idx] * np.sqrt(2*np.pi)))

        peak_result = peak_model.fit(peak, x=peak_times, **peak_params)

        if self.show:
            peak_result.plot()
            plt.title("Peak params via gaussian fit")
            plt.show()

        self.peak_center = peak_result.params["center"].value
        self.dpeak_center = may_be_nan(peak_result.params["center"].stderr)

        self.peak_fwhm = peak_result.params["sigma"].value * 2.3548
        self.dpeak_fwhm = may_be_nan(peak_result.params["sigma"].stderr) * 2.3548

        peak_idx = np.searchsorted(self.times, self.peak_center)

        if left_fit_idx is None:
            left_fit_idx = peak_idx - fit_channels_left_of_peak

        if right_fit_idx is None:
            right_fit_idx = peak_idx + fit_channels_right_of_peak

        data_range = slice(max(left_fit_idx, 0), right_fit_idx)

        self.trimmed_times = self.times[data_range]
        self.trimmed_spectrum = self.spectrum[data_range]

        model, params = self.make_model(n, self.peak_center)

        weights = np.sqrt(1/np.where(self.trimmed_spectrum > 0, self.trimmed_spectrum, 1))

        if self.show:
            plt.figure()
            plt.title("Cut out spectrum and initial guess for fit")
            plt.semilogy(self.trimmed_times, self.trimmed_spectrum, label="Data")
            plt.semilogy(self.trimmed_times, model.eval(t=self.trimmed_times, **params), label="Initial Guess")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Fit
        self.fit_result = model.fit(
            self.trimmed_spectrum[channel_mask],
            t=self.trimmed_times[channel_mask],
            weights=weights[channel_mask],
            **params, **kwargs
        )

        params = self.fit_result.params

        l_int = [
            params[f"h_{i}"].value if v else params[f"intensity_{i}"].value
            for i, v in enumerate(self.l_vary, 1)
        ]
        l_dint = [
            may_be_nan(params[f"h_{i}"].stderr)
            for i in range(1, self.n_l+1)
        ]
        l_cumul = sum(i * (not v) for i, v in zip(l_int, self.l_vary))
        l_cumul_sq_err = 0
        a, b = 0, 0
        for i in range(self.n_l):
            if self.l_vary[i]:
                h_i = l_int[i]
                l_int[i] *= max(1 - l_cumul, 0)
                l_cumul += l_int[i]
                l_cumul_sq_err += (l_dint[i]/h_i)**2
                l_dint[i] = l_int[i] * np.sqrt(l_cumul_sq_err)
                a, b = b, l_dint[i]
        l_dint[self.l_last_vary_idx] = a

        for i in range(1, self.n_l+1):
            params[f"intensity_{i}"] = param = params.pop(f"h_{i}")
            param.name = f"intensity_{i}"
            param.value = l_int[i-1]
            param.vary = self.l_vary[i-1]
            param.stderr = l_dint[i-1]

        r_int = [
            params[f"res_h_{j}"].value if v else params[f"res_intensity_{j}"].value
            for j, v in enumerate(self.r_vary, 1)
        ]
        r_dint = [
            may_be_nan(params[f"res_h_{j}"].stderr)
            for j in range(1, self.n_r+1)
        ]
        r_cumul = sum(i * (not v) for i, v in zip(r_int, self.r_vary))
        r_cumul_sq_err = 0
        a, b = 0, 0
        for j in range(self.n_r):
            if self.r_vary[j]:
                h_j = r_int[j]
                r_int[j] *= max(1 - r_cumul, 0)
                r_cumul += r_int[j]
                r_cumul_sq_err += (r_dint[j]/h_j)**2
                r_dint[j] = r_int[j] * np.sqrt(r_cumul_sq_err)
                a, b = b, r_dint[j]
        r_dint[self.r_last_vary_idx] = a

        for j in range(1, self.n_r+1):
            params[f"res_intensity_{j}"] = param = params.pop(f"res_h_{j}")
            param.name = f"res_intensity_{j}"
            param.value = r_int[j-1]
            param.vary = self.r_vary[j-1]
            param.stderr = r_dint[j-1]

        for i in range(1, self.n_l+1):
            setattr(self, f"lifetime_{i}", params[f"lifetime_{i}"].value)
            setattr(self, f"dlifetime_{i}", may_be_nan(params[f"lifetime_{i}"].stderr))
            setattr(self, f"intensity_{i}", params[f"intensity_{i}"].value)
            setattr(self, f"dintensity_{i}", may_be_nan(params[f"intensity_{i}"].stderr))

        for j in range(1, self.n_r+1):
            setattr(self, f"res_sigma_{j}", params[f"res_sigma_{j}"].value)
            setattr(self, f"dres_sigma_{j}", may_be_nan(params[f"res_sigma_{j}"].stderr))
            setattr(self, f"res_intensity_{j}", params[f"res_intensity_{j}"].value)
            setattr(self, f"dres_intensity_{j}", may_be_nan(params[f"res_intensity_{j}"].stderr))
            setattr(self, f"res_t0_{j}", params[f"res_t0_{j}"].value)
            setattr(self, f"dres_t0_{j}", may_be_nan(params[f"res_t0_{j}"].stderr))

        setattr(self, f"n", params[f"N"].value)
        setattr(self, f"dn", may_be_nan(params[f"N"].stderr))
        setattr(self, f"t0", params[f"t0"].value)
        setattr(self, f"dt0", may_be_nan(params[f"t0"].stderr))
        setattr(self, f"background", params[f"background"].value)
        setattr(self, f"dbackgroud", may_be_nan(params[f"background"].stderr))

        self.times -= params[f"t0"].value
        self.peak_center -= params[f"t0"].value

        lifetimes = [params[f"lifetime_{i}"].value for i in range(1, self.n_l+1)]
        intensities = [params[f"intensity_{i}"].value for i in range(1, self.n_l+1)]
        dlifetimes = [may_be_nan(params[f"lifetime_{i}"].stderr) for i in range(1, self.n_l+1)]
        dintensities = [may_be_nan(params[f"intensity_{i}"].stderr) for i in range(1, self.n_l+1)]

        self.mean_lifetime = np.nan
        self.dmean_lifetime = np.nan

        self.mean_lifetime = sum([l * i for l, i in zip(lifetimes, intensities)])
        self.dmean_lifetime = np.sqrt(np.sum(np.square([l * di + dl * i for l, dl, i, di in zip(lifetimes, dlifetimes, intensities, dintensities)])))
        params["mean_lifetime"] = lmfit.Parameter(
            "mean_lifetime",
            value=self.mean_lifetime,
        )
        params["mean_lifetime"].stderr = self.dmean_lifetime

        if self.verbose:
            self.fit_report()

        if self.show_fits or self.show:
            self.plot_fit_result()

        self.model = parse_model({
            k: [v.value, v.vary, v.min, v.max] for k, v in params.items()
        })

        return self.fit_result

    def plot_fit_result(
        self,
        show_init: bool = True,
        show: bool = True,
    ) -> Tuple[Figure, Tuple[plt.Axes, ...]]:
        assert self.trimmed_spectrum is not None
        assert self.trimmed_times is not None
        assert self.fit_result is not None

        fig = plt.figure()
        fig.suptitle(f"Fit Result\n{self.name}")

        gs = GridSpec(2, 1, height_ratios=[1, 4])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax2.set_xlabel("Time [ps]")
        ax2.set_ylabel("Counts")

        ax1.set_ylabel("Residuals")

        ax1.scatter(self.trimmed_times, self.fit_result.residual)

        ax2.scatter(self.trimmed_times, self.trimmed_spectrum, label="Data")
        if show_init:
            ax2.plot(
                self.trimmed_times, self.fit_result.init_fit, linestyle="--",
                c="C1", label="Initial Fit"
            )
        ax2.plot(
            self.trimmed_times, self.fit_result.best_fit, linestyle="-",
            c="C2", label="Best Fit"
        )

        ax2.set_yscale("log")
        ax2.set_ylim([1, np.max(self.trimmed_spectrum) * 1.5])

        if show:
            ax1.grid()
            ax2.grid()
            ax2.legend()
            plt.show()

        return fig, (ax1, ax2)

    def fit_report(self, sep: str = "") -> None:
        err = "No fit has been performed yet and/or the fit prerequisites are not fulfilled"
        assert self.spectrum is not None, err
        assert self.model is not None, err
        assert self.fit_result is not None, err

        result = self.fit_result
        rparams = result.params

        n_l = len(self.model.lifetime_components)
        n_r = len(self.model.resolution_components)

        headers = ["Parameter", "Value", "Stderr (abs)", "Stderr (rel)",
                   "Min", "Max", "Fixed"]
        formats = ["<", ">", ">", ">", ">", ">", "<"]

        def print_params(params):
            table_data = np.array([
                [
                    p.name if not p.name.startswith("res_") else p.name[4:],
                    f"{100 * p.value:.1f}" if "int" in p.name else f"{p.value:.4g}",
                    (
                        (f"{100 * p.stderr:.1f}" if "int" in p.name else f"{p.stderr:.4g}")
                        if p.stderr is not None else "nan"
                    ),
                    f"{abs(100 * p.stderr / p.value):.3f} %" if p.value != 0 and p.stderr is not None else "nan",
                    f"{p.min:.2f}",
                    f"{p.max:.2f}",
                    "" if p.vary else "Fixed",
                ]
                for p in params
            ], dtype=str)

            widths = np.max([np.max(np.vectorize(len)(table_data), axis=0),
                            np.vectorize(len)(headers)], axis=0)

            for i, head in enumerate(headers):
                print(f"{sep} {head:^{widths[i]}} ", end="")
            print(sep)
            for i, head in enumerate(headers):
                print(f"{sep} {'-' * widths[i]} ", end="")
            print(sep)

            for row in table_data:
                for col, fmt, wid in zip(row, formats, widths):
                    print(f"{sep} {col:{fmt}{wid}} ", end="")
                print(sep)

        print("#"*100)
        print(f"{' Fit report ':^100}")
        print("#"*100)
        print()
        print(f"Spectrum Name: {self.name}")
        print()
        print(f"Statistics: {np.sum(self.spectrum):,}")
        print(f"Number of Data Points: {result.ndata}")
        print()
        print(f"Lifetime Components:   {n_l}")
        print(f"Resolution Components: {n_r}")
        print(f"Background Components: {int(self.model.background_component is not None)}")
        print()
        print(f"Fit Method: {result.method}")
        print(f"Fit Status: {'Success' if result.success else 'Failure'}")
        print(f"Solver Message: {result.message}")
        print(f"Reason for Termination: {result.lmdif_message}")
        print(f"Number of Function Evaluations: {result.nfev}")
        print(f"Reduced Chi squared: {result.redchi}")
        print()
        print("-"*100)
        print(f"{'Parameters':^100}")
        print("-"*100)
        print()
        print(f"{' General Parameters ':=^100}")
        print()

        print_params([rparams[p] for p in ["N", "t0", "background", "mean_lifetime"]])

        print()
        print(f"{' Lifetime Components ':=^100}")
        print()

        params = ["lifetime", "intensity"]
        params = [f"{p}_{i}" for p in params for i in range(1, n_l+1)]
        print_params([rparams[p] for p in params])

        print()
        print(f"{' Resolution Components ':=^100}")
        print()

        params = ["res_sigma", "res_intensity", "res_t0"]
        params = [f"{p}_{i}" for p in params for i in range(1, n_r+1)]
        print_params([rparams[p] for p in params])

        print()
        print("-"*100)

    def dump_components(self, filepath=None):
        err = "No fit has been performed successfully"
        assert self.model is not None, err
        assert self.fit_result is not None, err

        out = dump_model(self.model)

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)

        return out

    def dump_resolution_components(self, filepath=None):
        assert self.model is not None

        out = dump_model(self.model)

        out = {k: v for k, v in out.items() if k.startswith("res")}

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)

        return out

    def dump_lifetime_components(self, filepath=None):
        assert self.model is not None

        out = dump_model(self.model)

        out = {k: v for k, v in out.items() if not k.startswith("res")}

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)

        return out
