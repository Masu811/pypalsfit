from typing import List, Tuple, Dict, Self, Callable
from types import EllipsisType
from collections.abc import Sequence
from copy import deepcopy
import inspect
from itertools import product
import json

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RangeSlider
from scipy.special import erfc
import lmfit

from .model import LifetimeModel, combine_models, parse_model, dump_model, gauss
from .utils import may_be_nan


sqrt_2 = np.sqrt(2)
sqrt_2_pi = np.sqrt(2*np.pi)
sqrt_2_inv = 1 / sqrt_2
sqrt_2_pi_inv = 1 / sqrt_2_pi


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
            plt.title(f"Imported Spectrum\n{self.name}")
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

        self.bg = np.nan

        self.trimmed_times = None
        self.trimmed_spectrum = None

        self.n_l = None
        self.n_r = None
        self.l_vary = None
        self.r_vary = None
        self.l_last_vary_idx = None
        self.r_last_vary_idx = None

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
        assert self.spectrum is not None and other.spectrum is not None
        new = LifetimeSpectrum(
            spectrum = self.spectrum + other.spectrum,
            detname = self.detname if self.detname == other.detname else "A",
            tcal = (
                None if self.tcal is None or other.tcal is None
                else np.average((self.tcal, other.tcal), axis=0)
            ),
            name = (
                None if self.name is None or other.name is None
                else " + ".join([self.name, other.name])
            ),
            show_fits = self.show_fits or other.show_fits,
            show = self.show or other.show,
            verbose = self.verbose or other.verbose,
            autocompute = self.autocompute and other.autocompute,
            dtype = None,
        )
        new.model = self.model
        return new

    def make_model_func(self):
        assert self.model is not None
        assert len(self.model.lifetime_components) > 0
        assert len(self.model.resolution_components) > 0

        self.n_l = n_l = len(self.model.lifetime_components)
        self.n_r = n_r = len(self.model.resolution_components)

        self.l_vary = l_vary = [lt.intensity.vary for lt in self.model.lifetime_components]
        l_vary_idcs = [i for i, vary in enumerate(self.l_vary) if vary]
        self.l_last_vary_idx = l_last_vary_idx = max(l_vary_idcs) if len(l_vary_idcs) > 0 else 0

        self.r_vary = r_vary = [res.intensity.vary for res in self.model.resolution_components]
        r_vary_idcs = [j for j, vary in enumerate(self.r_vary) if vary]
        self.r_last_vary_idx = r_last_vary_idx = max(r_vary_idcs) if len(r_vary_idcs) > 0 else 0

        def func(t, N, t0, background, **kwargs):
            y = np.zeros_like(t, dtype=float)

            l_int = [
                kwargs[f"h_{i}"] if v else kwargs[f"intensity_{i}"]
                for i, v in enumerate(l_vary, 1)
            ]
            l_cumul = sum(i for i, v in zip(l_int, l_vary) if not v)
            for i in range(n_l):
                if l_vary[i]:
                    l_int[i] *= max(1 - l_cumul, 0)
                    l_cumul += l_int[i]

            r_int = [
                kwargs[f"res_h_{j}"] if v else kwargs[f"res_intensity_{j}"]
                for j, v in enumerate(r_vary, 1)
            ]
            r_cumul = sum(i for i, v in zip(r_int, r_vary) if not v)
            for j in range(n_r):
                if r_vary[j]:
                    r_int[j] *= max(1 - r_cumul, 0)
                    r_cumul += r_int[j]

            for i, j in product(range(1, n_l+1), range(1, n_r+1)):
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
                for i in range(1, n_l+1):
                    l = kwargs[f"lifetime_{i}"]
                    h = kwargs[f"h_{i}"]
                    print(f"lifetime_{i} = {l}")
                    print(f"h_{i} = {h}")
                    print(f"intensity_{i} = {l_int[i-1]}")
                for j in range(1, n_r+1):
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

        setattr(func, "__signature__", inspect.Signature([
            p("t"), p("N"), p(f"t0"), p(f"background"),
            *[p(f"lifetime_{i}") for i in range(1, n_l+1)],
            *[p(f"intensity_{i}") for i in range(1, n_l+1)],
            *[p(f"h_{i}") for i in range(1, n_l+1)],
            *[p(f"res_sigma_{j}") for j in range(1, n_r+1)],
            *[p(f"res_intensity_{j}") for j in range(1, n_r+1)],
            *[p(f"res_h_{j}") for j in range(1, n_r+1)],
            *[p(f"res_t0_{j}") for j in range(1, n_r+1)],
        ]))

        def dfunc(params, data, weights, t):
            N = params["N"]
            t0 = params["t0"]
            background = params["background"]

            l_int = [
                params[f"h_{i}"] if v else params[f"intensity_{i}"]
                for i, v in enumerate(l_vary, 1)
            ]
            l_cumul = sum(i for i, v in zip(l_int, l_vary) if not v)
            for i in range(n_l):
                if l_vary[i]:
                    l_int[i] *= max(1 - l_cumul, 0)
                    l_cumul += l_int[i]

            r_int = [
                params[f"res_h_{j}"] if v else params[f"res_intensity_{j}"]
                for j, v in enumerate(r_vary, 1)
            ]
            r_cumul = sum(i for i, v in zip(r_int, r_vary) if not v)
            for j in range(n_r):
                if r_vary[j]:
                    r_int[j] *= max(1 - r_cumul, 0)
                    r_cumul += r_int[j]

            terms = {}
            deriv = {}

            y = np.zeros_like(t)

            for i, j in product(range(1, n_l+1), range(1, n_r+1)):
                l_tau = params[f"lifetime_{i}"]
                l_i = l_int[i-1]
                r_sigma = params[f"res_sigma_{j}"]
                r_i = r_int[j-1]
                r_t0 = params[f"res_t0_{j}"]
                # e becomes numerically unstable for lifetimes ~ tcal[1]
                _t = t - (t0 + r_t0)
                a = -_t / l_tau + (r_sigma/(sqrt_2*l_tau))**2
                b = r_sigma / (l_tau * sqrt_2) - _t / (r_sigma * sqrt_2)
                e = np.exp(a)
                c = erfc(b)
                decay = N * l_i * r_i / (2*l_tau) * e * c
                y += decay
                exp_comb = N * (l_i * r_i) / (sqrt_2_pi * l_tau) * np.exp(a - b**2)

                df_dt0 = -exp_comb / r_sigma + decay / l_tau
                df_dtau = exp_comb * r_sigma / l_tau**2 + decay * (-r_sigma**2/l_tau**3 + _t/l_tau**2 - 1/l_tau)
                df_dsigma = decay * r_sigma / l_tau**2 - exp_comb * (1/l_tau + _t/r_sigma**2)
                df_dr_t0 = df_dt0

                terms[f"{i}_{j}"] = decay

                deriv[f"{i}_{j}"] = [
                    df_dt0,
                    df_dtau,
                    df_dsigma,
                    df_dr_t0,
                ]

            df_dN = (y - background) / N
            df_dbackground = np.ones_like(t, dtype=float)
            df_dt0 = np.sum([
                deriv[f"{i}_{j}"][0]
                for j in range(1, n_r+1)
                for i in range(1, n_l+1)
            ], axis=0)

            l = l_vary.copy()
            l[l_last_vary_idx] = False
            r = r_vary.copy()
            r[r_last_vary_idx] = False

            df_dh_i = []

            for k in range(1, n_l+1):
                df_dh_k = []
                if not l[k-1]:
                    df_dh_i.append(np.zeros_like(t, dtype=float))
                    continue
                for n, m in product(range(1, n_l+1), range(1, n_r+1)):
                    if n < k:
                        continue
                    elif n == k:
                        df_dh_k.append(terms[f"{n}_{m}"] / (params[f"h_{k}"] + 1e-4))
                    else:
                        df_dh_k.append(-terms[f"{n}_{m}"] / (1 - params[f"h_{k}"] + 1e-4))

                df_dh_i.append(np.sum(df_dh_k, axis=0))

            df_dres_h_j = []

            for k in range(1, n_r+1):
                df_dres_h_k = []
                if not r[k-1]:
                    df_dres_h_j.append(np.zeros_like(t, dtype=float))
                    continue
                for n, m in product(range(1, n_l+1), range(1, n_r+1)):
                    if m < k:
                        continue
                    elif m == k:
                        df_dres_h_k.append(terms[f"{n}_{m}"] / (params[f"res_h_{k}"] + 1e-4))
                    else:
                        df_dres_h_k.append(-terms[f"{n}_{m}"] / (1 - params[f"res_h_{k}"] + 1e-4))

                df_dres_h_j.append(np.sum(df_dres_h_k, axis=0))

            out = {
                "N": df_dN,
                "t0": df_dt0,
                "background": df_dbackground,
                **{
                    f"lifetime_{i}": np.sum(
                        [deriv[f"{i}_{j}"][1] for j in range(1, n_r+1)], axis=0
                    ) for i in range(1, n_l+1)
                },
                **{
                    f"h_{i}": df_dh_i[i-1] for i in range(1, n_l+1)
                },
                **{
                    f"res_sigma_{j}": np.sum(
                        [deriv[f"{i}_{j}"][2] for i in range(1, n_l+1)], axis=0
                    ) for j in range(1, n_r+1)
                },
                **{
                    f"res_h_{j}": df_dres_h_j[j-1] for j in range(1, n_r+1)
                },
                **{
                    f"res_t0_{j}": np.sum(
                        [deriv[f"{i}_{j}"][3] for i in range(1, n_l+1)], axis=0
                    ) for j in range(1, n_r+1)
                }
            }

            out = np.array([out[name] for name, param in params.items() if param.vary])

            return -out * weights

        return func, dfunc

    def make_model(
        self,
    ) -> Tuple[lmfit.Model, lmfit.Parameters, Callable, Callable]:
        assert self.spectrum is not None
        assert self.times is not None
        assert self.model is not None
        assert len(self.model.lifetime_components) > 0
        assert len(self.model.resolution_components) > 0
        assert self.l_last_vary_idx is not None
        assert self.r_last_vary_idx is not None

        func, dfunc = self.make_model_func()

        model = lmfit.Model(func)
        params = lmfit.Parameters()

        l = np.min(self.times)
        r = np.max(self.times)
        d = r - l

        # TODO: implement possibility to provide Parameters for N and t0
        n = np.sum(self.spectrum) * 2

        params.add("N", n, min=0)
        params.add("t0", self.peak_center, min=l-d, max=r+d)

        if self.model.background_component is not None:
            param = self.model.background_component.background
            param.name = "background"
            if not np.isnan(self.bg):
                param.value = self.bg
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
            if i == self.l_last_vary_idx + 1:
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
            if j == self.r_last_vary_idx + 1:
                param.value = 1
                param.vary = False
            params.add(deepcopy(param))

            param = res.t0
            param.name = f"res_t0_{j}"
            if j == 1:
                param.value = 0
                param.vary = False
            params.add(deepcopy(param))

        return model, params, func, dfunc

    def get_fit_range(
        self,
        left_fit_idx: None | int = None,
        right_fit_idx: None | int = None,
    ) -> Tuple[int, int]:
        # TODO: Outsource plotting code to a shared function (share with self.get_bg)
        assert self.spectrum is not None

        if left_fit_idx is None or right_fit_idx is None:
            fig, axs = plt.subplots(1, 2)

            fig.suptitle(f"Fit Range Determination (Close Plot to Confirm)\n{self.name}")

            left_fit_idx = left_fit_idx or 0
            right_fit_idx = right_fit_idx or len(self.spectrum) - 1

            axs[0].semilogy(self.spectrum)
            l_lim = axs[0].axvline(left_fit_idx, c="C1")
            u_lim = axs[0].axvline(right_fit_idx, c="C1")
            axs[0].set_xlabel("Channel")
            axs[0].set_ylabel("Counts")
            axs[0].set_title("Entire Spectrum")

            bg_img, = axs[1].semilogy(
                np.linspace(0, 1, right_fit_idx - left_fit_idx),
                self.spectrum[left_fit_idx:right_fit_idx]
            )
            axs[1].set_title("Fit Region")

            fig.subplots_adjust(bottom=0.25)

            slider_ax = fig.add_axes((0.20, 0.1, 0.60, 0.03))
            slider = RangeSlider(
                slider_ax,
                "Fit Range",
                left_fit_idx,
                right_fit_idx,
                valinit=(left_fit_idx, right_fit_idx),
            )

            lower_lim = [left_fit_idx]
            upper_lim = [right_fit_idx]

            def update(val):
                lower_lim[0] = left_fit_idx = int(slider.val[0])
                upper_lim[0] = right_fit_idx = int(slider.val[1])

                l_lim.set_xdata([left_fit_idx] * 2)
                u_lim.set_xdata([right_fit_idx] * 2)

                ydata = self.spectrum[left_fit_idx:right_fit_idx]
                xdata = np.linspace(0, 1, right_fit_idx - left_fit_idx)

                axs[1].set_title("Fit Region")

                bg_img.set_ydata(ydata)
                bg_img.set_xdata(xdata)

                fig.canvas.draw_idle()

            slider.on_changed(update)

            plt.show()

            left_fit_idx = lower_lim[0]
            right_fit_idx = upper_lim[0]

        return (left_fit_idx, right_fit_idx)

    def get_bg(
        self,
        bg_start_idx: None | int = None,
        bg_end_idx: None | int = None,
    ) -> None:
        # TODO: Outsource plotting code to a shared function (share with self.get_fit_range)
        assert self.spectrum is not None

        if bg_start_idx is None or bg_end_idx is None:
            fig, axs = plt.subplots(1, 2)

            fig.suptitle(f"Background Determination (Close Plot to Confirm)\n{self.name}")

            bg_start_idx = bg_start_idx or 0
            bg_end_idx = bg_end_idx or len(self.spectrum) - 1

            axs[0].semilogy(self.spectrum)
            l_lim = axs[0].axvline(bg_start_idx, c="C1")
            u_lim = axs[0].axvline(bg_end_idx, c="C1")
            axs[0].set_xlabel("Channel")
            axs[0].set_ylabel("Counts")
            axs[0].set_title("Entire Spectrum")

            bg_img, = axs[1].semilogy(
                np.linspace(0, 1, bg_end_idx - bg_start_idx),
                self.spectrum[bg_start_idx:bg_end_idx]
            )
            avg = np.mean(self.spectrum[bg_start_idx:bg_end_idx])
            axs[1].set_title(f"Background Region\nAverage Counts: {avg:.4f}")

            fig.subplots_adjust(bottom=0.25)

            slider_ax = fig.add_axes((0.20, 0.1, 0.60, 0.03))
            slider = RangeSlider(
                slider_ax,
                "Background Range",
                bg_start_idx,
                bg_end_idx,
                valinit=(bg_start_idx, bg_end_idx),
            )

            lower_lim = [bg_start_idx]
            upper_lim = [bg_end_idx]

            def update(val):
                lower_lim[0] = bg_start_idx = int(slider.val[0])
                upper_lim[0] = bg_end_idx = int(slider.val[1])

                l_lim.set_xdata([bg_start_idx] * 2)
                u_lim.set_xdata([bg_end_idx] * 2)

                ydata = self.spectrum[bg_start_idx:bg_end_idx]
                xdata = np.linspace(0, 1, bg_end_idx - bg_start_idx)

                axs[1].set_title(f"Background Region\nAverage Counts: {np.mean(ydata):.4f}")

                bg_img.set_ydata(ydata)
                bg_img.set_xdata(xdata)

                fig.canvas.draw_idle()

            slider.on_changed(update)

            plt.show()

            bg_start_idx = lower_lim[0]
            bg_end_idx = upper_lim[0]

        self.bg = float(np.mean(self.spectrum[bg_start_idx:bg_end_idx]))

    def get_peak_center(self) -> None:
        assert self.spectrum is not None
        assert self.times is not None

        # Initial Guesses
        peak_center_guess_idx = np.argmax(self.spectrum)
        t0_idx = np.argmax(self.spectrum)
        n = np.sum(self.spectrum) * 2
        shift = self.times[t0_idx]

        peak_range = slice(peak_center_guess_idx-15, peak_center_guess_idx+15)
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

    def prepare_fit(
        self,
        fit_time_left_of_peak: int | float = 300,
        fit_time_right_of_peak: int | float = 3000,
        fit_channels_left_of_peak: int | None = None,
        fit_channels_right_of_peak: int | None = None,
        fit_start_time: int | float | None = None,
        fit_end_time: int | float | None = None,
        fit_start_idx: int | None = None,
        fit_end_idx: int | None = None,
        fit_start_counts: int | None = None,
        fit_end_counts: int | None = None,
        get_fit_range: bool = False,
        bg_start_idx: int | None = None,
        bg_end_idx: int | None = None,
        get_bg: bool = False,
    ) -> None:
        assert self.spectrum is not None
        assert self.times is not None

        if np.isnan(self.peak_center):
            self.get_peak_center()

        peak_idx = np.searchsorted(self.times, self.peak_center)

        def time_to_ch(time):
            return int(time / self.tcal[1] - self.tcal[0])

        def get_abs_idx(sign, thr, abs_ch, abs_time, rel_ch, rel_time):
            if thr is not None:
                start = peak_idx
                stop = -1 if sign < 0 else len(self.spectrum)
                for idx in range(start, stop, sign):
                    if self.spectrum[idx] < thr:
                        return idx - sign

            if abs_ch is not None:
                return abs_ch

            if abs_time is not None:
                return time_to_ch(abs_time)

            if rel_ch is not None:
                return peak_idx + sign * rel_ch

            if rel_time is not None:
                return time_to_ch(self.peak_center + sign * rel_time)

            return None

        if get_fit_range:
            left_fit_idx, right_fit_idx = self.get_fit_range(fit_start_idx, fit_end_idx)
        else:
            left_fit_idx = get_abs_idx(
                -1, fit_start_counts, fit_start_idx, fit_start_time,
                fit_channels_left_of_peak, fit_time_left_of_peak
            ) or 0

            right_fit_idx = get_abs_idx(
                1, fit_end_counts, fit_end_idx, fit_end_time,
                fit_channels_right_of_peak, fit_time_right_of_peak
            ) or len(self.spectrum) - 1

        data_range = slice(left_fit_idx, right_fit_idx)

        self.trimmed_times = self.times[data_range]
        self.trimmed_spectrum = self.spectrum[data_range]

        if get_bg or bg_start_idx is not None or bg_end_idx is not None:
            self.get_bg(bg_start_idx, bg_end_idx)

    def fit(
        self,
        fit_time_left_of_peak: int | float = 300,
        fit_time_right_of_peak: int | float = 3000,
        fit_channels_left_of_peak: int | None = None,
        fit_channels_right_of_peak: int | None = None,
        fit_start_time: int | float | None = None,
        fit_end_time: int | float | None = None,
        fit_start_idx: int | None = None,
        fit_end_idx: int | None = None,
        fit_start_counts: int | None = None,
        fit_end_counts: int | None = None,
        get_fit_range: bool = False,
        channel_mask: EllipsisType | Sequence[bool | int] = ...,
        use_jacobian: bool = True,
        bg_start_idx: int | None = None,
        bg_end_idx: int | None = None,
        get_bg: bool = False,
        **kwargs
    ) -> lmfit.model.ModelResult:
        """Fit the spectrum to determine lifetimes and intensities.

        Parameters
        ----------
        fit_time_left_of_peak : int
            Time in units of the time calibration to the left of the peak
            maximum to use for fitting. Is overridden by any of the following
            start parameters. The default is 300.
        fit_time_right_of_peak : int
            Same as `fit_time_left_of_peak` but to the right of the peak. Is
            overridden by any of the following end parameters. The default is
            3000.
        fit_channels_left_of_peak : int or None, optional
            Number of channels to the left of the peak maximum to use for
            fitting. Overrides any previous start parameters. Is overridden by
            any of the following start parameters.
        fit_channels_right_of_peak : int or None, optional
            Same as `fit fit_channels_left_of_peak` but to the right of the
            peak. Overrides any previous end parameters. Is overridden by any
            of the following end parameters.
        fit_start_time : int or float or None, optional
            Time in units of the time calibration to use as the beginning of
            the range used for fitting. Overrides any previous start
            parameters. Is overridden by any of the following start parameters.
        fit_end_time : int or float or None, optional
            Time in units of the time calibration to use as the end of
            the range used for fitting. Overrides any previous end parameters.
            Is overridden by any of the following end parameters.
        fit_start_idx : int or None, optional
            Channel index to use as the beginning of the range used for
            fitting. Overrides any previous start parameters. Is overridden by
            any of the following start parameters.
        fit_end_idx : int or None, optional
            Channel index to use as the end of the range used for fitting.
            Overrides any previous end parameters. Is overridden by any of the
            following end parameters.
        fit_start_counts : int or None, optional
            Threshold of counts defining the beginning of the fit range to the
            left of the peak maximum. Overrides any previous start parameters.
        fit_end_counts : int or None, optional
            Threshold of counts defining the end of the fit range to the right
            of the peak maximum. Overrides any previous end parameters.
        get_fit_range: bool, optional
            Whether to open a fit range determination prompt. The default is
            False.
        channel_mask : Sequence of bool or Sequence of int or Ellipsis, optional
            Boolean mask or fancy index set to determine, which parts in the
            cut spectrum to use in the fit and which parts to ignore.
            The default is Ellipsis (use all).
        use_jacobian : bool, optional
            Whether to use the analytical Jacobian matrix during fitting.
            Using the analytical Jacobian generally reduces the number of
            function evaluations required to find the optimum. If disabled,
            the Jacobian is numerically computed. For more info see the docs of
            the underlying lmfit.Model.fit and scipy.optimize.leastsq
            function. The default is True.
        bg_start_idx : int or None, optional
            Channel index in the uncut spectrum that marks the beginning of
            the region to use to determine the background level. If
            `bg_start_idx` and `bg_end_idx` are given and `get_bg` is True,
            the background is taken as the average count per bin in the defined
            region. If either or both parameters are None while `get_bg` is
            True, a background determination prompt is opened. The default is
            None.
        bg_end_idx : int or None, optional
            Same as `bg_start_idx` but for marking the end of the background
            region. The default is None.
        get_bg : bool, optional
            Whether to compute the background level from a given region in the
            spectrum. If `bg_start_idx` and `bg_end_idx` are given and `get_bg`
            is True, the background is taken as the average count per bin in the
            defined region. If either or both parameters are None while `get_bg`
            is True, a background determination prompt is opened. If `get_bg` is
            False, the background level is not determined. The default is False.
        **kwargs:
            Other keyword arguments are passed to lmfit.Model.fit.
        """
        err = "LifetimeSpectrum is missing {}. Could not fit"
        assert self.spectrum is not None, err.format("spectrum data")
        assert self.times is not None and self.tcal is not None, err.format("time calibration")
        assert self.model is not None, err.format("fit model")

        self.prepare_fit(
            fit_time_left_of_peak, fit_time_right_of_peak,
            fit_channels_left_of_peak, fit_channels_right_of_peak,
            fit_start_time, fit_end_time,
            fit_start_idx, fit_end_idx,
            fit_start_counts, fit_end_counts,
            get_fit_range,
            bg_start_idx, bg_end_idx,
            get_bg,
        )

        assert self.trimmed_spectrum is not None
        assert self.trimmed_times is not None

        model, params, func, dfunc = self.make_model()

        weights = np.sqrt(1/np.where(self.trimmed_spectrum > 0, self.trimmed_spectrum, 1))

        if self.show:
            plt.figure()
            plt.title("Cut out spectrum and initial guess for fit")
            plt.semilogy(
                self.trimmed_times, self.trimmed_spectrum, label="Data"
            )
            plt.semilogy(
                self.trimmed_times,
                model.eval(t=self.trimmed_times, **params),
                label="Initial Guess"
            )
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        if use_jacobian:
            if "fit_kws" in kwargs:
                kwargs["fit_kws"].update({"Dfun": dfunc, "col_deriv": True})
            else:
                kwargs["fit_kws"] = {"Dfun": dfunc, "col_deriv": True}

        self.fit_result = model.fit(
            data=self.trimmed_spectrum[channel_mask],
            params=params,
            t=self.trimmed_times[channel_mask],
            weights=weights[channel_mask],
            **kwargs
        )

        self.post_fit()

        if self.verbose:
            self.fit_report()

        if self.show_fits or self.show:
            self.plot_fit_result()

        return self.fit_result

    def post_fit(self) -> None:
        assert self.fit_result is not None
        assert self.n_l is not None and self.n_r is not None
        assert self.l_vary is not None and self.r_vary is not None

        params = self.fit_result.params

        l_int = [
            params[f"h_{i}"].value if v else params[f"intensity_{i}"].value
            for i, v in enumerate(self.l_vary, 1)
        ]
        h = l_int.copy()
        l_dint = [
            may_be_nan(params[f"h_{i}"].stderr)
            for i in range(1, self.n_l+1)
        ]
        dh = l_dint.copy()
        l_cumul = sum(i for i, v in zip(l_int, self.l_vary) if not v)
        l_cumul_sq_err = 0
        for i in range(self.n_l):
            if self.l_vary[i]:
                l_int[i] *= max(1 - l_cumul, 0)
                l_cumul += l_int[i]
                l_dint[i] = l_int[i] * np.sqrt(
                    l_cumul_sq_err + (dh[i] / (h[i] + 1e-12))**2
                )
                if i == self.l_last_vary_idx:
                    break
                l_cumul_sq_err += (dh[i] / (1 - h[i] + 1e-12))**2

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
        h = r_int.copy()
        r_dint = [
            may_be_nan(params[f"res_h_{j}"].stderr)
            for j in range(1, self.n_r+1)
        ]
        dh = r_dint.copy()
        r_cumul = sum(i for i, v in zip(r_int, self.r_vary) if not v)
        r_cumul_sq_err = 0
        for j in range(self.n_r):
            if self.r_vary[j]:
                r_int[j] *= max(1 - r_cumul, 0)
                r_cumul += r_int[j]
                r_dint[j] = r_int[j] * np.sqrt(
                    r_cumul_sq_err + (dh[j] / (h[j] + 1e-12))**2
                )
                if j == self.r_last_vary_idx:
                    break
                r_cumul_sq_err += (dh[j] / (1 - h[j] + 1e-12))**2

        for j in range(1, self.n_r+1):
            params[f"res_intensity_{j}"] = param = params.pop(f"res_h_{j}")
            param.name = f"res_intensity_{j}"
            param.value = r_int[j-1]
            param.vary = self.r_vary[j-1]
            param.stderr = r_dint[j-1]

        lifetime_order = np.argsort([params[f"lifetime_{i}"].value for i in range(1, self.n_l+1)])+1

        for i in range(1, self.n_l+1):
            setattr(self, f"lifetime_{i}", params[f"lifetime_{lifetime_order[i-1]}"].value)
            setattr(self, f"dlifetime_{i}", may_be_nan(params[f"lifetime_{lifetime_order[i-1]}"].stderr))
            setattr(self, f"intensity_{i}", params[f"intensity_{lifetime_order[i-1]}"].value)
            setattr(self, f"dintensity_{i}", may_be_nan(params[f"intensity_{lifetime_order[i-1]}"].stderr))

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

        self.model = parse_model({
            k: (v.value, v.vary, v.min, v.max) for k, v in params.items()
        })

    def get_component_arrays(self) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        assert self.trimmed_spectrum is not None
        assert self.trimmed_times is not None
        assert self.fit_result is not None
        assert self.n_l is not None and self.n_r is not None

        ### Resolution Components

        params = self.fit_result.params
        t = self.trimmed_times

        t_shifts = [params[f"res_t0_{i}"].value for i in range(1, self.n_r+1)]
        sigmas = [params[f"res_sigma_{i}"].value for i in range(1, self.n_r+1)]

        left_ranges = [t_shift - sig for t_shift, sig in zip(t_shifts, sigmas)]
        right_ranges = [t_shift + sig for t_shift, sig in zip(t_shifts, sigmas)]

        t_res = np.arange(np.min(left_ranges) - 100, np.max(right_ranges) + 100)
        res = np.zeros_like(t_res)
        res_components = []

        for j in range(1, self.n_r+1):
            sig = params[f"res_sigma_{j}"].value
            t0 = params[f"res_t0_{j}"].value
            i = params[f"res_intensity_{j}"].value

            c = i * gauss(t_res, t0, sig)
            res += c
            res_components.append(c)

        ### Lifetime Components

        t = self.trimmed_times
        lt_components = []

        for i in range(1, self.n_l+1):
            y = np.zeros_like(t)
            for j in range(1, self.n_r+1):
                l_tau = params[f"lifetime_{i}"]
                l_i = params[f"intensity_{i}"]
                r_sigma = params[f"res_sigma_{j}"]
                r_i = params[f"res_intensity_{j}"]
                r_t0 = params[f"res_t0_{j}"]
                # e becomes numerically unstable for lifetimes ~ tcal[1]
                _t = t - r_t0
                a = -_t / l_tau + (r_sigma/(sqrt_2*l_tau))**2
                b = r_sigma / (l_tau * sqrt_2) - _t / (r_sigma * sqrt_2)
                e = np.exp(a)
                c = erfc(b)
                y += l_i * r_i / (2*l_tau) * e * c
            lt_components.append(y * params["N"] + params["background"])

        return t_res, lt_components, res_components

    def plot_fit_result(
        self,
        show_init: bool = False,
        show: bool = True,
    ) -> Tuple[Figure, Tuple[plt.Axes, ...]]:
        assert self.trimmed_spectrum is not None
        assert self.trimmed_times is not None
        assert self.fit_result is not None
        assert self.n_l is not None and self.n_r is not None

        fig = plt.figure()
        fig.suptitle(f"Fit Result\n{self.name}")

        gs = GridSpec(2, 2, height_ratios=[1, 4])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[:, 1])

        t = self.trimmed_times

        t_res, lt_components, res_components = self.get_component_arrays()
        res = np.sum(res_components, axis=0)

        ### Residuals

        ax1.set_ylabel("Residuals")

        if self.fit_result.residual is not None:
            ax1.scatter(t, self.fit_result.residual, s=1)

        ### Resolution Function and Components

        for i, c in enumerate(res_components, 1):
            ax3.semilogy(t_res, c, label=f"Component {i}")
        ax3.semilogy(t_res, res, label="Total")
        ax3.set_title("Resolution Components")
        ax3.set_xlabel("Time [ps]")
        ax3.set_ylim((1e-6, np.max(res)*1.1))

        ### Lifetime Spectrum and Components

        ax2.set_xlabel("Time [ps]")
        ax2.set_ylabel("Counts")

        ax2.scatter(t, self.trimmed_spectrum, label="Data", s=1)

        if show_init:
            ax2.plot(
                t, self.fit_result.init_fit, linestyle="--",
                c="C1", label="Initial Fit"
            )

        ax2.plot(
            t, self.fit_result.best_fit, linestyle="-",
            c="C2", label="Best Fit"
        )

        ax2.set_yscale("log")

        ylim = ax2.get_ylim()

        for i, c in enumerate(lt_components, 1):
            ax2.semilogy(t, c, label=f"Component {i}", c=f"C{i+2}")

        ax2.set_ylim(ylim)

        if show:
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax2.legend()
            ax3.legend()
            plt.tight_layout()
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
                   "Min", "Max", "Fixed", ""]
        formats = ["<", ">", ">", ">", ">", ">", "<", "<"]

        def print_params(params):
            table_data = np.array([
                [
                    p.name if not p.name.startswith("res_") else p.name[4:],
                    (
                        f"{100 * p.value:.1f}" if "int" in p.name else
                        f"{p.value:.2f}" if "life" in p.name else
                        f"{p.value:.4g}"
                    ),
                    (
                        (f"{100 * p.stderr:.1f}" if "int" in p.name else f"{p.stderr:.4g}")
                        if p.stderr is not None else "nan"
                    ),
                    f"{abs(100 * p.stderr / p.value):.3f} %" if p.value != 0 and p.stderr is not None else "nan",
                    f"{100 * p.min:.1f}" if "int" in p.name else f"{p.min:.1f}",
                    f"{100 * p.max:.1f}" if "int" in p.name else f"{p.max:.1f}",
                    "" if p.vary else "Fixed",
                    "At Boundary" if abs(p.value - p.min) < 1e-3 or abs(p.value - p.max) < 1e-3 else "",
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
