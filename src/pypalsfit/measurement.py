from typing import Dict, List
import json
from copy import deepcopy
import os
import re
import pickle
import time
from collections import namedtuple, Counter
from pathlib import Path
import warnings

import numpy as np
import scipy
import matplotlib.pyplot as plt

from .lifetime import LifetimeSpectrum
from .model import LifetimeModel
from .utils import Filter, isfinite, _check_input, _split_params, progress


class LifetimeMeasurement:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        lt_model: LifetimeModel | Dict | str | None = None,
        lt_keys: List[str] | None = None,
        res_model: LifetimeModel | Dict | str | None = None,
        res_keys: List[str] | None = None,
        calibrate: bool = False,
        name: str | None = None,
        dtype: type | None = None,
        show_fits: bool = True,
        show: bool = False,
        verbose: bool = False,
        autocompute: bool = True,
        debug: bool = False,
        **kwargs
    ) -> None:

        from .importer import import_elbe

        self.show = show
        self.verbose = verbose
        self.name = name

        self.path = None
        self.directory = None
        self.filename = None
        self.filetype = None

        self.spectra = []

        self.metadata = {}

        if path is not None:
            self.path = path = Path(path)
            self.directory = path.parent.name
            self.filename = path.stem
            self.filetype = path.suffix
            self.name = name or self.filename

            filetype_lookup = {".dat": import_elbe,}

            if self.filetype not in filetype_lookup.keys():
                err = f"Filetype '{self.filetype}' not supported."
                raise NotImplementedError(err)

            import_function = filetype_lookup[self.path.suffix]
            m = import_function(self.path, dtype)
            self.spectra = m.spectra
            self.metadata = m.metadata

            if not isinstance(lt_model, str):
                lt_model = deepcopy(lt_model)
            if not isinstance(res_model, str):
                res_model = deepcopy(res_model)

            if isinstance(lt_model, str):
                with open(lt_model, "r") as f:
                    lt_model = json.load(f)
            if isinstance(res_model, str):
                with open(res_model, "r") as f:
                    res_model = json.load(f)

            for kind, models, keys in zip(
                ["lifetime", "resolution"], [lt_model, res_model], [lt_keys, res_keys]
            ):
                if not isinstance(models, dict) or keys is None or not any(key.startswith("Measurement") for key in models):
                    continue

                params = {k: str(self.metadata[k]) for k in keys}

                if debug:
                    print(f"Picking a {kind} model for measurement with parameters {params}...")

                for name, model in models.items():
                    metadata = model["Metadata"]
                    model_params = {k: str(metadata[k]) for k in keys}
                    if params == model_params:
                        models.clear()
                        models.update(model)
                        if debug:
                            print(f"Picking {name}")
                        break
                else:
                    err = (
                        f"No provided {kind} model matches this "
                        "LifetimeMeasurement's parameters:"
                        f"\nMeasurement: {self.name}"
                        f"\nParameters: {params}"
                    )
                    raise ValueError(err)

            if not isinstance(lt_model, dict) or not any(key.startswith("Detector") for key in lt_model):
                lt_model = {
                    f"Detector {s.detname}": lt_model for s in self.spectra
                }

            if not isinstance(res_model, dict) or not any(key.startswith("Detector") for key in res_model):
                res_model = {
                    f"Detector {s.detname}": res_model for s in self.spectra
                }

            for s in self.spectra:
                s.__init__(
                    spectrum = s.spectrum,
                    detname = s.detname,
                    tcal = s.tcal,
                    name = self.name,
                    lt_model = lt_model[f"Detector {s.detname}"],
                    res_model = res_model[f"Detector {s.detname}"],
                    calibrate = calibrate,
                    show_fits = show_fits,
                    show = show,
                    verbose = verbose,
                    autocompute = autocompute,
                    dtype = dtype,
                    **kwargs
                )

    def __getitem__(self, item):
        if isinstance(item, str):
            for spectrum in self.spectra:
                if spectrum.detname == item:
                    return spectrum
            try:
                return self.metadata[item]
            except KeyError:
                pass
            try:
                return getattr(self, item.lower())
            except AttributeError:
                pass
        else:
            return self.spectra[item]

        raise KeyError(item)

    def __setitem__(self, key, value):
        try:
            self.metadata[key] = value
            return
        except KeyError:
            pass
        try:
            for s in self.spectra:
                s[key] = value
            return
        except (AttributeError, KeyError):
            pass

        raise KeyError(key)

    def __len__(self):
        return len(self.spectra)

    def __iter__(self):
        return (x for x in self.spectra)

    @property
    def detnames(self):
        return {s.detname for s in self}

    def show_spectra(self, time_axis=True):
        if any(s.times is None for s in self):
            time_axis = False
        for det in self.detnames:
            s = self[det]
            if time_axis:
                plt.semilogy(s.times, s.spectrum, label=det)
            else:
                plt.semilogy(s.spectrum, label=det)

        if time_axis:
            plt.xlabel("Time [ns]")
        else:
            plt.xlabel("Channel")
        plt.ylabel("Counts")
        plt.grid()
        plt.legend()
        plt.show()

    def dump_components(self, filepath=None):
        out = {"Metadata": {k: str(v) for k, v in self.metadata.items()}}
        for i, s in enumerate(self):
            out[f"Detector {s.detname or i+1}"] = s.dump_components()

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)

        return out

    def dump_resolution_components(self, filepath=None):
        out = {"Metadata": {k: str(v) for k, v in self.metadata.items()}}
        for i, s in enumerate(self):
            out[f"Detector {s.detname or i+1}"] = s.dump_resolution_components()

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)

        return out

    def dump_lifetime_components(self, filepath=None):
        out = {"Metadata": {k: str(v) for k, v in self.metadata.items()}}
        for i, s in enumerate(self):
            out[f"Detector {s.detname or i+1}"] = s.dump_lifetime_components()

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)

        return out


class MeasurementCampaign:
    def __init__(
        self,
        path: str | Path | List[str | Path] | None = None,
        *,
        measurements: List[LifetimeMeasurement] | None = None,
        lt_model: LifetimeModel | Dict | str | None = None,
        lt_keys: List[str] | None = None,
        res_model: LifetimeModel | Dict | str | None = None,
        res_keys: List[str] | None = None,
        calibrate: bool = False,
        name: str | None = None,
        dtype: type | None = None,
        show_fits: bool = True,
        show: bool = False,
        verbose: bool = False,
        autocompute: bool = True,
        pbar: bool = True,
        cache: bool = False,
        cache_path: str = "/tmp/pals_cache.pkl",
        **kwargs
    ) -> None:

        if pbar:
            start = time.time()

        self.show = show
        self.verbose = verbose
        self.name = name

        self.path = path
        self.directory = None

        self.measurements = []

        cache_hit = False
        self.cache_path = cache_path
        if cache and os.path.exists(cache_path):
            if pbar:
                progress(0, 1, start, 20, "Importing")
            cached_mc = self.load_cache(cache_path, inplace=False)
            cached_path = cached_mc.path

            if ((
                    path is None or cached_path is None
                ) or (
                    isinstance(path, (str, Path)) and isinstance(cached_path, (str, Path))
                    and Path(path) == Path(cached_path)
                ) or (
                    isinstance(path, (list)) and isinstance(cached_path, list)
                    and len(path) == len(cached_path)
                    and np.all(np.equal(list(map(Path, path)), list(map(Path, cached_path))))
                )):
                cache_hit = True
                measurements = cached_mc.measurements
                self.path = cached_path
                self.directory = cached_mc.directory
                self.name = name or cached_mc.directory
                if pbar:
                    progress(1, 1, start, 20, "Importing")

        supported_filetypes = (".dat",)

        def alphanum_key(s):
            return [int(part) if part.isdigit() else part
                    for part in re.split(r"(\d+)", s)]

        def get_filepaths(path):
            path = Path(path)
            files, folders = [], []

            for entry in path.iterdir():
                if entry.is_file() and entry.suffix in supported_filetypes and "PALS" in str(entry):
                    files.append(entry)
                elif entry.is_dir():
                    folders.append(entry)

            files = sorted(files, key=lambda f: alphanum_key(f.name))
            folders = sorted(folders, key=lambda f: alphanum_key(f.name))

            if len(folders) > 0:
                for folder in folders:
                    files.extend(get_filepaths(folder))

            return files

        if measurements is not None:
            self.measurements = list(measurements)
        elif path is not None:
            if isinstance(path, (str, Path)):
                self.path = path = Path(path)
                self.directory = path.name
                self.name = name or self.directory
                files = get_filepaths(path)
            elif isinstance(path, list):
                self.path = list(map(Path, path))
                files = []
                for subpath in path:
                    files += get_filepaths(subpath)
            else:
                raise TypeError

            if isinstance(lt_model, str):
                with open(lt_model, "r") as f:
                    lt_model = json.load(f)
            if isinstance(res_model, str):
                with open(res_model, "r") as f:
                    res_model = json.load(f)

            total = len(files)
            for done, f in enumerate(files):
                self.measurements.append(LifetimeMeasurement(
                    f, lt_model=lt_model, res_model=res_model,
                    lt_keys=lt_keys, res_keys=res_keys,
                    calibrate=calibrate, dtype=dtype, show_fits=show_fits,
                    show=self.show, verbose=self.verbose,
                    autocompute=autocompute, **kwargs
                ))
                if pbar:
                    progress(done+1, total, start, 20, "Importing")

        if cache and not cache_hit:
            self.cache()

    def __getitem__(self, i):
        if isinstance(i, str):
            for measurement in self:
                if i == measurement.name:
                    return measurement
            return [m[i] for m in self]
        elif isinstance(i, list) and isinstance(i[0], str):
            return [self[j] for j in i]
        return self.measurements[i]

    def __add__(self, other):
        new = MeasurementCampaign()

        new.measurements = self.measurements + other.measurements
        new.show = self.show or other.show
        new.verbose = self.verbose or other.verbose

        return new

    def __len__(self):
        return len(self.measurements)

    def __iter__(self):
        return (m for m in self.measurements)

    def __truediv__(self, param):
        return self.split(param)

    def _get_data(self, params, parsers, targets):
        """Extract values of params from measurements.

        This method is for internal use only.
        """
        err = "An unexpected error occured. Be sure to report this issue."
        for input in [params, parsers, targets]:
            assert isinstance(input, list), err

        x = np.full((len(params),
                     2,
                     max(len(self.detnames), 1),
                     len(self)),
                    np.nan, dtype=object)

        def find(param, parser, target, x, key, key_id, meas, meas_keys):
            found = False
            if target is None or key in target:
                for j, m in enumerate(meas):
                    for k, det in enumerate(meas_keys):
                        try:
                            x[key_id, k, j] = parser(m[det][param])
                            found = True
                        except (AttributeError, KeyError):
                            pass
            return found

        arglist = [["meas", 0, [[m] for m in self.measurements], [0]],
                   ["spec", 1, self, self.detnames]]

        for i, p in enumerate(params):
            if p == "":
                x[i, 0, 0, :] = parsers[i](np.arange(len(self)))
                continue
            for args in arglist:
                if find(p, parsers[i], targets[i], x[i], *args):
                    break

        return x

    @property
    def detnames(self):
        return {detname for m in self for detname in m.detnames}

    @property
    def shape(self):
        shape = namedtuple("Shape", ["m", "s"])

        n_m = len(self)

        n_s = Counter([len(m.spectra) for m in self])
        if len(n_s) == 0:
            n_s = np.nan
        elif len(n_s) == 1:
            n_s = list(n_s.keys())[0]
        elif max(n_s.values()) == 1:
            n_s = tuple(n_s.keys())

        return shape(n_m, n_s)

    def is_homogeneous(self):
        """Check if this instance is homogeneous.

        A MeasurementCampaign is homogeneous if all its DopplerMeasurements
        contain the same number of SingleSpectra, the same number of
        CoincidenceSpectra and those Spectra are of the same detectors.

        Parameters
        ----------
        single : bool, optional
            Whether to check singles for homogeneity. The default is true.
        coinc : bool, optional
            Whether to check coincs for homogeneity. The default is true.

        Returns
        -------
        bool
            True if this instance is homogeneous, otherwise false.

        Examples
        --------
        Homogeneous MeasurementCampaign:

        >>> mc = {m0: {spectra: {A, B}},
        ...       m1: {spectra: {A, B}}}

        Inhomogeneous MeasurementCampaign (different singles count):

        >>> mc = {m0: {spectra: {A, B}},
        ...       m1: {spectra: {A}}}

        Inhomogeneous MeasurementCampaign (different detectors):

        >>> mc = {m0: {spectra: {A, B}},
        ...       m1: {spectra: {C, D}}}
        """
        for m in self:
            if m.detnames != self.detnames:
                return False

        return True

    # Declare method alias:
    hom = is_homogeneous

    def copy(self, deep=True, copy_spectra=True):
        """Return a copy of this instance."""
        mc = MeasurementCampaign()
        for attr, value in self.__dict__.items():
            try:
                setattr(mc, attr, value.copy())
            except AttributeError:
                setattr(mc, attr, value)

        if deep:
            for i, m in enumerate(self):
                mc.measurements[i] = m.copy(copy_spectra=copy_spectra)

        return mc

    def cache(self, path=None):
        self.cache_path = path or self.cache_path

        with open(self.cache_path, "wb") as f:
            pickle.dump(self, f)

    def load_cache(self, path=None, inplace=True):
        self.cache_path = path or self.cache_path

        with open(self.cache_path, "rb") as f:
            cached_mc = pickle.load(f)

        if inplace:
            self.measurements = cached_mc.measurements
            self.path = cached_mc.path
            self.directory = cached_mc.directory
            self.name = cached_mc.directory
            self.model = cached_mc.model
        else:
            return cached_mc

    def get(self, params, parsers=None, targets=None, slice=..., use_pd=True):
        """Extract values of params from measurements.

        Parameters
        ----------
        params : str or list of str
            Parameters to extract.
        parsers : callable or list of callable, optional
            Functions to apply to parameters when extracting. If `params` is a
            list, `parsers` can be too, then parsers[i] is applied to
            params[i]. `parsers` can be None if no function is to be applied.
            If `parsers` is a single value, it is applied to all parameters.
            The default is None.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.

        Returns
        -------
        data : list of numpy.ndarray
            Extracted data.
        """
        params, parsers, targets = _check_input(params, parsers, targets)

        x = self._get_data(params, parsers, targets)

        if slice is not None and slice is not Ellipsis:
            x = x[:, :, :, slice]

        m_params, m_parsers, m_mask = _split_params(x, params, parsers, 0)
        s_params, s_parsers, s_mask = _split_params(x, params, parsers, 1)

        if len(self.detnames) > 1:
            foo = lambda p: headers.append([" ".join([p, d]) for d in self.detnames])
        else:
            foo = lambda p: headers.append([p])

        headers = []
        for p, t in zip(params, targets):
            if p in m_params and (t is None or "meas" in t):
                headers.append([p])
            elif p in s_params and (t is None or "spec" in t):
                foo(p)
            else:
                headers.append([])

        if "" in headers:
            headers[headers.index("")] = "Index"

        columns = []

        for i, (param, header, p) in enumerate(zip(params, headers, x)):
            mask = np.any(isfinite(p), axis=2)
            column = p[mask]
            if param in s_params:
                headers[i] = np.array(header)[mask[1]]
            if column.ndim == 1:
                columns.append(column)
            elif column.ndim == 0:
                columns.append([column])
            else:
                columns.extend(column)

        headers = [header for param_headers in headers for header in param_headers]

        data = {h: c for h, c in zip(headers, columns)}

        if use_pd:
            try:
                import pandas as pd
                data = pd.DataFrame(data)
            except ModuleNotFoundError:
                wrn = ("No Pandas installation found. "
                       "Pandas is recommended for handling large data sets")
                warnings.warn(wrn)

        return data

    def print(self, params, parsers=None, targets=None,
              index=True, header=True, sep="", fmt="<", slice=...):
        """Print a list of parameters of measurements.

        Parameters
        ----------
        params : str or list of str
            Parameters to print values of. Can be parameter of
            DopplerMeasurement, SingleSpectrum or CoincidenceSpectrum. If a
            parameter is from a Single- or CoincidenceSpectrum, a column for
            each detector (pair) is created.
        parsers : callable or list of callable, optional
            Functions to apply to parameters when printing. If `params` is a
            list, `parsers` can be too, then parsers[i] is applied to
            params[i]. `parsers` can be None if no function is to be applied.
            If `parsers` is a single value, it is applied to all parameters.
            The default is None.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.
        index : bool, optional
            Whether to print an index column at the beginning of the table.
            The default is true.
        header : bool, optional
            Whether to print the header row at the beginning of the table.
            The default is true.
        sep : str, optional
            Column separator character. The default is "" (no separator).
        fmt : str, optional
            f-string format specifier to go before the width. Can e.g. be used
            to control alignment or padding. The default is "<" (left-aligned).
        slice : index, optional
            Slice of measurements to print. The default is Ellipsis (print all).
        """
        data = self.get(params, parsers, targets, slice=slice, use_pd=False)

        if len(data) == 0:
            raise ValueError("MeasurementCampaign is empty "
                             "or parameters do not exist")

        columns = list(data.values())
        headers = list(data.keys())

        if index:
            columns.insert(0, np.arange(1, len(columns[0]) + 1))
            headers.insert(0, "")

        columns = np.where(isfinite(columns), np.vectorize(str)(columns), "")
        widths = np.max([np.max(np.vectorize(len)(columns), axis=1),
                         np.vectorize(len)(headers)], axis=0)

        if header is True:
            for i, head in enumerate(headers):
                print(f"{sep} {head:{fmt}{widths[i]}} ", end="")
            print(sep)
            for i, head in enumerate(headers):
                print(f"{sep} {'-' * widths[i]} ", end="")
            print(sep)

        for i in range(len(columns[0])):
            for j, col in enumerate(columns):
                print(f"{sep} {col[i]:{fmt}{widths[j]}} ", end="")
            print(sep)

    def parse(self, params, parsers, targets=None, inplace=True):
        """Parse parameters.

        Parameters
        ----------
        params : str or list of str
            Parameters to parse.
        parsers : callable or list of callable
            Functions to apply to parameters. If `params` is a list, `parsers`
            can be too, then parsers[i] is applied to params[i]. `parsers` can
            be None if no function is to be applied. If `parsers` is a single
            value, it is applied to all parameters.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.
        inplace : bool, optional
            If true, overwrite parameters of this instance.
            If false, overwrite parameters of a copy of this instance.

        Returns
        -------
        new_mc : MeasurementCampaign
            If `inplace` is true, this instance is returned.
            If `inplace` is false, the copy is returned.
        """
        params, parsers, targets = _check_input(params, parsers, targets)

        if inplace:
            new_mc = self
        else:
            new_mc = self.copy()

        def _parse(param, parser, target, key, key_id, meas, meas_keys):
            found = False
            if target is None or key in target:
                for j, m in enumerate(meas):
                    for k, det in enumerate(meas_keys):
                        try:
                            m[det][param] = parser(m[det][param])
                            found = True
                        except (AttributeError, KeyError):
                            pass
            return found

        arglist = [["meas", 0, np.reshape(self.measurements, (-1, 1)), [0]],
                   ["spec", 1, self, self.detnames]]

        for i, p in enumerate(params):
            for args in arglist:
                if _parse(p, parsers[i], targets[i], *args):
                    break

        return new_mc

    def derive(self, param, arguments, formula=None,
               parsers=None, targets=None):
        """Add a derived parameter to measurements.

        The parameter can be derived from existing parameters of
        DopplerMeasurements, Single- or CoincidenceSpectra.
        The parameter is added as attribute (lower case name) to the highest
        possible structure in the MeasurementCampaign (details below).

        If `arguments` contains attributes or metadata of DopplerMeasurements
        only, the parameter is added to DopplerMeasurements.
        If `arguments` additionally contains at least one attribute of
        SingleSpectra, the parameter is added to SingleSpectra.
        The same holds for CoincidenceSpectra accordingly.
        If `arguments` contains attributes of both SingleSpectra and
        CoincidenceSpectra, an error is raised.

        Parameters
        ----------
        param : str
            Name of derived parameter. May contain upper case letters,
            those are then converted to lower case.
        arguments : str or list of str
            Parameter names of the arguments of formula.
        formula : callable
            Function that computes the derived parameter from `arguments`.
        parsers : callable or list of callable, optional
            Functions to apply to `arguments` before use in `formula`. If
            `arguments` is a list, `parsers` can be too, then parsers[i] is
            applied to arguments[i]. `parsers` can be None if no function is to
            be applied. If `parsers` is a single value, it is applied to all
            parameters. The default is None.
        targets : str or list of str, optional
            Specifies if `arguments` is meant as attribute of
            DopplerMeasurements (then `targets` must contain "doppler"),
            SingleSpectra (then `targets` must contain "single") or
            CoincidenceSpectra (then `targets` must contain "coinc").
            If `arguments` is a list, `targets` can be too, then targets[i] is
            applied to arguments[i]. `targets` can be None, then `arguments` is
            taken from the first class in the order above where it is found. If
            `targets` is a single value, it is applied to all parameters.
            The default is None.

        Raises
        ------
        AttributeError:
            If any element of `arguments` cannot be found.
        RuntimeError:
            If `arguments` contains attribute names of both Single- and
            CoincidenceSpectra.

        Example
        -------
        >>> mc.derive('countrate',
        ...           ['Counts', 'RealTimeDuration'],
        ...           lambda a, b: a / b)
        """
        if formula is None:
            err = "If `formula` is None, `arguments` must be a single parameter"
            assert (isinstance(arguments, str) or
                    (isinstance(arguments, list) and len(arguments) == 1)), err
            formula = lambda x: x

        err = "Derived parameter name must be of type str"
        assert isinstance(param, str), err
        err = "`formula` must be a callable function"
        assert callable(formula), err

        arguments, parsers, targets = \
            _check_input(arguments, parsers, targets)

        x = self._get_data(arguments, parsers, targets)

        m_params, m_parsers, m_mask = _split_params(x, arguments, parsers, 0)
        s_params, s_parsers, s_mask = _split_params(x, arguments, parsers, 1)

        n_args = len(arguments)

        if n_args != len(m_params) + len(s_params):
            args = set(arguments) - (set(m_params)|set(s_params))
            raise AttributeError(f"Could not find parameter(s) {args}")

        arg_mask = np.argmax([m_mask, s_mask], axis=0)

        def _derive(param, meas, attr, len, x, n_args, arg_mask):
            for i, m in enumerate(meas):
                for j, s in (enumerate(getattr(m, attr)) if len else [(0, m)]):
                    setattr(s, param.lower(),
                            formula(*[x[k, arg_mask[k], j*(arg_mask[k]>0), i]
                                      for k in range(n_args)]))

        if n_args == len(m_params):
            _derive(param, self, None, False, x, n_args, arg_mask)
        else:
            _derive(param, self, "spec", True, x, n_args, arg_mask)

    def sort(self, params, parsers=None, targets=None,
             ret_params=False):
        """Sort a MeasurementCampaign.

        Sorting a MeasurementCapmaign involves sorting the measurements list
        (list of DopplerMeasurements), as well as singles (list of
        SingleSpectra) and coinc (list of CoincidenceSpectra) of individual
        DopplerMeasurements.

        Parameters
        ----------
        params : str or list of str
            Parameters to sort by. If `params` is a list of values, sort first
            by the first element, then the next, ... without destroying the
            order of the previous element.
        parsers : callable or list of callable, optional
            Functions to apply to parameters when sorting. If `params` is a
            list, `parsers` can be too, then parsers[i] is applied to
            params[i]. `parsers` can be None if no function is to be applied.
            If `parsers` is a single value, it is applied to all parameters.
            The default is None.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.
        ret_params : bool
            If true, return the extracted parameter array used for sorting.

        Returns
        -------
        x : numpy.ndarray
            If `ret_params` is true return the extracted parameter array used
            for sorting.
            If `ret_params` is false, nothing is returned.
        """
        params, parsers, targets = _check_input(params, parsers, targets)

        x = self._get_data(params, parsers, targets)[::-1]

        m_params, m_parsers, m_mask = _split_params(x, params, parsers, 0)
        s_params, s_parsers, s_mask = _split_params(x, params, parsers, 1)

        m, s = x[:, 0][m_mask], x[:, 1][s_mask]

        if len(m_params) > 0:
            m_order = np.lexsort(m[:, 0])
        else:
            m_order = np.arange(len(self))

        if len(s_params) > 0:
            s_order = [np.lexsort(i.T) for i in s.T]
        else:
            s_order = np.arange(len(self.detnames))

        for i, m in enumerate(self):
            m.spectra = np.array(m.spectra)
            try:
                # Sort singles
                m.spectra[:] = m.spectra[s_order[i][::-1]]
            except IndexError:
                # Happens when singles is empty
                pass
            m.spectra = list(m.spectra)

        # Sort DopplerMeasurements
        self.measurements = [self.measurements[i] for i in m_order]

        if ret_params:
            return self._get_data(params, parsers, targets)

    def filter(self, params, values=None, *, min=-np.inf, max=np.inf,
               negative=False, parsers=None, targets=None, inplace=True):
        """Filter a MeasurementCampaign.

        Parameters
        ----------
        params : str
            Parameters to filter.
        values : any type or list of any type, optional
            Allowed values. All parameters whose values don't fit, will be
            removed (opposite if `negative` is true). The default is None.
        min : any type, optional
            Minimum value of `param` that the target must have.
            The default is -np.inf.
        max : any type, optional
            Maximum value of `param` that the target must have.
            The default is np.inf.
        negative : bool
            Flag that indicates whether matches (True) or non-matches (False)
            are to be filtered out. The default is False
        parsers : callable or list of callable, optional
            Functions to apply to parameters when filtering. If `params` is a
            list, `parsers` can be too, then parsers[i] is applied to
            params[i]. `parsers` can be None if no function is to be applied.
            If `parsers` is a single value, it is applied to all parameters.
            The default is None.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.
        inplace : bool, optional
            If `inplace` is true, this method operates on this instance.
            If `inplace` is false, this method operates on a copy of this
            instance.

        Returns
        -------
        filtered_mc : MeasurementCampaign
            If `inplace` is true, this instance is returned.
            If `inplace` is false, the copy is returned.
        """
        raise NotImplementedError("Filtering does not yet work for LifetimeMeasurements")
        if params is None:
            return self

        err = "Parameters must be of type str or list of str"
        assert isinstance(params, str) or isinstance(params, list), err

        params, parsers, targets = _check_input(params, parsers, targets)

        filter = Filter(self)

        for param, parser, target in zip(params, parsers, targets):
            filter.add(param, values, min=min, max=max, negative=negative,
                       parser=parser, target=target)

        return filter.apply(inplace=inplace)

    def split(self, params, parsers=None, targets=None):
        """Split a MeasurementCampaign into subcampaigns.

        Parameters
        ----------
        params : str or list of str
            Parameters to split by.
        parsers : callable or list of callable
            Functions to apply to parameters. If `params` is a list, `parsers`
            can be too, then parsers[i] is applied to params[i]. `parsers` can
            be None if no function is to be applied. If `parsers` is a single
            value, it is applied to all parameters.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.

        Returns
        -------
        new_mcs : List of MeasurementCampaign
            List of MeasurementCampaigns split by the parameters. If the
            campaign is split only by parameters of DopplerMeasurements, the
            references to Single- and CoincidenceSpectra will remain the same.
            If the campaign is split by parameters of SingleSpectra, the
            references to CoincidenceSpectra will remain the same and vice
            versa.
        """
        params, parsers, targets = _check_input(params, parsers, targets)

        x = self._get_data(params, parsers, targets)

        m_params, m_parsers, m_mask = _split_params(x, params, parsers, 0)
        s_params, s_parsers, s_mask = _split_params(x, params, parsers, 1)

        m, s = x[:, 0][m_mask], x[:, 1][s_mask]
        m, s = m.astype(str), s.astype(str)

        n_params, _, n_dets, n_meas = x.shape

        new_mcs = [self]

        # Split DopplerMeasurements

        if len(m_params) > 0:
            m = m[np.any(isfinite(m), axis=-1)].T

            unique = np.unique(m, axis=0)
            idcs = np.arange(len(self))

            if unique.ndim == 1:
                unique = unique.reshape(-1, 1)

            unique_idcs = [idcs[np.all(m == u, axis=1)] for u in unique]

            new_mcs = [MeasurementCampaign(measurements=[self.measurements[i] for i in u_idcs],
                                           name=np.squeeze(u))
                       for u, u_idcs in zip(unique, unique_idcs)]
        else:
            unique_idcs = [np.arange(len(self))]

        # Split Spectra

        def split_spectra(params, x, attr, new_mcs, m_params=m_params,
                          unique_idcs=unique_idcs):
            if len(params) == 0:
                return new_mcs
            overwrite_mcs = []
            x = np.moveaxis(x, 0, -1)

            if len(m_params) > 0:
                x = [x[:, u_idcs, :] for u_idcs in unique_idcs]
            else:
                x = [x]

            for i, (mc, data) in enumerate(zip(new_mcs, x)):
                # n_dets, n_sub_meas, n_s_params = data.shape
                unique = np.unique(np.concatenate(data, axis=0), axis=0)

                new_sub_mc = []
                idcs = np.arange(len(mc))

                for u in unique:
                    new_mc = MeasurementCampaign()

                    x_mask = np.all(data == u, axis=-1)
                    meas_mask = np.any(x_mask, axis=0)

                    for j, m in zip(idcs[meas_mask], mc[meas_mask]):
                        new_m = m.copy(False, False)

                        setattr(new_m, attr, list(np.array(getattr(m, attr))[x_mask[..., j]]))

                        new_mc.measurements.append(new_m)

                    new_sub_mc.append(new_mc)

                overwrite_mcs.extend(new_sub_mc)

            return overwrite_mcs

        new_mcs = split_spectra(s_params, s, "spec", new_mcs)

        mcs = MultiCampaign(campaigns=new_mcs)

        return mcs

    def merge(self, others, inplace=True):
        """Merge split MeasurementCampaigns.

        This is the inverse operation to split.

        Parameters
        ----------
        others : MeasurementCampaign or list of MeasurementCampaign
            MeasurementCampaign(s) to merge into this campaign.

        Returns
        -------
        self : MeasurementCampaign
            For convenience self is returned to be able to chain method calls.
        """
        if isinstance(others, type(self)):
            others = [others]
        else:
            err = ("`others` must be of type MeasurementCampaign or "
                "list of MeasurementCampaign")
            assert isinstance(others, list), err
            for m in others:
                assert isinstance(m, type(self)), err

        for other in others:
            for m2 in other.measurements:
                found = False
                if m2.name is not None or m2.path is not None:
                    for m1 in self.measurements:
                        if ((m2.name is None or m1.name == m2.name) and
                                (m2.path is None or m1.path == m2.path)):
                            m1.spectra += m2.spectra
                            found = True
                            break
                if not found:
                    self.measurements.append(m2)

        return self

    def cluster(self, params, clusters=None, *, n_clusters=None, guess=None,
                parsers=None, targets=None, metric="euclidean",
                show_assignment=False, inplace=True, ret_clusters=False):
        """Overwrite parameters with the cluster they belong to.

        Parameters
        ----------
        params : any type or list of any type
            Parameters to cluster. Can be a list of parameters for higher
            dimensional clusters. The type of any parameter must be castable
            to float, otherwise a suited parser must be provided.
        clusters : list of any type, optional
            Parameters (coordinates) of the clusters. If provided, each
            measurement is assigned to its closest cluster and corresponding
            parameters overwritten accordingly. The default is None.
        n_clusters : int, optional
            Number of clusters in the parameters. If provided, the parameters
            of the clusters (i.e. the `clusters` argument) are determined with
            the kmeans algorithm. Only works well if the scattering within each
            cluster is much smaller than the distance between the clusters.
            `clusters` has precedence over `n_clusters`.
            The default is None.
        guess : list of any type, optional
            Initial guess of the clusters. Must be of shape n_clusters by
            len(params). If provided, the parameters of the clusters (i.e. the
            `clusters` argument) are determined with the kmeans algorithm.
            `clusters` has precedence over `guess`. The default is None.
        parsers : callable or list of callable, optional
            Functions to apply to parameters when clustering. If `params` is a
            list, `parsers` can be too, then parsers[i] is applied to
            params[i]. `parsers` can be None if no function is to be applied.
            If `parsers` is a single value, it is applied to all parameters.
            The default is None.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.
        metric : str or callable, optional
            See scipy.spatial.distance.cdist. The default is "euclidean".
        show_assignment : bool, optional
            If true and the dimensionality of clusters is at most two, show a
            scatter plot of parameters for each measurement with color
            indicating what cluster they were assigned to.
            The default is False.
        inplace : bool, optional
            If true, overwrite parameters of this instance.
            If false, overwrite parameters of a copy of this instance.
            The default is true.
        ret_clusters : bool, optional
            Whether to return the computed cluster coordinates. The default
            is false.

        Returns
        -------
        new_mc : MeasurementCampaign
            If `inplace` is true, this instance is returned.
            If `inplace` is false, the copy is returned.
        clusters : numpy.ndarray
            If `ret_clusters` is true, the computed cluster coordinates are
            returned.
        """
        if inplace is True:
            new_mc = self
        else:
            new_mc = self.copy()

        if clusters is None and n_clusters is None and guess is None:
            return new_mc

        params, parsers, targets = _check_input(params, parsers, targets)

        x = new_mc._get_data(params, parsers, targets)

        n_params, _, n_dets, n_meas = x.shape

        m_params, m_parsers, m_mask = _split_params(x, params, parsers, 0)
        s_params, s_parsers, s_mask = _split_params(x, params, parsers, 1)

        if n_params != len(m_params) + len(s_params):
            args = set(params) - (set(m_params)|set(s_params))
            raise AttributeError(f"Could not find parameter(s) {args}")

        m, s = x[:, 0][m_mask], x[:, 1][s_mask]

        if n_params == len(m_params):
            meas_coords = m[:, 0].T.squeeze()
        else:
            m[:, 1:] = m[:, 0:1]
            meas_coords = np.vstack((m, s)).T.squeeze()

        meas_coords = meas_coords.reshape(-1, n_params).astype(float)

        if not np.all(isfinite(meas_coords)):
            raise RuntimeError("Cannot cluster inhomogeneous MeasurementCampaign")

        if clusters is None:
            if guess is not None:
                guess = np.reshape(guess, (-1, meas_coords.shape[1]))
                clusters = scipy.cluster.vq.kmeans(meas_coords, guess)[0]
            else:
                clusters = scipy.cluster.vq.kmeans(meas_coords, n_clusters)[0]

        distances = scipy.spatial.distance.cdist(clusters, meas_coords,
                                                 metric=metric)
        cluster_idcs = np.argmin(distances, axis=0)

        if n_params == len(m_params):
            cluster_idcs = cluster_idcs.reshape(n_meas)
        else:
            cluster_idcs = cluster_idcs.reshape(n_meas, n_dets)

        if meas_coords.shape[1] in [1, 2] and show_assignment:
            p = np.concatenate((m_params, s_params))
            fig, axs = plt.subplots(2, 1)
            fig.suptitle("Cluster Assignment")
            for ax in axs:
                if meas_coords.shape[1] == 1:
                    ax.set_xlabel(p[0])
                    ax.scatter(x=meas_coords[:, 0],
                               y=[0 for _ in meas_coords],
                               c=[f"C{i}" for i in cluster_idcs.reshape(-1)])
                    for c in clusters:
                        ax.scatter(c, 0, marker="X", c="k", edgecolor="w")
                else:
                    ax.set_xlabel(p[0])
                    ax.set_ylabel(p[1])
                    ax.scatter(x=meas_coords[:, 0],
                               y=meas_coords[:, 1],
                               c=[f"C{i}" for i in cluster_idcs.reshape(-1)])
                    for cx, cy in clusters:
                        ax.scatter(cx, cy, marker="X", c="k", edgecolor="w")

            axs[0].set_title("Aspect Ratio: Zoomed")
            axs[1].set_title("Aspect Ratio: Equal")
            axs[1].set_aspect('equal')
            plt.tight_layout()
            plt.show()

        err = ("Cluster value of DopplerMeasurement parameter {} cannot be "
               "uniquely determined:\n{m} 0 clusters to {}, but {m} {} to {}."
               "\nConsider changing the (number of) clusters, "
               "parsing the parameters to similar order of magnitude or "
               "using a different metric")

        if n_params == len(m_params):
            for i, p in enumerate(m_params):
                for j, m in enumerate(new_mc):
                    m[p] = clusters[cluster_idcs[j], i]
        else:
            for i, p in enumerate(m_params):
                for j, m in enumerate(new_mc):
                    eq = np.equal(clusters[cluster_idcs[j, :], i],
                                  clusters[cluster_idcs[j, 0], i])
                    if not np.all(eq):
                        a = clusters[cluster_idcs[j, 0], i]
                        idx = np.argmax(~eq)
                        b = clusters[cluster_idcs[j, idx], i]
                        raise RuntimeError(err.format(p, a, idx, b, m="Spectrum"))
                    m[p] = clusters[cluster_idcs[j, 0], i]
            for i, p in enumerate(s_params):
                for j, m in enumerate(new_mc):
                    for k, det in enumerate(new_mc.detnames):
                        m[det][p] = clusters[cluster_idcs[j, k], i]

        if ret_clusters:
            return new_mc, clusters.squeeze()
        else:
            return new_mc

    def _aggregate(self, func, s_default,
                   do="aggregate", inplace=True,
                   params=None, parsers=None, targets=None,
                   keep_params=None, keep_parsers=None, keep_targets=None,
                   analyze=False, **kwargs):
        """Aggregate iterated measurements.

        This method is for internal use only.

        Parameters
        ----------
        func : callable
            Function that computes the aggregates from spectra with signature
            (mc, params, parsers, target, n_meas, dets, avg_dets, Spectrum)
            where
            mc : MeasurementCampaign
                Campaign split by param.
            params : str or list of str
                `params` targeted to `Spectrum`.
            parsers :
                Same as outer `parsers` targeted to `Spectrum`.
            target :
                Specifies target of `params`.
            n_meas : int
                Number of measurements in `mc`.
            dets : iterable
                Detectors.
            avg_dets : bool
                Whether to aggregate detectors.
            Spectrum : type
                Class of return values.
            func must yield the Spectrum type or return an empty list if `dets`
            or `params` are empty.
        s_default : list of str
            List of parameters to aggregate.
        c_default : list of str
            List of parameters to aggregate.
        do : str, optional
            String that describes what this method is doing, e.g. "average".
        inplace : bool, optional
            If true, overwrite measurements with averages.
            If false, store averaged measurements in a new MeasurementCampaign.
        analyze : bool, optional
            Whether to reanalyze the computed spectra.
        params : str or list of str, optional
            Parameters to aggregate by. Those DopplerMeasurements are agg'ed
            where all values of `params` are equal. To agg Single- and
            CoincidenceSpectra within each DopplerMeasurement, `params` must
            contain "single" or "coinc". If `params` is None, all
            DopplerMeasurements are agg'ed. The default is None.
        parsers : callable or list of callable
            Functions to apply to parameters. If `params` is a list, `parsers`
            can be too, then parsers[i] is applied to params[i]. `parsers` can
            be None if no function is to be applied. If `parsers` is a single
            value, it is applied to all parameters.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.
        keep_params : str or list of str, optional
            Metadata to keep. The value will be the average and a corresponding
            "d{param}" metadata item will be added. The default is None.
        keep_parsers : callable or list of callable, optional
            Same as `parsers` but for `keep_params`. The default is None.
        keep_targets : callable or list of callable, optional
            Same as `targets` but for `keep_params`. The default is None.
        analyze : bool, optional
            Whether to reevaluate the resulting spectra. The default is true.
        single_kwargs : dict, optional
            Keyword arguments for SingleSpectrum.analyze.
        coinc_kwargs : dict, optional
            Keyword arguments for CoincidenceSpectrum.coinc_s_param.

        Returns
        -------
        MeasurementCampaign
            Computed aggregated MeasurementCampaign.
        """
        avg_singles = False
        avg_coinc = False

        params, parsers, targets = _check_input(params, parsers, targets)

        if any(a := ["spec" in i for i in params]):
            avg_singles = True
            params = list(np.array(params)[~np.array(a)])
            parsers = list(np.array(parsers)[~np.array(a)])
            targets = list(np.array(targets)[~np.array(a)])

        if not self.is_homogeneous():
            err = f"Cannot {do} inhomogeneous MeasurementCampaign"
            raise RuntimeError(err)

        x = self._get_data(params, parsers, targets)

        n_params, _, n_dets, n_meas = x.shape

        m_params, m_parsers, m_mask = _split_params(x, params, parsers, 0)
        s_params, s_parsers, s_mask = _split_params(x, params, parsers, 1)

        del x

        if n_params != len(m_params):
            args = set(params) - set(m_params)
            raise AttributeError(f"Could not find parameter(s) {args}")

        keep_params, keep_parsers, keep_targets = \
            _check_input(keep_params, keep_parsers, keep_targets)

        x = self._get_data(keep_params, keep_parsers, keep_targets)

        n_params, _, n_dets, n_meas = x.shape

        m_keep_params, m_keep_parsers, m_keep_mask = \
            _split_params(x, keep_params, keep_parsers, 0)
        s_keep_params, s_keep_parsers, s_keep_mask = \
            _split_params(x, keep_params, keep_parsers, 1)

        del x

        if n_params != (len(m_keep_params) + len(s_keep_params)):
            args = set(keep_params)\
                   - (set(m_keep_params)|set(s_keep_params))
            raise AttributeError(f"Could not find parameter(s) {args}")

        m_keep_params = m_params + m_keep_params
        m_keep_parsers = m_parsers + m_keep_parsers

        s_keep_params += s_default
        s_keep_parsers += [None for _ in s_default]

        split_mcs = self.split(m_params, m_parsers, targets="meas")

        new_mc = MeasurementCampaign()

        for mc in split_mcs:
            new_m = LifetimeMeasurement()
            n_meas = len(mc)

            new_m.spectra = list(func(mc, s_keep_params, s_keep_parsers,
                                      "spec", n_meas, self.detnames,
                                      avg_singles, LifetimeSpectrum))

            if len(m_keep_params) > 0:
                m_keep_params, m_keep_parsers, m_target = \
                    _check_input(m_keep_params, m_keep_parsers, "meas")
                m = mc._get_data(m_keep_params, m_keep_parsers, m_target)[:, 0, 0, :]
                dm = np.full(m.shape, np.nan)

                for i in range(len(m)):
                    try:
                        dm[i, :] = np.std(m[i].astype(float))
                        m[i, :] = np.mean(m[i].astype(float))
                    except ValueError:
                        pass

                for i, param in enumerate(m_keep_params):
                    new_m.metadata[param] = m[i, 0]
                    new_m.metadata[f"d{param}"] = dm[i, 0]

            if analyze:
                for s in new_m.spectra:
                    s.fit(**kwargs)

            new_mc.measurements.append(new_m)

        if inplace:
            self.measurements = new_mc.measurements
            new_mc = self

        new_mc.sort(params, parsers, targets)

        return new_mc

    def average(self, params=None, parsers=None, targets=None,
                keep_params=None, keep_parsers=None, keep_targets=None,
                inplace=True):
        """Average measurements with equal parameters.

        This method averages Single- and CoincidenceSpectra by averaging their
        attributes. The result is stored in a new DopplerMeasurement. The
        actual spectrum arrays, as well as all metadata, are discarded in the
        process (except if explicitly kept).

        For each averaged parameter, a corresponding "d{param}" attribute is
        added for the standard deviation of the mean. If a "d{param}" attribute
        exists, it is overwritten.

        Only homogeneous MeasurementCampaigns can be averaged.

        Parameters
        ----------
        params : str or list of str, optional
            Parameters to average by. Those DopplerMeasurements are averaged
            where all values of `params` are equal. To average Single- and
            CoincidenceSpectra within each DopplerMeasurement, `params` must
            contain "single" or "coinc". If `params` is None, all
            DopplerMeasurements are averaged. The default is None.
        parsers : callable or list of callable
            Functions to apply to parameters. If `params` is a list, `parsers`
            can be too, then parsers[i] is applied to params[i]. `parsers` can
            be None if no function is to be applied. If `parsers` is a single
            value, it is applied to all parameters.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.
        keep_params : str or list of str, optional
            Metadata to keep. The value will be the average and a corresponding
            "d{param}" metadata item will be added. The default is None.
        keep_parsers : callable or list of callable, optional
            Same as `parsers` but for `keep_params`. The default is None.
        keep_targets : callable or list of callable, optional
            Same as `targets` but for `keep_params`. The default is None.
        inplace : bool, optional
            If true, overwrite measurements with averages.
            If false, store averaged measurements in a new MeasurementCampaign.

        Returns
        -------
        new_mc : MeasurementCampaign
            If `inplace` is true, this instance is returned.
            If `inplace` is false, the copy is returned.
        """
        def avg(mc, params, parsers, target, n_meas, dets, avg_dets, Spectrum):
            if len(dets) == 0 or len(params) == 0:
                return []

            idx = 0 if target == "meas" else 1

            params, parsers, target = _check_input(params, parsers, target)

            x = mc._get_data(params, parsers, target).astype(float)

            data = []

            for p, param in enumerate(params):
                a = x[p, idx, :, :]
                data.append(a[np.any(isfinite(a), axis=1)])

            x = data

            if avg_dets:
                dx = np.std(x, axis=(-1, -2)) / np.sqrt(len(dets) * n_meas)
                x = np.mean(x, axis=(-1, -2))
                dets = [" % ".join(dets)]
            else:
                dx = np.std(x, axis=-1) / np.sqrt(n_meas)
                x = np.mean(x, axis=-1)

            if avg_dets:
                for i, det in enumerate(dets):
                    new_x = Spectrum(None, det)
                    for j, param in enumerate(params):
                        setattr(new_x, param, x[j])
                        setattr(new_x, f"d{param}", dx[j])
                    yield new_x
            else:
                for i, det in enumerate(dets):
                    new_x = Spectrum(None, det)
                    for j, param in enumerate(params):
                        setattr(new_x, param, x[j, i])
                        setattr(new_x, f"d{param}", dx[j, i])
                    yield new_x

        # TODO: adapt
        s_default = ["s", "w", "v2p", "valley", "counts", "peak_counts"]

        return self._aggregate(
            avg, s_default, do="average", inplace=inplace,
            params=params, parsers=parsers, targets=targets,
            keep_params=keep_params, keep_parsers=keep_parsers,
            keep_targets=keep_targets)

    # Declare method alias:
    avg = average

    def sum(self, params=None, parsers=None, targets=None,
            keep_params=None, keep_parsers=None, keep_targets=None,
            inplace=True, analyze=True, **kwargs):
        """Sum up spectra of measurements with equal parameters and reevaluate.

        This method sums up the spectra of Single- and CoincidenceSpectra. The
        result is stored in a new DopplerMeasurement. All metadata is discarded
        in the process (except if explicitly kept).

        Only homogeneous MeasurementCampaigns can be summed up and also only
        if the summed spectra are of the same shape.

        Parameters
        ----------
        params : str or list of str, optional
            Parameters to sum by. Those DopplerMeasurements are summed up where
            all values of `params` are equal. To sum Single- and Coincidence-
            Spectra within each DopplerMeasurement, `params` must contain
            "single" or "coinc". If `params` is None, all DopplerMeasurements
            are summed up. The default is None.
        parsers : callable or list of callable
            Functions to apply to parameters. If `params` is a list, `parsers`
            can be too, then parsers[i] is applied to params[i]. `parsers` can
            be None if no function is to be applied. If `parsers` is a single
            value, it is applied to all parameters.
        targets : str or list of str, optional
            Specifies if `params` is meant as attribute of DopplerMeasurements
            (then `targets` must contain "doppler"), SingleSpectra (then
            `targets` must contain "single") or CoincidenceSpectra (then
            `targets` must contain "coinc"). If `params` is a list, `targets`
            can be too, then targets[i] is applied to params[i]. `targets` can
            be None, then `params` is taken from the first class in the order
            above where it is found. If `targets` is a single value, it is
            applied to all parameters. The default is None.
        keep_params : str or list of str, optional
            Metadata to keep. The value will be the average and a corresponding
            "d{param}" metadata item will be added. The default is None.
        keep_parsers : callable or list of callable, optional
            Same as `parsers` but for `keep_params`. The default is None.
        keep_targets : callable or list of callable, optional
            Same as `targets` but for `keep_params`. The default is None.
        inplace : bool, optional
            If true, overwrite measurements with averages.
            If false, store averaged measurements in a new MeasurementCampaign.
        analyze : bool, optional
            Whether to reevaluate the resulting spectra. The default is true.
        single_kwargs : dict, optional
            Keyword arguments for SingleSpectrum.analyze.
        coinc_kwargs : dict, optional
            Keyword arguments for CoincidenceSpectrum.coinc_s_param.

        Returns
        -------
        new_mc : MeasurementCampaign
            If `inplace` is true, this instance is returned.
            If `inplace` is false, the copy is returned.
        """
        def add(mc, params, parsers, target, n_meas, dets, avg_dets, Spectrum):
            if len(dets) == 0 or len(params) == 0:
                return []

            idx = 0 if target == "meas" else 1

            params, parsers, target = _check_input(params, parsers, target)

            x = mc._get_data(params, parsers, target)#.astype(float)

            data = []

            for p, param in enumerate(params):
                a = x[p, idx, :, :]
                if param == "tcal":
                    tcal = a[np.any(isfinite(a), axis=1)]
                elif param == "spectrum":
                    spectra = a[np.any(isfinite(a), axis=1)]
                else:
                    data.append(a[np.any(isfinite(a), axis=1)])

            x = data

            attrs = set(params) - set(["cal", "spectrum"])

            if avg_dets:
                spectra = np.sum(spectra, axis=(0, 1))  # TODO adapt
                tcal = np.mean(tcal, axis=(0, 1))
                x = np.sum(x, axis=(1, 2))
                dets = [" + ".join(dets)]
            else:
                summed_spectra = np.empty(spectra.shape[0], dtype=object)
                for i in range(spectra.shape[0]):
                    summed = np.zeros(spectra[i, 0].shape, dtype=np.uint64)
                    for spectrum in spectra[i]:
                        summed += spectrum
                    summed_spectra[i] = summed
                tcal = np.mean(tcal, axis=1)
                x = np.sum(x, axis=-1)

            for i, det in enumerate(dets):
                new_x = Spectrum(summed_spectra[i], det, autocompute=False)
                for j, attr in enumerate(attrs):
                    setattr(new_x, attr, x[j, i])
                new_x.tcal = tcal[i]
                yield new_x

        s_default = ["spectrum", "tcal"]

        return self._aggregate(
            add, s_default, do="sum", inplace=inplace,
            params=params, parsers=parsers, targets=targets,
            keep_params=keep_params, keep_parsers=keep_parsers,
            keep_targets=keep_targets, analyze=analyze,
            **kwargs)

    def normalize(
        self, normalize_to=1, normalize_with=None, n=None, x_param=None,
        x_parser=None, avg=True, clusters=None, show_assignment=False,
        keep_params=None, keep_parsers=None, keep_targets=None, inplace=True,
    ):
        """Normalize lineshape parameters of this MeasurementCampaign's
        SingleSpectra.

        This method scales the values of lineshape parameters according to
        the provided arguments. The normalization uses the formula
        >>> normalized = normalize_to * original_value / normalize_with

        Parameters
        ----------
        normalize_to : numpy.ndarray or int or float or None, optional
            Value(s) that this campaign should be normalized to.

            - If `normalize_to` is an int or float, then all values are
              normalized to that value.
            - If `normalize_to` is a numpy.ndarray, then its shape must be
              (number of lineshape parameters, number of single detectors,
              number of DopplerMeasurements) and its elements are applied to
              the corresponding SingleSpectrum's lineshape parameters. The
              order of lineshape parameters is s, w, v2p, valley, counts,
              peak_counts. The order of detector is that in `self.detnames`.
              The order of DopplerMeasurements is that in `self.measurements`.
            - If `normalize_to` is None, then no normalization is applied.

            The default is 1.
        normalize_with : numpy.ndarray or int or float or None, optional
            Value(s) that this campaign should be normalized with.

            - If `normalize_with` is None (default), then it is computed from
              the DopplerMeasurements specified by `n`.
            - If `normalize_with` is a numpy.ndarray, then its shape must be
              (number of lineshape parameters, number of single detectors,
              number of DopplerMeasurements) and its elements are applied to
              the corresponding SingleSpectrum's lineshape parameters. The
              order of lineshape parameters is s, w, v2p, valley, counts,
              peak_counts. The order of detector is that in `self.detnames`.
              The order of DopplerMeasurements is that in `self.measurements`.
        n : int or list of int or numpy.ndarray or range or slice or Ellipsis or None, optional
            Indices of DopplerMeasurements to use for the computation of
            `normalize_with`. Only takes effect if `normalize_with` is None.
            The options below apply to each lineshape parameter and each
            detector separately.

            - If `n` is None (default), then `normalize_with` is equal to the
              original value, meaning the above equation becomes
              `normalized = normalize_to`.
            - If `n` is a positive int, then `normalize_with` is the average of
              the first `n` DopplerMeasurements.
            - If `n` is a negative int, then `normalize_with` is the average of
              the last `n` DopplerMeasurements.
            - If `n` is Ellipsis, then `normalize_with` is the average of
              all DopplerMeasurements.
            - If `n` is a range, slice or list of int, then the elements are
              taken as indices of DopplerMeasurements to compute the average
              with.
            - If `n` is a numpy.ndarray, then it is taken as index array.

            See `x_param` for the option to average and sort DopplerMeasurements
            before applying the indices.
        x_param : str or None
            Parameter to sort and optionally average DopplerMeasurements by.
            Accordingly, `x_params` must be an attribute of DopplerMeasurements
            or key in metadata. If `x_params` is None (default), then the
            DopplerMeasurements are neither sorted nor averaged and `n` refers
            to index in this MeasurementCampaign's `measurements` list. If
            `x_param` is provided, then the DopplerMeasurements are at least
            sorted by that parameter. See `avg` to specify whether to also
            average DopplerMeasurements by `x_param`.
        x_parser : callable, optional
            Function to apply to `x_param` before using it to sort and average
            by. The default is None.
        avg : bool, optional
            Whether to average by `x_param` before sorting and indexing. Only
            applies if `x_params` is provided. The default is True.
        clusters : list, optional
            If provided, use the cluster method before averaging and pass
            `clusters` as `guess`. The default is None (no clustering).
        show_assignment : bool, optinal
            Whether to show cluster assignment. The default is False.
        keep_params : str or list of str, optional
            Metadata to keep when averaging. The value will be the average and
            a corresponding "d{param}" metadata item will be added.
            The default is None.
        keep_parsers : callable or list of callable, optional
            Function to apply to `keep_params` before averaging.
            The default is None.
        keep_targets : callable or list of callable, optional
            Specifies what object `keep_params` is meant to belong to.
            'doppler' for DopplerMeasurements, 'single' for SingleSpectra and
            'coinc' for CoincidenceSpectra'. If None, then the first one of
            these in this order where it is found is used. The default is None.
        inplace : bool, optional
            If True, overwrite lineshape parameters with their normalized
            values. If False, store normalized values in a new
            MeasurementCampaign.

        Returns
        -------
        numpy.ndarray
            Array of values used for `normalize_with` in the top equation. Its
            shape is (number of lineshape parameters, number of single
            detectors, number of DopplerMeasurements). The order of lineshape
            parameters is s, w, v2p, valley, counts, peak_counts. The order of
            detector is that in `self.detnames`. The order of
            DopplerMeasurements is that in `self.measurements`.
        MeasurementCampaign
            If `inplace` is False, the new MeasurementCampaign is returned.
        """
        raise NotImplementedError()
        err = "Only homogeneous MeasurementCampaign may be normalized"
        assert self.is_homogeneous(), err
        assert normalize_with is None or isinstance(normalize_with, np.ndarray)

        keep_params, keep_parsers, keep_targets = \
            _check_input(keep_params, keep_parsers, keep_targets)

        keep_params.append(x_param)
        keep_parsers.append(x_parser)
        keep_targets.append("meas")

        if not inplace:
            self = self.copy()

        if x_param is not None:
            if avg is True:
                if clusters is not None:
                    self.cluster(
                        x_param, parsers=x_parser, targets="meas",
                        guess=clusters, show_assignment=show_assignment
                    )
                self.average(
                    x_param, x_parser, "meas", keep_params=keep_params,
                    keep_parsers=keep_parsers, keep_targets=keep_targets
                )
            self.sort(x_param, x_parser)

        params = ["s", "w", "v2p", "valley", "counts", "peak_counts"]
        dparams = ["d" + p for p in params]

        if normalize_with is None:
            normalize_with = self._get_data(*_check_input(params, None, "single"))

            if n is not None:
                normalize_with = normalize_with[:, 1, :, n]
            else:
                normalize_with = normalize_with[:, 1, :, :]

            # factors.shape:
            #     (n_params, n_detectors, len(self))   if n is None
            #     (n_params, n_detectors)              if n is scalar
            #     (n_params, n_detectors, len(n))      otherwise

            if normalize_with.ndim > 2 and n is not None:
                normalize_with = np.average(normalize_with, axis=-1)

        normalize_with = np.where(normalize_with == 0, 1, normalize_with)

        if normalize_to is None:
            if not inplace:
                return normalize_with, self
            else:
                return normalize_with

        if isinstance(normalize_to, (int, float)):
            normalize_to = np.full(normalize_with.shape, normalize_to)

        assert np.all(np.equal(normalize_to.shape, normalize_with.shape))

        dets = self.detnames

        for i, m in enumerate(self):
            for d, det in enumerate(dets):
                for p, (param, dparam) in enumerate(zip(params, dparams)):
                    if n is None:
                        a = normalize_to[p, d, i]
                        b = normalize_with[p, d, i]
                    else:
                        a = normalize_to[p, d]
                        b = normalize_with[p, d]
                    single = m[det]
                    y = getattr(single, param)
                    dy = getattr(single, dparam)
                    _y = a * y / b
                    _dy = a * dy / b
                    setattr(single, param, _y)
                    setattr(single, dparam, _dy)

        if not inplace:
            return normalize_with, self
        else:
            return normalize_with

    def plot(self, x_params=None, y_params=None, z_params=None, *,
             x_parsers=None, y_parsers=None, z_parsers=None,
             x_targets=None, y_targets=None, z_targets=None,
             errorbars=None, scatter=None, filled=None,
             show=True, ret_data=False, fig=None, axs=None,
             label="", label_is_param=False, detnames_in_label=None,
             full_xlabels=False, **kwargs):
        """Plot any numerical aspect against any other numerical aspect.

        This method can create line, errorbar, scatter and tricontour plots
        depending on the arguments `errorbars`, `scatter` and `filled`:

        - If `scatter` is false or if `scatter` is None and `z_params` is None:
            - If `errorbars` is false, use `plt.plot`.
            - Else use `plt.errorbar`.
        - Else:
            - If `filled` is None, use `plt.scatter`.
            - If `filled` is true, use `plt.tricontourf`.
            - If `filled` is false, use `plt.tricontour`.

        This method orients itself on matplotlib's plot function when it comes
        to positional arguments: If only one positional argument is given, it is
        taken as `y_params`. If two positional arguments are given, the first
        one is taken as `x_params` and the second one as `y_params`. If three
        positional arguments are given, the first one is taken as `x_params`,
        the second one as `y_params` and the third one as `z_params`.

        Parameters
        ----------
        x_params : str or list of str, optional
            Parameters to use as x-coordinates.

            - If `x_params` is None, the index of data points is taken as
              x-coordinates for all plots.
            - If `x_params` is a single value, it is used for all plots.
            - `x_params` can be a list of values, then its length must be equal
              to the number of plots and then `x_params[i]` is used in plot i.

            The number of plots is equal to the maximum number of parameters in
            `x_params`, `y_params` and `z_params`. The default is None.
        y_params : str or list of str
            Parameters to use as y-coordinates.

            - If `y_params` is None, `x_params` is taken as `y_params` and
              the index of data points is taken as x-coordinates for all plots.
            - If `y_params` is a single value, it is used for all plots.
            - `y_params` can be a list of values, then its length must be equal
              to the number of plots and then `y_params[i]` is used in plot i.

            The number of plots is equal to the maximum number of parameters in
            `x_params`, `y_params` and `z_params`.
        z_params : str or list of str, optional
            Parameters to use as z-coordinates.

            - If `z_params` is None, no coloring is applied.
            - If `z_params` is a single value, it is used for all plots.
            - `z_params` can be a list of values, then its length must be equal
              to the number of plots and then `z_params[i]` is used in plot i.

            The number of plots is equal to the maximum number of parameters in
            `x_params`, `y_params` and `z_params`. The default is None.
        x_parsers : callable or list of callable, optional
            Functions to apply to x-parameters.

            - If `x_parsers` is None, no function is applied.
            - If `x_parsers` is a single value, it is applied to all
              x parameters.
            - If `x_params` is a list, `x_parsers` can be too, then x_parsers[i]
              is applied to x_params[i].

            The default is None.
        y_parsers : callable or list of callable, optional
            Same as `x_parsers`, but for `y_params`. The default is None.
        z_parsers : callable or list of callable, optional
            Same as `x_parsers`, but for `z_params`. The default is None.
        x_targets : str or list of str, optional
            Specifies if `x_params` is meant as attribute of

            1. `DopplerMeasurement` (then `x_targets` must contain "doppler")
            2. `SingleSpectrum` (then `x_targets` must contain "single")
            3. `CoincidenceSpectrum` (then `x_targets` must contain "coinc")

            - If `x_targets` is None, `x_params` is taken from the first class
              in the order above where it is found.
            - If `x_targets` is a single value, it is applied to all
              parameters.
            - If `x_params` is a list, `x_targets` can be too, then
              x_targets[i] is applied to x_params[i].
            The default is None.
        y_targets : str or list of str, optional
            Same as `x_targets`, but for `y_params` and `y_parsers`.
            The default is None.
        z_targets : str or list of str, optional
            Same as `x_targets`, but for `z_params` and `z_parsers`.
            The default is None.
        errorbars : bool, optional
            Whether to plot uncertainties of parameters. Only applies to 1d
            plots (i.e. if `z_params` is None). If true or None, for each
            parameter in `y_params` and `x_params` look for a corresponding
            f"d{param}" attribute or metadata entry to take as error.
            The default is None.
        scatter : bool, optional
            - If true, force use of `plt.scatter` for 1d plots (i.e. `z_params`
              is None).
            - If false, force use of `plt.errorbar`
            - If None, use `plt.scatter` if `z_params` is provided,
              otherwise use `plt.errorbar`.
            The default is None.
        filled : bool, optional
            - If true, use `plt.tricontourf`.
            - If false, use `plt.tricontour`.
            - If None, use `plt.scatter`.
            The default is None.
        show : bool, optional
            If true, call `plt.grid`, `plt.legend`, `plt.tight_layout` and
            `plt.show` on the figure before returning it. The default is true.
        ret_data : bool, optional
            If true, return the data computed for plotting. The default is
            false.
        fig : matplotlib.figure.Figure
            Figure to plot on. If None and `axs` is None, a new Figure instance
            is created. The default is None.
        axs : (numpy.ndarray of) matplotlib.axes.Axes, optional
            Axes to plot on. If None and `fig` is None, a new Figure instance
            is created. If None and `fig` is not None, use axes of `fig`.
            If `axs` is a numpy.ndarray, it is raveled and `axs[i]` is used for
            plot i.
            The default is None.
        label : str, optional
            Labels to add to legend entries for each detector.
            If `label` is a parameter name and `label_is_param` is true, lookup
            the parameter value in the first `DopplerMeasurement` and use its
            value as label. The default is "" (no labels).
        label_is_param : bool, optional
            Whether to interpret `label` as parameter name to extract or to
            use as label directly. The default is false (use directly).
        detnames_in_label : bool, optional
            - If true, add the detector (pair) name to each legend entry.
            - If None, the names are added only if there is more than one
              detector (pair) to plot.
            - Otherwise, suppress detnames.
            The default is None.
        full_xlabels : bool, optional
            Whether to plot x-labels on all subplots.

            - If false and `x_params` contains only one parameter, only the
              bottom row of plots reveices x-labels.
            - If `x_params` contains more than one parameter, `full_xlabels`
              is automatically true.
            The default is false.
        **kwargs:
            Keyword arguments are passed along to the matplotlib plotting
            function used.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure created for plotting.
        axs : numpy.ndarray
            (numpy.ndarray of) matplotlib.axes.Axes
        data : pandas.DataFrame or dict of numpy.ndarray
            If `ret_data` is true, return the data computed for plotting.
        """
        err = "Not all input parameters can be None"
        assert (x_params is not None or y_params is not None
                or z_params is not None), err

        err = ("Configuration argument `{}` has incompatible type {}. "
               "Must be None or bool")
        for var, x in zip(["errorbars", "scatter", "filled"],
                          [errorbars, scatter, filled]):
            assert x is None or isinstance(x, bool), err.format(var, type(x))

        x_params, x_parsers, x_targets = \
            _check_input(x_params, x_parsers, x_targets)

        y_params, y_parsers, y_targets = \
            _check_input(y_params, y_parsers, y_targets)

        z_params, z_parsers, z_targets = \
            _check_input(z_params, z_parsers, z_targets)

        n_x, n_y, n_z = len(x_params), len(y_params), len(z_params)
        n_plots = max(n_x, n_y, n_z)
        n_dets = len(self.detnames)
        n_meas = len(self)

        err = (f"Input parameters of `plot` have incompatible lengths: "
               f"len(x)={n_x}, len(y)={n_y}, len(z)={n_z}")
        for n in [n_x, n_y, n_z]:
            assert n in [0, 1, n_plots], err

        def make_index():
            return [""], [lambda x: x], [None]

        match tuple(i > 0 for i in (n_x, n_y, n_z)):
            case (True, True, True):
                pass
            case (True, True, False):
                pass
            case (True, False, True):
                # replace y with index
                y_params, y_parsers, y_targets = make_index()
            case (False, True, True):
                # replace x with index
                x_params, x_parsers, x_targets = make_index()
            case (True, False, False):
                # replace y with x
                y_params, y_parsers, y_targets = x_params, x_parsers, x_targets
                # replace x with index
                x_params, x_parsers, x_targets = make_index()
            case (False, True, False):
                # replace x with index
                x_params, x_parsers, x_targets = make_index()
            case (False, False, True):
                # replace x and y with index
                x_params, x_parsers, x_targets = make_index()
                y_params, y_parsers, y_targets = make_index()

        n_x, n_y, n_z = len(x_params), len(y_params), len(z_params)

        full_xlabels = full_xlabels or (n_x > 1)

        if n_x < n_plots:
            for p in [x_params, x_parsers, x_targets]:
                p *= n_plots

        if n_y < n_plots:
            for p in [y_params, y_parsers, y_targets]:
                p *= n_plots

        if n_z < n_plots and n_z > 0:
            for p in [z_params, z_parsers, z_targets]:
                p *= n_plots

        x = self._get_data(x_params, x_parsers, x_targets)
        x = np.broadcast_to(x, (n_plots, 2, n_dets, n_meas))

        y = self._get_data(y_params, y_parsers, y_targets)
        y = np.broadcast_to(y, (n_plots, 2, n_dets, n_meas))

        if n_plots < 4:
            n_rows = n_plots
            n_cols = 1
        else:
            n_rows = int(np.ceil(np.sqrt(n_plots)))
            n_cols = int(np.ceil(n_plots / n_rows))

        x_mask = np.any(isfinite(x), axis=(2, 3))
        y_mask = np.any(isfinite(y), axis=(2, 3))

        labels = np.full(y.shape[1:-1], "", dtype=object)

        if label_is_param:
            label = str(self[0][label])

        if (detnames_in_label is True or
                (detnames_in_label is None and y.shape[2] > 1)):
            if label != "":
                label += " "

            for i, det in enumerate(self.detnames):
                labels[1, i] = label + det
        else:
            for i, det in enumerate(self.detnames):
                labels[1, i] = label

        labels = np.array([labels] * n_plots)

        if fig is None and axs is None:
            fig, axs = plt.subplots(n_rows, n_cols)
        elif axs is None:
            axs = fig.get_axes()
        else:
            from matplotlib.axes import Axes
            if not isinstance(axs, np.ndarray):
                axs = np.array(axs)
            err = ("Provided axes contain not enough subplots for requested "
                   f"plots (got {axs.size} subplots for {n_plots} plots)")
            assert axs.size >= n_plots, err
            fig = axs.flat[0].get_figure()

        axs = np.ravel(axs)

        warnings.simplefilter("ignore", RuntimeWarning)
        if scatter is False or (scatter is None and n_z == 0):
            if n_x == 0:
                dx = np.full(x.shape, np.nan)
            else:
                dx = self._get_data([f"d{p}" for p in x_params],
                                    x_parsers, x_targets)
                dx = np.broadcast_to(dx, (n_plots, 2, n_dets, n_meas))

            dy = self._get_data([f"d{p}" for p in y_params],
                                y_parsers, y_targets)
            dy = np.broadcast_to(dy, (n_plots, 2, n_dets, n_meas))
            if errorbars is False:
                for i, param in enumerate(y_params):
                    axs[i].set_ylabel(param)
                    mx = x_mask[0].argmax()
                    my = y_mask[i].argmax()
                    for j in range(y.shape[2]):
                        axs[i].plot(x[i, mx, 0, :], y[i, my, j, :],
                                    label=labels[i, my, j], **kwargs)

            else:
                for i, param in enumerate(y_params):
                    axs[i].set_ylabel(param)
                    mx = x_mask[0].argmax()
                    my = y_mask[i].argmax()
                    for j in range(y.shape[2]):
                        axs[i].errorbar(x[i, mx, 0, :], y[i, my, j, :],
                                        dy[i, my, j, :], dx[i, mx, 0, :],
                                        label=labels[i, my, j], **kwargs)
        else:
            if n_z == 0:
                z = np.full((n_plots, 2, n_dets, n_meas), 0)
                z_mask = np.full((n_plots, 2, n_dets, n_meas), True)
            else:
                z = self._get_data(z_params, z_parsers, z_targets)
                z_mask = np.any(isfinite(z), axis=(2, 3))

            x = x.astype(str).astype(float)
            y = y.astype(str).astype(float)
            z = z.astype(str).astype(float)

            for i, param in enumerate(y_params):
                axs[i].set_ylabel(param)
                mx = x_mask[0].argmax()
                my = y_mask[i].argmax()
                mz = z_mask[i].argmax()
                if filled is None or n_z == 0:
                    im = axs[i].scatter(x[i, mx, 0, :], y[i, my, 0, :],
                                        c=z[i, mz, 0, :], **kwargs)
                elif filled is True:
                    im = axs[i].tricontourf(x[i, mx, 0, :], y[i, my, 0, :],
                                            z[i, mz, 0, :], **kwargs)
                else:
                    im = axs[i].tricontour(x[i, mx, 0, :], y[i, my, 0, :],
                                           z[i, mz, 0, :], **kwargs)
                if n_z > 0:
                    fig.colorbar(im, ax=axs[i])
        warnings.simplefilter("default", RuntimeWarning)

        if full_xlabels is True:
            for i in range(n_plots):
                axs[i].set_xlabel(x_params[i] or "Measurement")
        else:
            for i in range(-n_cols, 0, 1):
                axs[i].set_xlabel(x_params[0] or "Measurement")

        if show:
            for i, param in enumerate(y_params):
                axs[i].grid()
                if n_z == 0:
                    axs[i].legend()
            plt.tight_layout()
            plt.show()

        params = []
        parsers = []
        targets = []

        def get_dlist(x):
            return list(np.ravel([*zip(x, [f"d{p}" for p in x])]))

        def get_list(x):
            return list(np.ravel([*zip(x, x)]))

        for i_params, i_parsers, i_targets in zip(
                [x_params, y_params, z_params],
                [x_parsers, y_parsers, z_parsers],
                [x_targets, y_targets, z_targets]):
            params.extend(get_dlist(i_params))
            parsers.extend(get_list(i_parsers))
            targets.extend(get_list(i_targets))

        data = self.get(params, parsers, targets)

        if ret_data:
            return fig, axs, data
        else:
            return fig, axs

    def plot_metadata(self, key_string="",
                      x_param="", x_parser=None, x_target=None, show=True):
        """Generate plots of metadata, like power supply voltages and currents.

        Using the __getitem__ function and the metadata attribute of the
        DopplerMeasurements contained in this MeasurementCampaign extract
        meta-datasets and plot them. Very useful to check if power supplies
        are stable.

        Parameters
        ----------
        key_string : str
            Sequence of characters that must be contained in the metadata key.
        x_param : str, optional
            Parameter to use as x-coordinates. If None, take index of
            measurements. The default is None.
        x_parser : callable or list of callable, optional
            Functions to apply to x-parameters. The default is None.
        x_target : str or list of str, optional
            Specifies if `x_params` is meant as attribute of

            1. `DopplerMeasurement` (then `x_targets` must contain "doppler")
            2. `SingleSpectrum` (then `x_targets` must contain "single")
            3. `CoincidenceSpectrum` (then `x_targets` must contain "coinc")

            If `x_targets` is None, `x_params` is taken from the first class
            in the order above where it is found. The default is None.
        """
        x = list(self.get(x_param, x_parser, x_target, use_pd=False).values())[0]
        meas = self.measurements[0]

        fig, ax = plt.subplots()

        for key, val in meas.metadata.items():
            if key_string in key:
                all_values_are_numbers = (
                        all([isinstance(x, (int, float)) for x in self[key]])
                        and not all([np.isnan(x) for x in self[key]]))
                if all_values_are_numbers:
                    ax.plot(x, self[key], label=key)
                else:
                    continue

        ax.set_xlabel(x_param or "Measurement")
        ax.set_ylabel("Metadata")

        if show:
            ax.grid()
            ax.legend()
            plt.show()

        return fig, ax

    def depth_profiles(self, x_param, y_params,
                       x_parsers=None, y_parsers=None,
                       x_targets=None, y_targets=None,
                       keep_params=None, keep_parsers=None, keep_targets=None,
                       energies=None, show_assignment=False,
                       label="", label_is_param=False, detnames_in_label=None,
                       avg_equalpoints=True, avg_method="avg",
                       single_kwargs=None, coinc_kwargs=None,
                       show=True, ret_data=False, inplace=True,
                       makhov_params=None, xunit_order=True,
                       full_xlabels=False, **kwargs):
        """Plot depthprofiles.

        Parameters
        ----------
        x_param : str
            Parameter name of implantation energy.
        y_params : str or list of str
            Parameters to plot against energy.
        parsers : callable or list of callable
            Functions to apply to `y_params`. If `y_params` is a list,
            `parsers` can be too, then parsers[i] is applied to y_params[i].
            `parsers` can be None if no function is to be applied. If `parsers`
            is a single value, it is applied to all parameters.
            The default is None.
        targets : str or list of str, optional
            Specifies if `y_params` is meant as attribute of
            DopplerMeasurements (then `targets` must contain "doppler"),
            SingleSpectra (then `targets` must contain "single") or
            CoincidenceSpectra (then `targets` must contain "coinc"). If
            `y_params` is a list, `targets` can be too, then targets[i] is
            applied to y_params[i]. `targets` can be None, then `y_params` is
            taken from the first class in the order above where it is found.
            If `targets` is a single value, it is applied to all parameters.
            The default is None.
        energies : list of float
            Energies in the depth profiles. Each measurement is assigned to its
            closest energy (uses MeasurementCampaign.cluster).
            The default is None.
        show_assignment : bool
            Whether to show the assignment of measurements to `energies`.
        label : str
            Labels to add to legend entries for each detector.
            If `label` is a parameter name and `label_is_param` is true, lookup
            the parameter value in the first DopplerMeasurement and use its
            value as label. The default is "" (no labels).
        label_is_param : bool
            Whether to interpret `label` as parameter name to extract or to
            use as label directly. The default is false (use directly).
        detnames_in_label : bool
            If true, add the detector (pair) name to each legend entry.
            If None, the names are added only if there is more than one
            detector (pair) to plot.
            Otherwise, suppress detnames. The default is None.
        avg_equalpoints : bool
            Whether to average measurements with equal energy.
            The default is true.
        avg_method : str
            Method to use for averaging. Options are "avg" to average
            parameters of individual measurements or "sum" to sum measurements
            and evaluate parameters from the resulting spectra.
            The default is "avg".
        show : bool
            If true, call `plt.show` on the figure before returning it.
            The default is true.
        ret_data : bool, optional
            If true, return the data computed for plotting. The default is
            false.
        inplace : bool
            If true, averaging operations are performed on this instance,
            otherwise on a copy of this instance. The default is true.
        makhov_params : tuple of float, optional
            Density [g/cm^3] and Makhov parameters A [µg cm^{-2} keV^{-n}]
            and n [1] to compute the mean implantation depth in nm. See
            `xunit_order` for options on which units to show on which axes.
            The default is None.
        xunit_order : bool, optional
            Takes effect if `makhov_params` is provided.

            - If true, show energy on top and depth on bottom axis.
            - If false, show depth on top and energy on bottom axis.
            - If None, hide energy.

            The default is true.
        full_xlabels : bool, optional
            Whether to plot x-labels on all subplots.

            - If false and `x_params` contains only one parameter, only the
              bottom row of plots reveices x-labels.
            - If `x_params` contains more than one parameter, `full_xlabels`
              is automatically true.
            The default is false.
        **kwargs:
            Keyword arguments to be passed along to `plt.errorbar` or
            `plt.scatter`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure created for plotting.
        axs : numpy.ndarray
            Array of figure axes.
        data : tuple of numpy.ndarray
            If `ret_data` is true, return the data computed for plotting.
        """
        err = "`x_param` can only be a singular parameter of type str, is {}"
        assert isinstance(x_param, str), err.format(type(x_param))

        fig = None
        data = []

        if label_is_param:
            label = str(self[0][label])

        if avg_equalpoints:
            self = self.cluster(x_param, parsers=x_parsers, targets=x_targets,
                                guess=energies, show_assignment=show_assignment,
                                inplace=inplace)
            if avg_method == "avg":
                self = self.average(x_param, x_parsers, x_targets,
                                    keep_params, keep_parsers, keep_targets,
                                    inplace=inplace)
            elif avg_method == "sum":
                self = self.sum(x_param, x_parsers, x_targets,
                                keep_params, keep_parsers, keep_targets,
                                inplace=inplace, single_kwargs=single_kwargs,
                                coinc_kwargs=coinc_kwargs)
            else:
                raise ValueError(f"Unknown averaging method {avg_method}")

        if makhov_params is not None:
            rho, A, n = makhov_params
            x2_param = "MeanImplantationDepth"
            depth = lambda e: 10*A * (e/1000)**n / rho
            energy = lambda z: 1000 * (z * rho / (10 * A)) ** (1/n)
            self.derive(x2_param, x_param, depth)
            full_xlabels = True

            if xunit_order is True:
                x_param, x2_param =  x2_param, x_param
                conv_f, _ = energy, depth
            elif xunit_order is False:
                conv_f, _ = depth, energy
            elif xunit_order is None:
                x_param =  x2_param
            else:
                raise ValueError(xunit_order)

        fig, axs, data = self.plot(
            x_param, y_params, x_parsers=x_parsers, y_parsers=y_parsers,
            x_targets=x_targets, y_targets=y_targets, show=False, fig=fig,
            ret_data=True, label=label, detnames_in_label=detnames_in_label,
            full_xlabels=full_xlabels, **kwargs)

        if makhov_params is not None and xunit_order is not None:
            for ax in axs:
                x_ticks_bottom = ax.get_xticks()

                x_ticks_top = conv_f(x_ticks_bottom)

                mask = np.isfinite(x_ticks_top)
                x_ticks_bottom = x_ticks_bottom[mask]
                x_ticks_top = x_ticks_top[mask]

                ax_top = ax.secondary_xaxis("top")
                ax_top.set_xticks(x_ticks_bottom)
                ax_top.set_xticklabels([f"{val:.0f}" for val in x_ticks_top])

                ax_top.set_xlabel(x2_param)

        if show:
            for ax in axs:
                ax.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        if ret_data:
            return fig, axs, data
        else:
            return fig, axs

    def dump_components(self, filepath=None):
        out = {}
        for i, m in enumerate(self):
            out[f"Measurement {m.name or i+1}"] = m.dump_components()

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)
        else:
            return out

    def dump_resolution_components(self, filepath=None):
        out = {}
        for i, m in enumerate(self):
            out[f"Measurement {m.name or i+1}"] = m.dump_resolution_components()

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)
        else:
            return out

    def dump_lifetime_components(self, filepath=None):
        out = {}
        for i, m in enumerate(self):
            out[f"Measurement {m.name or i+1}"] = m.dump_lifetime_components()

        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(out, f, indent=4)
        else:
            return out


class MultiCampaign:
    def __init__(
        self,
        path: str | Path | List[str | Path] | None = None,
        *,
        campaigns: List[MeasurementCampaign] | None = None,
        lt_model: LifetimeModel | Dict | str | None = None,
        res_model: LifetimeModel | Dict | str | None = None,
        calibrate: bool = False,
        name: str | None = None,
        dtype: type | None = None,
        show_fits: bool = True,
        show: bool = False,
        verbose: bool = False,
        autocompute: bool = True,
        pbar: bool = True,
        cache: bool = False,
        cache_path: str = "/tmp/pals_cache.pkl",
        **kwargs
    ) -> None:

        if pbar:
            start = time.time()

        self.show = show
        self.verbose = verbose
        self.name = name

        self.path = path
        self.directory = None

        self.campaigns = []

        cache_hit = False
        self.cache_path = cache_path
        if cache and os.path.exists(cache_path):
            if pbar:
                progress(0, 1, start, 20, "Importing")
            cached_mcs = self.load_cache(cache_path, inplace=False)
            cached_path = cached_mcs.path

            if ((
                    path is None or cached_path is None
                ) or (
                    isinstance(path, (str, Path)) and isinstance(cached_path, (str, Path))
                    and Path(path) == Path(cached_path)
                ) or (
                    isinstance(path, (list)) and isinstance(cached_path, list)
                    and len(path) == len(cached_path)
                    and np.all(np.equal(list(map(Path, path)), list(map(Path, cached_path))))
                )):
                cache_hit = True
                campaigns = cached_mcs.campaigns
                self.path = cached_path
                self.directory = cached_mcs.directory
                self.name = cached_mcs.name
                if pbar:
                    progress(1, 1, start, 20, "Importing")

        if campaigns is not None:
            self.campaigns = list(campaigns)
        elif path is not None:
            if isinstance(path, (str, Path)):
                self.path = path = Path(path)
                self.directory = path.name
                self.name = name or self.directory
            elif isinstance(path, list):
                self.path = list(map(Path, path))
            else:
                raise TypeError

            self.campaigns = MeasurementCampaign(
                path, lt_model=lt_model, res_model=res_model,
                calibrate=calibrate, dtype=dtype, show_fits=show_fits,
                show=self.show, verbose=self.verbose,
                autocompute=autocompute, **kwargs
            ).split("directory").campaigns

            for mc in self.campaigns:
                mc.path = None if mc[0].path is None else mc[0].path.parent
                mc.directory = mc[0].directory
                mc.name = mc[0].directory

        if cache and not cache_hit:
            self.cache()

    def __len__(self):
        return len(self.campaigns)

    def __iter__(self):
        return (mc for mc in self.campaigns)

    def __getitem__(self, i):
        if isinstance(i, str):
            for mc in self:
                if mc.name == i:
                    return mc

        return self.campaigns[i]

    def __add__(self, other):
        new = MultiCampaign()

        new.campaigns = self.campaigns + other.campaigns
        new.show = self.show or other.show
        new.verbose = self.verbose or other.verbose

        return new

    @property
    def detnames(self):
        return {detname for mc in self for detname in mc.detnames}

    @property
    def shape(self):
        shape = namedtuple("Shape", ["mc", "m", "s"])

        n_mc = len(self)

        n_m = Counter([len(mc) for mc in self])
        if len(n_m) == 0:
            n_m = np.nan
        elif len(n_m) == 1:
            n_m = list(n_m.keys())[0]
        elif max(n_m.values()) == 1:
            n_m = tuple(n_m.keys())

        n_s = Counter([len(m.spectra) for mc in self for m in mc])
        if len(n_s) == 0:
            n_s = np.nan
        elif len(n_s) == 1:
            n_s = list(n_s.keys())[0]
        elif max(n_s.values()) == 1:
            n_s = tuple(n_s.keys())

        return shape(n_mc, n_m, n_s)

    def is_homogeneous(self):
        for mc in self:
            if not mc.is_homogeneous():
                return False

        return True

    def copy(self, deep=True, copy_spectra=True):
        mcs = MultiCampaign()
        for attr, value in self.__dict__.items():
            try:
                setattr(mcs, attr, value.copy())
            except AttributeError:
                setattr(mcs, attr, value)

        if deep:
            for i, mc in enumerate(self):
                mcs.campaigns[i] = mc.copy(deep, copy_spectra)

        return mcs

    def cache(self, path=None):
        self.cache_path = path or self.cache_path

        with open(self.cache_path, "wb") as f:
            pickle.dump(self, f)

    def load_cache(self, path=None, inplace=True):
        self.cache_path = path or self.cache_path

        with open(self.cache_path, "rb") as f:
            cached_mcs = pickle.load(f)

        if inplace:
            self.campaigns = cached_mcs.campaigns
            self.path = cached_mcs.path
            self.directory = cached_mcs.directory
            self.name = cached_mcs.directory
            self.model = cached_mcs.model
        else:
            return cached_mcs

    def get(self, params, parsers=None, targets=None, slice=..., use_pd=True):
        return [mc.get(params, parsers, targets, slice, use_pd) for mc in self]

    def print(self, params, parsers=None, targets=None,
              index=True, header=True, sep="", fmt="<", slice=...):
        for i, mc in enumerate(self):
            name = f"Measurement Campaign {i+1} ({mc.name})"
            print(f"{name}:")
            print()

            mc.print(params, parsers, targets, index, header, sep, fmt, slice)

            print()

    def parse(self, params, parsers, targets=None, inplace=True):
        new_mcs = [mc.parse(params, parsers, targets=targets, inplace=inplace)
                   for mc in self]

        if inplace:
            self.campaigns = new_mcs
            new_mcs = self
        else:
            new_mcs = MultiCampaign(campaigns=new_mcs)

        return new_mcs

    def derive(self, param, arguments, formula, parsers=None, targets=None):
        for mc in self:
            mc.derive(param, arguments, formula, parsers, targets)

    def sort(self, params, parsers=None, targets=None, ret_params=False):
        ret_vals = []
        for mc in self:
            ret_vals.append(mc.sort(params, parsers, targets, ret_params=True))

        x = self.get(params, parsers, targets, use_pd=False)

        lex = np.array([np.array(list(m.values()))[::-1, 0] for m in x]).T

        self.campaigns = [self[i] for i in np.lexsort(lex)]

        if ret_params:
            return ret_vals

    def filter(self, params, values=None, *, min=-np.inf, max=np.inf,
               negative=False, parsers=None, targets=None, inplace=True):
        if inplace:
            mcs = self
        else:
            mcs = self.copy()

        for mc in mcs:
            mc.filter(params, values, min=min, max=max, negative=negative,
                      parsers=parsers, targets=targets)

        mcs.campaigns = [mc for mc in mcs if len(mc) > 0]

        return mcs

    def split(self, params, parsers=None, targets=None, inplace=True):
        split_campaigns = []

        for mc in self:
            split_campaigns.extend(mc.split(params, parsers, targets))

        if inplace:
            mcs = self
        else:
            mcs = MultiCampaign()

        mcs.campaigns = split_campaigns

        return mcs

    def merge(self):
        return MeasurementCampaign().merge(self.campaigns)

    def cluster(self, params, clusters=None, *, n_clusters=None, guess=None,
                parsers=None, targets=None, metric="euclidean",
                show_assignment=False, inplace=True, ret_clusters=False):
        if inplace:
            mcs = self
        else:
            mcs = self.copy()

        campaigns = []

        for mc in self:
            campaigns.append(mc.cluster(
                params, clusters=clusters, n_clusters=n_clusters, guess=guess,
                parsers=parsers, targets=targets, metric=metric,
                show_assignment=show_assignment, inplace=inplace,
                ret_clusters=ret_clusters))

        mcs.campaigns = campaigns

        return mcs

    def average(self, params=None, parsers=None, targets=None,
                keep_params=None, keep_parsers=None, keep_targets=None,
                inplace=True):
        if inplace:
            mcs = self
        else:
            mcs = self.copy()

        campaigns = []

        for mc in self:
            campaigns.append(mc.average(
                params, parsers, targets,
                keep_params, keep_parsers, keep_targets, inplace))

        mcs.campaigns = campaigns

        return mcs

    def sum(self, params=None, parsers=None, targets=None,
            keep_params=None, keep_parsers=None, keep_targets=None,
            inplace=True, analyze=True, **kwargs):
        if inplace:
            mcs = self
        else:
            mcs = self.copy()

        campaigns = []

        for mc in self:
            campaigns.append(mc.sum(
                params, parsers, targets,
                keep_params, keep_parsers, keep_targets,
                inplace, analyze, **kwargs
            ))

        mcs.campaigns = campaigns

        return mcs

    def normalize(
        self, ref=None, normalize_to=1, normalize_with=None, n=None,
        x_param=None, x_parser=None, avg=True, clusters=None,
        show_assignment=False, keep_params=None, keep_parsers=None,
        keep_targets=None, inplace=True,
    ):
        """Normalize lineshape parameters of this MultiCampaign's
        SingleSpectra.

        This method scales the values of lineshape parameters according to
        the provided arguments. The normalization uses the formula
        >>> normalized = normalize_to * original_value / normalize_with

        Parameters
        ----------
        ref : int or str or None, optional
            Identifier of the reference's MeasurementCampaign in this
            MultiCampaign's `campaigns` list. The reference is normalized
            according to `normalize_to` and `normalize_with`. All other
            measurements are normalized with the same factors.

            - If `ref` is None (default), then there is no designated reference and
            all lineshape parameters of all measurements are normalized to 1.
            - If `ref` is an int or str, it is taken as the index or key of the
            reference in `self.campaigns`.
        normalize_to : numpy.ndarray or int or float or None, optional
            Value(s) that this campaign should be normalized to.

            - If `normalize_to` is an int or float, then all values are
              normalized to that value.
            - If `normalize_to` is a numpy.ndarray, then its shape must be
              (number of lineshape parameters, number of single detectors,
              number of DopplerMeasurements) and its elements are applied to
              the corresponding SingleSpectrum's lineshape parameters. The
              order of lineshape parameters is s, w, v2p, valley, counts,
              peak_counts. The order of detector is that in `self.detnames`.
              The order of DopplerMeasurements is that in `self.measurements`.
            - If `normalize_to` is None, then no normalization is applied.

            The default is 1.
        normalize_with : numpy.ndarray or int or float or None, optional
            Value(s) that this campaign should be normalized with.

            - If `normalize_with` is None (default), then it is computed from
              the DopplerMeasurements specified by `n`.
            - If `normalize_with` is a numpy.ndarray, then its shape must be
              (number of lineshape parameters, number of single detectors,
              number of DopplerMeasurements) and its elements are applied to
              the corresponding SingleSpectrum's lineshape parameters. The
              order of lineshape parameters is s, w, v2p, valley, counts,
              peak_counts and their uncertainties. The order of detector is that
              in `self.detnames`. The order of DopplerMeasurements is that in
              `self.measurements`.
        n : int or list of int or numpy.ndarray or range or slice or Ellipsis or None, optional
            Indices of DopplerMeasurements to use for the computation of
            `normalize_with`. Only takes effect if `normalize_with` is None.
            The options below apply to each lineshape parameter and each
            detector separately.

            - If `n` is None (default), then `normalize_with` is equal to the
              original value, meaning the above equation becomes
              `normalized = normalize_to`.
            - If `n` is a positive int, then `normalize_with` is the average of
              the first `n` DopplerMeasurements.
            - If `n` is a negative int, then `normalize_with` is the average of
              the last `n` DopplerMeasurements.
            - If `n` is Ellipsis, then `normalize_with` is the average of
              all DopplerMeasurements.
            - If `n` is a range, slice or list of int, then the elements are
              taken as indices of DopplerMeasurements to compute the average
              with.
            - If `n` is a numpy.ndarray, then it is taken as index array.

            See `x_param` for the option to average and sort DopplerMeasurements
            before applying the indices.
        x_param : str or None
            Parameter to sort and optionally average DopplerMeasurements by.
            Accordingly, `x_params` must be an attribute of DopplerMeasurements
            or key in metadata. If `x_params` is None (default), then the
            DopplerMeasurements are neither sorted nor averaged and `n` refers
            to index in this MeasurementCampaign's `measurements` list. If
            `x_param` is provided, then the DopplerMeasurements are at least
            sorted by that parameter. See `avg` to specify whether to also
            average DopplerMeasurements by `x_param`.
        x_parser : callable, optional
            Function to apply to `x_param` before using it to sort and average
            by. The default is None.
        avg : bool, optional
            Whether to average by `x_param` before sorting and indexing. Only
            applies if `x_params` is provided. The default is True.
        clusters : list, optional
            If provided, use the cluster method before averaging and pass
            `clusters` as `guess`. The default is None (no clustering).
        show_assignment : bool, optinal
            Whether to show cluster assignment. The default is False.
        keep_params : str or list of str, optional
            Metadata to keep when averaging. The value will be the average and
            a corresponding "d{param}" metadata item will be added.
            The default is None.
        keep_parsers : callable or list of callable, optional
            Function to apply to `keep_params` before averaging.
            The default is None.
        keep_targets : callable or list of callable, optional
            Specifies what object `keep_params` is meant to belong to.
            'doppler' for DopplerMeasurements, 'single' for SingleSpectra and
            'coinc' for CoincidenceSpectra'. If None, then the first one of
            these in this order where it is found is used. The default is None.
        inplace : bool, optional
            If True, overwrite lineshape parameters with their normalized
            values. If False, store normalized values in a new
            MeasurementCampaign.

        Returns
        -------
        numpy.ndarray
            Array of values used for `normalize_with` in the top equation. Its
            shape is (number of lineshape parameters, number of single
            detectors, number of DopplerMeasurements). The order of lineshape
            parameters is s, w, v2p, valley, counts, peak_counts and then their
            uncertainties. The order of detector is that in `self.detnames`.
            The order of DopplerMeasurements is that in `self.measurements`.
        MultiCmpaign
            If `inplace` is False, the new MultiCampaign is returned.
        """
        assert all(mc.is_homogeneous() for mc in self)

        kwargs = {
            "n": n, "x_param": x_param, "x_parser": x_parser, "avg": avg,
            "clusters": clusters, "show_assignment": show_assignment,
            "keep_params": keep_params, "keep_parsers": keep_parsers,
            "keep_targets": keep_targets
        }

        if not inplace:
            self = self.copy()

        if ref is not None:
            if isinstance(ref, str):
                for i, mc in enumerate(self):
                    if mc.name == ref:
                        ref = i
                        break

            if normalize_with is None:
                normalize_with = self[ref].normalize(
                    normalize_to=normalize_to, **kwargs
                )

                if normalize_to is None:
                    if not inplace:
                        return normalize_with, self
                    else:
                        return normalize_with
            else:
                self[ref].normalize(normalize_to=normalize_to, **kwargs)

        for i, mc in enumerate(self):
            if i == ref:
                continue
            mc.normalize(
                normalize_to=normalize_to, normalize_with=normalize_with, **kwargs
            )

        if not inplace:
            return normalize_with, self
        else:
            return normalize_to

    def plot(self, x_params=None, y_params=None, z_params=None, *,
             x_parsers=None, y_parsers=None, z_parsers=None,
             x_targets=None, y_targets=None, z_targets=None,
             errorbars=None, scatter=None, filled=None, separate=False,
             show=True, ret_data=False, fig=None, axs=None,
             labels="", labels_is_param=False, detnames_in_labels=None,
             iter_kwargs=None, full_xlabels=False, **kwargs):

        x_params, x_parsers, x_targets = \
            _check_input(x_params, x_parsers, x_targets)

        y_params, y_parsers, y_targets = \
            _check_input(y_params, y_parsers, y_targets)

        z_params, z_parsers, z_targets = \
            _check_input(z_params, z_parsers, z_targets)

        n_x, n_y, n_z = len(x_params), len(y_params), len(z_params)
        n_plots = max(n_x, n_y, n_z)

        if iter_kwargs is None:
            iter_kwargs = {}

        err = "Item {} in `iter_kwargs` has length {}, need {}"
        if len(iter_kwargs) > 0:
            for key, val in iter_kwargs.items():
                assert (a:=len(val)) == (b:=len(self)), err.format(key, a, b)

        if labels == "" or labels is None:
            labels = ["" for _ in self]
        elif labels_is_param:
            labels = [str(mc[0][labels]) for mc in self]

        data = []

        if separate is True:
            fig, axs = plt.subplots(len(self), n_plots)
            labels = ["" for _ in self]

        axes = axs

        for i, mc in enumerate(self):
            if separate is True:
                axs = axes[i]
            fig, axs, d = mc.plot(
                x_params, y_params, z_params,
                x_parsers=x_parsers, y_parsers=y_parsers, z_parsers=z_parsers,
                x_targets=x_targets, y_targets=y_targets, z_targets=z_targets,
                errorbars=errorbars, scatter=scatter, filled=filled,
                show=False, ret_data=True, fig=fig, axs=axs, label=labels[i],
                detnames_in_label=detnames_in_labels, full_xlabels=full_xlabels,
                **{k: v[i] for k, v in iter_kwargs.items()}, **kwargs)

            data.append(d)

        if show:
            for ax in axs:
                ax.grid()
                ax.legend()
            plt.tight_layout()
            plt.show()

        if ret_data:
            return fig, axs, data
        else:
            return fig, axs

    def depth_profiles(self, x_param, y_params,
                       x_parsers=None, y_parsers=None,
                       x_targets=None, y_targets=None,
                       keep_params=None, keep_parsers=None, keep_targets=None,
                       energies=None, show_assignment=False,
                       labels="", labels_is_param=False, detnames_in_labels=None,
                       avg_equalpoints=True, avg_method="avg",
                       single_kwargs=None, coinc_kwargs=None, scatter=None,
                       show=True, ret_data=False, inplace=True,
                       iter_kwargs=None, makhov_params=None, xunit_order=True,
                       full_xlabels=False, **kwargs):

        err = "`x_param` can only be a singular parameter of type str, is {}"
        assert isinstance(x_param, str), err.format(type(x_param))

        if iter_kwargs is None:
            iter_kwargs = {}

        err = "Item {} in `iter_kwargs` has length {}, need {}"
        if len(iter_kwargs) > 0:
            for key, val in iter_kwargs.items():
                assert (a:=len(val)) == (b:=len(self)), err.format(key, a, b)

        fig = None
        data = {}

        if avg_equalpoints:
            self = self.cluster(x_param, parsers=x_parsers, targets=x_targets,
                                guess=energies, show_assignment=show_assignment,
                                inplace=inplace)

            if labels_is_param:
                if keep_params is None:
                    keep_params = labels
                elif isinstance(keep_params, str):
                    keep_params = [keep_params, labels]
                else:
                    keep_params.extend(labels)

            if avg_method == "avg":
                self = self.average(x_param, x_parsers, x_targets,
                                    keep_params, keep_parsers, keep_targets,
                                    inplace=inplace)
            elif avg_method == "sum":
                self = self.sum(x_param, x_parsers, x_targets,
                                keep_params, keep_parsers, keep_targets,
                                inplace=inplace, single_kwargs=single_kwargs,
                                coinc_kwargs=coinc_kwargs)
            else:
                raise ValueError(f"Unknown averaging method {avg_method}")

        if labels == "":
            labels = [str(i) for i in range(len(self))]
        elif labels_is_param:
            labels = [str(mc[0][labels]) for mc in self]

        err = ("Number of labels does not match number of campaigns "
               "(len(labels)={} and len(campaigns)={})")
        assert (a := len(labels)) == (b := len(self)), err.format(a, b)

        if makhov_params is not None:
            rho, A, n = makhov_params
            x2_param = "MeanImplantationDepth"
            depth = lambda e: 10*A * (e/1000)**n / rho
            energy = lambda z: 1000 * (z * rho / (10 * A)) ** (1/n)
            self.derive(x2_param, x_param, depth)
            full_xlabels = True

            if xunit_order is True:
                x_param, x2_param =  x2_param, x_param
                conv_f, _ = energy, depth
            elif xunit_order is False:
                conv_f, _ = depth, energy
            elif xunit_order is None:
                x_param =  x2_param
            else:
                raise ValueError(xunit_order)

        for i, mc in enumerate(self):
            fig, axs, d = mc.plot(
                x_param, y_params, x_parsers=x_parsers, y_parsers=y_parsers,
                x_targets=x_targets, y_targets=y_targets, scatter=scatter,
                show=False, fig=fig, ret_data=True, label=labels[i],
                detnames_in_labels=detnames_in_labels, full_xlabels=full_xlabels,
                **{k: v[i] for k, v in iter_kwargs.items()}, **kwargs)

            data[labels[i]] = d

        if makhov_params is not None and xunit_order is not None:
            for ax in axs:
                x_ticks_bottom = ax.get_xticks()

                x_ticks_top = conv_f(x_ticks_bottom)

                mask = np.isfinite(x_ticks_top)
                x_ticks_bottom = x_ticks_bottom[mask]
                x_ticks_top = x_ticks_top[mask]

                ax_top = ax.secondary_xaxis("top")
                ax_top.set_xticks(x_ticks_bottom)
                ax_top.set_xticklabels([f"{val:.0f}" for val in x_ticks_top])

                ax_top.set_xlabel(x2_param)

        if show:
            for ax in axs:
                ax.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        if ret_data:
            return fig, axs, data
        else:
            return fig, axs
