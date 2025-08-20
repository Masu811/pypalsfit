import numpy as np
import typing
import time
import warnings


def gauss(x, mu, sig):
    sig2 = sig**2
    return 1 / np.sqrt(2 * np.pi * sig2) * np.exp(-0.5 / sig2 * (x - mu)**2)


def may_be_nan(x):
    return x if x is not None else np.nan


class Spectrum:
    """Wrapper for np.ndarrays that handles efficient memory usage."""

    def __init__(self, arr):
        self.arr = np.array(arr, dtype=min_dtype(np.max(arr)))

    def __add__(self, other):
        """Handle addition of Spectrum with type promotion if necessary."""
        if isinstance(other, Spectrum):
            # Addition with another Spectrum: other shall be the array
            other = other.arr
        elif np.isscalar(other):
            # Addition with a scalar: other can remain as is
            pass
        elif isinstance(other, np.ndarray):
            # Addition with a np.ndarray: other can remain as is
            pass
        else:
            raise ValueError("Can only add Spectrum, np.ndarray or scalar "
                             f"to Spectrum, not {type(other)}")

        # The statement c = a + b => max(c) = max(a) + max(b) is not generally
        # true because the maxima of a and b could be at different indices.
        # But since we are working with spectra whose maxima should be roughly
        # at the same position, we do it here because it is convenient

        max_self = self.arr.max().astype(np.uint64)
        max_other = np.max(other).astype(np.uint64)

        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try:
                max = max_self + max_other
            except RuntimeWarning:
                raise OverflowError("Overflow encountered in Spectrum addition")

        result_arr = np.zeros(self.arr.shape, dtype=min_dtype(max))

        result_arr += self.arr
        result_arr += other

        return result_arr

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Handle subtraction of Spectrum with type promotion if necessary."""
        if isinstance(other, Spectrum):
            # Subtraction with another Spectrum: other shall be the array
            other = other.arr
        elif np.isscalar(other):
            # Subtraction with a scalar: other can remain as is
            pass
        elif isinstance(other, np.ndarray):
            # Subtraction with a np.ndarray: other can remain as is
            pass
        else:
            raise ValueError("Can only add Spectrum, np.ndarray or scalar "
                             f"to Spectrum, not {type(other)}")

        result_arr = np.where(other > self.arr, 0, self.arr - other)

        return result_arr.astype(min_dtype(result_arr.max()))

    def __mul__(self, other):
        """Handle multiplication of Spectrum with type promotion if necessary."""
        if np.isscalar(other):
            pass
        else:
            raise ValueError("Can only perform scalar multiplication of Spectrum")

        max_self = self.arr.max().astype(np.uint64)

        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try:
                max = max_self * other
            except RuntimeWarning:
                raise OverflowError("Overflow encountered in Spectrum multiplication")

        result_arr = self.arr.copy().astype(min_dtype(max))

        result_arr *= other

        return result_arr

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Handle division of Spectrum with type promotion if necessary."""
        if np.isscalar(other):
            pass
        else:
            raise ValueError("Can only perform scalar division of Spectrum")

        result_arr = self.arr / other

        return result_arr.astype(min_dtype(result_arr.max()))

    def __getattr__(self, attr):
        """Make all np.ndarray attributes and methods available."""
        return getattr(self.arr, attr)

    def __getstate__(self):
        """Define how to pickle the object."""
        return {'arr': self.arr}

    def __setstate__(self, state):
        """Define how to unpickle the object."""
        self.__dict__.update(state)

    # Below are dunder methods that cannot be intercepted by __getattr__

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return self.arr[idx]

    def __setitem__(self, idx, value):
        self.arr[idx] = value

    def __iter__(self):
        return iter(self.arr)

    def __eq__(self, other):
        return self.arr == other

    def __neg__(self):
        return -self.arr

    def __repr__(self):
        return f"<Spectrum, {self.arr}, {self.dtype}>"

    def promote(self, dtype):
        self.arr = self.arr.astype(dtype)


class Filter:
    """Class for handling filtering of MeasurementCampaign.measurements.

    Parameters
    ----------
    parent : MeasurementCampaign or DopplerMeasurement
        Object that this filter should act upon.

    Attributes
    ----------
    parent : MeasurementCampaign or DopplerMeasurement
        Object that this filter acts upon.
    params : set of str
        Set of parameter names for which filter conditions were set.
    dynamic:
        For each parameter a Parameter instance is added as an attribute
        accessible via the parameter name.
    """

    def __init__(self, parent):
        self.parent = parent
        self.params = set()

    def __getitem__(self, key):
        return getattr(self, key)

    def __getattr__(self, name):
        setattr(self, name, Parameter(name))
        self.params.add(name)
        return getattr(self, name)

    def __iter__(self):
        return (getattr(self, name) for name in self.params)

    def __len__(self):
        return len(self.params)

    def __repr__(self):
        return f"{[getattr(self, name) for name in self.params]}"

    def copy(self):
        f = Filter(self.parent)
        f.params = self.params.copy()
        return f

    def add(self, param_name, values=None, *, min=-np.inf, max=np.inf,
            negative=False, parser=None, target=None):
        """Add a new parameter to the filter.

        Parameters
        ----------
        param_name : str
            Parameter to filter.
        values : any type or list of any type, optional
            Allowed values. All parameters whose values don't fit, will be
            removed (opposite if 'negative' is true). The default is None.
        min : any type, optional
            Minimum value of 'param' that the target must have.
            The default is -np.inf.
        max : any type, optional
            Maximum value of 'param' that the target must have.
            The default is np.inf.
        negative : bool
            Flag that indicates whether matches (True) or non-matches (False)
            are to be filtered out. The default is False
        parsers : callable or list of callable, optional
            Functions to apply to parameters when filtering. If 'params' is a
            list, 'parsers' can be too, then parsers[i] is applied to
            params[i]. 'parsers' can be None if no function is to be applied.
            If 'parsers' is a single value, it is applied to all parameters.
            The default is None.
        targets : str or list of str, optional
            Specifies if 'params' is meant as attribute of DopplerMeasurements
            (then 'targets' must contain "doppler"), SingleSpectra (then
            'targets' must contain "single") or CoincidenceSpectra (then
            'targets' must contain "coinc"). If 'params' is a list, 'targets'
            can be too, then targets[i] is applied to params[i]. 'targets' can
            be None, then 'params' is taken from the first class in the order
            above where it is found. If 'targets' is a single value, it is
            applied to all parameters. The default is None.
        """
        parameter = Parameter(param_name)
        parameter.set(values, min=min, max=max,
                      target=target, negative=negative, parser=parser)
        setattr(self, param_name, parameter)
        self.params.add(param_name)

    def remove(self, param_name):
        """Remove an existing parameter from the filter.

        Parameters
        ----------
        param_name : str
            Name of the parameter to remove.
        """
        delattr(self, param_name)
        self.params.remove(param_name)

    def clear(self):
        """Remove all parameters from self."""
        params = self.params.copy()
        for param in params:
            self.remove(param)

    def apply(self, inplace=True):
        """Apply parameter attributes to corresponding MeasurementCampaign.

        Parameters
        ----------
        inplace : bool
            If inplace is False, return a copy of this instance on which
            the filter was applied.
            If inplace is True, nothing is returned and the filter applied to
            this instance.
            The default is True.

        Returns
        -------
        parent : MeasurementCampaign or DopplerMeasurement
            If inplace is False, return a copy of this instance on which
            the filter was applied.
            If inplace is True, nothing is returned and the filter applied to
            this instance.
        """
        from .measurement import MeasurementCampaign, LifetimeMeasurement

        if inplace:
            parent = self.parent
        else:
            parent = self.parent.copy()

        if isinstance(parent, MeasurementCampaign):
            meas = parent.measurements

            for param in self:
                if param.target is None or "doppler" in param.target.lower():
                    match = param.match(meas)
                    meas[:] = list(np.array(meas)[match])
                if (
                    param.values == "" and
                    (param.target is None or "doppler" in param.target.lower())
                ):
                    continue
                for m in meas:
                    s = m.spectra
                    if param.target is None or "single" in param.target.lower():
                        match = param.match(m.spectra)
                        m.spectra[:] = list(np.array(s)[match])

        elif isinstance(parent, LifetimeMeasurement):
            s = parent.spectra

            for param in self:
                if param.target is None or "single" in param.target.lower():
                    match = param.match(parent.spectra)
                    parent.spectra[:] = list(np.array(s)[match])

        return parent


class Parameter:
    """Class representing a parameter filter condition.

    Parameters
    ----------
    name : str
        Attribute name or metadata key that this instance represents.

    Attributes
    ----------
    name : str
        Attribute name or metadata key that this instance represents.
    values : any type or list of any type, optional
        Allowed value(s). The default is None.
    min : any type, optional
        Minimum value of parameter that the target must have.
        The default is -np.inf.
    max : any type, optional
        Maximum value of parameter that the target must have.
        The default is np.inf.
    target : str, optional
        Specifies if the parameter is meant to apply to
        DopplerMeasurements (then target must contain 'doppler'),
        SingleSpectra (then target must contain 'single') or
        CoincidenceSpectra (then target must contain 'coinc').
        If target is None, the parameter is filtered on all three
        of those. The default is None.
    negative : bool
        Flag that indicates whether matches (True) or non-matches (False)
        are to be filtered out. The default is False
    parser : callable
        Function to parse parameter into a comparable value.
        The default is None.
    """

    def __init__(self, name):
        self.name = name
        self.values = None
        self.min = None
        self.max = None
        self.target = None
        self.negative = False
        self.parser = None

    def __repr__(self):
        return f"{self.__dict__}".replace("'", "")

    def copy(self):
        p = Parameter(self.name)
        for attr, value in self.__dict__.items():
            setattr(p, attr, value)
        return p

    def set(self, values=None, *, min=-np.inf, max=np.inf,
            negative=False, parser=None, target=None):
        """Change attributes of parameter.

        Parameters
        ----------
        values : any type or list of any type, optional
            Allowed values. All parameters whose values don't fit, will be
            removed (opposite if 'negative' is true). The default is None.
        min : any type, optional
            Minimum value of 'param' that the target must have.
            The default is -np.inf.
        max : any type, optional
            Maximum value of 'param' that the target must have.
            The default is np.inf.
        negative : bool
            Flag that indicates whether matches (True) or non-matches (False)
            are to be filtered out. The default is False
        parsers : callable or list of callable, optional
            Functions to apply to parameters when filtering. If 'params' is a
            list, 'parsers' can be too, then parsers[i] is applied to
            params[i]. 'parsers' can be None if no function is to be applied.
            If 'parsers' is a single value, it is applied to all parameters.
            The default is None.
        targets : str or list of str, optional
            Specifies if 'params' is meant as attribute of DopplerMeasurements
            (then 'targets' must contain "doppler"), SingleSpectra (then
            'targets' must contain "single") or CoincidenceSpectra (then
            'targets' must contain "coinc"). If 'params' is a list, 'targets'
            can be too, then targets[i] is applied to params[i]. 'targets' can
            be None, then 'params' is taken from the first class in the order
            above where it is found. If 'targets' is a single value, it is
            applied to all parameters. The default is None.
        """
        if values is not None and (min != -np.inf or max != np.inf):
            values = None
        self.values = values
        self.min = min
        self.max = max
        self.target = target
        self.negative = negative
        self.parser = parser
        self.check_eq = True if values is not None else False

    def match(self, ref):
        """Check a list for filter conditions.

        Parameters
        ----------
        ref : (list of) DopplerMeasurement or (list of) SingleSpectra
              or (list of) CoincidenceSpectra
            Object to be checked for the conditions. Can be a list of
            objects, then its elements are checked.

        Returns
        -------
        match : (list of) bool
            (List of) flags indicating if the object(s) match the
            condition(s).
        """
        if isinstance(ref, (list, tuple, np.ndarray)):
            if self.name == "":
                parser = self.parser or (lambda x: x)
                if self.check_eq:
                    if isinstance(self.values, (list, tuple, np.ndarray)):
                        if not self.negative:
                            return [(parser(i) in self.values)
                                    for i in range(len(ref))]
                        else:
                            return [(parser(i) not in self.values)
                                    for i in range(len(ref))]
                    else:
                        if not self.negative:
                            return [(parser(i) == self.values)
                                    for i in range(len(ref))]
                        else:
                            return [(parser(i) != self.values)
                                    for i in range(len(ref))]
                else:
                    minidx = max(self.min, 0)
                    maxidx = min(self.max, len(ref))
                    if not self.negative:
                        return [(minidx <= parser(i) and parser(i) <= maxidx)
                                for i in range(len(ref))]
                    else:
                        return [(minidx > parser(i) or parser(i) > maxidx)
                                for i in range(len(ref))]
            else:
                return [self.match(i) for i in ref]

        try:
            parser = self.parser or (lambda x: x)
            if self.check_eq:
                if isinstance(self.values, (list, tuple, np.ndarray)):
                    match = False
                    for val in self.values:
                        match |= (val == parser(ref[self.name]))
                    if not self.negative:
                        return match
                    else:
                        return not match
                else:
                    if not self.negative:
                        return self.values == parser(ref[self.name])
                    else:
                        return self.values != parser(ref[self.name])
            else:
                x = parser(ref[self.name])
                if not self.negative:
                    return (self.min <= x and x <= self.max)
                else:
                    return (self.min > x or x > self.max)
        except (KeyError, AttributeError):
            return True


def min_dtype(val):
    bits = np.ceil(np.log2(val or 1))

    if bits <= 8:
        dtype = np.uint8
    elif bits <= 16:
        dtype = np.uint16
    elif bits <= 32:
        dtype = np.uint32
    else:
        dtype = np.uint64

    return dtype


def isnan(x):
    return np.vectorize(lambda x: str(x) == "nan", otypes=[bool])(x)


def isfinite(x):
    return np.vectorize(lambda x: str(x) != "nan", otypes=[bool])(x)


def _check_input(params, parsers=None, targets=None, dim=""):
    """Check input for the right dtype and format it conveniently.

    This method is for internal use only.
    """
    def check_input(input, length, name,
                    dtype=str, default=None, no_none=False):
        err = f"{dim}{name} has incompatible type {type(input)}"
        if no_none:
            assert isinstance(input, (dtype, list)), err
        else:
            assert input is None or isinstance(input, (dtype, list)), err

        if input is None:
            return [default for _ in range(length)]

        if isinstance(input, dtype):
            if no_none:
                return [input]
            else:
                return [input for _ in range(length)]

        else:
            err = f"Number of {dim}{name} does not equal number of params"
            assert len(input) == length, err

            for i in input:
                err = (f"{dim}{name} contain element of incompatible type "
                       f"{type(i)}")
                if no_none:
                    assert isinstance(i, dtype), err
                else:
                    assert i is None or isinstance(i, dtype), err

            return [i if i is not None else default for i in input]

    if params is None:
        return [], [], []

    params = check_input(params, len(params), "params", no_none=True)
    parsers = check_input(parsers, len(params), "parsers",
                          typing.Callable, lambda x: x)
    targets = check_input(targets, len(params), "targets")

    return params, parsers, targets


def _split_params(x, params, parsers, i):
    """Return params only belonging to level.

    This method is for internal use only.
    """
    params = np.array(params)
    parsers = np.array(parsers)
    mask = np.any(isfinite(x[:, i]), axis=(1, 2))
    return list(params[mask]), list(parsers[mask]), mask


def progress(done, total, start, width=20, info_text="Progress", block="="):
    fraction = done / total
    left = int(width * fraction)
    right = int(width - left)
    bar = "[" + block * left + " " * right + "]"

    elapsed = time.time() - start
    eta = elapsed / fraction - elapsed if fraction > 0 else 0
    elapsed = time.strftime('%M:%S', time.gmtime(elapsed))
    eta = time.strftime('%M:%S', time.gmtime(eta))

    progress_bar = " ".join([f"\r{info_text}: {bar}",
                             f"{done:.0f}/{total:.0f} ({fraction*100:.0f}%)",
                             f"Elapsed: {elapsed} ETA: {eta}"])

    print(progress_bar, end="", flush=True)

    if done == total:
        print()
