from typing import List, Dict, Tuple

import numpy as np
import lmfit


def parse_parameter_tuple(
    params: tuple | list | float | int,
    *,
    vary: bool = True,
    min: float | int = -np.inf,
    max: float | int = np.inf,
) -> list:
    """Parse parameter tuple.

    Converts a tuple of possibly missing parameter attributes into a list of
    four parameter attributes: value, vary, min and max. The omission rules are

    - Max can be omitted. Then it defaults to `max`.
    - Min and max can be omitted. Then they default to `min` and `max`.
    - Vary, min and max can be omitted. Then they default to `vary`, `min` and
      `max`.
    - Vary can be omitted. Then it defaults to `vary`.

    Parameters
    ----------
    params : tuple or list or float or int
        The possibly incomplete parameter attributes.
    min : float or int, optional
        Default minimum if not in `params`. The default is -numpy.inf.
    max : float or int, optional
        Default maximum if not in `params`. The default is numpy.inf.
    vary : float or int, optional
        Default minimum if not in `params`. The default is -numpy.inf.

    Returns
    -------
    list of [float or int or None, bool, float, float]
    """
    assert isinstance(params, (tuple, list, float, int))

    args = [None, vary, min, max]

    if isinstance(params, (float, int)):
        args[0] = params
        return args

    params = list(params)

    n = len(params)

    if n == 1:
        args[0] = params[0]
        return args

    if not isinstance(params[1], bool):
        params.insert(1, True)
        n += 1

    for i in range(n):
        args[i] = params[i]

    return args


class LifetimeComponent:
    """Class representing a lifetime component.

    A lifetime component consists of
    - a lifetime
    - an intensity.

    Parameters
    ----------
    lifetime : float or int or lmfit.Parameter or sequence, optional
        Lifetime. Gets converted to an lmfit.Parameter using
        `parse_parameter_tuple` and stored under the `lifetime` attribute.
        The default value is 100.
    intensity : float or int or lmfit.Parameter or sequence, optional
        Intensity. Gets converted to an lmfit.Parameter using
        `parse_parameter_tuple` and stored under the `lifetime` attribute.
        The default value is 0.5.
    """
    def __init__(
        self,
        lifetime: float | int | lmfit.Parameter | Tuple | List = 100,
        intensity: float | int | lmfit.Parameter | Tuple | List = 0.5,
    ) -> None:

        self.lifetime = (
            lifetime if isinstance(lifetime, lmfit.Parameter) else
            lmfit.Parameter(
                "lifetime",
                *parse_parameter_tuple(lifetime, min=0)
            )
        )
        self.intensity = (
            intensity if isinstance(intensity, lmfit.Parameter) else
            lmfit.Parameter(
                "intensity",
                *parse_parameter_tuple(intensity, min=0)
            )
        )


class LifetimeComponents:
    """Class representing a collection of `LifetimeComponent`s.

    Parameters
    ----------
    components : int or LifetimeComponent or List of LifetimeComponent or None, optional
        Lifetime components. Get stored under the `components` attribute which
        is a list of `LifetimeComponent`s.

        - If None, `self.components` remains empty.
        - If an int, specifies with how many default `LifetimeComponent`s
          `self.components` is filled.
        - If a `LifetimeComponent`, gets put into `self.components`.
        - If a list of `LifetimeComponent`s, `self.components` is a reference
          to that list.

        The default is None.
    """
    def __init__(
        self,
        components: None
            | int
            | LifetimeComponent
            | List[LifetimeComponent] = None,
    ) -> None:

        self.components: List[LifetimeComponent] = (
            [] if components is None else
            [LifetimeComponent() for _ in range(components)]
                if isinstance(components, int) else
            [components] if isinstance(components, LifetimeComponent) else
            components
        )

        assert all(isinstance(x, LifetimeComponent) for x in self.components)

    def __iter__(self):
        return (x for x in self.components)

    def __len__(self):
        return len(self.components)

    def __getitem__(self, i):
        return self.components[i]


class ResolutionComponent:
    """Class representing a resolution component.

    A resolution component (modelled by a Gaussian) consists of
    - a standard deviation (Gaussian sigma)
    - an intensity
    - an offset in time.

    Parameters
    ----------
    sigma : float or int or lmfit.Parameter or sequence, optional
        Standard deviation. Gets converted to an lmfit.Parameter using
        `parse_parameter_tuple` and stored under the `sigma` attribute.
        The default value is 100.
    intensity : float or int or lmfit.Parameter or sequence, optional
        Intensity. Gets converted to an lmfit.Parameter using
        `parse_parameter_tuple` and stored under the `lifetime` attribute.
        The default value is 0.5.
    t0 : float or int or lmfit.Parameter or sequence, optional
        Time offset. Gets converted to an lmfit.Parameter using
        `parse_parameter_tuple` and stored under the `t0` attribute.
        The default value is 0.
    """
    def __init__(
        self,
        sigma: float | int | lmfit.Parameter | Tuple | List = 100,
        intensity: float | int | lmfit.Parameter | Tuple | List = 0.5,
        t0: float | int | lmfit.Parameter | Tuple | List = 0,
    ) -> None:

        self.sigma = (
            sigma if isinstance(sigma, lmfit.Parameter) else
            lmfit.Parameter(
                "sigma",
                *parse_parameter_tuple(sigma, min=0)
            )
        )
        self.intensity = (
            intensity if isinstance(intensity, lmfit.Parameter) else
            lmfit.Parameter(
                "intensity",
                *parse_parameter_tuple(intensity, min=0)
            )
        )
        self.t0 = (
            t0 if isinstance(t0, lmfit.Parameter) else
            lmfit.Parameter(
                "t0",
                *parse_parameter_tuple(t0)
            )
        )


class ResolutionComponents:
    """Class representing a collection of `ResolutionComponent`s.

    Parameters
    ----------
    components : int or ResolutionComponent or List of ResolutionComponent or None, optional
        Resolution components. Get stored under the `components` attribute which
        is a list of `ResolutionComponent`s.

        - If None, `self.components` remains empty.
        - If an int, specifies with how many default `ResolutionComponent`s
          `self.components` is filled.
        - If a `ResolutionComponent`, gets put into `self.components`.
        - If a list of `ResolutionComponent`s, `self.components` is a reference
          to that list.

        The default is None.
    """
    def __init__(
        self,
        components: None
            | int
            | ResolutionComponent
            | List[ResolutionComponent] = None,
    ) -> None:

        self.components: List[ResolutionComponent] = (
            [] if components is None else
            [ResolutionComponent() for _ in range(components)]
                if isinstance(components, int) else
            [components] if isinstance(components, ResolutionComponent) else
            components
        )

        assert all(isinstance(x, ResolutionComponent) for x in self.components)

    def __iter__(self):
        return (x for x in self.components)

    def __len__(self):
        return len(self.components)

    def __getitem__(self, i):
        return self.components[i]


class BackgroundComponent:
    """Class representing a background component.

    A background component consists of a single value.

    Parameters
    ----------
    value : float or int or lmfit.Parameter or Tuple or List, optional
        Value of the background. Gets converted to an lmfit.Parameter using
        `parse_parameter_tuple` and stored under the `value` attribute.
        The default value is 0.
    """
    def __init__(
        self,
        value: float | int | lmfit.Parameter | Tuple | List = 0,
    ) -> None:

        self.value = (
            value if isinstance(value, lmfit.Parameter) else
            lmfit.Parameter(
                "background",
                *parse_parameter_tuple(value)
            )
        )


class ScaleComponent:
    """Class representing a scale component.

    A scale component consists of a single value.

    Parameters
    ----------
    value : float or int or lmfit.Parameter or Tuple or List, optional
        Value of the scale. Gets converted to an lmfit.Parameter using
        `parse_parameter_tuple` and stored under the `value` attribute.
        The default value is 1.
    """
    def __init__(
        self,
        value: float | int | lmfit.Parameter | Tuple | List = 1,
    ) -> None:

        self.value = (
            value if isinstance(value, lmfit.Parameter) else
            lmfit.Parameter(
                "N",
                *parse_parameter_tuple(value)
            )
        )


class ShiftComponent:
    """Class representing a shift component.

    A shift component consists of a single value.

    Parameters
    ----------
    value : float or int or lmfit.Parameter or Tuple or List, optional
        Value of the shift. Gets converted to an lmfit.Parameter using
        `parse_parameter_tuple` and stored under the `value` attribute.
        The default value is 0.
    """
    def __init__(
        self,
        value: float | int | lmfit.Parameter | Tuple | List = 0,
    ) -> None:

        self.value = (
            value if isinstance(value, lmfit.Parameter) else
            lmfit.Parameter(
                "t0",
                *parse_parameter_tuple(value)
            )
        )


class LifetimeModel:
    """Class representing the theoretical model to describe a real lifetime
    spectrum.

    A lifetime model consists of

    - at least one `LifetimeComponent`
    - at least one `ResolutionComponent`
    - an optional `BackgroundComponent`
    - an optional `ScaleComponent`
    - an optional `ShiftComponent`

    Parameters
    ----------
    lifetime_components : int or LifetimeComponent or list of LifetimeComponent or LifetimeComponents, optional
        Lifetime components of this model. Get stored under the
        `lifetime_components` attribute which is of type `LifetimeComponents`.

        - If an int, specifies with how many default `LifetimeComponent`s
          `lifetime_components` is filled.
        - If a `LifetimeComponent`, gets put into `lifetime_components`.
        - If a list of `LifetimeComponent`s, gets converted to a
          `LifetimeComponents` object.
        - If `LifetimeComponents`, `lifetime_components` is a reference
          to object.

        The default is 1.
    resolution_components : int or ResolutionComponent or list of ResolutionComponent or ResolutionComponents, optional
        Resolution components of this model. Get stored under the
        `resolution_components` attribute which is of type
        `ResolutionComponents`.

        - If an int, specifies with how many default `ResolutionComponent`s
          `resolution_components` is filled.
          - If a `ResolutionComponent`, gets put into `resolution_components`.
        - If a list of `ResolutionComponent`s, gets converted to a
        `ResolutionComponents` object.
        - If `ResolutionComponents`, `resolution_components` is a reference
          to object.

        The default is 1.
    background_component : BackgroundComponent or None, optional
        Background component of this model. Gets stored under the
        `background_component` attribute. The default is None.
    scale_component : ScaleComponent or None, optional
        Scale component of this model. gets stored under the
        `scale_component` attribute. The default is None.
    shift_component : ShiftComponent or None, optional
        Shift component of this model. Gets stored under the
        `shift_component` attribute. The default is None.
    """
    def __init__(
        self,
        lifetime_components: int
            | LifetimeComponent
            | List[LifetimeComponent]
            | LifetimeComponents = 1,
        resolution_components: int
            | ResolutionComponent
            | List[ResolutionComponent]
            | ResolutionComponents = 1,
        background_component: None
            | BackgroundComponent = None,
        scale_component: None
            | ScaleComponent = None,
        shift_component: None
            | ShiftComponent = None,
    ) -> None:

        self.lifetime_components = (
            lifetime_components
                if isinstance(lifetime_components, LifetimeComponents) else
            LifetimeComponents(lifetime_components)
        )

        self.resolution_components = (
            resolution_components
                if isinstance(resolution_components, ResolutionComponents) else
            ResolutionComponents(resolution_components)
        )

        self.background_component = background_component
        self.scale_component = scale_component
        self.shift_component = shift_component


def parse_model(model: Dict[str, Tuple] | None) -> LifetimeModel | None:
    """Convert a dictionary of parameter attributes to a LifetimeModel.

    Parameters
    ----------
    model : dict or None
        Must be of the form `name: attributes`, where `name` is the parameter
        name and `attributes` is a tuple of the four parameter attributes
        `value` (float), `vary` (bool), `min` (float) and `max` (float).

        The following names are recognized:

        - `lifetime_{i}` for the lifetime value of `LifetimeComponent` i
        - `intensity_{i}` for the intensity of `LifetimeComponent` i
        - `res_sigma_{i}` for the standard deviation of `ResolutionComponent` i
        - `res_intensity_{i}` for the intensity of `ResolutionComponent` i
        - `res_t0_{i}` for the time offset of `ResolutionComponent` i
        - `background` for the value of a `BackgroundComponent`
        - `N` for the value of a `ScaleComponent`
        - `t0` for the value of a `ShiftComponent`.

        Attribute tuples can be abbreviated by omitting elements accoring to the
        following rules. The fallback values are defined in the corresponding
        component's class.

        - `max` can be omitted. E.g. `(1, True, 0)` defines\\
           `value=1, vary=True, min=0, max=<default>`.
        - `min` and `max` can be omitted. E.g. `(1, True)` defines\\
          `value=1, vary=True, min=<default>, max=<default>`.
        - `vary`, `min` and `max` can be omitted. In that case `attributes` does
          not need to be a tuple but can be a single float. E.g. `1` defines\\
          `value=1, vary=<default>, min=<default>, max=<default>`.
        - `vary` can be omitted. E.g. `(1, 0, 2)` defines\\
          `value=1, vary=<default>, min=0, max=2`.
        - `vary` and `max` can be omitted. E.g. `(1, 0)` defines\\
          `value=1, vary=<default>, min=0, max=<default>`.

    Returns
    -------
    LifetimeModel or None
        Assembled `LifetimeModel` or None if the input is None.
    """
    if model is None or len(model) == 0:
        return None

    model = {key.lower(): val for key, val in model.items()}

    lifetime_components = []
    resolution_components = []
    background_component = None
    scale_component = None
    shift_component = None

    if "background" in model:
        background_component = BackgroundComponent(model.pop("background"))

    if "n" in model:
        scale_component = ScaleComponent(model.pop("n"))

    if "t0" in model:
        shift_component = ShiftComponent(model.pop("t0"))

    resolution_params = [key for key in model if key.startswith("res")]
    lifetime_params = [
        key for key in model
        if key.startswith("lif") or key.startswith("int")
    ]

    r_indices = {int(p.split("_")[2]) for p in resolution_params}
    l_indices = {int(p.split("_")[1]) for p in lifetime_params}

    n_r = 0 if len(r_indices) == 0 else max(r_indices)
    n_l = 0 if len(l_indices) == 0 else max(l_indices)

    for j in range(1, n_r+1):
        resolution_components.append(
            ResolutionComponent(
                sigma=parse_parameter_tuple(model.pop(f"res_sigma_{j}", 100)),
                intensity=parse_parameter_tuple(model.pop(f"res_intensity_{j}", 0.5)),
                t0=parse_parameter_tuple(model.pop(f"res_t0_{j}", 0))
            )
        )

    for i in range(1, n_l+1):
        lifetime_components.append(
            LifetimeComponent(
                lifetime=parse_parameter_tuple(model.pop(f"lifetime_{i}", 100)),
                intensity=parse_parameter_tuple(model.pop(f"intensity_{i}", 0.5)),
            )
        )

    return LifetimeModel(
        lifetime_components=lifetime_components,
        resolution_components=resolution_components,
        background_component=background_component,
        scale_component=scale_component,
        shift_component=shift_component,
    )


def dump_model(model: LifetimeModel) -> Dict:
    """Convert a LifetimeModel to a dictionary of parameter attributes.

    Parameters
    ----------
    model : LifetimeModel
        The `LifetimeModel` to convert.

    Returns
    -------
    dict
        The dictionary of the disassembled `LifetimeModel`.
    """
    out = {}

    for i in range(len(model.lifetime_components)):
        p = model.lifetime_components[i].lifetime
        out[f"lifetime_{i+1}"] = (p.value, p.vary, p.min, p.max)
        p = model.lifetime_components[i].intensity
        out[f"intensity_{i+1}"] = (p.value, p.vary, p.min, p.max)

    for j in range(len(model.resolution_components)):
        p = model.resolution_components[j].sigma
        out[f"res_sigma_{j+1}"] = (p.value, p.vary, p.min, p.max)
        p = model.resolution_components[j].intensity
        out[f"res_intensity_{j+1}"] = (p.value, p.vary, p.min, p.max)
        p = model.resolution_components[j].t0
        out[f"res_t0_{j+1}"] = (p.value, p.vary, p.min, p.max)

    if model.background_component is not None:
        p = model.background_component.value
        out["background"] = (p.value, p.vary, p.min, p.max)

    if model.scale_component is not None:
        p = model.scale_component.value
        out["N"] = (p.value, p.vary, p.min, p.max)

    if model.shift_component is not None:
        p = model.shift_component.value
        out["t0"] = (p.value, p.vary, p.min, p.max)

    return out


def combine_models(
    model1: LifetimeModel | Dict | None,
    model2: LifetimeModel | Dict | None
) -> LifetimeModel | None:
    """Merge two LifetimeModels into one.

    Parameters
    ----------
    model1 : LifetimeModel or dict or None
        The first `LifetimeModel`. Can be a dictionary of parameter attributes,
        then it is assembled before it is merged. Can be None, then the two
        models are not merged.
    model2 : LifetimeModel or dict or None
        The second `LifetimeModel`. Can be a dictionary of parameter attributes,
        then it is assembled before it is merged. Can be None, then the two
        models are not merged.

    Returns
    -------
    LifetimeModel or None
        The merged LifetimeModel or None if both inputs are None.
    """
    if model1 is None and model2 is None:
        return None
    elif model1 is None:
        return parse_model(model2) if not isinstance(model2, LifetimeModel) else model2
    elif model2 is None:
        return parse_model(model1) if not isinstance(model1, LifetimeModel) else model1

    model1_dict = dump_model(model1) if isinstance(model1, LifetimeModel) else model1
    model2_dict = dump_model(model2) if isinstance(model2, LifetimeModel) else model2

    return parse_model(model1_dict | model2_dict)


def pick_model(
    models: dict | LifetimeModel | None,
    keys: list[str] | None,
    metadata: dict[str, str],
    kind: str,
    name: str | None = None,
    debug: bool = False,
) -> dict | LifetimeModel | None:
    if not isinstance(models, dict) or keys is None or not any(key.startswith("Measurement") for key in models):
        return models

    params = {k: str(metadata[k]) for k in keys}

    if debug:
        print(f"Picking a {kind} model for measurement with parameters {params}...")

    for name, model in models.items():
        metadata = model["Metadata"]
        model_params = {k: str(metadata[k]) for k in keys}
        if params == model_params:
            if debug:
                print(f"Picking {name}")
            return model

    err = (
        f"No provided {kind} model matches this "
        "LifetimeMeasurement's parameters:"
        f"\nMeasurement: {name}"
        f"\nParameters: {params}"
    )
    raise ValueError(err)

