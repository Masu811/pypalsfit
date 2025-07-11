from typing import List, Dict, Tuple

import numpy as np
import lmfit


def gauss(x, mu, sig):
    sig2 = sig**2
    return 1 / np.sqrt(2 * np.pi * sig2) * np.exp(-0.5 / sig2 * (x - mu)**2)


def parse_parameter_tuple(
    params: tuple | list | float | int,
    *,
    min=-np.inf,
    max=np.inf,
    vary=True,
) -> list:
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
        self.__name__ = "Lifetime Component"

    def __call__(self, t, intensity, lifetime, t0):
        return np.where(t < t0, 0, intensity / lifetime * np.exp(-(t - t0) / lifetime))


class LifetimeComponents:
    def __init__(
        self,
        components: None | int | LifetimeComponent | List[LifetimeComponent] = None,
        vary: bool = True,
    ) -> None:

        self.vary = vary

        self.components: List[LifetimeComponent] = (
            [] if components is None else
            [LifetimeComponent() for _ in range(components)] if isinstance(components, int) else
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
                *parse_parameter_tuple(t0, vary=False)
            )
        )
        self.__name__ = "Resolution Component"

    def __call__(self, t, intensity, sigma):
        return intensity * gauss(t, t[t.shape[0]//2], sigma)


class ResolutionComponents:
    def __init__(
        self,
        components: None | int | ResolutionComponent | List[ResolutionComponent] = None,
        vary: bool = True,
    ) -> None:

        self.vary = vary

        self.components: List[ResolutionComponent] = (
            [] if components is None else
            [ResolutionComponent() for _ in range(components)] if isinstance(components, int) else
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
    def __init__(
        self,
        value: float | int | lmfit.Parameter | Tuple | List = 0,
    ) -> None:

        self.background = (
            value if isinstance(value, lmfit.Parameter) else
            lmfit.Parameter(
                "background",
                *parse_parameter_tuple(value)
            )
        )
        self.__name__ = "Background Component"

    def __call__(self, t: np.ndarray, background: float | int):
        return np.full(t.shape, background)


class LifetimeModel:
    def __init__(
        self,
        lifetime_components: int
            | LifetimeComponent
            | List[LifetimeComponent]
            | LifetimeComponents = 1,
        resolution_components: None
            | ResolutionComponent
            | List[ResolutionComponent]
            | ResolutionComponents = None,
        background_component: None
            | BackgroundComponent = None,
    ) -> None:

        self.lifetime_components = (
            lifetime_components if isinstance(lifetime_components, LifetimeComponents) else
            LifetimeComponents(lifetime_components)
        )

        self.resolution_components = (
            resolution_components if isinstance(resolution_components, ResolutionComponents) else
            ResolutionComponents(resolution_components)
        )

        self.background_component = background_component


def parse_model(model: Dict[str, Tuple] | None) -> LifetimeModel | None:
    if model is None or len(model) == 0:
        return None

    model = {key.lower(): val for key, val in model.items()}

    lifetime_components = []
    resolution_components = []
    background_component = None

    if "background" in model:
        background_component = BackgroundComponent(model.pop("background"))

    resolution_params = [key for key in model if key.startswith("res")]
    lifetime_params = [
        key for key in model
        if key.startswith("lif") or key.startswith("int")
    ]

    r_indices = {int(p.split("_")[2]) for p in resolution_params}
    l_indices = {int(p.split("_")[1]) for p in lifetime_params}

    n_r = max(r_indices)
    n_l = max(l_indices)

    for j in range(1, n_r+1):
        resolution_components.append(
            ResolutionComponent(
                sigma=parse_parameter_tuple(model.pop(f"res_sigma_{j}", 100)),
                intensity=parse_parameter_tuple(model.pop(f"res_intensity_{j}", 0.5)),
                t0=parse_parameter_tuple(model.pop(f"res_t0_{j}", 0), vary=False)
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
        background_component=background_component
    )


def dump_model(model: LifetimeModel) -> Dict:
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
        p = model.background_component.background
        out["background"] = (p.value, p.vary, p.min, p.max)

    return out


def combine_models(model1: LifetimeModel | Dict | None, model2: LifetimeModel | Dict | None) -> LifetimeModel | None:
    if model1 is None and model2 is None:
        return None
    elif model1 is None:
        return parse_model(model2) if not isinstance(model2, LifetimeModel) else model2
    elif model2 is None:
        return parse_model(model1) if not isinstance(model1, LifetimeModel) else model1

    model1_dict = dump_model(model1) if isinstance(model1, LifetimeModel) else model1
    model2_dict = dump_model(model2) if isinstance(model2, LifetimeModel) else model2

    model1_dict.update(model2_dict)

    return parse_model(model1_dict)