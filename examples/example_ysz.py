import matplotlib.pyplot as plt
from pypalsfit import MeasurementCampaign

top = "data/YSZ/"

### Calibration

ysz_model = {
    "lifetime_1": (178, False),
}

res_model = {
    "res_sigma_1": (105, 30, 1000),
    "res_intensity_1": (0.78),

    "res_sigma_2": (81, 30, 1000),
    "res_intensity_2": (0.2),
    "res_t0_2": (-16, -500, 500),

    "res_sigma_3": (350, 30, 1000),
    "res_intensity_3": (0.03),
    "res_t0_3": (138, -500, 500),

    "res_sigma_4": (350, 30, 1000),
    "res_intensity_4": (0.03),
    "res_t0_4": (138, -500, 500),
}

mc = MeasurementCampaign(
    top,

    lt_model = "models/ysz_model.json",

    res_model = res_model,
    calibrate = True,

    show_fits = True,
    verbose = True,

    fit_time_left_of_peak = 500,
    fit_time_right_of_peak = 10000,
)

fig, axs = plt.subplots(4, 2)

for ax in axs[:-1]:
    ax[1].set_ylim([0, 1])

mc.plot(
    "PositronImplantationEnergy", [
        "res_sigma_1", "res_intensity_1",
        "res_sigma_2", "res_intensity_2",
        "res_sigma_3", "res_intensity_3",
        "res_t0_2", "res_t0_3",
    ],
    fig=fig, marker="x", errorbars=False,
)

mc.dump_resolution_components(top + "ysz_resolution_components.json")


### Backwards Validation

mc = MeasurementCampaign(
    top,

    lt_model = "models/ysz_model.json",
    lt_keys = ["PositronImplantationEnergy"],

    res_model = top + "ysz_resolution_components.json",
    res_keys = ["PositronImplantationEnergy"],

    show_fits = True,
    verbose = True,

    # different fit range than the calibration
    fit_time_left_of_peak = 100,
    fit_time_right_of_peak = 10000,
)

mc.plot(
    "PositronImplantationEnergy", [
        "lifetime_1", "intensity_1",
        "lifetime_2", "intensity_2",
    ],
    marker="x", errorbars=False,
)

