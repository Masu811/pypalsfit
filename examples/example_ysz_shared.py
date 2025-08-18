import matplotlib.pyplot as plt
from pypalsfit import MeasurementCampaign

top = "small_data/"

### Calibration

ysz_model = {
    "lifetime_1": (178, False),
    "intensity_1": (0.93, 0.0005),

    "lifetime_2": (300, 200, 600),
    "intensity_2": (0.07, 0.0005),
}

res_model = {
    "res_sigma_1": (80, 50, 110),
    "res_intensity_1": (0.5, 0.02),

    "res_sigma_2": (125, 100, 150),
    "res_intensity_2": (0.15, 0.05, 0.4),
    "res_t0_2": (10, -70, 30),

    "res_sigma_3": (70, 50, 120),
    "res_intensity_3": (0.1, 0.05, 0.4),
    "res_t0_3": (90, 30, 150),
}

# This dict was determined via the background determination prompt
ysz_background = {
    0.5: 0.0338, 1.0: 0.0155, 1.5: 0.0186, 2.0: 0.0184, 2.5: 0.0168,
    3.0: 0.0159, 3.5: 0.0182, 4.0: 0.0174, 4.5: 0.0184, 5.0: 0.0178,
    5.5: 0.0183, 6.0: 0.0172, 6.5: 0.0169, 7.0: 0.0192, 7.5: 0.0201,
    8.0: 0.0182, 9.0: 0.0187, 10.0: 0.0200, 12.0: 0.0176, 14.0: 0.0197,
    16.0: 0.0198, 18.0: 0.0183
}

def e_dependent_bg(model):
    return {
        f"Measurement {e}_keV": {
            "Metadata": {"PositronImplantationEnergy": 1000 * e},
            "Detector A": model.copy() | {"background": (bg, False)}
        } for e, bg in ysz_background.items()
    }

mc = MeasurementCampaign(
    top,

    lt_model = e_dependent_bg(ysz_model),
    lt_keys = ["PositronImplantationEnergy"],

    res_model = res_model,
    calibrate = True,

    show_fits = True,
    verbose = True,

    fit_start_counts = 10,
    fit_end_counts = 900,

    autocompute = False,
)

mc.shared_fit(
    {
        "lifetime_1": (178, 172, 190),
        "lifetime_2": (378, 300, 450),
    },

    use_jacobian=False,

    fit_start_counts = 10,
    fit_end_counts = 900,
)

fig, axs = plt.subplots(6, 2)

for ax in axs[:-1]:
    ax[1].set_ylim([0, 1])

mc.plot(
    "PositronImplantationEnergy", [
        "lifetime_1", "intensity_1",
        "lifetime_2", "intensity_2",
        "res_sigma_1", "res_intensity_1",
        "res_sigma_2", "res_intensity_2",
        "res_sigma_3", "res_intensity_3",
        "res_t0_2", "res_t0_3",
    ],
    fig=fig, marker="x",
)

mc.dump_resolution_components(top + "ysz_resolution_components_shared.json")


### Backwards Validation

new_ysz_model = {
    "lifetime_1": (178, 100, 200),
    "intensity_1": (0.92, 0.0005),

    "lifetime_2": (350, 250, 450),
    "intensity_2": (0.05, 0.0005),
}

mc = MeasurementCampaign(
    top,

    lt_model = e_dependent_bg(new_ysz_model),
    lt_keys = ["PositronImplantationEnergy"],

    res_model = top + "ysz_resolution_components.json",
    res_keys = ["PositronImplantationEnergy"],

    show_fits = True,
    verbose = True,

    # different fit range than the calibration
    fit_time_left_of_peak = 100,
    fit_time_right_of_peak = 1500,
)

mc.plot(
    "PositronImplantationEnergy", [
        "lifetime_1", "intensity_1",
        "lifetime_2", "intensity_2",
    ],
    marker="x",
)
