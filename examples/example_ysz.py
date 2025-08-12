from pypalsfit import MeasurementCampaign


lt_model = {
    "lifetime_1": (186, 176, 196),
    "intensity_1": 0.99,

    "lifetime_2": (300, 200, 750),
    "intensity_2": 0.01,

    "lifetime_3": (5000, 750),
    "intensity_3": 0.0,

    "background": (0.57, 0, 1),
}

res_model = {
    "res_sigma_1": (96, 50, 120),
    "res_intensity_1": 0.86,

    "res_sigma_2": (156, 100, 1000),
    "res_intensity_2": 0.14,
    "res_t0_2": (-50, -500, 500),
}

mc = MeasurementCampaign(
    "data",
    lt_model = lt_model,
    res_model = res_model,
    calibrate = True,
    verbose = True,
)

mc.dump_resolution_components("data/resolution_components.json")


### Samples

folders = ["8-SC104 - adjusted energies", "9-W1Re5", "10-W3Re5"]

mult = MultiCampaign(
    [top + folder for folder in folders],

    lt_model = "models/w_model.json",

    res_model = top + "ysz_resolution_components.json",
    res_keys = ["PositronImplantationEnergy"],

    show = False,
    verbose = True,
)

mult.sort("PositronImplantationEnergy")

mult.plot(
    "PositronImplantationEnergy", [
        "lifetime_1", "intensity_1",
        "lifetime_2", "intensity_2",
        "lifetime_3", "intensity_3"
    ],
    labels="MeasName", labels_is_param=True,
    marker="x"
)
