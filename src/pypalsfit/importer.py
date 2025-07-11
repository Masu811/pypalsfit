from pathlib import Path
import warnings
import datetime

from .lifetime import LifetimeSpectrum
from .measurement import LifetimeMeasurement


def import_elbe(
    filepath: str | Path,
    dtype: type | None = None
) -> LifetimeMeasurement:
    import pandas as pd
    err = "The provided file path does not lead to an ELBE spectrum file"
    assert str(filepath).endswith(".dat") and "PALS" in str(filepath), err

    with open(filepath, "r") as f:
        header = f.readline()

    file_dir = Path(filepath).parent
    filename = Path(filepath).stem

    f_info = filename.split("_")

    f_info[-3] = "Measurement_Parameter"

    param_filename = "_".join(f_info) + ".dat"

    param_filepath = file_dir / Path(param_filename)

    f_meas_idx = f_info.pop(0)
    f_meas_time = f_info.pop(-1)
    f_meas_date = f_info.pop(-1)
    _ = f_info.pop(-1)
    f_stat = f_info.pop(-1)
    f_temp = f_info.pop(-1)
    f_eimp = f_info.pop(-1)
    f_sample = "_".join(f_info)

    h_info = header.split("_")

    h_meas_idx = h_info.pop(0)
    tcal = h_info.pop(-1)
    h_stat = h_info.pop(-1)
    h_temp = h_info.pop(-1)
    h_eimp = h_info.pop(-1)
    h_sample = "_".join(h_info)

    commons = ["Meas Idx", "Sample", "Energy", "Temp"]
    f_common = [f_meas_idx, f_sample, f_eimp, f_temp]
    h_common = [h_meas_idx, h_sample, h_eimp, h_temp]
    common_equal = [f == h for f, h in zip(f_common, h_common)]

    if not all(common_equal):
        wrn = (
            f"{filename}:\n"
            f"File name and content header contain inconsistent metadata\n"
        )
        for i, eq in enumerate(common_equal):
            if not eq:
                wrn += f"{commons[i]}: File: {f_common[i]}, Header: {h_common[i]}\n"
        wrn += "Using header information"
        warnings.warn(wrn)

    spectrum = pd.read_csv(filepath, dtype=dtype, index_col=False, skiprows=1).to_numpy().squeeze()
    tcal = [0, 1000*float(tcal[:-3])]

    s = LifetimeSpectrum(spectrum=spectrum, tcal=tcal, name=filename)

    m = LifetimeMeasurement()

    m.spectra.append(s)

    split_date = list(map(int, f_meas_date.split(".")[::-1]))
    split_date[0] += 2000
    conv_date = datetime.date(*split_date)
    conv_time = datetime.time(*map(int, f_meas_time.split(".")))

    m.metadata["StartDateTime"] = datetime.datetime.combine(conv_date, conv_time)
    m.metadata["PositronImplantationEnergy"] = 1000*float(f_eimp[:-3])
    m.metadata["SampleTemperature"] = float(f_temp[:-1])
    m.metadata["MeasName"] = h_sample

    if not param_filepath.is_file():
        wrn = f"{filename}: Could not find parameter file"
        warnings.warn(wrn)
        return m

    with open(param_filepath, "r") as f:
        for line in f.readlines():
            if line.startswith("-") or len(line) < 29 or line.startswith("PMT"):
                continue
            key = line[:29].strip()
            val = line[29:].strip()
            m.metadata[key] = val

    return m