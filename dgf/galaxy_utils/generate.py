from importlib.resources import is_resource
import astropy.io.ascii as asc
import pandas as pd
import numpy as np
from scipy.stats import norm
import astropy.units as u
import dgf.galaxy_utils.isochrone as isochroneModel

from IPython import embed

# ------- DEFAULT VARIABLES ------- #
# ic_file = "Isochrones/iso_age_12_feh_-1.8.txt"
# lf_file = "Isochrones/lf_age_12_feh_-1.8_hires.txt"
# fg_file = "Foregrounds/formatted_gcd_dwarf.h5"
# -------------------------------- #


def create_dwarf(
    x,
    y,
    ic_file,
    lf_file,
    distance=100 * u.kpc,
    coordinates=(0, 0),
    hrv=0,
    dhrv=2,
    dhrv_scale=0.4,
    metal=-1.8,
    metal_scale=0.2,
):
    N = len(x)

    x = x.to(u.kpc)
    y = y.to(u.kpc)
    distance = distance.to(u.kpc)

    # Generate dwarf components
    relative_dec = ((np.arctan((y.value / (2 * (distance.value))))) * u.rad).to(
        u.degree
    )

    relative_ra = (
        (np.arctan((x.value / (2 * (distance.to(u.kpc).value)))))
        * np.cos(relative_dec)
        * u.rad
    ).to(u.degree)

    ra = coordinates[0] + relative_ra
    dec = coordinates[1] + relative_dec

    isochrone = asc.read(ic_file, format="commented_header", header_start=13)
    lum_func = asc.read(lf_file, format="commented_header", header_start=13)
    cmd = isochroneModel.sample(isochrone, lum_func, N, noise=0.05, dist=distance)

    rmag = np.array(cmd["mag"])
    color = np.array(cmd["color"])
    gmag = np.array(cmd["mag"]) + np.array(cmd["color"])

    df = pd.DataFrame(
        {
            "RA": ra,
            "DEC": dec,
            "HRV": hrv + norm.rvs(0, dhrv, N),
            "dHRV": norm.rvs(dhrv, dhrv_scale, N),
            "[Fe/H]": norm.rvs(metal, metal_scale, N),
            "d[Fe/H]": 0.1 * np.ones(N),
            "r": rmag,
            "g": gmag,
            "color": color,
            "member": np.ones(N),
        }
    )

    return df


def create_observation(
    dwarf_df,
    fg_file,
    dwarf_coord,
    dhrv,
    dhrv_scale,
    slit=None,
    mag_limit=None,
    cmd_window=False,
    ic_file=None,
    window=0.2,
    distance=100 * u.kpc,
    slit_sample_count=None,
):

    fg_data = asc.read(fg_file, format="commented_header")

    mw_ra = fg_data["RAJ2000"]
    mw_dec = fg_data["DECJ2000"]

    mw_hrv = fg_data["HRV"]
    mw_dhrv = fg_data["errHrv"]

    mw_c = fg_data["g-r"]

    mw_r = mw_g = None

    if "r" in fg_data.colnames:
        mw_r = fg_data["r"]
        mw_g = fg_data["g-r"] + mw_r
    elif "g" in fg_data.colnames:
        mw_g = fg_data["g"]
        mw_r = mw_g - fg_data["g-r"]

    mw_feh = fg_data["[M/H]"]
    mw_feh_err = fg_data["errMet"]

    N = len(mw_ra)

    mw_df = pd.DataFrame(
        {
            "RA": mw_ra,
            "DEC": mw_dec,
            "HRV": mw_hrv,
            "dHRV": mw_dhrv,
            "[Fe/H]": mw_feh,
            "d[Fe/H]": mw_feh_err,
            "r": mw_r,
            "g": mw_g,
            "color": mw_c,
            "member": np.zeros(N),
        }
    )

    merged_df = dwarf_df.merge(mw_df, how="outer")

    if slit != None:
        merged_df = merged_df.loc[
            (merged_df.RA < (dwarf_coord[0] + (slit[0].to(u.deg))))
            & (merged_df.RA > (dwarf_coord[0] - (slit[0].to(u.deg))))
            & (merged_df.DEC < (dwarf_coord[1] + (slit[1].to(u.deg))))
            & (merged_df.DEC > (dwarf_coord[1] - (slit[1].to(u.deg))))
        ]

    if mag_limit != None:
        merged_df = merged_df.loc[merged_df.r < mag_limit]

    if cmd_window:
        isochrone = asc.read(ic_file, format="commented_header", header_start=13)
        iso_r = 5 * np.log10(distance.to(u.pc).value) - 5 + isochrone["rmag"]
        iso_c = isochrone["gmag"] - isochrone["rmag"]
        mask = np.zeros(len(merged_df.color))
        for i in np.arange(len(merged_df.color)):
            distances = np.sqrt(
                (merged_df.iloc[i]["r"] - iso_r) ** 2
                + (merged_df.iloc[i]["color"] - iso_c) ** 2
            )
            if np.min(distances) < window:
                mask[i] = 1
        merged_df = merged_df.iloc[mask == 1]

    if slit_sample_count != None:
        random_indices = np.random.randint(
            low=0, high=len(merged_df.r), size=slit_sample_count
        )
        merged_df = merged_df.iloc[random_indices]

    return merged_df


### Gaia Challenge Data Helper Functions ###


def format_gcd(data_path, outfile="formatted_data.hdf"):
    data = asc.read(data_path)
    # err  = asc.read(err_path)

    data.rename_columns(
        ["col1", "col2", "col3", "col4", "col5", "col6"],
        ["x", "y", "z", "vx", "vy", "vz"],
    )
    #   err.rename_columns(
    #     ["col1", "col2", "col3", "col4", "col5", "col6"],
    #     ["x_err", "y_err", "z_err", "vx_err", "vy_err", "vz_err"],
    # )
    data.write(outfile, overwrite=True)


### Misc


def diameterToAngle(diameter, distance, dec=0 * u.deg):
    angular_size = (
        (np.arctan((diameter / (2 * (distance.to(u.kpc)))).decompose().value))
        * np.cos(dec)
        * u.rad
    ).to(u.degree)
    return angular_size
