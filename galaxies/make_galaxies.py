import dgf
import sys
import numpy as np
import asdf
from IPython import embed
import astropy.units as u
from astropy.io.misc.hdf5 import read_table_hdf5
import json
from dune import GeneratePlummerNFW

from IPython import embed

gals = ["kona_a", "kona_b", "kona_c", "kona", "draco_a", "draco_b", "draco_c", "draco"]

for gal_name in gals:
    galaxy_config_path = f"./gals/{gal_name}.json"
    file = open(galaxy_config_path)
    galaxy_config = json.load(file)
    slit = None

    x = y = vz = None

    if galaxy_config["phase_space_path"] != 0:
        ps_df = read_table_hdf5(galaxy_config["phase_space_path"])
        x = ps_df["x"] * u.kpc
        y = ps_df["y"] * u.kpc
        vz = ps_df["vz"] * u.km / u.s
    else:
        ps = GeneratePlummerNFW(
            galaxy_config["N_stars"],
            galaxy_config["plummer_radius"] * u.pc,
            galaxy_config["scale_radius"] * u.kpc,
            mass=galaxy_config["dyn_mass"] * u.solMass,
        )
        ps["x"] = ps["x"].to(u.kpc)
        ps["y"] = ps["y"].to(u.kpc)
        ps["vz"] = ps["vz"] * (u.km / u.s)

        x = ps["x"]
        y = ps["y"]
        vz = ps["vz"]

    coords = (galaxy_config["RA"] * u.deg, galaxy_config["DEC"] * u.deg)

    galaxy = dgf.create_dwarf(
        x,
        y,
        galaxy_config["isochrone_path"],
        galaxy_config["lumfunc_path"],
        galaxy_config["distance"] * u.kpc,
        coords,
        galaxy_config["HRV"],
        galaxy_config["dHRV"],
        galaxy_config["dHRV_err"],
        galaxy_config["metal"],
        galaxy_config["metal_err"],
    )

    if (galaxy_config["slit_x"] == 0) or (galaxy_config["slit_y"] == 0):
        slit = None
    else:
        slit = (galaxy_config["slit_x"] * u.arcmin, galaxy_config["slit_y"] * u.arcmin)

    slit_sample_count = (
        None if (galaxy_config["slit_sample"] <= 0) else galaxy_config["slit_sample"]
    )

    observation = dgf.create_observation(
        galaxy,
        galaxy_config["foreground_path"],
        coords,
        galaxy_config["dHRV"],
        galaxy_config["dHRV_err"],
        cmd_window=True,
        slit=slit,
        ic_file=galaxy_config["isochrone_path"],
        distance=galaxy_config["distance"] * u.kpc,
        slit_sample_count=slit_sample_count,
        mag_limit=23,
    )

    galaxy.to_hdf(f"./gals/{gal_name}_stars.hdf", key="w")
    observation.to_hdf(f"./gals/{gal_name}_observation.hdf", key="w")
