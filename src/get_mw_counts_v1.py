import json
import multiprocessing as mp
import os
from collections import Counter

import numpy as np
import pandas as pd

from mw_projections import MW_Observation


def get_counts(gid, dat_dir, df_gal, class_dict: dict = {}):

    sb_frac = 0.5

    bands = {"JC": ["V"]}
    # save_dict = {}
    # for gid in range(1, 37):
    # for gid in range(1, 3):
    # print(gid)
    d_mpc = df_gal.loc[df_gal["gid"] == gid]["d_mpc"].values[0]
    if d_mpc <= 20:
        psf_fwhm = 1
    elif d_mpc <= 40:
        psf_fwhm = 0.8
    else:
        psf_fwhm = 0.7

    obss = MW_Observation(
        galaxy=gid,
        dat_dir=dat_dir,
        thetas=10,
        bands=bands,
        sb_key="JC_V",
        psf_fwhm_arcsec=psf_fwhm,
        sb_min=24.5,
        sb_frac=sb_frac,
        verbose=True,
    )

    # save_dict[gid] = obss

    ####################################################################################

    masks = ["pos", "sb", "ext"]

    # class_dict = {}
    # for gid in range(1, 3):
    class_dict[gid] = {"Bulge": [], "Disc": [], "Halo": [], "Unknown": []}

    for obs in obss:
        gc_data = obs.gc_data

        mask = np.ones(len(gc_data["ID"]), dtype=bool)
        if "pos" in masks:
            mask &= gc_data["pos_mask"]
        if "sb" in masks:
            mask &= gc_data["sb_mask"]
        if "ext" in masks:
            mask &= gc_data["ext_mask"]

        classifications = gc_data["class"][mask]

        # Count each class
        counts = Counter(classifications)

        # Append counts to class_dict
        for cls in class_dict[gid]:
            class_dict[gid][cls].append(counts.get(cls, 0))

    return class_dict


if __name__ == "__main__":
    cores = 2
    # dat_dir = "../../data/"
    dat_dir = "../data/"
    gal_dir = os.path.join(dat_dir, "geckos/")
    df_gal = pd.read_csv(gal_dir + "galaxy_details.csv")

    # gids = np.arange(1, 37)
    gids = np.arange(1, 3)

    ####################################################################################

    with mp.Manager() as manager:
        shared_dict = manager.dict()  # Shared dictionary across processes

        args = [(gid, dat_dir, df_gal, shared_dict) for gid in gids]

        with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
            pool.starmap(get_counts, args, chunksize=1)

        res_dict = dict(shared_dict)

    ####################################################################################

    print()
    for gid in gids:
        print("-------------------------")
        print(gid, "-", df_gal["object"][gid - 1], "\n")
        for cls in res_dict[gid]:
            mean = np.round(np.mean(res_dict[gid][cls]), 1)
            std = np.round(np.std(res_dict[gid][cls]), 1)
            print(f"{cls}: {mean:.1f} +/- {std:.1f}")

        print()
