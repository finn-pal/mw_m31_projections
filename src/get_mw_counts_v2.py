import multiprocessing as mp
import os
from collections import Counter

import numpy as np
import pandas as pd

from mw_projections import MW_Observation


def get_counts(gid, dat_dir):
    """
    Compute class counts for one galaxy.
    Returns (gid, class_dict).
    """

    gid = int(gid)

    sb_frac = 0.5
    bands = {"JC": ["V"]}

    gal_dir = os.path.join(dat_dir, "geckos")
    df_gal = pd.read_csv(os.path.join(gal_dir, "galaxy_details.csv"))

    d_mpc = df_gal.loc[df_gal["gid"] == gid, "d_mpc"].values[0]
    if d_mpc <= 20:
        psf_fwhm = 1.0
    elif d_mpc <= 40:
        psf_fwhm = 0.8
    else:
        psf_fwhm = 0.7

    obss = MW_Observation(
        galaxy=gid,
        dat_dir=dat_dir,
        thetas=100,
        bands=bands,
        sb_key="JC_V",
        psf_fwhm_arcsec=psf_fwhm,
        sb_min=24.5,
        sb_frac=sb_frac,
        verbose=True,
    )

    class_dict = {"Bulge": [], "Disc": [], "Halo": [], "Unknown": []}

    for obs in obss:
        gc_data = obs.gc_data
        mask = gc_data["pos_mask"] & gc_data["sb_mask"] & gc_data["ext_mask"]
        counts = Counter(gc_data["class"][mask])

        for cls in class_dict:
            class_dict[cls].append(counts.get(cls, 0))

    class_dict["Total"] = np.sum(list(class_dict.values()), axis=0).tolist()

    return gid, class_dict


if __name__ == "__main__":
    CORES = 8

    dat_dir = "../data/"
    # dat_dir = "/Users/z5114326/Documents/other_projects/mw_m31_projections/data/"
    gids = np.arange(1, 37)

    with mp.Pool(processes=CORES, maxtasksperchild=1) as pool:
        results = pool.starmap(get_counts, [(gid, dat_dir) for gid in gids])

    res_dict = dict(results)

    df_gal = pd.read_csv(os.path.join(dat_dir, "geckos", "galaxy_details.csv"))

    print()
    for gid in gids:
        gid = int(gid)
        print("-------------------------")
        print(f"{gid} - {df_gal['object'].iloc[gid - 1]}\n")

        for cls in res_dict[gid]:
            mean = np.mean(res_dict[gid][cls])
            std = np.std(res_dict[gid][cls])
            print(f"{cls}: {mean:.1f} +/- {std:.1f}")

        print()

    # --------------------------------------------------
    # Build summary table
    # --------------------------------------------------
    rows = []

    for gid in gids:
        gid = int(gid)

        row = {
            "gid": gid,
            "object": df_gal["object"].iloc[gid - 1],
        }

        # class-wise mean/std
        for cls in ["Bulge", "Disc", "Halo", "Unknown"]:
            vals = np.array(res_dict[gid][cls])
            row[cls] = np.mean(vals)
            row["d" + cls] = np.std(vals)

        # total mean/std (already computed per observation)
        total_vals = np.array(res_dict[gid]["Total"])
        row["Total"] = np.mean(total_vals)
        row["dTotal"] = np.std(total_vals)

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    # enforce column order
    df_summary = df_summary[
        [
            "gid",
            "object",
            "Disc",
            "dDisc",
            "Halo",
            "dHalo",
            "Bulge",
            "dBulge",
            "Unknown",
            "dUnknown",
            "Total",
            "dTotal",
        ]
    ]

    df_summary.to_csv(dat_dir + "results/mw_gc_cnt.csv", index=False)
