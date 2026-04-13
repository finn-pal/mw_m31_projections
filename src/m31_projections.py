import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fsps.filters import FILTERS
from scipy.integrate import quad, quad_vec

from field import Field
from utils import Conversions, Transforms

#############################################################################################################


class Galaxy_Surface_Brightness:
    """
    Set of functions used to calculate galaxy surface brightness.
    Based on Courteau et al. (2011)
    """

    @staticmethod
    def sersic_profile(r: float, n: float, mu_e: float, R_e: float) -> float:
        b_n = 1.9992 * n - 0.3271
        I_e = 10 ** (-0.4 * mu_e)
        I_r = I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1))
        return I_r

    @staticmethod
    def exp_profile(r: float, mu_0: float, R_d: float) -> float:
        I_0 = 10 ** (-0.4 * mu_0)
        I_r = I_0 * np.exp(-r / R_d)
        return I_r

    @staticmethod
    def power_law(r: float, mu_star: float, a_h: float, alpha: float, R_star: float = 30) -> float:
        I_star = 10 ** (-0.4 * mu_star)
        I_r = I_star * ((1 + (R_star / a_h) ** 2) / (1 + (r / a_h) ** 2)) ** alpha
        return I_r

    @staticmethod
    def get_sb_profile(
        r: float | np.ndarray, dat_dir: str, model: str = "U", sb_key: str = "JC_I"
    ) -> float | np.ndarray:

        band_type, band = sb_key.split("_")

        # If using Johnson-Cousins then zeropoints is Vega else if using SDSS then zeropoints is AB
        assert (band_type == "JC") & (band == "I"), (
            "Currently only supporting Johnson-Cousins I-band for M31 surface brightness profiles"
        )

        sb_dict = {"Power-Law": ["R", "S", "T", "U"], "Sersic": ["V", "W"]}
        mkey = None
        for key in sb_dict.keys():
            if model in sb_dict[key]:
                mkey = key
        assert mkey is not None, "Model must be chosen from [R, S, T, U, V, W]"

        # print(f"{band_type} {band}-band, Model: {model}, Disk - Bulge - {mkey} Faint (Halo)")

        sb_model_path = dat_dir + "m31/surface_brightness/sb_profiles.json"
        with open(sb_model_path, "r") as f:
            sb_models = json.load(f)
        sb_dict = sb_models[band_type][band][model]

        I_b = Galaxy_Surface_Brightness.sersic_profile(r, sb_dict["n"], sb_dict["mu_eb"], sb_dict["R_eb"])
        I_d = Galaxy_Surface_Brightness.exp_profile(r, sb_dict["mu_0"], sb_dict["R_d"])

        if mkey == "Power-Law":
            I_h = Galaxy_Surface_Brightness.power_law(r, sb_dict["mu_star"], sb_dict["a_h"], sb_dict["alpha"])
        elif mkey == "Sersic":
            I_h = Galaxy_Surface_Brightness.sersic_profile(
                r, sb_dict["n_f"], sb_dict["mu_ef"], sb_dict["R_ef"]
            )
        else:
            assert False, f"Unexpected mkey: {mkey}"

        I_tot = I_b + I_d + I_h
        # if I_tot <= 0:
        #     mu_tot = np.inf
        # else:
        #     mu_tot = -2.5 * np.log10(I_tot)

        mu_gal = np.where(I_tot > 0, -2.5 * np.log10(I_tot), np.inf)

        return mu_gal


#############################################################################################################


class Magntiudes:
    """
    Set of functions used to calculate GC magntiudes and surface brightness.
    """

    @staticmethod
    def get_gc_sb(gc_dict: dict, gecko_dist_kpc: float, sb_key: str, pixel_scale: float = 0.2):
        gecko_dist_pc = gecko_dist_kpc * 1000

        m_abs = gc_dict[sb_key]
        m_app = m_abs + 5 * np.log10(gecko_dist_pc) - 5

        mu_gc = m_app + 2.5 * np.log10(pixel_scale**2)

        return mu_gc

    def sb_limits(mu_gc: np.ndarray, mu_gal: np.ndarray, sb_frac: float = 1) -> np.ndarray:
        """
        Returns True if the GC surface brightness is brighter than a fraction (sb_frac) of the galaxy surface
        brightness.

        Args:
            mu_gc (np.ndarray): _description_
            mu_gal (np.ndarray): _description_
            sb_frac (float, optional): If 1 then GC is brighter than galaxy. If 0.5 then GC is brighter
                than a galaxy at half the brightness. If 2 then GC is brighter than a galaxy at double the
                brightness. Defaults to 1.

        Returns:
            np.ndarray: _description_
        """

        return mu_gc < mu_gal - 2.5 * np.log10(sb_frac)

    @staticmethod
    def observation_limit(
        gc_dict: dict, gecko_dist_kpc: float, sb_key: str, sb_min: float, pixel_scale: float
    ) -> np.ndarray:
        # calculate absolute magntiude observational limit
        gecko_dist_pc = gecko_dist_kpc * 1000

        m_abs_lim = sb_min - 2.5 * np.log10(pixel_scale**2) - 5 * np.log10(gecko_dist_pc) + 5
        m_abs = gc_dict[sb_key]

        return m_abs < m_abs_lim


#############################################################################################################


@dataclass
class M31_Observation:
    galaxy: str | int  # galaxy identifier (name or gid)
    dat_dir: str  # base directory containing geckos/ and mw/gcs/
    pixel_scale: float = 0.2  # arcsec
    sb_min: float = 24.5  # surface brightness limit (mag arcsec^-2)
    sb_model: str = "U"  # can be R, S, T, U, V, W
    get_gc_mags: bool = True  # time crunch
    sb_key: str = "JC_I"
    sb_frac: float = 1

    gc_data: dict = field(init=False)
    field_data: dict = field(init=False)

    def __post_init__(self):

        # 1. Load field dictionary from Field.get_field()
        self.field_data = Field.get_field(
            gal=self.galaxy,
            gal_dir=os.path.join(self.dat_dir, "geckos/"),
            get_archival=True,
            view_mode="wide",
        )

        self.m31_dist_kpc = 785  # kpc (McConnachie et al. (2005))

        # 2. Extract gal_dis_kpc
        # gal_dis_kpc = self.field_data["gal_dis_kpc"]

        # 3. Load M31 GC catalogue
        m31_path = os.path.join(self.dat_dir, "m31/gcs/m31_gcs.csv")
        m31_gcs = pd.read_csv(m31_path)

        # 5. Build GC dataset

        bands_path = os.path.join(self.dat_dir, "supplementary/bands.json")
        with open(bands_path, "r") as f:
            bands_dict = json.load(f)
        self.bands_dict = bands_dict

        self.gc_data = self._build_gc_data(m31_gcs)

    # =============================================================
    # Internal GC data builder (your original logic)
    # =============================================================
    def _build_gc_data(self, m31_gcs: pd.DataFrame):

        xs_m31_arcmin = m31_gcs["x"].values  # arcmin
        ys_m31_arcmin = m31_gcs["y"].values  # arcmin

        xs_m31_kpc = xs_m31_arcmin * self.m31_dist_kpc * np.pi / 10800
        ys_m31_kpc = ys_m31_arcmin * self.m31_dist_kpc * np.pi / 10800

        m31_dist_pc = self.m31_dist_kpc * 1000

        gecko_dist_kpc = self.field_data["gal_dis_kpc"]
        # gecko_dist_pc = gecko_dist_kpc * 1000

        R_offset_arcsec = 206265 * (xs_m31_kpc / gecko_dist_kpc)
        z_offset_arcsec = 206265 * (ys_m31_kpc / gecko_dist_kpc)

        rp_kpc = m31_gcs["Rp"].values

        # get region masking
        pos_mask = Field.positional_masking(xs_m31_kpc, ys_m31_kpc, self.field_data, "kpc")

        gc_dict = {
            "ID": m31_gcs["Name"].values,
            "JC_U": Conversions.app_to_abs(m31_gcs["Umag"].values, m31_dist_pc),
            "JC_B": Conversions.app_to_abs(m31_gcs["Bmag"].values, m31_dist_pc),
            "JC_V": Conversions.app_to_abs(m31_gcs["Vmag"].values, m31_dist_pc),
            "JC_R": Conversions.app_to_abs(m31_gcs["Rmag"].values, m31_dist_pc),
            "JC_I": Conversions.app_to_abs(m31_gcs["Imag"].values, m31_dist_pc),
            "Vr": m31_gcs["Vr"].values,  # radial velocity [km/s]
            "x_m31_arcmin": xs_m31_arcmin,  # m31 projected x [arcmin]
            "y_m31_arcmin": ys_m31_arcmin,  # m31 projected x [arcmin]
            "x_kpc": xs_m31_kpc,  # m31 projected x [kpc]
            "y_kpc": ys_m31_kpc,  # m31 projected x [kpc]
            "rp_kpc": rp_kpc,  # projected distance from M31 [kpc]
            "R_offset_kpc": xs_m31_kpc,  # have repeated x_kpc for plotting consistency with mw data
            "z_offset_kpc": ys_m31_kpc,  # have repeated y_kpc for plotting consistency with mw data
            "R_offset_arcsec": R_offset_arcsec,
            "z_offset_arcsec": z_offset_arcsec,
            "pos_mask": pos_mask,
        }

        # galaxy surface brightness profile is not corrected for projection or extinction effects
        mu_gal = Galaxy_Surface_Brightness.get_sb_profile(rp_kpc, self.dat_dir, self.sb_model, self.sb_key)
        gc_dict["gal_sb" + "_" + self.sb_key] = mu_gal

        # using 2 positions (projected x, y) and observed (pre-reddening correction) magntiudes
        mu_gc = Magntiudes.get_gc_sb(gc_dict, gecko_dist_kpc, self.sb_key, self.pixel_scale)
        gc_dict["gc_sb" + "_" + self.sb_key] = mu_gc

        gc_dict["sb_mask"] = Magntiudes.sb_limits(mu_gc, mu_gal, self.sb_frac)
        gc_dict["ext_mask"] = Magntiudes.observation_limit(
            gc_dict, gecko_dist_kpc, self.sb_key, self.sb_min, self.pixel_scale
        )

        # extinction business can be ignored as taking observed colors
        # color_excess_path = os.path.join(self.dat_dir, "m31/colors_excess/color_excess.csv")
        # color_excess = pd.read_csv(color_excess_path)
        # color_excess["Name"] = [s.replace(" ", "") for s in color_excess["Name"]]

        # m31_gcs = m31_gcs.merge(color_excess[["Name", "EBV"]], on="Name", how="left")
        # gc_dict["EBV"] = np.array([EBV if ~np.isnan(EBV) else 0 for EBV in m31_gcs["EBV"]])

        return gc_dict


#############################################################################################################
