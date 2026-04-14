import os

# os.environ["SPS_HOME"] = "/Users/z5114326/Documents/GitHub/python-fsps/src/fsps/libfsps"
os.environ["SPS_HOME"] = "../../../GitHub/python-fsps/src/fsps/libfsps"

import json
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
    """

    @staticmethod
    def disc_3d_density_profile(x: float, y: float, z: float, disc_dict: dict):
        # x,y,z values are in pc
        # units for disc profile should be in [M_sol pc^-2] or [pc]
        # and so rho will be in units of [M_sol pc^-3]

        R = np.sqrt(x**2 + y**2)

        S0_tn, Rd_tn, zd_tn = disc_dict["thin"].values()
        S0_tk, Rd_tk, zd_tk = disc_dict["thick"].values()

        rho_thin = S0_tn / (2 * zd_tn) * np.exp(-np.abs(z) / zd_tn) * np.exp(-R / Rd_tn)
        rho_thick = S0_tk / (2 * zd_tk) * np.exp(-np.abs(z) / zd_tk) * np.exp(-R / Rd_tk)

        return rho_thin + rho_thick

    @staticmethod
    def disc_surface_density(
        y: float, z: float, disc_dict: dict, xmin: float = -np.inf, xmax: float = np.inf
    ):
        # x,y,z values are in pc
        # units for disc profile should be in [M_sol pc^-2] or [pc]
        # and so Sigma_mass will be in units of [M_sol pc^-2]

        args = (y, z, disc_dict)
        Sigma_mass, _ = quad(Galaxy_Surface_Brightness.disc_3d_density_profile, xmin, xmax, args=args)

        return Sigma_mass

    @staticmethod
    def surface_luminosity(Sigma_mass: float, dat_dir: str, band_type: str, band: str):
        # Sigma_mass will be in units of [M_sol pc^-2]
        # and so Sigma_lum will be in units of [L_sol pc^-2]

        mw_mags_path = dat_dir + "mw/mw_ml_ratios/mw_magnitudes.json"
        with open(mw_mags_path, "r") as f:
            mw_mags_dict = json.load(f)

        mass_to_light = mw_mags_dict[band_type][band]["mass-to-light"]
        Sigma_lum = Sigma_mass / mass_to_light

        return Sigma_lum

    @staticmethod
    def gal_surface_brightness(
        y: float, z: float, bands_dict: dict, dat_dir: str, band_type: str, band: str, sb_min: float = 24.5
    ):

        # If using Johnson-Cousins then zeropoints is Vega else if using SDSS then zeropoints is AB
        assert (band_type == "JC") | (band_type == "SDSS"), "band_type must be JC or SDSS"

        # values from best fitting model of McMillan (2017)
        # maintain otder of variables in this dictionry
        disc_dict = {
            "thin": {  # thin disc components
                "Sigma0": 896,  # central surface density [M_sol pc^-2]
                "Rd": 2500,  # scalelength [pc]
                "zd": 300,  # scaleheight [pc]
            },
            "thick": {  # thick disc components
                "Sigma0": 183,  # central surface density [M_sol pc^-2]
                "Rd": 3020,  # scalelength [pc]
                "zd": 900,  # scaleheight [pc]
            },
        }

        Sigma_mass = Galaxy_Surface_Brightness.disc_surface_density(y, z, disc_dict)
        Sigma_lum = Galaxy_Surface_Brightness.surface_luminosity(Sigma_mass, dat_dir, band_type, band)

        # get absolute magnitude of the sun in corresponding band
        fil = FILTERS[bands_dict[band_type][band]["fsps"]]
        if band_type == "JC":
            Mabs_sol = fil.msun_vega  # Vega
        elif band_type == "SDSS":
            Mabs_sol = fil.msun_ab  # AB

        # surface brightness
        mu = Mabs_sol - 2.5 * np.log10(Sigma_lum) + 21.572

        # min detection of halo light ~29.5 mag/arcsec https://ui.adsabs.harvard.edu/abs/2022ApJ...932...44G/abstract
        mu = np.min((mu, sb_min))

        return mu

    @staticmethod
    def process_gal_sb(
        ys_pc: np.ndarray, zs_pc: np.ndarray, bands: dict, band_dict: dict, dat_dir: str, sb_min: float = 29.5
    ):
        sb_dict = {}

        for band_type in bands.keys():
            for band in bands[band_type]:
                mu_hld = []
                for y, z in zip(ys_pc, zs_pc):
                    mu = Galaxy_Surface_Brightness.gal_surface_brightness(
                        y, z, band_dict, dat_dir, band_type, band, sb_min
                    )
                    mu_hld.append(mu)
                sb_dict["gal_sb_" + band_type + "_" + band] = np.array(mu_hld)

        return sb_dict


#############################################################################################################


class Magntiudes:
    """
    Set of functions used to calculate GC magntiudes and surface brightness.
    """

    @staticmethod
    def band_abs_mags(sp, age_gyr: float, feh: float, bands_dict: dict, band_type: str, band: str):
        # If using Johnson-Cousins then zeropoints is Vega else if using SDSS then zeropoints is AB
        assert (band_type == "JC") | (band_type == "SDSS"), "band_type must be JC or SDSS"

        band_fsps = bands_dict[band_type][band]["fsps"]
        sp.params["logzsol"] = feh
        # always returns absolute magnitude in AB.
        Mabs_ab = sp.get_mags(tage=age_gyr, bands=[band_fsps])[0]

        if band_type == "JC":
            # if in JC need to convert AB magntiude to Vega.
            Mabs = Conversions.ab_to_vega(Mabs_ab, bands_dict, band_type, band)
        else:
            Mabs = Mabs_ab

        return Mabs

    @staticmethod
    def get_absolute_magnitudes(dat_dir: str, bands: dict, ref_key: str, ref_mag: np.ndarray):

        col_path = os.path.join(dat_dir, "mw/colors/", "colors_reference.csv")
        df_col = pd.read_csv(col_path)

        # ref_band_type = next(iter(bands["ref"]))
        # ref_band = bands["ref"][ref_band_type]
        # ref_key = ref_band_type + "_" + ref_band

        # for band_type in bands.keys():
        #     for band in bands[band_type]:
        #         band_key = band_type + "_" + band
        #         col_name = ref_key + "-" + band_key

        #         mag_dict[col_name] = df_col[col_name]

        mag_dict = {}
        for band_type in bands.keys():
            for band in bands[band_type]:
                band_key = band_type + "_" + band
                if band_key == ref_key:
                    mag_dict[band_key] = ref_mag
                else:
                    col_name = ref_key + "-" + band_key
                    color = df_col[col_name]
                    mag_dict[band_key] = ref_mag - color

        return mag_dict

    @staticmethod
    def process_gc_sb(mag_dict: dict, bands: dict, gc_dist_pc: np.ndarray, pixel_scale: float = 0.2):
        # assume a single GC takes up a full pixel
        # pixel spatial scale = 0.2  # arcsec pixel^−1

        sb_dict = {}

        for band_type in bands.keys():
            for band in bands[band_type]:
                m_app = mag_dict[band_type + "_" + band] + 5 * np.log10(gc_dist_pc) - 5
                sb_dict["gc_sb" + "_" + band_type + "_" + band] = m_app + 2.5 * np.log10(pixel_scale**2)

        return sb_dict

    @staticmethod
    def sb_limits(sb_gc_dict: dict, sb_gal_dict: dict, sb_key: str, sb_frac: float = 1) -> bool:
        """
        Returns True if the GC surface brightness is brighter than a fraction (sb_frac) of the galaxy surface
        brightness.

        Args:
            sb_gc_dict (dict): _description_
            sb_gal_dict (dict): _description_
            sb_key (str): _description_
            sb_frac (float, optional): If 1 then GC is brighter than galaxy. If 0.5 then GC is brighter
                than a galaxy at half the brightness. If 2 then GC is brighter than a galaxy at double the
                brightness. Defaults to 1.

        Returns:
            bool: _description_
        """

        # ref_band_type = next(iter(bands["ref"]))
        # ref_band = bands["ref"][ref_band_type]

        # if GC surface brightness is brighter than background galaxy then return True
        gal_sb = sb_gal_dict["gal_sb_" + sb_key]
        gc_sb = sb_gc_dict["gc_sb_" + sb_key]

        return gc_sb < gal_sb - 2.5 * np.log10(sb_frac)


#############################################################################################################


class Extinction:
    """
    Set of functions used to calculate extinction along the LOS.
    """

    @staticmethod
    def z_warp_func(
        R: float,
        phi: float,
        gamma_warp: float = 0.18,
        R_warp: float = 8.4,
        phi_max: float = 90,
        theta: float = None,
    ):
        # we add theta correction to original equation for warp aligning in new frame
        if theta is None:
            theta = 0
        return (
            gamma_warp * np.min((R_warp, R - R_warp)) * np.cos(np.deg2rad(phi) - np.deg2rad(phi_max + theta))
        )

    @staticmethod
    def k_flare_func(R: float, gamma_flare: float = 0.0054, R_flare: float = 8.96):
        return 1 + gamma_flare * np.min((R_flare, R - R_flare))

    @staticmethod
    def dust_density_func(
        s: float,
        xyz_observer: np.ndarray,
        xyz_target: np.ndarray,
        theta: float = None,
        rho_0: float = 0.23,
        h_R: float = 4.200,
        h_z: float = 0.088,
        gamma_warp: float = 0.18,
        gamma_flare: float = 0.0054,
        phi_max=90,
        R_flare: float = 8.96,
        R_warp: float = 8.4,
        R_sol: float = 8.0,
    ):

        # Unit vector along LOS
        n = (xyz_target - xyz_observer) / np.linalg.norm(xyz_target - xyz_observer)
        xyz_vector = xyz_observer + s * n

        R_s = np.sqrt(xyz_vector[0] ** 2 + xyz_vector[1] ** 2)
        phi_s = np.arctan2(xyz_vector[1], xyz_vector[0])  # in radians
        phi_s = np.rad2deg(phi_s)  # convert to degress of consistency later on
        z_s = xyz_vector[2]

        z_warp = Extinction.z_warp_func(R_s, phi_s, gamma_warp, R_warp, phi_max, theta)
        k_flare = Extinction.k_flare_func(R_s, gamma_flare, R_flare)

        rho_dust = (
            (rho_0 / k_flare) * np.exp(-(R_s - R_sol) / h_R) * np.exp(-np.abs(z_s - z_warp) / (k_flare * h_z))
        )

        return rho_dust

    @staticmethod
    def LOS_dust_integration(xyz_rot: np.ndarray, gal_dis_kpc: float, theta: float, dust_dict: dict):
        const_args = (
            theta,
            dust_dict["rho_0"],
            dust_dict["h_R"],
            dust_dict["h_z"],
            dust_dict["gamma_warp"],
            dust_dict["gamma_flare"],
            dust_dict["phi_max"],
            dust_dict["R_flare"],
            dust_dict["R_warp"],
            dust_dict["R_sol"],
        )

        xyz_observer = np.array([gal_dis_kpc, 0, 0])
        EBV = np.zeros(len(xyz_rot))
        for i, xyz_target in enumerate(xyz_rot):
            s_max = np.linalg.norm(xyz_target - xyz_observer)

            # combine per-target args with constants
            args = (xyz_observer, xyz_target) + const_args

            # integrate from observer to target
            column_density, _ = quad_vec(Extinction.dust_density_func, 0, s_max, args=args)

            EBV[i] = column_density

        return EBV

    @staticmethod
    def get_kprime(wavelength: float, RVprime: float = 4.05):
        # RVprime = 4.05 +/- 0.80
        # wavelength must be in µm

        assert 0.12 <= wavelength <= 2.20, f"{wavelength}\u03bcm is outside of range [0.12, 2.20]\u03bcm"

        if (0.63 <= wavelength) & (wavelength <= 2.20):
            kp = 2.659 * (-1.857 + 1.040 / wavelength) + RVprime
        elif (0.12 <= wavelength) & (wavelength < 0.63):
            kp = (
                2.659 * (-2.156 + 1.509 / wavelength - 0.198 / wavelength**2 + 0.011 / wavelength**3)
                + RVprime
            )
        return kp

    @staticmethod
    def get_extinction(
        EBV: np.ndarray, bands: dict, bands_dict: dict, RVprime: float = 4.05, EBV_multiplier: float = 1
    ):
        # Calzetti et al. (2000) - https://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C/abstract
        # RVprime = 4.05 +/- 0.80

        # convert wavelength to micrometres (µm)
        lambda_convert = {
            "a": 1e-4,  # angstrom (Å) -> µm
            "nm": 1e-3,  # nanometre -> µm
            "um": 1.0,  # µm -> µm
            "mm": 1e3,  # millimetre -> µm
            "cm": 1e4,  # centimetre -> µm
            "m": 1e6,  # metre → µm
        }

        # EBV = gc_dict["EBV"]
        # EsBV = 0.44 * EBV  # 0.44 +/- 0.03
        EsBV = EBV * EBV_multiplier

        ext_dict = {"EBV": EBV}

        for band_type in bands.keys():
            for band in bands[band_type]:
                band_key = band_type + "_" + band
                band_units = bands_dict[band_type][band]["units"]
                assert band_units in lambda_convert, f"Unrecognized wavelength unit: {band_units}"

                wavelength = bands_dict[band_type][band]["wavelength"] * lambda_convert[band_units]
                kp = Extinction.get_kprime(wavelength, RVprime)
                A = kp * EsBV
                ext_dict[band_key + "_A"] = A
                # gc_dict[band_key + "_ex"] = np.array(gc_dict[band_key] + A, dtype=">f4")

        return ext_dict

    @staticmethod
    def process_extinction(
        xyz_rot: np.ndarray, gal_dis_kpc: float, theta: float, bands: dict, bands_dict: dict
    ):

        R_sol = 8.0  # kpc
        dust_dict = {
            "rho_0": 0.23,  # mag/kpc
            "h_R": 4.200,  # kpc
            "h_z": 0.088,  # kpc
            "R_sol": R_sol,  # kpc
            "gamma_warp": 0.18,  # 1/kpc
            "gamma_flare": 0.0054,  # 1/kpc
            "phi_max": 90,  # deg
            "R_flare": 1.12 * R_sol,
            "R_warp": 1.05 * R_sol,
        }

        EBV = Extinction.LOS_dust_integration(xyz_rot, gal_dis_kpc, theta, dust_dict)
        ext_dict = Extinction.get_extinction(EBV, bands, bands_dict)

        return ext_dict

    @staticmethod
    def apply_extinction(gc_dict: dict, bands: dict):

        # d_kpc = gc_dict["gc_dist_kpc"]
        # d_pc = d_kpc * 1000

        for band_type in bands.keys():
            for band in bands[band_type]:
                band_key = band_type + "_" + band
                m_abs = gc_dict[band_key]
                # m_app = Conversions.abs_to_app(m_abs, d_pc)
                # m_app_ext = m_app + gc_dict[band_key + "_A"]
                # m_abs_ext = Conversions.app_to_abs(m_app_ext, d_pc)
                m_abs_ext = m_abs + gc_dict[band_key + "_A"]
                gc_dict[band_key + "_ext"] = m_abs_ext

        return gc_dict

    @staticmethod
    def observation_limit(gc_dict: dict, bands: dict, sb_key: str, sb_min: float, pixel_scale: float):
        # calculate absolute magntiude observational limit
        gc_dist_kpc = gc_dict["gc_dist_kpc"]
        gc_dist_pc = gc_dist_kpc * 1000

        m_abs_lim = sb_min - 2.5 * np.log10(pixel_scale**2) - 5 * np.log10(gc_dist_pc) + 5

        for band_type in bands.keys():
            for band in bands[band_type]:
                band_key = band_type + "_" + band
                if band_key == sb_key:
                    m_abs_ext = gc_dict[band_key + "_ext"]
                    gc_dict["ext_mask"] = m_abs_ext < m_abs_lim

        return gc_dict


#############################################################################################################


@dataclass
class Single_Observation:
    galaxy: str | int  # galaxy identifier (name or gid)
    dat_dir: str  # base directory containing geckos/ and mw/gcs/
    bands: dict  # dictionary of bands to determine
    theta: float | None = None
    pixel_scale: float = 0.2  # arcsec
    sb_min: float = 24.5  # surface brightness limit (mag arcsec^-2)
    get_gc_mags: bool = True  # time crunch
    ref_key: str = "JC_V"
    sb_key: str = "JC_V"
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

        # 2. Extract gal_dis_kpc
        # gal_dis_kpc = self.field_data["gal_dis_kpc"]

        # 3. Load Milky Way GC catalogue
        mw_path = os.path.join(self.dat_dir, "mw/gcs/mw_gcs.csv")
        mw_gcs = pd.read_csv(mw_path)

        # 4. Randomize theta if needed
        if self.theta is None:
            self.theta = float(np.random.rand() * 360)

        # 5. Build GC dataset

        bands_path = os.path.join(self.dat_dir, "supplementary/bands.json")
        with open(bands_path, "r") as f:
            bands_dict = json.load(f)
        self.bands_dict = bands_dict

        self.gc_data = self._build_gc_data(mw_gcs)

    # =============================================================
    # Internal GC data builder (your original logic)
    # =============================================================
    def _build_gc_data(self, mw_gcs: pd.DataFrame):

        pos = np.column_stack((mw_gcs["x_gc"], mw_gcs["y_gc"], mw_gcs["z_gc"]))
        vel = np.column_stack((mw_gcs["u"], mw_gcs["v"], mw_gcs["w"]))

        pos_rot = Transforms.rotate(pos, self.theta)
        vel_rot = Transforms.rotate(vel, self.theta)

        dx = pos_rot[:, 0] - self.field_data["gal_dis_kpc"]
        dy = pos_rot[:, 1]
        dz = pos_rot[:, 2]

        gc_dist_kpc = np.sqrt(dx**2 + dy**2 + dz**2)
        gc_dist_pc = gc_dist_kpc * 1000

        # angular units
        R_offset_arcsec = 206265 * (dy / gc_dist_kpc)
        z_offset_arcsec = 206265 * (dz / gc_dist_kpc)

        # get region masking
        pos_mask = Field.positional_masking(dy, dz, self.field_data, "kpc")

        # get galaxy surface brightness at positions of GCs
        ys_pc = dy * 1000
        zs_pc = dz * 1000
        gal_sb_dict = Galaxy_Surface_Brightness.process_gal_sb(
            ys_pc, zs_pc, self.bands, self.bands_dict, self.dat_dir, self.sb_min
        )

        # get gc absolute magntiudes and surface brightness
        # ref_mag = Conversions.app_to_abs(mw_gcs["V"], gc_dist_pc)
        r_sun_mw_kpc = mw_gcs["r_sun"].values
        # r_sun_mw_pc = r_sun_mw_kpc * 1000

        # V-band apparent magntiudes from Baumgardt et al. (2020) are observed values. Need to correct to
        # intrinsic V-band apparent magntiudes using the color excess from Harris (2010).
        # Then can convert to absolute magntiudes
        # Instead of the above I have just taken absolute magntiudes from appendix of Baumgardt et al. (2020)
        # If GC didn't have it I calculated using E(B-V) from Harris (2010).
        # If Gc didn't have assumed no extinction and converted apparent to absolute directly.

        # ref_mag = Conversions.app_to_abs(mw_gcs["V"], r_sun_mw_pc)
        ref_mag = mw_gcs["V_abs"].values

        ages = mw_gcs["age_gyr"].values  # Gyr
        fehs = mw_gcs["feh"].values

        gc_dict = {
            "ID": mw_gcs["cluster_name"].values,
            "JC_V": ref_mag,
            "xyz": pos,
            "vxyz": vel,
            "xyz_rot": pos_rot,
            "vxyz_rot": vel_rot,
            "class": mw_gcs["class"].values,
            "age": ages,
            "feh": fehs,
            "r_sun_mw_kpc": r_sun_mw_kpc,
            "theta": self.theta,
            "gc_dist_kpc": gc_dist_kpc,
            "R_offset_kpc": dy,
            "z_offset_kpc": dz,
            "R_offset_arcsec": R_offset_arcsec,
            "z_offset_arcsec": z_offset_arcsec,
            "pos_mask": pos_mask,
        }

        gc_dict.update(gal_sb_dict)

        if self.get_gc_mags:
            mag_dict = Magntiudes.get_absolute_magnitudes(self.dat_dir, self.bands, self.ref_key, ref_mag)
            sb_dict = Magntiudes.process_gc_sb(mag_dict, self.bands, gc_dist_pc, self.pixel_scale)
            sb_lim = Magntiudes.sb_limits(sb_dict, gal_sb_dict, self.sb_key, self.sb_frac)

            gc_dict.update(mag_dict)
            gc_dict.update(sb_dict)
            gc_dict["sb_mask"] = sb_lim

            ext_dict = Extinction.process_extinction(
                pos_rot, self.field_data["gal_dis_kpc"], self.theta, self.bands, self.bands_dict
            )
            gc_dict.update(ext_dict)

            gc_dict = Extinction.apply_extinction(gc_dict, self.bands)
            gc_dict = Extinction.observation_limit(
                gc_dict, self.bands, self.sb_key, self.sb_min, self.pixel_scale
            )

        return gc_dict


#############################################################################################################


@dataclass
class MW_Observation:
    galaxy: str | int
    dat_dir: str
    bands: dict
    thetas: int | list[float]  # <--- can be int (N views) or list of angles
    pixel_scale: float = 0.2
    sb_min: float = 24.5
    get_gc_mags: bool = True
    ref_key: str = "JC_V"
    sb_key: str = "JC_V"
    sb_frac: float = 1

    observations: list = field(init=False)

    def __post_init__(self):

        # -------------------------------------------------------
        # 1) If user passed an integer → generate that many random θ
        # -------------------------------------------------------
        if isinstance(self.thetas, int):
            N = self.thetas
            self.thetas = list(np.random.rand(N) * 360)

        # Basic sanity for any other input
        if not hasattr(self.thetas, "__iter__"):
            raise ValueError("`thetas` must be a list of angles or an integer.")

        # -------------------------------------------------------
        # 2) Build all Observation_Single objects
        # -------------------------------------------------------
        self.observations = []

        for theta in self.thetas:
            obs = Single_Observation(
                galaxy=self.galaxy,
                dat_dir=self.dat_dir,
                bands=self.bands,
                theta=theta,
                pixel_scale=self.pixel_scale,
                sb_min=self.sb_min,
                get_gc_mags=self.get_gc_mags,
                ref_key=self.ref_key,
                sb_key=self.sb_key,
                sb_frac=self.sb_frac,
            )
            self.observations.append(obs)

    # -------------------------------------------------------
    # Make class behave like a list of Observation_Single
    # -------------------------------------------------------
    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx]
