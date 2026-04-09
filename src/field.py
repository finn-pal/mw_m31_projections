import astropy.units as u
import numpy as np
import pandas as pd
from regions import PixCoord, RectanglePixelRegion

from utils import Conversions

#############################################################################################################


class Field:
    """
    Functions used in obtaining field of observation
    """

    @staticmethod
    def process_fields(
        gal: str | int,
        df_poi: pd.DataFrame,
        regions: list[str],
        field_dict: dict,
        archival: bool = False,
        view_mode: str = "wide",
    ):
        # muse field size in arcsec
        muse_field = {"narrow": 7.5, "wide": 60}  # arcsec

        # galaxy distance in kpc
        gal_dis_kpc = field_dict["gal_dis_kpc"]

        for region in regions:
            # skip if no objects (for non-archival)
            if not archival:
                region_cnt = df_poi.loc[df_poi["object"] == gal, "nob_" + region].values[0]
                if region_cnt == 0:
                    continue

            # get x,y offsets in arcsec
            xoff_arcsec = df_poi.loc[df_poi["object"] == gal, "xoff_" + region].values[0]
            yoff_arcsec = df_poi.loc[df_poi["object"] == gal, "yoff_" + region].values[0]

            # convert to radians then kpc (small-angle approximation)
            xoff_rad = np.deg2rad(Conversions.arcsec_deg(xoff_arcsec))
            yoff_rad = np.deg2rad(Conversions.arcsec_deg(yoff_arcsec))

            # pointing centre (kpc)
            x_point = gal_dis_kpc * xoff_rad
            y_point = gal_dis_kpc * yoff_rad

            # view size in kpc
            view_size = gal_dis_kpc * np.deg2rad(Conversions.arcsec_deg(muse_field[view_mode]))

            # bounding box (kpc)
            x_l = x_point - view_size / 2
            x_u = x_point + view_size / 2
            y_l = y_point - view_size / 2
            y_u = y_point + view_size / 2

            # centre array (kpc)
            centre = np.array([x_point, y_point])

            # box corners before rotation (kpc)
            corners_norot = np.array(
                [
                    [x_l, y_l],
                    [x_u, y_l],
                    [x_u, y_u],
                    [x_l, y_u],
                    [x_l, y_l],
                ]
            )

            corners_shifted = corners_norot - centre

            # get PA
            if not archival:
                poi_pa = df_poi.loc[df_poi["object"] == gal, "pa_diamond"].values[0]
            else:
                poi_pa = df_poi.loc[df_poi["object"] == gal, "pa_" + region].values[0]

            # rotation matrix
            theta = np.deg2rad(poi_pa)
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            # rotated corners (kpc)
            corners_rot = corners_shifted @ R.T
            corners_rot += centre

            # ----------------------------------------------------
            # NEW: rotated geometry converted into arcsec
            # ----------------------------------------------------
            corners_rot_arcsec = (corners_rot / gal_dis_kpc) * 206265
            centre_arcsec = (centre / gal_dis_kpc) * 206265

            # ----------------------------------------------------
            # Store all required output
            # ----------------------------------------------------
            field_dict["regions"][region] = {
                "centre_kpc": centre,  # kpc
                "centre_arcsec": centre_arcsec,  # arcsec
                "corners_rot_kpc": corners_rot,  # kpc
                "corners_rot_arcsec": corners_rot_arcsec,  # arcsec
                "pa": poi_pa,
                "box_length": view_size,
            }

        return field_dict

    @staticmethod
    def get_field(gal: str | int, gal_dir: str, get_archival: bool = True, view_mode: str = "wide"):
        # get galaxy information and pointing details
        df_gal = pd.read_csv(gal_dir + "galaxy_details.csv")

        if type(gal) is int:
            gal = df_gal.loc[df_gal["gid"] == gal + 1, "object"].values[0]

        # get galaxy details
        gal_dis_mpc = df_gal.loc[df_gal["object"] == gal, "d_mpc"].values[0]
        gal_dis_kpc = gal_dis_mpc * 1000

        field_dict = {"gal": gal, "gal_dis_kpc": gal_dis_kpc, "regions": {}}

        # different posible regions of observation
        regions = ["central", "disk", "outflow", "ext", "far", "giant", "huge"]

        df_poi = pd.read_csv(gal_dir + "pointings.csv")
        df_poi.columns = df_poi.columns.str.replace(r"\s+", "", regex=True)
        df_poi["object"] = df_poi["object"].str.replace(" ", "", regex=False)

        field_dict = Field.process_fields(
            gal, df_poi, regions, field_dict, archival=False, view_mode=view_mode
        )

        if get_archival:
            df_poi_arc = pd.read_csv(gal_dir + "pointings_archival.csv")
            df_poi_arc.columns = df_poi_arc.columns.str.replace(r"\s+", "", regex=True)
            df_poi_arc["object"] = df_poi_arc["object"].str.replace(" ", "", regex=False)

            arc_cnt = df_poi_arc.loc[df_poi_arc["object"] == gal, "nob_archival"].values[0]

            if arc_cnt > 0:
                regions_arc = ["a" + str(i) for i in range(1, arc_cnt + 1)]

                field_dict = Field.process_fields(
                    gal, df_poi_arc, regions_arc, field_dict, archival=True, view_mode=view_mode
                )

        return field_dict

    @staticmethod
    def positional_masking(r_view: np.ndarray, z_view: np.ndarray, field_dict: dict, scale: str = "kpc"):
        pos_msk = np.zeros(len(r_view), dtype=bool)
        for region in field_dict["regions"]:
            centre = field_dict["regions"][region]["centre_" + scale]
            box_length = field_dict["regions"][region]["box_length"]
            angle = field_dict["regions"][region]["pa"]

            box_centre = PixCoord(*centre)
            rect = RectanglePixelRegion(
                center=box_centre, width=box_length, height=box_length, angle=angle * u.deg
            )

            positions = PixCoord(x=r_view, y=z_view)
            mask = rect.contains(positions)  # boolean array
            pos_msk |= mask

        return pos_msk
