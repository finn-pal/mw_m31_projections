import numpy as np

#############################################################################################################


class Conversions:
    """
    Useful set of functions that can be used for conversions
    """

    @staticmethod
    def arcsec_deg(arcsec):
        return arcsec / 3600

    @staticmethod
    def au_to_pc(au):
        return au / 206265

    @staticmethod
    def abs_to_app(m_abs, d_pc):
        m_app = m_abs + 5 * np.log10(d_pc / 10)
        return np.round(m_app, 2)

    @staticmethod
    def app_to_abs(m_app, d_pc):
        m_abs = m_app - 5 * np.log10(d_pc / 10)
        return np.round(m_abs, 2)


#############################################################################################################


class Transforms:
    """
    Useful set of functions that can be used for coordinate transformations
    """

    @staticmethod
    def rotate(vector, theta, inc=90):
        inc = np.deg2rad(90 - inc)  # correct so that 90deg is edge on
        theta = np.deg2rad(theta)

        cy = np.cos(inc)
        sy = np.sin(inc)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

        cz = np.cos(theta)
        sz = np.sin(theta)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R = Ry @ Rz
        vector_transform = (R @ vector.T).T

        return vector_transform

    @staticmethod
    def cartesian_to_cylindrical(x, y, z):
        """
        Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (R, phi, z)
        Supports scalar inputs or numpy arrays
        """
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        R = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)  # angle in radians, correct quadrant
        return R, phi, z
