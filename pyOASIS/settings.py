from pyproj import Transformer, CRS
import numpy as np
import math

def convert_coords(obs_x, obs_y, obs_z, to_radians=True):
    """
    New version (10/07/2025)
    Converts Cartesian (ECEF) coordinates to geodetic (longitude, latitude, altitude).
    Currently used only for receiver positions.
    Input parameters must be in meters [m].

    Example:
    >>> obs_x, obs_y, obs_z = 5043729.726, -3753105.556, -1072967.067
    >>> lon, lat, alt = convert_coords(obs_x, obs_y, obs_z)
    >>> print(lon, lat, alt)
    -36.6534, -9.7492, 266.23012
    """

    # Define source and target CRS (ECEF → Geodetic)
    crs_from = CRS(proj='geocent', ellps='WGS84', datum='WGS84')
    crs_to = CRS(proj='latlong', ellps='WGS84', datum='WGS84')

    # Create the transformer
    transformer = Transformer.from_crs(crs_from, crs_to)

    # Transform coordinates
    lon, lat, alt = transformer.transform(
        xx=obs_x,
        yy=obs_y,
        zz=obs_z,
        radians=False
    )

    if to_radians:
        # Normalize longitude to [0, 360) before converting to radians
        lon = np.radians(lon % 360.0)
        lat = np.radians(lat)

    return lon, lat, alt


# ============================================================
# Global Constants
# ============================================================
Re = 6371.0  # Earth radius [km]
hm = 450.0   # Ionospheric shell height [km]
dtor = 0.0174532925199433  # Degrees to radians conversion


class IonosphericPiercingPoint(object):
    """
    Computes Ionospheric Piercing Point (IPP) coordinates
    and related geometry using a single-layer (thin-shell) model.
    """

    def __init__(self, sat_x, sat_y, sat_z, obs_x, obs_y, obs_z):
        """
        Initialize with satellite and receiver positions (ECEF).
        Satellite positions are assumed to be in kilometers,
        while receiver positions are converted from meters to kilometers.
        """

        # Convert receiver coordinates from meters to kilometers
        obs_x /= 1000.0
        obs_y /= 1000.0
        obs_z /= 1000.0

        # ⚠️ Do NOT convert satellite coordinates — they are already in kilometers

        # Compute relative positions
        self.dx = sat_x - obs_x
        self.dy = sat_y - obs_y
        self.dz = sat_z - obs_z

        # Cross products (used in geometric computations)
        self.dxdy = (sat_x * obs_y - sat_y * obs_x)
        self.dxdz = (sat_x * obs_z - sat_z * obs_x)

    def positions(self, height="top"):
        """
        Compute the IPP position in Cartesian coordinates
        for a given shell height (default: Re + hm).
        """

        if height == "top":
            h = Re + hm  # 6371 + 450 = 6821 km
        else:
            h = Re + 300.0

        arg_1 = (self.dy * self.dxdy + self.dz * self.dxdz) ** 2
        arg_2 = self.dx**2 + self.dy**2 + self.dz**2
        arg_3 = self.dxdy**2 + self.dxdz**2 - self.dx**2 * h**2
        arg_4 = self.dy**2 + self.dz**2 + self.dx**2

        factor = np.sqrt(arg_1 - arg_2 * arg_3)

        sub_ion_x = (-1 * (self.dy * self.dxdy + self.dz * self.dxdz) - factor) / arg_4
        sub_ion_y = (self.dy * sub_ion_x + self.dxdy) / self.dx
        sub_ion_z = (self.dz * sub_ion_x + self.dxdz) / self.dx

        return sub_ion_x, sub_ion_y, sub_ion_z

    def relative_directions(self, lat, lon):
        """
        Compute relative direction vectors (meridional, zonal, vertical)
        based on receiver latitude and longitude (in radians).
        """

        meridional = (-np.cos(lon) * np.sin(lat) * self.dx
                      - np.sin(lon) * np.sin(lat) * self.dy
                      + np.cos(lat) * self.dz)

        zonal = -np.sin(lon) * self.dx + np.cos(lon) * self.dy

        vertical = (np.cos(lon) * np.cos(lat) * self.dx
                    + np.sin(lon) * np.cos(lat) * self.dy
                    + np.sin(lat) * self.dz)

        return meridional, zonal, vertical

    def zenital_angle(self, lat, lon):
        """
        Compute the zenith angle between satellite and receiver.
        """

        meridional, zonal, vertical = self.relative_directions(lat, lon)

        arg_1 = (np.cos(lon) * np.cos(lat)) ** 2
        arg_2 = (np.sin(lon) * np.cos(lat)) ** 2
        arg_3 = (np.sin(lat)) ** 2
        vertical_norm = np.sqrt(arg_1 + arg_2 + arg_3)
        r = np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)

        return np.arccos(vertical / (r * vertical_norm))

    def elevation(self, lat, lon):
        """
        Compute elevation angle (in degrees) from the zenith angle.
        """
        zangle = self.zenital_angle(lat, lon)
        return ((np.pi / 2.0) - zangle) / dtor  # radians → degrees

    def azimuth(self, lat, lon):
        """
        Compute azimuth angle (in radians) from relative directions.
        """
        meridional, zonal, vertical = self.relative_directions(lat, lon)
        azimuth_angle = np.arctan2(zonal, meridional)
        if azimuth_angle < 0:
            azimuth_angle += 2.0 * np.pi
        return azimuth_angle

    def zenital_iono_angle(self, lat, lon):
        """
        Compute the zenith angle projection in the ionosphere (ψ).
        """
        el = self.elevation(lat, lon) * dtor  # convert elevation to radians
        return ((np.pi / 2.0) - el -
                np.arcsin((Re / (Re + hm)) * np.cos(el)))

    def coordinates(self, lat, lon):
        """
        Compute the IPP coordinates (in degrees)
        using thin-shell approximation.
        """

        azimuth = self.azimuth(lat, lon)
        zangle_ion = self.zenital_iono_angle(lat, lon)

        lat_ip = np.arcsin(
            np.sin(lat) * np.cos(zangle_ion)
            + np.cos(lat) * np.sin(zangle_ion) * np.cos(azimuth)
        )

        lon_ip = lon + np.arcsin(
            (np.sin(zangle_ion) * np.sin(azimuth)) / np.cos(lat_ip)
        )

        # Normalize longitude to [0, 360)
        lat_ip_deg = np.degrees(lat_ip)
        lon_ip_deg = (np.degrees(lon_ip)) % 360.0

        return lat_ip_deg, lon_ip_deg



def mapfun(elevation_deg, re_km=6371.0, h_km=450.0):
    """
    Calculates the geometric mapping factor from sTEC to vTEC.
    Equivalent to MapFun.for in Fortran.

    :param elevation_deg: satellite elevation angle in degrees
    :param re_km: Earth's radius (default: 6371 km)
    :param h_km: ionospheric layer height (default: 450 km)
    :return: mapping factor (>= 1.0)
    """
    d2r = np.pi / 180.0
    mu = re_km / (re_km + h_km)
    return np.cos(np.arcsin(mu * np.cos(elevation_deg * d2r)))

# print(f"MapFun(30°) = {mapfun(30):.6f}")  # Should be around 1.1547

def cholesky_solve(a, y):
    n = len(y)
    expected_len = n * (n + 1) // 2
    if len(a) != expected_len:
        raise ValueError(f"❌ Incorrect vector 'a' size. Expected {expected_len}, but got {len(a)} for n={n}.")

    x = [0.0] * n
    a = a.copy()  # overwrite in-place like Fortran
    y = y.copy()

    def idx(i, j):
        return j * (j + 1) // 2 + i  # Fortran-style 1D indexing of lower triangle

    # Factorization step
    a[idx(0, 0)] = math.sqrt(a[idx(0, 0)])
    for j in range(1, n):
        a[idx(0, j)] = a[idx(0, j)] / a[idx(0, 0)]

    for i in range(1, n):
        sum_ = sum(a[idx(m, i)] ** 2 for m in range(i))
        diag_val = a[idx(i, i)] - sum_
        if diag_val <= 0:
            raise ValueError(f"Non-positive pivot detected at row {i+1}: {diag_val:.6e}")
        a[idx(i, i)] = math.sqrt(diag_val)
        for j in range(i + 1, n):
            sum_ = sum(a[idx(m, i)] * a[idx(m, j)] for m in range(i))
            a[idx(i, j)] = (a[idx(i, j)] - sum_) / a[idx(i, i)]

    # Forward substitution (solve L*y = b)
    y[0] = y[0] / a[idx(0, 0)]
    for i in range(1, n):
        sum_ = sum(a[idx(m, i)] * y[m] for m in range(i))
        y[i] = (y[i] - sum_) / a[idx(i, i)]

    # Backward substitution (solve L^T*x = y)
    x[n - 1] = y[n - 1] / a[idx(n - 1, n - 1)]
    for i in range(n - 2, -1, -1):
        sum_ = sum(a[idx(i, m)] * x[m] for m in range(i + 1, n))
        x[i] = (y[i] - sum_) / a[idx(i, i)]

    return x

