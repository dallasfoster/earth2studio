# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import numpy as np
import torch
import xarray as xr
from cucim.skimage.measure import label
from metpy.calc import vorticity

from earth2studio.models.dx import ClimateNet
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.type import CoordSystem

from .utils import haversine_torch

VARIABLES = ["tcwv", "u850", "v850", "msl", "u10m", "v10m", "z500", "z850", "z200"]

OUT_VARIABLES = ["lat", "lon", "mslp", "w10m"]


class TropicalCycloneTracker:
    """_summary_

    Parameters
    ----------
    mslp_threshold : float, optional
        _description_, by default 990
    radius_threshold : float, optional
        _description_, by default 278
    vorticity_threshold : float, optional
        _description_, by default 5e-5
    w10m_threshold : float, optional
        _description_, by default 8
    """

    def __init__(
        self,
        mslp_threshold: float = 990,
        radius_threshold: float = 278,
        vorticity_threshold: float = 5e-5,
        w10m_threshold: float = 8,
    ):
        self.climatenet = ClimateNet.load_model(ClimateNet.load_default_package())

        self.mslp_threshold = mslp_threshold
        self.radius_threshold = radius_threshold
        self.vorticity_threshold = vorticity_threshold
        self.w10m_threshold = w10m_threshold

    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            _description_
        coords : CoordSystem
            _description_

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            _description_
        """
        x_climatenet, coords_climatenet = map_coords(
            x, coords, self.climatenet.input_coords
        )
        self.climatenet.to(x.device)
        out, _ = self.climatenet(x_climatenet, coords_climatenet)

        mask = out[..., 1, :, :]
        lat = torch.as_tensor(coords["lat"], device=x.device)
        lon = torch.as_tensor(coords["lon"], device=x.device)

        centers, mslp, w10m = self._get_tc_data(lat, lon, mask, x, coords)

        out = torch.cat((centers, mslp, w10m), dim=-1)
        output_coords = coords.copy()
        output_coords.pop("variable")
        output_coords.pop("lat")
        output_coords.pop("lon")
        output_coords["event"] = np.arange(out.shape[-2])
        output_coords["variable"] = np.array(OUT_VARIABLES)

        return out, output_coords

    def _get_tc_data(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        mask: torch.Tensor,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Utility function for looping over input dimensions to collect Tropical Cyclone information."""

        if mask.ndim > 2:
            centers = []
            mslp = []
            w10m = []
            reduced_coords = coords.copy()
            reduced_coords.pop(list(reduced_coords)[0])
            for i in range(mask.shape[0]):
                c, m, w = self._get_tc_data(lat, lon, mask[i], x[i], reduced_coords)
                centers.append(c)
                mslp.append(m)
                w10m.append(w)

            centers = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(centers), float("nan")
            )

            mslp = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(mslp), float("nan")
            )

            w10m = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(w10m), float("nan")
            )

            return centers, mslp, w10m
        else:
            data, _, variables = split_coords(x, coords, dim="variable")
            variables = list(variables)
            centers, mslp, w10m = find_centers(
                mask,
                lat,
                lon,
                data[variables.index("msl")],
                data[variables.index("u850")],
                data[variables.index("v850")],
                data[variables.index("z850")],
                data[variables.index("z200")],
                data[variables.index("u10m")],
                data[variables.index("v10m")],
                mslp_threshold=self.mslp_threshold,
                radius_threshold=self.radius_threshold,
                vorticity_threshold=self.vorticity_threshold,
                w10m_threshold=self.w10m_threshold,
            )

        return centers, mslp, w10m


def find_centers(
    mask: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    msl: torch.Tensor,
    u850: torch.Tensor,
    v850: torch.Tensor,
    z850: torch.Tensor,
    z200: torch.Tensor,
    u10m: torch.Tensor,
    v10m: torch.Tensor,
    mslp_threshold: float = 990,
    radius_threshold: float = 278,
    vorticity_threshold: float = 5e-5,
    w10m_threshold: float = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the tropical cyclone centers given a set of
    geophysical variables.


    Note
    ----
    This algorithm is derived from the algorithm given in
    https://www.ecmwf.int/en/elibrary/78231-newsletter-no-102-winter-200405

    Parameters
    ----------
    mask : torch.Tensor
        0-1 Mask of possible tropical cyclones, typical derived from climatenet
    lat : torch.Tensor
        Latitudes defining the mask/variables arrays.
    lon : torch.Tensor
        Longitudes defining the mask/variables arrays.
    msl : torch.Tensor
        Mean Sea Level Pressure
    u850 : torch.Tensor
        u-winds at 850 hPa. See earth2stuio/lexicon
    v850 : torch.Tensor
        v-winds at 850 hPa. See earth2stuio/lexicon
    z850 : torch.Tensor
        geopotential at 850 hPa. See earth2stuio/lexicon
    z200 : torch.Tensor
        geopotential at 200 hPa. See earth2stuio/lexicon
    u10m : torch.Tensor
        u-winds at 10 meters. See earth2stuio/lexicon
    v10m : torch.Tensor
        v-winds at 10 meters. See earth2stuio/lexicon
    mslp_threshold : float, optional
        Mean sea level threshold (in hPa) for hurricane definition, by default 990 hPa
    radius_threshold : float, optional
        Radius for separating tropical cyclones (in km), by default 278 km
    vorticity_threshold : float, optional
        vorticity threshold for hurricane, by default 5e-5 1/s
    w10m_threshold : float, optional
        10 meter wind speed threshold for hurricane (in m/s), by default 8 m/s

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Return center positions (lat/lon), mean sea level pressure and wind speeds
    """
    # Create meshgrid
    LON, LAT = torch.meshgrid(lon, lat, indexing="xy")

    # Find labels
    mask = cp.asarray(mask)
    labels, num = label(mask, return_num=True)
    labels = torch.as_tensor(labels, device=lat.device)
    tc_centers = []
    tc_mslp = []
    tc_w10m = []
    # Loop over all positive labels (starting at i = 1)
    for possible_locations in range(1, num + 1):

        # Get mask indices
        mask_inds = labels == possible_locations
        # Get lon/lat for this subsection
        mask_lon, mask_lat = LON[mask_inds], LAT[mask_inds]

        # Get msl for this subsection and find minimum
        mask_msl = msl[mask_inds]
        msl_min_ind = torch.argmin(mask_msl)

        msl_min_lon, msl_min_lat = mask_lon[msl_min_ind], mask_lat[msl_min_ind]
        tc_center = [msl_min_lat, msl_min_lon]

        # Threshold MSL
        if mask_msl[msl_min_ind] / 100.0 > mslp_threshold:
            continue

        # Get Northern Hemisphere Flag
        NH_flag = msl_min_lat >= 0.0

        # Get radius around possible center
        valid_inds = torch.where(
            haversine_torch(tc_center[0], tc_center[1], mask_lat, mask_lon)
            < radius_threshold
        )[0]

        # Compute vorticity
        # Find vorticity (use metpy)
        uwind = xr.DataArray(
            data=u850.cpu().numpy(),
            dims=["lat", "lon"],
            coords=dict(lat=lat.cpu().numpy(), lon=lon.cpu().numpy()),
            attrs=dict(units="m s-1"),
        )
        vwind = xr.DataArray(
            data=v850.cpu().numpy(),
            dims=["lat", "lon"],
            coords=dict(lat=lat.cpu().numpy(), lon=lon.cpu().numpy()),
            attrs=dict(units="m s-1"),
        )
        mask_vort = torch.as_tensor(
            vorticity(uwind, vwind).values, device=labels.device, dtype=torch.float32
        )[mask_inds]

        vort_ext_ind = torch.argmax(mask_vort) if NH_flag else torch.argmin(mask_vort)
        vort_ext = mask_vort[vort_ext_ind]
        vort_threshold = (
            (vort_ext > vorticity_threshold)
            if NH_flag
            else (vort_ext < -vorticity_threshold)
        )
        if not ((vort_ext_ind in valid_inds) and (vort_threshold)):
            continue

        # Calculate z850 - z200 thickness
        mask_z850 = z850[mask_inds]
        mask_z200 = z200[mask_inds]
        dz = mask_z200 - mask_z850
        dz_max = torch.argmax(dz)
        if dz_max not in valid_inds:
            continue

        # Calculate 10m wind speed
        mask_u10m = u10m[mask_inds]
        mask_v10m = v10m[mask_inds]
        w10m = torch.sqrt(mask_u10m**2 + mask_v10m**2)
        w10m_max_ind = torch.argmax(w10m)
        w10m_threshold = w10m[w10m_max_ind] > w10m_threshold
        if not ((w10m_max_ind in valid_inds) and w10m_threshold):
            continue

        tc_centers.append(tc_center)
        tc_mslp.append(mask_msl[msl_min_ind])
        tc_w10m.append(w10m[w10m_max_ind])

    return (
        torch.as_tensor(tc_centers, device=labels.device).reshape(-1, 2),
        torch.as_tensor(tc_mslp, device=labels.device).reshape(-1, 1),
        torch.as_tensor(tc_w10m, device=labels.device).reshape(-1, 1),
    )


if __name__ == "__main__":

    from earth2studio.data import GFS, fetch_data
    from earth2studio.utils.time import to_time_array

    gfs = GFS()
    x, coords = fetch_data(
        gfs,
        to_time_array(["2023-09-08T00:00:00", "2023-09-08T06:00:00"]),
        variable=np.array(VARIABLES),
        device="cuda:0",
    )

    tc = TropicalCycloneTracker()

    y, y_coords = tc(x, coords)
    print(y.shape, list(y_coords))
    print(y, y_coords)
