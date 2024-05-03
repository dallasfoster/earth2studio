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

import torch

from earth2studio.utils.type import CoordSystem


def _broadcast_weights(
    weights: torch.Tensor, rd: list[str], coords: CoordSystem
) -> torch.Tensor:
    """
    Broadcast weights to appropriate dimensions. This is meant for internal use.

    Parameters
    ----------
    weights : torch.Tensor
        A tensor contains weights to broadcast to shape of coords. Can also
        be None, in which a 1.0 is returned.
    rd : List[str]
        A list of dimension names corresponding to the dimensions of weights.
    coords : CoordSystem
        An ordered dict containing dimensions and coordinates that the weights
        will be broadcasted to.
    """
    if weights is None:
        return torch.ones([len(v) if c in rd else 1 for c, v in coords.items()])
    else:
        dim_shapes = [len(v) for c, v in coords.items() if c in rd]
        if list(weights.shape) != dim_shapes:
            raise AssertionError(
                "Error, weights have don't have an appropriate shape.\n"
                + f"weights have shape {weights.shape} but should have shape {dim_shapes}."
            )
        rs = [slice(len(v)) if c in rd else None for c, v in coords.items()]
        return weights[rs]


def haversine_torch(
    lat1: torch.Tensor, lon1: torch.Tensor, lat2: torch.Tensor, lon2: torch.Tensor
) -> torch.Tensor:
    """Utility function for computing the haversine distance between points.

    Parameters
    ----------
    lat1: torch.Tensor
        Latitudes for point 1
    lon1: torch.Tensor
        Longitudes for point 1
    lat2: torch.Tensor
        Latitudes for point 2
    lon2: torch.Tensor
        Longitudes for point 2

    Returns
    -------
    torch.Tensor
        Tensor of haversine distances
    """
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    return (
        2
        * 6371
        * torch.arcsin(
            torch.sqrt(
                torch.sin(dlat / 2) ** 2
                + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
            )
        )
    )
