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

from collections import OrderedDict
from collections.abc import Generator, Iterator

import numpy as np
import torch
from modulus.models import Module
from modulus.utils.zenith_angle import cos_zenith_angle

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
]


class ModulusGraphcast(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """

    Note
    ----


    Parameters
    ----------
    core_model : Module
        Core Module model with loaded weights
    center : torch.Tensor
        Model center normalization tensor of size [26]
    scale : torch.Tensor
        Model scale normalization tensor of size [26]
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.variables = VARIABLES

        lat = np.linspace(90.0, -90.0, 721)
        lon = np.linspace(0, 360, 1440, endpoint=False)
        self.LON, self.LAT = np.meshgrid(lon, lat)

        self.input_coords = OrderedDict(
            {
                "time": np.empty(1),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
                "lat": lat,
                "lon": lon,
            }
        )

        self.output_coords = OrderedDict(
            {
                "time": np.empty(1),
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(VARIABLES),
                "lat": lat,
                "lon": lon,
            }
        )

    def __str__(self) -> str:
        return "modulus_graphcast_73ch"

    def to(self, device: str | torch.device | int) -> PrognosticModel:
        """Move model to device"""
        device = torch.device(device)
        if device.index is None:
            if device.type == "cuda":
                device = torch.device(device.type, torch.cuda.current_device())
            else:
                device = torch.device(device.type, 0)

        super().to(device)
        self.model.to(device)

        return self

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""

        # Download and cache the checkpoint file if needed
        import json
        import tarfile
        import tempfile

        cached_model_file = package.get("model.mdlus")
        cached_icospheres = package.get("icospheres.json")
        cached_static = package.get("static")
        # Use a temporary directory to extract the tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            from pathlib import Path

            local_path = Path(temp_dir)

            # Open the tar file and extract its contents to the temporary directory
            with tarfile.open(cached_model_file, "r") as tar:
                tar.extractall(path=local_path)

            # Check if the checkpoint is valid
            Module._check_checkpoint(local_path)

            # Load model arguments and instantiate the model
            with open(local_path.joinpath("args.json")) as f:
                args = json.load(f)

            args["__args__"]["meshgraph_path"] = cached_icospheres
            args["__args__"]["static_dataset_path"] = cached_static
            model = Module.instantiate(args)

            # Load the model weights
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=model.device
            )
            model.load_state_dict(model_dict)

        model.eval()

        # Load center and std normalizations
        local_center = torch.Tensor(np.load(package.get("global_means.npy")))[
            :, : len(VARIABLES)
        ]
        local_std = torch.Tensor(np.load(package.get("global_stds.npy")))[
            :, : len(VARIABLES)
        ]

        return cls(
            model,
            center=local_center,
            scale=local_std,
        )

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords.copy()
        output_coords["time"] = coords["time"]
        output_coords["lead_time"] = output_coords["lead_time"] + coords["lead_time"]

        x = x.squeeze(1)
        x = (x - self.center) / self.scale
        for (i, t) in enumerate(coords["time"]):
            t = timearray_to_datetime(t + coords["lead_time"])[0]
            # Add cosine zenith angle
            cosz = cos_zenith_angle(t, self.LON, self.LAT)
            cosz = cosz.astype(np.float32)
            z = (
                torch.as_tensor(cosz, device=x.device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            z = torch.cat((x[i : i + 1], z), dim=1)

            # Model forward pass
            x[i : i + 1] = self.model(z)

        x = self.scale * x + self.center
        x = x.unsqueeze(1)
        return x, output_coords

    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        ------
        x : torch.Tensor
        coords : CoordSystem
        """
        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "time":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        x, coords = self._forward(x, coords)

        return x, coords

    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "time":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        yield x, coords

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward is identity operator
            x, coords = self._forward(x, coords)

            # Rear hook
            x, coords = self.rear_hook(x, coords)

            yield x, coords.copy()

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system


        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
