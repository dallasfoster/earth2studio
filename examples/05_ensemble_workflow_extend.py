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

# %%
"""
Single Variable Perturbation Method
===================================

Intermediate ensemble inference using a custom perturbation method.

This example will demonstrate how to run a an ensemble inference workflow
with a custom perturbation method that only applies noise to a specific variable.

In this example you will learn:

- How to extend an existing pertubration method
- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Running a simple built in workflow
- Extend a built-in method using custom code.
- Post-processing results
"""

# %%
# Set Up
# ------
# All workflows inside Earth2Studio require constructed components to be
# handed to them. In this example, we will use the built in ensemble workflow
# :py:meth:`earth2studio.run.ensemble`.

# %%
# .. literalinclude:: ../../earth2studio/run.py
#    :language: python
#    :lines: 116-156

# %%
# We need the following:
#
# - Prognostic Model: Use the built in DLWP model :py:class:`earth2studio.models.px.DLWP`.
# - perturbation_method: Extend the Spherical Gaussian Method :py:class:`earth2studio.perturbation.SphericalGaussian`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import DLWP
from earth2studio.perturbation import PerturbationMethod, SphericalGaussian
from earth2studio.run import ensemble
from earth2studio.utils.type import CoordSystem

# Load the default model package which downloads the check point from NGC
package = DLWP.load_default_package()
model = DLWP.load_model(package)

# Create the data source
data = GFS()

# %%
# The perturbation method in :ref:`sphx_glr_examples_03_ensemble_workflow.py` is naive because it
# applies the same noise amplitude to every variable. We can create a custom wrapper
# that only applies the perturbation method to a particular variable instead.

# %%
class ApplyToVariable:
    """Apply a perturbation to only a particular variable."""

    def __init__(self, pm: PerturbationMethod, variable: str | list[str]):
        self.pm = pm
        if isinstance(variable, str):
            variable = [variable]
        self.variable = variable

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        # Construct perturbation
        dx, coords = self.pm(x, coords)
        # Find variable in data
        ind = np.in1d(coords["variable"], self.variable)
        dx[..., ~ind, :, :] = 0.0
        return dx, coords


# Generate a new noise amplitude that specifically targets 't2m' with a 1 K noise amplitude
avsg = ApplyToVariable(SphericalGaussian(noise_amplitude=1.0), "t2m")

# Create the IO handler, store in memory
chunks = {"ensemble": 1, "time": 1}
io = ZarrBackend(file_name="outputs/04_ensemble_avsg.zarr", chunks=chunks)

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the forecast we will predict for 10 steps (for FCN, this is 60 hours) with 8 ensemble
# members which will be ran in 2 batches with batch size 4.

# %%
nsteps = 10
nensemble = 8
batch_size = 4
io = ensemble(
    ["2024-01-01"],
    nsteps,
    nensemble,
    model,
    data,
    io,
    avsg,
    batch_size=batch_size,
    output_coords={"variable": np.array(["t2m", "tcwv"])},
)

# %%
# Post Processing
# ---------------
# The last step is to post process our results.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast = "2024-01-01"


def plot_(axi, data, title, cmap):
    """Convenience function for plotting pcolormesh."""
    # Plot the field using pcolormesh
    im = axi.pcolormesh(
        io["lon"][:],
        io["lat"][:],
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )
    plt.colorbar(im, ax=axi, shrink=0.6, pad=0.04)
    # Set title
    axi.set_title(title)
    # Add coastlines and gridlines
    axi.coastlines()
    axi.gridlines()


for variable, cmap in zip(["t2m", "tcwv"], ["coolwarm", "Blues"]):
    step = 4  # lead time = 24 hrs

    plt.close("all")
    # Create a Robinson projection
    projection = ccrs.Robinson()

    # Create a figure and axes with the specified projection
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, subplot_kw={"projection": projection}, figsize=(16, 3)
    )

    plot_(
        ax1,
        io[variable][0, 0, step],
        f"{forecast} - Lead time: {6*step}hrs - Member: {0}",
        cmap,
    )
    plot_(
        ax2,
        io[variable][1, 0, step],
        f"{forecast} - Lead time: {6*step}hrs - Member: {1}",
        cmap,
    )
    plot_(
        ax3,
        np.std(io[variable][:, 0, step], axis=0),
        f"{forecast} - Lead time: {6*step}hrs - Std",
        cmap,
    )

    plt.savefig(f"outputs/04_{forecast}_{variable}_{step}_ensemble.jpg")