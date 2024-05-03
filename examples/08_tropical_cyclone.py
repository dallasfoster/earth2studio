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
Statistical Inference
==========================

This example will demonstrate how to run a simple inference workflow to generate a
forecast and then to save a statistic of that data. There are a handful of built-in
statistics available in `earth2studio.statistics`, but here we will demonstrate how
to define a custom statistic and run inference.

In this example you will learn:

- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Create a custom statistic
- Running a simple built in workflow
- Post-processing results
"""

# %%
# Creating a statistical workflow
# -----------------------------------
#
# Start with creating a simple inference workflow to use. We encourage
# users to explore and experiment with their own custom workflows that borrow ideas from
# built in workflows inside :py:obj:`earth2studio.run` or the examples.
#
# Creating our own generalizable workflow to use with statistics is easy when we rely on
# the component interfaces defined in Earth2Studio (use dependency injection). Here we
# create a run method that accepts the following:
#
# - time: Input list of datetimes / strings to run inference for
# - nsteps: Number of forecast steps to predict
# - prognostic: Our initialized prognostic model
# - statistic: our custom statistic
# - data: Initialized data source to fetch initial conditions from
# - io: IOBackend
#
# We do not run an ensemble inference workflow here, even though it is common for statistical
# inference. See ensemble examples for details on how to extend this example for that purpose.

# %%
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend, KVBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.statistics.tc import TropicalCycloneTracker
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


def run_tc(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    tc_tracker: TropicalCycloneTracker,
    data: DataSource,
    io: IOBackend,
) -> IOBackend:
    """Simple statistics workflow

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    prognostic : PrognosticModel
        Prognostic models
    tc_tracker : TropicalCycloneTracker
        Tropical Cyclone Track producing model
    data : DataSource
        Data source
    io : IOBackend
    Returns
    -------
    IOBackend
        Output IO object
    """
    logger.info("Running simple statistics workflow!")
    # Load model onto the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    # Fetch data from data source and load onto device
    time = to_time_array(time)
    x, coords = fetch_data(
        source=data,
        time=time,
        lead_time=prognostic.input_coords["lead_time"],
        variable=prognostic.input_coords["variable"],
        device=device,
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend
    temp_io = KVBackend(device="cpu")

    total_coords = coords.copy()
    total_coords["lead_time"] = np.asarray(
        [prognostic.output_coords["lead_time"] * i for i in range(nsteps + 1)]
    ).flatten()

    temp_io.add_array(total_coords, "fields")

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic.input_coords)

    # Create prognostic iterator
    model = prognostic.create_iterator(x, coords)

    logger.info("Inference starting!")
    with tqdm(total=nsteps + 1, desc="Running inference") as pbar:
        for step, (x, coords) in enumerate(model):
            temp_io.write(x, coords, "fields")
            pbar.update(1)
            if step == nsteps:
                break

    logger.success("Inference complete")
    logger.info("Computing Tropical Cyclone tracks")

    x, coords = io["fields"], io.coords
    tc, tc_coords = tc_tracker(x, coords)
    io.add_array(tc_coords, "track", data=tc)
    return io


# %%
# Set Up
# ------
# With the statistical workflow defined, we now need to create the individual components.
#
# We need the following:
#
# - Prognostic Model: Use the built in Pangu 24 hour model :py:class:`earth2studio.models.px.Pangu24`.
# - statistic: We define our own statistic: the Southern Oscillation Index (SOI).
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a NetCDF4 store :py:class:`earth2studio.io.NetCDF4Backend`.

# %%
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO

# Load the default model package which downloads the check point from NGC
package = SFNO.load_default_package()
model = SFNO.load_model(package)

# Create the data source
data = GFS()

# Create the IO handler, store in memory
io = ZarrBackend(
    file_name="outputs/tropical_cyclone_tracks.zarr",
)

# Instantiate tropical cyclone tracker (see documentation for parameter settings)
tc = TropicalCycloneTracker()

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
# We simulate a trajectory of 60 time steps, or 2 months using Pangu24

# %%
nsteps = 60
nensemble = 1
io = run_tc(["2023-09-08T00:00:00"], nsteps, model, tc, data, io)


# %%
# Post Processing
# ---------------
# The last step is to post process our results.
#
# Notice that the ZarrBackend IO function has additional APIs to interact with the stored data.

# %%

import matplotlib.pyplot as plt

# times = io["time"][:].flatten() + io["lead_time"][:].flatten()

# fig = plt.figure(figsize=(12, 4))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(times, io["soi"][:].flatten())
# ax.set_title("Southern Oscillation Index")
# ax.grid("on")

plt.savefig("outputs/2023-09-08_tropical_cyclone_tracks.png")
io.close()
