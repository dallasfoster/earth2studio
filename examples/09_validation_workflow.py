# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from math import ceil
from typing import Any, Dict

import numpy as np
import torch
from loguru import logger
from modulus.distributed import DistributedManager
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.statistics import Metric
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array

# %%
"""
Running Validation Inference
========================

This example will demonstrate how to run a validation inference workflow to
generate errors of a forecast against a datasource. This example will be using
one of the built in models of Earth-2 Inference Studio.

In this example you will learn:

- How to instantiate a built in prognostic model
- Creating a custom data source
- Construct a metric
"""

# %%
# Set Up
# ------
def run(
    times: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    initial_condition_data: DataSource,
    validation_data: DataSource,
    metrics: Dict[str, Metric],
    output_coords: CoordSystem = OrderedDict({}),
    output_directory: str = os.path.join(os.curdir, "outputs"),
    time_batch_size: int = None,
    io_kwargs: dict[str, Any] = {},
) -> dict[str, ZarrBackend]:
    """Simple ensemble workflow

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    prognostic : PrognosticModel
        Prognostic models
    data : DataSource
        Data source
    metrics : List[Metrics]
        list_of_metric_options
    output_directory : str
        Parent directory to store outputs.
    device : Optional[torch.device], optional
        Device to run inference on, by default None

    Returns
    -------
    Dict[IOBackend]
        Output IO object
    """
    logger.info("Running validation workflow!")

    # Load model onto the device
    DistributedManager.initialize()
    manager = DistributedManager()
    rank = int(manager.rank)
    world_size = int(manager.world_size)
    device = manager.device
    logger.info(f"Inference rank {rank} out of world size {world_size}")
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)

    # Set up IO backend (create backend to hold data in memory.)
    local_times = to_time_array(times)
    local_times = np.array_split(local_times, world_size)[rank]
    total_coords = OrderedDict(
        {
            "time": local_times,
            "lead_time": np.asarray(
                [
                    np.array([np.timedelta64(0, "h")])
                    + prognostic.output_coords["lead_time"] * i
                    for i in range(nsteps + 1)
                ]
            ).flatten(),
        }
    )
    for coord in prognostic.output_coords:
        if coord not in ["batch", "time", "lead_time"]:
            total_coords[coord] = prognostic.output_coords[coord]

    for key, value in total_coords.items():
        total_coords[key] = output_coords.get(key, value)

    variables_to_save = total_coords.pop("variable")

    # Set up validation backends
    ios = {}
    for name, metric in metrics.items():
        val_coords = OrderedDict(
            {
                k: total_coords[k]
                for k in total_coords
                if k not in metric.reduction_dimensions
            }
        )
        kwargs = io_kwargs.get(str(metric), {})
        kwargs["file_name"] = os.path.join(
            output_directory,
            f"rank_{rank}_" + kwargs.get("file_name", name + ".zarr"),
        )
        ios[name] = ZarrBackend(**kwargs)
        ios[name].add_array(val_coords, variables_to_save)

    # Calculate batches

    if time_batch_size is None:
        time_batch_size = len(local_times)
    time_batch_size = min(len(local_times), time_batch_size)
    number_of_batches = ceil(len(local_times) / time_batch_size)

    logger.info(
        f"Rank {rank}: Starting to execute {number_of_batches} number of batches"
    )
    for time_index in tqdm(
        range(0, len(local_times), time_batch_size), desc="Running batch time inference"
    ):
        logger.info(
            f"Rank {rank}: Starting to execute times {time_index}:{time_index+time_batch_size}"
        )
        time = local_times[time_index : time_index + time_batch_size]
        # Fetch data from data source and load onto device
        x, coords = fetch_data(
            source=initial_condition_data,
            time=time,
            lead_time=prognostic.input_coords["lead_time"],
            variable=prognostic.input_coords["variable"],
            device=device,
        )
        logger.success(f"Rank {rank}: Fetched data from {data.__class__.__name__}")

        # Map lat and lon if needed
        x, coords = map_coords(x, coords, prognostic.input_coords)

        # Create prognostic iterator
        model = prognostic.create_iterator(x, coords)

        logger.info(f"Rank {rank}: Inference starting!")
        with tqdm(total=nsteps + 1, desc="Running inference") as pbar:
            for step, (x, coords) in enumerate(model):
                # Compute Metrics
                y, y_coords = fetch_data(
                    source=validation_data,
                    time=coords["time"] + coords["lead_time"],
                    variable=coords["variable"],
                    device=device,
                )
                for name, metric in metrics.items():
                    out, out_coords = metric(x, coords, y, y_coords)
                    out, out_coords = map_coords(out, out_coords, output_coords)
                    ios[name].write(*split_coords(out, out_coords))
                pbar.update(1)
                if step == nsteps:
                    break

    logger.success(f"Rank {rank}: Inference complete")
    return ios


# It's clear we need the following:
# - Prognostic Model: Use the built in FourCastNetv2 Model.
# - Datasource: Create our own datasource from file (EOS).
# - Metric: We will use a built-in rmse to compute lat/lon rmse.
#
# %%
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.data import DataSetFile
from earth2studio.models.px import SFNO
from earth2studio.statistics import lat_weight, rmse

# Load the default model package which downloads the check point from NGC
model = SFNO.load_model(SFNO.load_default_package())

# Create the data source
initial_data = DataSetFile(os.environ["dataset_path"], "fields")
val_data = DataSetFile(os.environ["dataset_path"], "fields")

# Construct the metric
metrics = {
    'rmse_lat_lon': rmse(
        reduction_dimensions=["lat", "lon"],
        weights=torch.as_tensor(lat_weight(model.input_coords["lat"]), device="cuda:0")
        .unsqueeze(1)
        .repeat(1, 1440),
    ),
    'rmse_map': rmse(reduction_dimensions=["time"], batch_update=True),
}

# Construct the kwargs for passing to the metric IOBackend (zarr)
io_kwargs = {
    list(metrics)[0]: {
        "file_name": "lat_lon_rmse.zarr",
        "chunks": {"time": 1, "lead_time": 1},
    },
    list(metrics)[1]: {
        "file_name": "rmse_map.zarr",
        "chunks": {"lead_time": 1},
    },
}
output_directory = os.path.join(
    os.environ.get("output_data_path", "./output/scoring"), "sfno_73ch_small"
)

# %%
# Execute the Workflow
# --------------------
# With all componments intialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the validation forecast we will predict for 15 days and 1400 time stamps covering
# the year 2018. (these will get executed as a batch)
# %%

# Construct the Times over which to perform inference

times = [datetime(2018, 1, 1) + timedelta(hours=6) * i for i in range(1400)]
time_batch_size = 16

# Length of simulation
nsteps = 60

io = run(
    times,
    nsteps,
    model,
    initial_data,
    val_data,
    metrics,
    output_coords={"variable": np.array(["t2m", "tcwv", "z500", "t850"])},
    output_directory=output_directory,
    io_kwargs=io_kwargs,
    time_batch_size=time_batch_size,
)

