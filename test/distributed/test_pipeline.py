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

import numpy as np
import pytest
import torch

from earth2studio.distributed.config import (
    DiagnosticConfig,
    IOConfig,
    PipelineConfig,
    PrognosticConfig,
)
from earth2studio.distributed.pipeline import Pipeline
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px.persistence import Persistence


class IdentityDiagnostic(torch.nn.Module, DiagnosticModel):
    """Simple diagnostic model that returns input unchanged"""

    def __init__(self):
        super().__init__()

    def input_coords(self):
        return OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))])

    def output_coords(self, input_coords):
        return input_coords.copy()

    def __call__(self, x, coords):
        return x, coords


class ScalingDiagnostic(torch.nn.Module, DiagnosticModel):
    """Diagnostic model that scales input by a factor"""

    def __init__(self, scale_factor: float = 2.0):
        super().__init__()
        self.scale_factor = scale_factor

    def input_coords(self):
        return OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))])

    def output_coords(self, input_coords):
        return input_coords.copy()

    def __call__(self, x, coords):
        return self.scale_factor * x, coords


@pytest.fixture
def domain_coords():
    return {"lat": np.arange(10), "lon": np.arange(20)}


@pytest.fixture
def base_coords(domain_coords):
    return OrderedDict(
        {
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(["t2m"]),
            **domain_coords,
        }
    )


@pytest.fixture
def input_tensor(base_coords):
    return torch.randn(1, 1, len(base_coords["lat"]), len(base_coords["lon"]))


@pytest.mark.parametrize(
    "device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
)
class TestPipeline:
    def test_single_prognostic(self, input_tensor, base_coords, domain_coords, device):
        """Test pipeline with single prognostic model."""
        prog_model = Persistence(base_coords["variable"], domain_coords)

        config = PipelineConfig(
            prognostic_stages=[
                PrognosticConfig(
                    name="prog", device=device, model=prog_model, output_stages=["diag"]
                )
            ],
            diagnostic_stages=[
                DiagnosticConfig(
                    name="diag",
                    device=device,
                    model=IdentityDiagnostic(),
                    input_stage="prog",
                    output_stages=["io"],
                )
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device="cpu",
                    backend=ZarrBackend(),
                    input_stage="diag",
                )
            ],
        )

        pipeline = Pipeline(config)
        print("starting pipeline")
        pipeline(input_tensor, base_coords, nsteps=1)
        print("pipeline finished")
        metrics = pipeline.get_metrics()
        print("metrics", metrics)
        assert "prog" in metrics
        assert "diag" in metrics
        assert "io" in metrics

    def test_multiple_diagnostic_chain(
        self, input_tensor, base_coords, domain_coords, device
    ):
        """Test pipeline with multiple diagnostic models in chain."""
        prog_model = Persistence(base_coords["variable"], domain_coords)

        config = PipelineConfig(
            prognostic_stages=[
                PrognosticConfig(
                    name="prog",
                    device=device,
                    model=prog_model,
                    output_stages=["diag1"],
                )
            ],
            diagnostic_stages=[
                DiagnosticConfig(
                    name="diag1",
                    device=device,
                    model=ScalingDiagnostic(2.0),
                    input_stage="prog",
                    output_stages=["diag2"],
                ),
                DiagnosticConfig(
                    name="diag2",
                    device=device,
                    model=ScalingDiagnostic(0.5),
                    input_stage="diag1",
                    output_stages=["io"],
                ),
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device="cpu",
                    backend=ZarrBackend(),
                    input_stage="diag2",
                )
            ],
        )

        pipeline = Pipeline(config)
        pipeline(input_tensor, base_coords, nsteps=1)

        metrics = pipeline.get_metrics()
        assert len(metrics) == 4  # prog + 2 diag + io

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multi_gpu_pipeline(self, input_tensor, base_coords, domain_coords, device):
        """Test pipeline with models on different GPUs."""
        prog_model = Persistence(base_coords["variable"], domain_coords)

        # Create two IO backends for different outputs
        io1 = ZarrBackend()
        io2 = ZarrBackend()

        config = PipelineConfig(
            prognostic_stages=[
                PrognosticConfig(
                    name="prog",
                    device="cuda:0",
                    model=prog_model,
                    output_stages=["diag1", "diag2"],
                )
            ],
            diagnostic_stages=[
                DiagnosticConfig(
                    name="diag1",
                    device="cuda:0",
                    model=ScalingDiagnostic(2.0),
                    input_stage="prog",
                    output_stages=["io1"],
                ),
                DiagnosticConfig(
                    name="diag2",
                    device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0",
                    model=ScalingDiagnostic(0.5),
                    input_stage="prog",
                    output_stages=["io2"],
                ),
            ],
            io_stages=[
                IOConfig(
                    name="io1",
                    device="cpu",
                    backend=io1,
                    input_stage="diag1",
                ),
                IOConfig(
                    name="io2",
                    device="cpu",
                    backend=io2,
                    input_stage="diag2",
                ),
            ],
        )

        pipeline = Pipeline(config)
        pipeline(input_tensor, base_coords, nsteps=1)
        # Check results
        assert np.allclose(io1["t2m"][:] * 0.5, io2["t2m"][:] * 2.0)

    def test_error_handling(self, input_tensor, base_coords, domain_coords, device):
        """Test pipeline error handling."""
        prog_model = Persistence(base_coords["variable"], domain_coords)

        # Test invalid stage name reference
        with pytest.raises(ValueError):
            config = PipelineConfig(
                prognostic_stages=[
                    PrognosticConfig(
                        name="prog",
                        device="cpu",
                        model=prog_model,
                        output_stages=["nonexistent"],
                    )
                ],
                diagnostic_stages=[],
                io_stages=[],
            )
            Pipeline(config)

        # Test invalid memory fraction
        with pytest.raises(ValueError):
            config = PipelineConfig(
                prognostic_stages=[
                    PrognosticConfig(
                        name="prog",
                        device="cpu",
                        model=prog_model,
                        output_stages=[],
                        memory_fraction=1.5,  # Invalid
                    )
                ],
                diagnostic_stages=[],
                io_stages=[],
            )
            Pipeline(config)

    def test_memory_based_queue_sizing(
        self, input_tensor, base_coords, domain_coords, device
    ):
        """Test that queue sizes are set based on memory constraints."""
        if device == "cpu":
            pytest.skip("Memory-based queue sizing only relevant for GPU")

        prog_model = Persistence(base_coords["variable"], domain_coords)

        config = PipelineConfig(
            prognostic_stages=[
                PrognosticConfig(
                    name="prog",
                    device=device,
                    model=prog_model,
                    output_stages=["diag"],
                    memory_fraction=0.5,
                )
            ],
            diagnostic_stages=[
                DiagnosticConfig(
                    name="diag",
                    device=device,
                    model=IdentityDiagnostic(),
                    input_stage="prog",
                    output_stages=["io"],
                    memory_fraction=0.3,
                )
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device="cpu",
                    backend=ZarrBackend(),
                    input_stage="diag",
                )
            ],
        )

        pipeline = Pipeline(config)
        pipeline(input_tensor, base_coords, nsteps=1)

        # Check metrics for memory usage
        metrics = pipeline.get_metrics()
        assert metrics["prog"]["max_memory_usage"] <= 0.5
        assert metrics["diag"]["max_memory_usage"] <= 0.3
