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
from queue import Queue

import numpy as np
import pytest
import torch

from earth2studio.distributed.config import (
    DiagnosticConfig,
    IOConfig,
    PipelineConfig,
    PrognosticConfig,
)
from earth2studio.distributed.pipeline import (
    Pipeline,
    PipelineStage,
    StageMetrics,
    estimate_tensor_memory,
)
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


class RenameDiagnostic(torch.nn.Module, DiagnosticModel):
    """Diagnostic model that renames input"""

    def __init__(self, output_variable: str):
        super().__init__()
        self.output_variable = output_variable

    def input_coords(self):
        return OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))])

    def output_coords(self, input_coords):
        output_coords = input_coords.copy()
        output_coords["variable"] = np.array([self.output_variable])
        return output_coords

    def __call__(self, x, coords):
        output_coords = coords.copy()
        output_coords["variable"] = self.output_variable
        return x, output_coords


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


def test_estimate_tensor_memory():
    tensor = torch.randn(1, 1, 10, 20)
    assert estimate_tensor_memory(tensor) == 10 * 20 * 4


class TestStageMetrics:
    def test_add_metric(self):
        metrics = StageMetrics()
        metrics.add_metric(1.0, 0.5, 100.0, 10)
        assert metrics.processing_times == [1.0]
        assert metrics.queue_wait_times == [0.5]
        assert metrics.memory_usage == [100.0]
        assert metrics.batch_sizes == [10]
        assert metrics.get_summary() == {
            "avg_processing_time": 1.0,
            "avg_queue_wait": 0.5,
            "avg_memory_usage": 100.0,
            "avg_batch_size": 10,
            "max_memory_usage": 100.0,
        }


class TestPipelineStage:
    def test_initialize(self):
        stage = PipelineStage("test", torch.device("cpu"))
        assert stage.name == "test"
        assert stage.device == torch.device("cpu")
        assert stage.input_queues == {}
        assert stage.output_queues == {}

    def test_set_error(self):
        stage = PipelineStage("test", torch.device("cpu"))
        stage._set_error(ValueError("test error"))
        assert stage.error is not None
        assert stage.error_traceback is not None
        assert stage.stop_event.is_set()
        assert stage.completed.is_set()

        stage = PipelineStage("test", torch.device("cuda:0"))
        stage._set_error(None)
        stage._check_error()
        assert stage.error is None

    def test_check_error(self):
        stage = PipelineStage("test", torch.device("cpu"))
        stage._set_error(ValueError("test error"))
        with pytest.raises(Exception):
            stage._check_error()

    def test_cleanup_queues(self):
        stage = PipelineStage("test", torch.device("cpu"))
        stage.input_queues["input"] = Queue()
        stage.output_queues["output"] = Queue()
        stage._cleanup_queues()
        assert stage.input_queues["input"].get() is None
        assert stage.output_queues["output"].get() is None

    def test_mark_complete(self):
        stage = PipelineStage("test", torch.device("cpu"))
        stage._mark_complete()
        assert stage.completed.is_set()

    def test_get_memory_usage(self):
        stage = PipelineStage("test", torch.device("cpu"))
        assert stage.get_memory_usage() == 0.0

    def test_get_memory_usage_cuda(self):
        stage = PipelineStage("test", torch.device("cuda:0"))
        a = torch.tensor([1, 2, 3], device="cuda:0")
        a.share_memory_()
        assert stage.get_memory_usage() > 0.0


class TestConfig:
    def test_io_config(self):
        config = IOConfig(
            name="io",
            device="cpu",
            backend=ZarrBackend(),
            input_stages=["diag"],
        )
        assert config.output_coords == {}
        assert not config._is_initialized
        assert config.array_name is None
        assert config.split_variable_key == "variable"

    def test_io_config_with_array_name(self):
        config = IOConfig(
            name="io",
            device="cpu",
            backend=ZarrBackend(),
            array_name="t2m",
            split_variable_key=None,
            input_stages=["diag"],
        )
        assert config.output_coords == {}
        assert not config._is_initialized

        with pytest.raises(ValueError):
            config = IOConfig(
                name="io",
                device="cpu",
                backend=ZarrBackend(),
                array_name="t2m",
                split_variable_key="variable",
                input_stages=["diag"],
            )

        with pytest.raises(ValueError):
            config = IOConfig(
                name="io",
                device="cpu",
                backend=ZarrBackend(),
                array_name=None,
                split_variable_key=None,
                input_stages=["diag"],
            )

    def test_pipeline_config(self):
        config = PipelineConfig(
            prognostic_stages=[
                PrognosticConfig(
                    name="prog",
                    device="cpu",
                    model=IdentityDiagnostic(),
                    output_stages=["diag"],
                )
            ],
            diagnostic_stages=[
                DiagnosticConfig(
                    name="diag",
                    device="cpu",
                    model=IdentityDiagnostic(),
                    input_stages=["prog"],
                    output_stages=["io"],
                )
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device="cpu",
                    backend=ZarrBackend(),
                    input_stages=["diag"],
                )
            ],
        )
        assert config.prognostic_stages[0].name == "prog"
        assert config.diagnostic_stages[0].name == "diag"
        assert config.io_stages[0].name == "io"

    def test_pipeline_config_with_invalid_stage_references(self):

        with pytest.raises(ValueError):
            config = PipelineConfig(
                prognostic_stages=[],
                diagnostic_stages=[],
                io_stages=[],
            )
            config.validate()

        with pytest.raises(ValueError):
            config = PipelineConfig(
                prognostic_stages=[
                    PrognosticConfig(
                        name="prog",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        output_stages=["nonexistent"],
                    )
                ],
                diagnostic_stages=[
                    DiagnosticConfig(
                        name="diag",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        input_stages=["prog"],
                        output_stages=["io"],
                    )
                ],
                io_stages=[
                    IOConfig(
                        name="io",
                        device="cpu",
                        backend=ZarrBackend(),
                        input_stages=["diag"],
                    )
                ],
            )
            config.validate()

        with pytest.raises(ValueError):
            config = PipelineConfig(
                prognostic_stages=[
                    PrognosticConfig(
                        name="prog",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        output_stages=["diag"],
                    )
                ],
                diagnostic_stages=[
                    DiagnosticConfig(
                        name="diag",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        input_stages=["nonexistent"],
                        output_stages=["io"],
                    )
                ],
                io_stages=[
                    IOConfig(
                        name="io",
                        device="cpu",
                        backend=ZarrBackend(),
                        input_stages=["diag"],
                    )
                ],
            )
            config.validate()

        with pytest.raises(ValueError):
            config = PipelineConfig(
                prognostic_stages=[
                    PrognosticConfig(
                        name="prog",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        output_stages=["diag"],
                    )
                ],
                diagnostic_stages=[
                    DiagnosticConfig(
                        name="diag",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        input_stages=["prog"],
                        output_stages=["nonexistent"],
                    )
                ],
                io_stages=[
                    IOConfig(
                        name="io",
                        device="cpu",
                        backend=ZarrBackend(),
                        input_stages=["diag"],
                    )
                ],
            )
            config.validate()

        with pytest.raises(ValueError):
            config = PipelineConfig(
                prognostic_stages=[
                    PrognosticConfig(
                        name="prog",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        output_stages=["diag"],
                    )
                ],
                diagnostic_stages=[],
                io_stages=[],
            )
            config.validate()

        with pytest.raises(ValueError):
            config = PipelineConfig(
                prognostic_stages=[
                    PrognosticConfig(
                        name="prog",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        output_stages=["diag"],
                    )
                ],
                diagnostic_stages=[
                    DiagnosticConfig(
                        name="diag",
                        device="cpu",
                        model=IdentityDiagnostic(),
                        input_stages=["prog"],
                        output_stages=["io"],
                    )
                ],
                io_stages=[
                    IOConfig(
                        name="diag",
                        device="cpu",
                        backend=ZarrBackend(),
                        input_stages=["prog"],
                    ),
                    IOConfig(
                        name="io",
                        device="cpu",
                        backend=ZarrBackend(),
                        input_stages=["diag"],
                    ),
                ],
            )
            config.validate()


@pytest.mark.parametrize(
    "device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
)
class TestPipeline:
    def test_pipeline_step_error(
        self, input_tensor, base_coords, domain_coords, device
    ):
        """Test pipeline error handling."""
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
                    input_stages=["prog"],
                    output_stages=["io"],
                )
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device=device,
                    backend=ZarrBackend(),
                    input_stages=["diag"],
                )
            ],
        )

        pipeline = Pipeline(config)
        with pytest.raises(Exception):
            pipeline.step()

    def test_pipeline_step_with_cuda_stream(
        self, input_tensor, base_coords, domain_coords, device
    ):
        """Test pipeline step with CUDA stream."""
        if device == "cpu":
            pytest.skip("CUDA stream only relevant for GPU")

        prog_model = Persistence(base_coords["variable"], domain_coords)
        stream1 = torch.cuda.Stream(device=device)
        stream2 = torch.cuda.Stream(device=device)
        config = PipelineConfig(
            prognostic_stages=[
                PrognosticConfig(
                    name="prog",
                    device=device,
                    model=prog_model,
                    output_stages=["diag"],
                    stream=stream1,
                )
            ],
            diagnostic_stages=[
                DiagnosticConfig(
                    name="diag",
                    device=device,
                    model=IdentityDiagnostic(),
                    input_stages=["prog"],
                    output_stages=["io"],
                    stream=stream2,
                )
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device=device,
                    backend=ZarrBackend(),
                    input_stages=["diag"],
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

    def test_pipeline_determine_io_backend_coords(
        self, input_tensor, base_coords, domain_coords, device
    ):
        """Test pipeline determine IO backend coordinates."""

        # Test with single chain
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
                    input_stages=["prog"],
                    output_stages=["io"],
                )
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device=device,
                    backend=ZarrBackend(),
                    input_stages=["diag"],
                )
            ],
        )
        pipeline = Pipeline(config)
        pipeline._setup_queues(input_tensor)

        total_coords = pipeline._determine_io_backend_coords(
            pipeline.prognostic_stages,
            pipeline.diagnostic_stages,
            pipeline.io_stages["io"],
            1,
            base_coords,
        )
        assert len(total_coords) == 1
        output_coords = total_coords[0]
        assert np.allclose(output_coords["lat"], base_coords["lat"])
        assert np.allclose(output_coords["lon"], base_coords["lon"])
        assert np.all(
            output_coords["lead_time"]
            == np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
        )
        assert output_coords["variable"] == base_coords["variable"]

        # Test with multiple chains
        config = PipelineConfig(
            prognostic_stages=[
                PrognosticConfig(
                    name="prog",
                    device=device,
                    model=prog_model,
                    output_stages=["diag1", "diag2"],
                )
            ],
            diagnostic_stages=[
                DiagnosticConfig(
                    name="diag1",
                    device=device,
                    model=IdentityDiagnostic(),
                    input_stages=["prog"],
                    output_stages=["io"],
                ),
                DiagnosticConfig(
                    name="diag2",
                    device=device,
                    model=RenameDiagnostic("tcwv"),
                    input_stages=["prog"],
                    output_stages=["io"],
                ),
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device=device,
                    backend=ZarrBackend(),
                    input_stages=["diag1", "diag2"],
                )
            ],
        )
        pipeline = Pipeline(config)
        pipeline._setup_queues(input_tensor)
        total_coords = pipeline._determine_io_backend_coords(
            pipeline.prognostic_stages,
            pipeline.diagnostic_stages,
            pipeline.io_stages["io"],
            1,
            base_coords,
        )
        assert len(total_coords) == 2
        for i, output_coords in enumerate(total_coords):
            assert np.allclose(output_coords["lat"], base_coords["lat"])
            assert np.allclose(output_coords["lon"], base_coords["lon"])
            assert np.all(
                output_coords["lead_time"]
                == np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
            )
            if i == 0:
                assert output_coords["variable"] == np.array(["t2m"])
            else:
                assert output_coords["variable"] == np.array(["tcwv"])

        # Test with multiple chains
        config = PipelineConfig(
            prognostic_stages=[
                PrognosticConfig(
                    name="prog",
                    device=device,
                    model=prog_model,
                    output_stages=["diag", "io"],
                )
            ],
            diagnostic_stages=[
                DiagnosticConfig(
                    name="diag",
                    device=device,
                    model=RenameDiagnostic("tcwv"),
                    input_stages=["prog"],
                    output_stages=["io"],
                ),
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device=device,
                    backend=ZarrBackend(),
                    input_stages=["prog", "diag"],
                )
            ],
        )
        pipeline = Pipeline(config)
        pipeline._setup_queues(input_tensor)
        total_coords = pipeline._determine_io_backend_coords(
            pipeline.prognostic_stages,
            pipeline.diagnostic_stages,
            pipeline.io_stages["io"],
            1,
            base_coords,
        )
        assert len(total_coords) == 2
        for i, output_coords in enumerate(total_coords):
            assert np.allclose(output_coords["lat"], base_coords["lat"])
            assert np.allclose(output_coords["lon"], base_coords["lon"])
            assert np.all(
                output_coords["lead_time"]
                == np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
            )
            if i == 0:
                assert output_coords["variable"] == np.array(["t2m"])
            else:
                assert output_coords["variable"] == np.array(["tcwv"])

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
                    input_stages=["prog"],
                    output_stages=["io"],
                )
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device="cpu",
                    backend=ZarrBackend(),
                    input_stages=["diag"],
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
                    input_stages=["prog"],
                    output_stages=["diag2"],
                ),
                DiagnosticConfig(
                    name="diag2",
                    device=device,
                    model=ScalingDiagnostic(0.5),
                    input_stages=["diag1"],
                    output_stages=["io"],
                ),
            ],
            io_stages=[
                IOConfig(
                    name="io",
                    device="cpu",
                    backend=ZarrBackend(),
                    input_stages=["diag2"],
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
                    input_stages=["prog"],
                    output_stages=["io1"],
                ),
                DiagnosticConfig(
                    name="diag2",
                    device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0",
                    model=ScalingDiagnostic(0.5),
                    input_stages=["prog"],
                    output_stages=["io2"],
                ),
            ],
            io_stages=[
                IOConfig(
                    name="io1",
                    device="cpu",
                    backend=io1,
                    input_stages=["diag1"],
                ),
                IOConfig(
                    name="io2",
                    device="cpu",
                    backend=io2,
                    input_stages=["diag2"],
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
