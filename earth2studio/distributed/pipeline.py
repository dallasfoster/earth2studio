# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Queue

import numpy as np
import torch
from loguru import logger

from earth2studio.distributed.config import (
    DiagnosticConfig,
    IOConfig,
    PipelineConfig,
    PrognosticConfig,
)
from earth2studio.utils.coords import CoordSystem, split_coords

# Configure loguru format to include thread name
logger.remove()
logger.add(
    lambda msg: print(msg),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<blue>{thread.name}</blue> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>",
)


def estimate_tensor_memory(tensor: torch.Tensor) -> int:
    """Estimate memory usage of a tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


def calculate_queue_size(
    memory_fraction: float, device: torch.device, sample_tensor: torch.Tensor
) -> int:
    """Calculate safe queue size based on available GPU memory and tensor size."""
    if device.type != "cuda":
        return 2  # Default size for CPU

    total_memory = torch.cuda.get_device_properties(device).total_memory
    available_memory = total_memory * memory_fraction
    tensor_memory = estimate_tensor_memory(sample_tensor)

    # Calculate how many tensors we can safely store
    # Use 20% of available memory for queues, rest for model
    queue_memory = available_memory * 0.2
    safe_size = max(2, int(queue_memory / tensor_memory))
    queue_size = min(safe_size, 10)  # Cap at 10 to prevent excessive memory usage
    logger.debug(
        f"Calculated queue size {queue_size} for device {device} (tensor size: {tensor_memory/1e6:.2f}MB)"
    )
    return queue_size


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage."""

    processing_times: list[float] = field(default_factory=list)
    queue_wait_times: list[float] = field(default_factory=list)
    memory_usage: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.processing_times = []
        self.queue_wait_times = []
        self.memory_usage = []
        self.batch_sizes = []

    def add_metric(
        self, processing_time: float, queue_wait: float, memory: float, batch_size: int
    ) -> None:
        """Add a new set of metrics."""
        self.processing_times.append(processing_time)
        self.queue_wait_times.append(queue_wait)
        self.memory_usage.append(memory)
        self.batch_sizes.append(batch_size)

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics of the metrics."""
        return {
            "avg_processing_time": np.mean(self.processing_times),
            "avg_queue_wait": np.mean(self.queue_wait_times),
            "avg_memory_usage": np.mean(self.memory_usage),
            "avg_batch_size": np.mean(self.batch_sizes),
            "max_memory_usage": max(self.memory_usage),
        }


class PipelineStage:
    """Base class for pipeline stages with enhanced monitoring and error handling."""

    def __init__(self, name: str, device: torch.device) -> None:
        self.name = name
        self.device = device
        self.input_queues: dict[str, Queue] = {}
        self.output_queues: dict[str, Queue] = {}
        self.metrics = StageMetrics()
        self.error: Exception | None = None
        self.error_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.completed = threading.Event()  # New flag to track completion
        logger.info(
            f"Initialized {self.__class__.__name__} '{name}' on device {device}"
        )

    def _set_error(self, error: Exception) -> None:
        """Thread-safe error setting."""
        with self.error_lock:
            if self.error is None:
                self.error = error
                self.error_traceback = traceback.format_exc()
                logger.error(
                    f"Error in stage {self.name}: {error}\n{self.error_traceback}"
                )
                self.stop_event.set()  # Signal all stages to stop on error

    def _check_error(self) -> None:
        """Check and raise any stored error."""
        with self.error_lock:
            if self.error is not None:
                raise Exception(
                    f"Error in stage {self.name}: {self.error}\n{self.error_traceback}"
                )

    def _cleanup_queues(self) -> None:
        """Clean up all queues."""
        logger.debug(f"Cleaning up queues for stage {self.name}")
        for queue in list(self.input_queues.values()) + list(
            self.output_queues.values()
        ):
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except Empty:  # noqa: PERF203
                    pass
            queue.put(None)  # Signal downstream stages

    def _mark_complete(self) -> None:
        """Mark this stage as completed."""
        self.completed.set()
        logger.debug(f"Stage {self.name} marked as complete")

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage for this stage's device."""
        if self.device.type == "cuda":
            return (
                torch.cuda.memory_allocated(self.device)
                / torch.cuda.get_device_properties(self.device).total_memory
            )
        return 0.0


class PrognosticStage(PipelineStage):
    """Enhanced prognostic stage with monitoring and CUDA stream support."""

    def __init__(self, config: PrognosticConfig) -> None:
        super().__init__(config.name, config.device)
        self.model = config.model.to(self.device)
        self.stream = config.stream
        self.memory_fraction = config.memory_fraction
        self.iterator = None
        logger.info(
            f"Initialized prognostic model {self.model.__class__.__name__} for stage {self.name}"
        )

    def initialize(self, x: torch.Tensor, coords: CoordSystem) -> None:
        """Initialize the prognostic iterator."""
        logger.debug(
            f"Initializing prognostic stage {self.name} with input shape {x.shape}"
        )
        x = x.to(self.device)
        self.iterator = self.model.create_iterator(x, coords)

    def step(self) -> bool:
        """Execute one step of the prognostic model."""
        if self.iterator is None:
            raise RuntimeError("Must initialize prognostic stage before stepping")

        try:
            start_time = time.time()
            logger.debug(f"Starting step for prognostic stage {self.name}")

            # Use CUDA stream if available
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    try:
                        x, coords = next(self.iterator)
                        if self.device.type == "cuda":
                            self.stream.synchronize()
                    except StopIteration:
                        self._signal_completion()
                        return False
            else:
                try:
                    x, coords = next(self.iterator)
                except StopIteration:
                    self._signal_completion()
                    return False

            processing_time = time.time() - start_time
            memory_usage = self.get_memory_usage()

            # Send output to all downstream stages
            for stage_name, queue in self.output_queues.items():
                logger.debug(f"Stage {self.name} sending output to {stage_name}")
                queue.put((x.clone(), coords.copy()))

            # Record metrics
            self.metrics.add_metric(
                processing_time=processing_time,
                queue_wait=0.0,  # Prognostic stage doesn't wait for input
                memory=memory_usage,
                batch_size=x.shape[0] if len(x.shape) > 3 else 1,
            )

            return True

        except Exception as e:
            self._set_error(e)
            self._cleanup_queues()
            return False

    def _signal_completion(self) -> None:
        """Signal completion to all downstream stages."""
        logger.info(f"Prognostic stage {self.name} completed iteration")
        for stage_name, queue in self.output_queues.items():
            logger.debug(f"Sending completion signal from {self.name} to {stage_name}")
            queue.put(None)
        self._mark_complete()


class DiagnosticStage(PipelineStage):
    """Enhanced diagnostic stage with monitoring and CUDA stream support."""

    def __init__(self, config: DiagnosticConfig) -> None:
        super().__init__(config.name, config.device)
        self.model = config.model.to(self.device)
        self.stream = config.stream
        self.memory_fraction = config.memory_fraction
        logger.info(
            f"Initialized diagnostic model {self.model.__class__.__name__} for stage {self.name}"
        )

    def run(self) -> None:
        """Main processing loop for the diagnostic stage."""
        try:
            while not self.stop_event.is_set():
                queue_start = time.time()
                logger.debug(f"Diagnostic stage {self.name} waiting for input")

                # Get input data
                data = next((queue.get() for queue in self.input_queues.values()), None)

                if data is None:
                    # Propagate termination signal and mark complete
                    logger.info(
                        f"Diagnostic stage {self.name} received completion signal"
                    )
                    for stage_name, queue in self.output_queues.items():
                        logger.debug(
                            f"Sending completion signal from {self.name} to {stage_name}"
                        )
                        queue.put(None)
                    for queue in self.input_queues.values():
                        queue.task_done()
                    break

                queue_wait = time.time() - queue_start
                start_time = time.time()

                x, coords = data
                x = x.to(self.device)
                logger.debug(
                    f"Stage {self.name} processing input tensor of shape {x.shape}"
                )

                # Process data using CUDA stream if available
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        output, out_coords = self.model(x, coords)
                        if self.device.type == "cuda":
                            self.stream.synchronize()
                else:
                    output, out_coords = self.model(x, coords)

                processing_time = time.time() - start_time
                memory_usage = self.get_memory_usage()

                # Send output to downstream stages
                for stage_name, queue in self.output_queues.items():
                    logger.debug(f"Stage {self.name} sending output to {stage_name}")
                    queue.put((output.clone(), out_coords.copy()))

                # Record metrics
                self.metrics.add_metric(
                    processing_time=processing_time,
                    queue_wait=queue_wait,
                    memory=memory_usage,
                    batch_size=x.shape[0] if len(x.shape) > 3 else 1,
                )

                for queue in self.input_queues.values():
                    queue.task_done()

        except Exception as e:
            self._set_error(e)
            self._cleanup_queues()
        finally:
            self._mark_complete()


class IOStage(PipelineStage):
    """Enhanced IO stage with monitoring."""

    def __init__(self, config: IOConfig) -> None:
        super().__init__(config.name, config.device)
        self.backend = config.backend
        self.array_name = config.array_name
        self.split_variable_key = config.split_variable_key
        self.memory_fraction = config.memory_fraction
        logger.info(
            f"Initialized IO stage {self.name} with backend {self.backend.__class__.__name__}"
        )

    def run(self) -> None:
        """Main processing loop for the IO stage."""
        try:
            while not self.stop_event.is_set():
                queue_start = time.time()
                logger.debug(f"IO stage {self.name} waiting for input")

                # Get input data
                data = next((queue.get() for queue in self.input_queues.values()), None)

                if data is None:
                    logger.info(f"IO stage {self.name} received completion signal")
                    for queue in self.input_queues.values():
                        queue.task_done()
                    break

                queue_wait = time.time() - queue_start
                start_time = time.time()

                x, coords = data
                x = x.detach().cpu()
                logger.debug(f"Stage {self.name} writing tensor of shape {x.shape}")

                # Split variables if needed
                if self.array_name is not None:
                    array_name = self.array_name
                else:
                    x, coords, array_name = split_coords(
                        x, coords, self.split_variable_key
                    )

                self.backend.write(x, coords, array_name)
                logger.debug(f"Stage {self.name} wrote data to arrays {array_name}")

                processing_time = time.time() - start_time
                memory_usage = self.get_memory_usage()

                # Record metrics
                self.metrics.add_metric(
                    processing_time=processing_time,
                    queue_wait=queue_wait,
                    memory=memory_usage,
                    batch_size=x[0].shape[0],
                )

                for queue in self.input_queues.values():
                    queue.task_done()

        except Exception as e:
            self._set_error(e)
        finally:
            self._mark_complete()


class Pipeline:
    """Pipeline implementation with improved monitoring and performance."""

    def __init__(self, config: PipelineConfig) -> None:
        logger.info("Initializing Pipeline")
        self.config = config
        self.config.validate()

        # Initialize stages
        self.prognostic_stages: dict[str, PrognosticStage] = {}
        self.diagnostic_stages: dict[str, DiagnosticStage] = {}
        self.io_stages: dict[str, IOStage] = {}

        # Create stages
        self._create_stages()

        self.executor = ThreadPoolExecutor()
        logger.info("Pipeline initialization complete")

    def _create_stages(self) -> None:
        """Create all pipeline stages."""
        logger.info("Creating pipeline stages")
        for prog_stage in self.config.prognostic_stages:
            self.prognostic_stages[prog_stage.name] = PrognosticStage(prog_stage)

        for diag_stage in self.config.diagnostic_stages:
            self.diagnostic_stages[diag_stage.name] = DiagnosticStage(diag_stage)

        for io_stage in self.config.io_stages:
            self.io_stages[io_stage.name] = IOStage(io_stage)

    def _setup_queues(self, sample_tensor: torch.Tensor) -> None:
        """Set up queues with sizes based on memory constraints."""
        logger.info("Setting up pipeline queues")
        # Connect prognostic stages to their outputs
        for prog_stage in self.config.prognostic_stages:
            for output_name in prog_stage.output_stages:
                if output_name in self.diagnostic_stages:
                    target_stage = self.diagnostic_stages[output_name]
                    queue_size = calculate_queue_size(
                        target_stage.memory_fraction, target_stage.device, sample_tensor
                    )
                else:
                    queue_size = 2  # Fixed size for IO stages

                queue: Queue = Queue(maxsize=queue_size)
                self.prognostic_stages[prog_stage.name].output_queues[
                    output_name
                ] = queue
                logger.debug(
                    f"Connected {prog_stage.name} -> {output_name} with queue size {queue_size}"
                )

                if output_name in self.diagnostic_stages:
                    self.diagnostic_stages[output_name].input_queues[
                        prog_stage.name
                    ] = queue
                elif output_name in self.io_stages:
                    self.io_stages[output_name].input_queues[prog_stage.name] = queue

        # Connect diagnostic stages to their outputs
        for diag_stage in self.config.diagnostic_stages:
            for output_name in diag_stage.output_stages:
                if output_name in self.diagnostic_stages:
                    target_stage = self.diagnostic_stages[output_name]
                    queue_size = calculate_queue_size(
                        target_stage.memory_fraction, target_stage.device, sample_tensor
                    )
                else:
                    queue_size = 2  # Fixed size for IO stages

                queue = Queue(maxsize=queue_size)
                self.diagnostic_stages[diag_stage.name].output_queues[
                    output_name
                ] = queue
                logger.debug(
                    f"Connected {diag_stage.name} -> {output_name} with queue size {queue_size}"
                )

                if output_name in self.diagnostic_stages:
                    self.diagnostic_stages[output_name].input_queues[
                        diag_stage.name
                    ] = queue
                elif output_name in self.io_stages:
                    self.io_stages[output_name].input_queues[diag_stage.name] = queue

    def _initialize_io_backend(
        self,
        prognostic_stages: dict[str, PrognosticStage],
        diagnostic_stages: dict[str, DiagnosticStage],
        io_stages: dict[str, IOStage],
        nsteps: int,
        coords: CoordSystem,
    ) -> None:
        """Initialize IO backends for all IO stages.

        This sets up the coordinate systems and arrays for each IO backend based on
        the prognostic model outputs and pipeline configuration.

        The coordinates are propagated through the pipeline:
        1. Start with input coordinates
        2. Transform through prognostic model's output_coords
        3. Transform through each diagnostic model's output_coords in the chain
        4. Use final coordinates for IO backend initialization
        """
        logger.info("Initializing IO backends")

        time = coords.get("time", None)

        def trace_pipeline_chain(
            io_stage_name: str,
        ) -> tuple[PrognosticStage | None, list[DiagnosticStage]]:
            """Trace back through the pipeline to find the chain of stages that feed into an IO stage."""
            chain: list[DiagnosticStage] = []
            current_stage_name = io_stage_name
            prog_stage = None

            # Work backwards through diagnostic stages until we find the prognostic source
            while current_stage_name:
                print(current_stage_name)
                # Check if any prognostic stage outputs to current_stage
                for prog in prognostic_stages.values():
                    if current_stage_name in prog.output_queues:
                        prog_stage = prog
                        return prog_stage, chain

                # If not found in prognostic, look through diagnostic stages
                found = False
                for diag in diagnostic_stages.values():
                    print(diag, diag.output_queues)
                    if current_stage_name in diag.output_queues:
                        chain.insert(0, diag)  # Insert at front to maintain order
                        current_stage_name = diag.name
                        found = True
                        break

                if not found:
                    break

            return prog_stage, chain

        for io_stage in io_stages.values():
            logger.debug(f"Initializing IO backend for stage {io_stage.name}")

            # Find the chain of stages that feed into this IO stage
            prog_stage, diag_chain = trace_pipeline_chain(io_stage.name)

            if prog_stage is None:
                logger.warning(
                    f"Could not find prognostic stage for IO stage {io_stage.name}"
                )
                continue

            # Start with prognostic model's input coordinates
            prog_input_coords = prog_stage.model.input_coords()

            # Transform through prognostic model's output coordinates
            current_coords = prog_stage.model.output_coords(prog_input_coords).copy()
            logger.debug(f"Coordinates after prognostic stage: {current_coords}")

            # Transform through each diagnostic model in the chain
            for diag_stage in diag_chain:
                # Map to diagnostic input coordinates first
                # Then get diagnostic output coordinates
                current_coords = diag_stage.model.output_coords(current_coords)
                logger.debug(
                    f"Coordinates after diagnostic stage {diag_stage.name}: {current_coords}"
                )

            # Remove batch dimensions
            for key, value in list(current_coords.items()):
                if value.shape == (0,) or key == "batch":
                    del current_coords[key]

            # Add time dimensions
            if time is not None:
                current_coords["time"] = time

            # Add lead time dimension
            if "lead_time" in current_coords:
                base_lead_time = current_coords["lead_time"]
                current_coords["lead_time"] = np.asarray(
                    [base_lead_time * i for i in range(nsteps + 1)]
                ).flatten()

            # Move time dimensions to front
            if "lead_time" in current_coords:
                current_coords.move_to_end("lead_time", last=False)
            if "time" in current_coords:
                current_coords.move_to_end("time", last=False)

            # Get variable names and remove from coords
            if io_stage.array_name is not None:
                var_names = io_stage.array_name
            else:
                var_names = current_coords.pop(io_stage.split_variable_key)

            logger.debug(
                f"Final coordinates for IO stage {io_stage.name}: {current_coords}"
            )
            logger.debug(f"Array names: {var_names}")

            # Add arrays to IO backend
            io_stage.backend.add_array(current_coords, var_names)

    def _wait_for_completion(self, timeout: float = 30.0) -> bool:
        """Wait for all stages to complete or timeout.

        Returns True if all stages completed, False if timeout occurred.
        """
        all_stages = (
            list(self.prognostic_stages.values())
            + list(self.diagnostic_stages.values())
            + list(self.io_stages.values())
        )
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check for errors
            for stage in all_stages:
                stage._check_error()

            # Check if all stages are complete
            if all(stage.completed.is_set() for stage in all_stages):
                return True

            time.sleep(0.1)

        return False

    def __call__(self, x: torch.Tensor, coords: CoordSystem, nsteps: int) -> None:
        """Run the pipeline for nsteps iterations."""
        logger.info(f"Starting pipeline execution for {nsteps} steps")
        try:
            # Set up queues based on input tensor size
            self._setup_queues(x)

            # Initialize IO backends
            self._initialize_io_backend(
                self.prognostic_stages,
                self.diagnostic_stages,
                self.io_stages,
                nsteps,
                coords,
            )

            # Initialize prognostic stages
            for prog_stage in self.prognostic_stages.values():
                prog_stage.initialize(x, coords)

            # Start diagnostic and IO stages
            futures = [  # noqa: F841
                self.executor.submit(stage.run)
                for stage in (
                    list(self.diagnostic_stages.values())
                    + list(self.io_stages.values())
                )
            ]

            # Run prognostic stages for requested steps
            for step in range(nsteps + 1):  # +1 for initial condition
                logger.info(f"Processing step {step}/{nsteps}")
                for prog_stage in self.prognostic_stages.values():
                    if not prog_stage.step():
                        break

            for prog_stage in self.prognostic_stages.values():
                prog_stage._signal_completion()

            # Wait for completion or timeout
            if not self._wait_for_completion():
                logger.error("Pipeline timed out waiting for completion")
                raise TimeoutError("Pipeline execution timed out")

            logger.info("Pipeline execution completed successfully")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            # Stop all stages
            all_stages = (
                list(self.prognostic_stages.values())
                + list(self.diagnostic_stages.values())
                + list(self.io_stages.values())
            )
            for stage in all_stages:
                stage.stop_event.set()
            raise e

        finally:
            # Clean up executor
            self.executor.shutdown(wait=False)

    def get_metrics(self) -> dict[str, dict[str, float]]:
        """Get performance metrics for all stages."""
        metrics = {}
        for name, stage in (
            list(self.prognostic_stages.items())
            + list(self.diagnostic_stages.items())
            + list(self.io_stages.items())
        ):
            metrics[name] = stage.metrics.get_summary()
        logger.debug(f"Pipeline metrics: {metrics}")
        return metrics

    def __del__(self) -> None:
        if hasattr(self, "executor") and not self.executor._shutdown:
            self.executor.shutdown(wait=False)
