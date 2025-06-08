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
from dataclasses import dataclass, field

import torch

from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel


@dataclass
class StageConfig:
    """Base configuration for a pipeline stage."""

    name: str
    device: str | torch.device


@dataclass
class PrognosticConfig(StageConfig):
    """Configuration for a prognostic stage."""

    model: PrognosticModel
    output_stages: list[str] = field(
        default_factory=list
    )  # Names of stages to send output to
    memory_fraction: float = 0.9  # Maximum GPU memory fraction to use
    stream: torch.cuda.Stream | None = None


@dataclass
class DiagnosticConfig(StageConfig):
    """Configuration for a diagnostic stage."""

    model: DiagnosticModel
    input_stage: str  # Name of stage to receive input from
    output_stages: list[str] = field(
        default_factory=list
    )  # Names of stages to send output to
    memory_fraction: float = 0.9  # Maximum GPU memory fraction to use
    stream: torch.cuda.Stream | None = None


@dataclass
class IOConfig(StageConfig):
    """Configuration for an IO stage."""

    backend: IOBackend
    input_stage: str  # Name of stage to receive input from
    array_name: str | None = None
    split_variable_key: str | None = "variable"
    memory_fraction: float = 0.9  # Maximum GPU memory fraction to use

    def __post_init__(self) -> None:
        if self.array_name is None and self.split_variable_key is None:
            raise ValueError("Either array_name or split_variable_key must be provided")

        if self.array_name is not None and self.split_variable_key is not None:
            raise ValueError(
                "Either array_name or split_variable_key must be provided, not both"
            )


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""

    prognostic_stages: list[PrognosticConfig]
    diagnostic_stages: list[DiagnosticConfig]
    io_stages: list[IOConfig]
    enable_profiling: bool = False
    enable_memory_monitoring: bool = False
    profile_output_file: str | None = None

    def validate(self) -> None:
        """Validate the pipeline configuration."""
        # Check that all referenced stage names exist
        all_stage_names = set()
        for prog_stage in self.prognostic_stages:
            all_stage_names.add(prog_stage.name)
        for diag_stage in self.diagnostic_stages:
            all_stage_names.add(diag_stage.name)
        for io_stage in self.io_stages:
            all_stage_names.add(io_stage.name)

        # Validate stage connections
        for prog_stage in self.prognostic_stages:
            for output_stage in prog_stage.output_stages:
                if output_stage not in all_stage_names:
                    raise ValueError(
                        f"Prognostic stage {prog_stage.name} references non-existent output stage {output_stage}"
                    )

        for diag_stage in self.diagnostic_stages:
            if diag_stage.input_stage not in all_stage_names:
                raise ValueError(
                    f"Diagnostic stage {diag_stage.name} references non-existent input stage {diag_stage.input_stage}"
                )
            for output_stage in diag_stage.output_stages:
                if output_stage not in all_stage_names:
                    raise ValueError(
                        f"Diagnostic stage {diag_stage.name} references non-existent output stage {output_stage}"
                    )

        for io_stage in self.io_stages:
            if io_stage.input_stage not in all_stage_names:
                raise ValueError(
                    f"IO stage {io_stage.name} references non-existent input stage {io_stage.input_stage}"
                )

        # Validate device configurations
        for stage in self.prognostic_stages + self.diagnostic_stages + self.io_stages:
            if isinstance(stage.device, str):
                stage.device = torch.device(stage.device)
            if stage.memory_fraction <= 0 or stage.memory_fraction > 1:
                raise ValueError(
                    f"Invalid memory fraction {stage.memory_fraction} for stage {stage.name}"
                )
