"""Load balancing strategies for distributing work across multiple devices."""

import random
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

T = TypeVar("T")  # Generic type for device selection


class LoadBalancer(ABC, Generic[T]):
    """Abstract base class for load balancing strategies."""

    def __init__(self, devices: list[torch.device]):
        self.devices = [
            torch.device(dev) if isinstance(dev, str) else dev for dev in devices
        ]
        self._current_idx = 0
        self.device_queues: list[list[T]] = [[] for _ in devices]

    @abstractmethod
    def select_device(self, data: T) -> int:
        """Select which device should handle the given data.

        Returns the index of the selected device.
        """
        pass

    def add_to_queue(self, device_idx: int, data: T) -> None:
        """Add data to the queue for the specified device."""
        self.device_queues[device_idx].append(data)

    def remove_from_queue(self, device_idx: int, data: T) -> None:
        """Remove data from the queue for the specified device."""
        self.device_queues[device_idx].remove(data)


class RoundRobinBalancer(LoadBalancer):
    """Distribute work in a circular order among devices."""

    def select_device(self, data: T) -> int:
        """Select the next device in the circular order."""
        selected = self._current_idx
        self._current_idx = (self._current_idx + 1) % len(self.devices)
        return selected


class LeastBusyBalancer(LoadBalancer):
    """Send work to the device with the shortest queue."""

    def select_device(self, data: T) -> int:
        """Select the device with the shortest queue."""
        queue_lengths = [len(q) for q in self.device_queues]
        return queue_lengths.index(min(queue_lengths))


class MemoryAwareBalancer(LoadBalancer):
    """Consider device memory usage when selecting a device."""

    def __init__(self, devices: list[torch.device]):
        super().__init__(devices)
        cuda_devices = [
            (i, dev) for i, dev in enumerate(self.devices) if dev.type == "cuda"
        ]
        if not cuda_devices:
            self.round_robin: RoundRobinBalancer | None = RoundRobinBalancer(
                self.devices
            )
        else:
            self.round_robin = None

    def select_device(self, data: T) -> int:
        """Select the device with the most available memory."""
        # Only consider CUDA devices
        cuda_devices = [
            (i, dev) for i, dev in enumerate(self.devices) if dev.type == "cuda"
        ]

        if not cuda_devices and self.round_robin is not None:
            # Fall back to round robin if no CUDA devices
            return self.round_robin.select_device(data)

        # Find device with most available memory
        memory_available = []
        for idx, device in cuda_devices:
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            memory_available.append((idx, total - allocated))

        return max(memory_available, key=lambda x: x[1])[0]


class RandomBalancer(LoadBalancer):
    """Randomly distribute work among devices."""

    def select_device(self, data: T) -> int:
        """Select a random device."""
        return random.randrange(len(self.devices))
