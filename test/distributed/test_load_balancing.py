import pytest
import torch

from earth2studio.distributed.config import LoadBalancingStrategy
from earth2studio.distributed.load_balancing import (
    LeastBusyBalancer,
    LoadBalancer,
    MemoryAwareBalancer,
    RandomBalancer,
    RoundRobinBalancer,
    create_load_balancer,
)


@pytest.fixture
def mock_devices():
    return [torch.device("cpu"), torch.device("cpu")]


@pytest.fixture
def cuda_devices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


class TestLoadBalancer:
    def test_abstract_class(self):
        """Test that LoadBalancer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LoadBalancer([torch.device("cpu")])

    def test_queue_management(self, mock_devices):
        """Test queue management methods of base class."""
        balancer = RoundRobinBalancer(mock_devices)  # Use concrete implementation
        data = torch.randn(10, 10)

        # Test add_to_queue
        balancer.add_to_queue(0, data)
        assert len(balancer.device_queues[0]) == 1
        assert len(balancer.device_queues[1]) == 0

        # Test remove_from_queue
        balancer.remove_from_queue(0, data)
        assert len(balancer.device_queues[0]) == 0

        # Test with multiple add_to_queue
        balancer.add_to_queue(0, data)
        balancer.add_to_queue(0, data)
        balancer.add_to_queue(1, data)
        assert len(balancer.device_queues[0]) == 2
        assert len(balancer.device_queues[1]) == 1


class TestRoundRobinBalancer:
    def test_device_selection(self, mock_devices):
        """Test round robin distribution pattern."""
        balancer = RoundRobinBalancer(mock_devices)
        selections = [balancer.select_device(None) for _ in range(4)]
        assert selections == [0, 1, 0, 1]

    def test_single_device(self):
        """Test behavior with single device."""
        balancer = RoundRobinBalancer([torch.device("cpu")])
        assert all(balancer.select_device(None) == 0 for _ in range(3))


class TestLeastBusyBalancer:
    def test_empty_queues(self, mock_devices):
        """Test selection with empty queues."""
        balancer = LeastBusyBalancer(mock_devices)
        assert (
            balancer.select_device(None) == 0
        )  # Should select first device when equal

    def test_uneven_queues(self, mock_devices):
        """Test selection with different queue lengths."""
        balancer = LeastBusyBalancer(mock_devices)
        data = torch.randn(10, 10)

        # Add data to first device queue
        balancer.add_to_queue(0, data)
        balancer.add_to_queue(0, data)

        # Should select second device as it's empty
        assert balancer.select_device(None) == 1


class TestMemoryAwareBalancer:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_selection(self, cuda_devices):
        """Test memory-based selection with CUDA devices."""
        balancer = MemoryAwareBalancer(cuda_devices)

        # Allocate some memory on first device
        with torch.cuda.device(cuda_devices[0]):
            torch.empty(int(1e8), device=cuda_devices[0])  # Allocate ~400MB

        # Should prefer device with more free memory
        selected = balancer.select_device(None)
        if torch.cuda.device_count() > 1:
            assert selected != 0
        else:
            assert selected == 0

    def test_cpu_fallback(self, mock_devices):
        """Test fallback to round robin for CPU devices."""
        balancer = MemoryAwareBalancer(mock_devices)
        print(balancer.round_robin)
        selections = [balancer.select_device(None) for _ in range(4)]
        assert selections == [0, 1, 0, 1]  # Should use round robin pattern


class TestRandomBalancer:
    def test_distribution(self, mock_devices):
        """Test random distribution pattern."""
        balancer = RandomBalancer(mock_devices)
        selections = [balancer.select_device(None) for _ in range(100)]

        # Both devices should be selected at least once
        assert 0 in selections and 1 in selections

        # Selections should not be identical (very low probability)
        assert len(set(selections[:10])) > 1

    def test_single_device(self):
        """Test behavior with single device."""
        balancer = RandomBalancer([torch.device("cpu")])
        assert all(balancer.select_device(None) == 0 for _ in range(3))


class TestCreateLoadBalancer:
    def test_strategy_selection(self, mock_devices):
        """Test creation of different balancer types."""
        strategies = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinBalancer,
            LoadBalancingStrategy.LEAST_BUSY: LeastBusyBalancer,
            LoadBalancingStrategy.MEMORY_AWARE: MemoryAwareBalancer,
            LoadBalancingStrategy.RANDOM: RandomBalancer,
        }

        for strategy, expected_class in strategies.items():
            balancer = create_load_balancer(strategy, mock_devices)
            assert isinstance(balancer, expected_class)

    def test_invalid_strategy(self, mock_devices):
        """Test fallback to round robin for invalid strategy."""
        balancer = create_load_balancer("invalid_strategy", mock_devices)  # type: ignore
        assert isinstance(balancer, RoundRobinBalancer)
