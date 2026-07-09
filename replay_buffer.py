import random
import pickle
from collections import deque
from pathlib import Path
import numpy as np


class ReplayBuffer:
    """
    FIFO replay buffer for storing self-play positions.

    Each entry is a tuple: (board_tensor, policy_target, value_target, material_target)
    - board_tensor: (18, 8, 8) numpy array
    - policy_target: (4672,) numpy array (MCTS visit distribution)
    - value_target: scalar float (game outcome: +1, 0, -1)
    - material_target: scalar float (material balance: White - Black)
    """

    def __init__(self, capacity: int = 7000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, board_tensor: np.ndarray, policy: np.ndarray, value: float, material: float):
        """
        Add a single position to the buffer.

        Args:
            board_tensor: (18, 8, 8) board encoding
            policy: (4672,) MCTS visit distribution
            value: Game outcome from this position's perspective
            material: Material balance (White - Black material)
        """
        self.buffer.append((board_tensor, policy, value, material))

    def sample(self, batch_size: int):
        """
        Sample a batch uniformly at random from the buffer.

        Args:
            batch_size: Number of positions to sample

        Returns:
            board_tensors: (batch_size, 18, 8, 8) numpy array
            policy_targets: (batch_size, 4672) numpy array
            value_targets: (batch_size,) numpy array
            material_targets: (batch_size,) numpy array
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        samples = random.sample(self.buffer, batch_size)

        # Unzip samples
        board_tensors = np.array([s[0] for s in samples], dtype=np.float32)
        policy_targets = np.array([s[1] for s in samples], dtype=np.float32)
        value_targets = np.array([s[2] for s in samples], dtype=np.float32)
        material_targets = np.array([s[3] for s in samples], dtype=np.float32)

        return board_tensors, policy_targets, value_targets, material_targets

    def __len__(self):
        return len(self.buffer)

    def is_empty(self):
        return len(self.buffer) == 0

    def save(self, filepath: str):
        """
        Save replay buffer to disk using pickle.

        Args:
            filepath: Path to save the buffer (e.g., 'buffer.pkl')
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),
                'capacity': self.capacity,
            }, f)

    @classmethod
    def load(cls, filepath: str):
        """
        Load replay buffer from disk.

        Args:
            filepath: Path to the saved buffer file

        Returns:
            ReplayBuffer: Loaded replay buffer instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        buffer = cls(capacity=data['capacity'])
        buffer.buffer = deque(data['buffer'], maxlen=data['capacity'])

        return buffer
