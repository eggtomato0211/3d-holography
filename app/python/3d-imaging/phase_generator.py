import numpy as np

class PhaseGenerator:
    def __init__(self, random_mode):
        self.random_mode = random_mode

    def generate_random_phase(self, shape):
        if self.random_mode:
            return (np.random.rand(*shape) - 0.5) * 2.0 * 2.0 * np.pi
        np.random.seed(42)
        return (np.random.rand(*shape) - 0.5) * 2.0 * 2.0 * np.pi
