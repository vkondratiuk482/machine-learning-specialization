import numpy as np
from normalizers.base_normalizer import BaseNormalizer

class MeanNormalizer(BaseNormalizer):
    def __init__(self):
        self.mu = None
        self.max = None
        self.min = None
        self.epsilon = 1e-8  # extremely small number to avoid division by zero

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.max = np.max(X, axis=0)
        self.min = np.min(X, axis=0)

    def transform(self, X):
        # this works with both 1d and 2d arrays because of the numpy broadcasting
        return (X - self.mu) / (self.max - self.min + self.epsilon)