import numpy as np
from normalizers.base_normalizer import BaseNormalizer

class ZScoreNormalizer(BaseNormalizer):
    def __init__(self):
        self.mu = None
        self.std = None

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        # this works with both 1d and 2d arrays because of the numpy broadcasting
        return (X - self.mu) / self.std