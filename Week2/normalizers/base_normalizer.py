from abc import ABC, abstractmethod

class BaseNormalizer(ABC):
    @abstractmethod
    def fit(x_train):
        pass

    @abstractmethod
    def transform(x_train):
        pass