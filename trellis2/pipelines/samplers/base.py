from typing import *
from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    A base class for samplers.
    """

    @abstractmethod
    def sample(
        self,
        model,
        **kwargs
    ):
        """
        Sample from a model.
        """
        pass
    