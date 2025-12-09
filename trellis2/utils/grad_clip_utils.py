from typing import *
import torch
import numpy as np
import torch.utils


class AdaptiveGradClipper:
    """
    Adaptive gradient clipping for training.
    """
    def __init__(
        self,
        max_norm=None,
        clip_percentile=95.0,
        buffer_size=1000,
    ):
        self.max_norm = max_norm
        self.clip_percentile = clip_percentile
        self.buffer_size = buffer_size
        
        self._grad_norm = np.zeros(buffer_size, dtype=np.float32)
        self._max_norm = max_norm
        self._buffer_ptr = 0
        self._buffer_length = 0

    def __repr__(self):
        return f'AdaptiveGradClipper(max_norm={self.max_norm}, clip_percentile={self.clip_percentile})'
        
    def state_dict(self):
        return {
            'grad_norm': self._grad_norm,
            'max_norm': self._max_norm,
            'buffer_ptr': self._buffer_ptr,
            'buffer_length': self._buffer_length,
        }

    def load_state_dict(self, state_dict):
        self._grad_norm = state_dict['grad_norm']
        self._max_norm = state_dict['max_norm']
        self._buffer_ptr = state_dict['buffer_ptr']
        self._buffer_length = state_dict['buffer_length']

    def log(self):
        return {
            'max_norm': self._max_norm,
        }

    def __call__(self, parameters, norm_type=2.0, error_if_nonfinite=False, foreach=None):
        """Clip the gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        max_norm = self._max_norm if self._max_norm is not None else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach)
        
        if torch.isfinite(grad_norm):
            self._grad_norm[self._buffer_ptr] = grad_norm
            self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_size
            self._buffer_length = min(self._buffer_length + 1, self.buffer_size)
            if self._buffer_length == self.buffer_size:
                self._max_norm = np.percentile(self._grad_norm, self.clip_percentile)
                self._max_norm = min(self._max_norm, self.max_norm) if self.max_norm is not None else self._max_norm
        
        return grad_norm