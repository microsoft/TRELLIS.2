import torch
import torch.nn as nn

# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def str_to_dtype(dtype_str: str):
    return {
        'f16': torch.float16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'f32': torch.float32,
        'fp32': torch.float32,
        'float32': torch.float32,
    }[dtype_str]
    

def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )
    

def zero_grad(model_params):
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()
            

# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0
        