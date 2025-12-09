import os
import io
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_dist(rank, local_rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    

def read_file_dist(path):
    """
    Read the binary file distributedly.
    File is only read once by the rank 0 process and broadcasted to other processes.

    Returns:
        data (io.BytesIO): The binary data read from the file.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        # read file
        size = torch.LongTensor(1).cuda()
        if dist.get_rank() == 0:
            with open(path, 'rb') as f:
                data = f.read()
            data = torch.ByteTensor(
                torch.UntypedStorage.from_buffer(data, dtype=torch.uint8)
            ).cuda()
            size[0] = data.shape[0]
        # broadcast size
        dist.broadcast(size, src=0)
        if dist.get_rank() != 0:
            data = torch.ByteTensor(size[0].item()).cuda()
        # broadcast data
        dist.broadcast(data, src=0)
        # convert to io.BytesIO
        data = data.cpu().numpy().tobytes()
        data = io.BytesIO(data)
        return data
    else:
        with open(path, 'rb') as f:
            data = f.read()
        data = io.BytesIO(data)
        return data
    

def unwrap_dist(model):
    """
    Unwrap the model from distributed training.
    """
    if isinstance(model, DDP):
        return model.module
    return model


@contextmanager
def master_first():
    """
    A context manager that ensures master process executes first.
    """
    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
            

@contextmanager
def local_master_first():
    """
    A context manager that ensures local master process executes first.
    """
    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() % torch.cuda.device_count() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
    