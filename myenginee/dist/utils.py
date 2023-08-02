from typing import Optional
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup

def is_distributed() -> bool:
    """"""
    return torch_dist.is_available() and torch_dist.is_initialized()

def get_default_group() -> Optional[ProcessGroup]:
    
    return torch_dist.distributed_c10d._get_default_group()

def get_rank(group: Optional[ProcessGroup] = None) -> int:
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0