import torch
from accelerate import PartialState


def get_rank():
    state = PartialState()
    rank = state.process_index  # The index of the current process.
    return rank


def get_device(require_cuda=True):
    # LLMs require a GPU to run
    is_cuda = torch.cuda.is_available()
    if require_cuda and not is_cuda:
        raise RuntimeError("This script requires a GPU to run.")

    rank = get_rank()
    device = f"cuda:{rank}"
    return device
