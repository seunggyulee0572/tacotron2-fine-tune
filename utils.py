import numpy as np
from scipy.io.wavfile import read
import torch


# def get_mask_from_lengths(lengths):
#     max_len = torch.max(lengths).item()
#     ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
#     mask = (ids < lengths.unsqueeze(1)).bool()
#     return mask
def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """
    lengths: (B,) int64 on any device (cpu/cuda/mps)
    returns: (B, T_max) boolean mask where True means "valid (not padded)"
    """
    
    if lengths.ndim != 1:
        lengths = lengths.view(-1)
    max_len = int(lengths.max().item())
    # arange on the SAME device & dtype
    ids = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    # (B, T_max) broadcast; boolean mask
    mask = ids.unsqueeze(0) < lengths.unsqueeze(1)
    return mask

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
