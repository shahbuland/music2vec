from dataclasses import dataclass
from  typing import Iterable

import math
import einops as eo

import torch
from torchtyping import TensorType

@dataclass
class BatchElement:
    wf : TensorType["batch", "samples"] # Waveform batch
    mask : None

def batch_waveform(wf : TensorType["channels", "samples"],
                   samples_per_batch : int) -> TensorType["batch", "channels", "samples//batch"]:
    """
    Batch waveform into batches of size samples_per_batch
    """

    c, n = wf.shape
    n_batches = math.ceil(n / samples_per_batch)
    # Need to pad final batch to make it have samples_per_batch samples
    pad_len = (n_batches * samples_per_batch) - n
    pad = torch.zeros_like(wf[:, :pad_len])

    wf = torch.cat([wf, pad], dim=1)

    return eo.rearrange(wf, "c (b n) -> b c n", n = (n_batches * samples_per_batch))

def pad_to_max(L : Iterable[TensorType["channels", "samples_i"]]):
    """
    Given a list of waveform arrays, pad them to the length of the longest one
    """

def discard_silent_batches(wf : TensorType["batch", "channels", "samples"]):
    """
    Discard batches that are entirely silent
    """
    
    is_silent = torch.all(wf == 0, dim=2)
    return wf[~is_silent]


    

