
import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torchtyping import TensorType

from music2vec.configs import EncoderConfig

# Dictionary of all possible encoder types
_ENCODERS: Dict[str, any] = {}


def register_encoder(name):
    """Decorator used to register in encoder in the above dictionary
    Args:
        name: Name of the encoder
    """

    def register_class(cls, name):
        _ENCODERS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

class Encoder(nn.Module):
    def __init__(self, cfg : EncoderConfig):
        self.cfg = cfg

    @abstractmethod
    def prep(self, wf : TensorType["batch", "samples", "channels"]) -> torch.Tensor:
        """
        Preprocess waveform batch for encoding
        """
        pass

    @abstractmethod
    def forward(self, x : torch.Tensor) -> TensorType["batch", "latent"]:
        """
        Take processed waveform batch and get encoding
        """
        pass

    def encode(self, wf : TensorType["batch", "samples", "channels"]) \
        -> TensorType["batch", "latent"]:
        """
        Encode waveform directly into latent
        """
        return self.forward(self.prep(wf))

    def save(cls, obj, path : str, obj_name : str):
        """
        Attempt to save to path. Alert but do not interrupt if failure occurs.
        """
        try:
            torch.save(obj, path + obj_name)
        except:
            print(f"Warning: failed to save {obj_name}")
    
    def load(cls, obj, path : str, obj_name : str):
        """
        Attempt to load. Alert but do not interrupt if failure occurs.
        """
        try:
            return torch.load(path + obj_name, map_location = "cpu") # apparently load cpu -> gpu better
        except:
            print(f"Warning: failed to load {obj_name}")
            
def get_encoder(name):
    return _ENCODERS[name.lower()]

def get_encoder_names():
    return _ENCODERS.keys()