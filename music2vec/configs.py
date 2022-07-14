from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

@dataclass
class EncoderConfig:
    latent_dim : int
    encoder_type : str
    path : str = None # For transformers.AutoModel

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

@dataclass
class TrainConfig:
    lr_ramp_steps : int
    lr_decay_steps : int
    lr_init : float
    lr_target : float
    learning_rate : float
    batch_size : int

    log_interval : int
    checkpoint_interval : int
    validate_interval : int

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

@dataclass
class Config:
    encoder : EncoderConfig
    training : TrainConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls(
            EncoderConfig.from_dict(config["encoder"]),
            TrainConfig.from_dict(config["training"]),
        )

    def to_dict(self):
        data = self.model.__dict__.copy()
        data.update(self.train_job.__dict__)
        return data

