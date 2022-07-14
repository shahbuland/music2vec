from transformers import AutoModel, AutoConfig
from transformers import Wav2Vec2FeatureExtractor

from music2vec.configs import EncoderConfig
from music2vec.model.encoders import (
    Encoder,
    register_encoder
)

import torch
from torch import nn
from torchtyping import TensorType

@register_encoder
class Data2VecEncoder(Encoder):
    def __init__(self, cfg : EncoderConfig):
        super().__init__(cfg)

        model_config = AutoConfig.from_pretrained(cfg.path)
        self.model = AutoModel.from_config(model_config)
        self.extractor = Wav2Vec2FeatureExtractor()
    
    def prep(self, wf : TensorType["batch", "samples", "channels"]) -> torch.Tensor:
        """
        Preprocess waveform batch for encoding
        """
        

