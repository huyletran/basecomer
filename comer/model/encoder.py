import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import ImgPosEnc

from efficientnet_pytorch import EfficientNet
class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.model = EfficientNet.from_name('efficientnet-b0',in_channels=1)
        self.feature_proj = nn.Conv2d(self.model._fc.in_features, d_model, kernel_size=1)
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature
        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, h, w, d], [b, h, w]
        """
        # extract feature
        feature = self.model.extract_features(img)  # EfficientNet feature extraction
        feature = self.feature_proj(feature)
        # proj
        feature = feature.permute(0, 2, 3, 1)
        # positional encoding
        feature = self.pos_enc_2d(feature, img_mask)
        feature = self.norm(feature)
        # flat to 1-D
        feature = feature.permute(0, 3, 1, 2)
        return feature, img_mask
