import logging
import math
from typing import Optional

import torch
from torch import nn

<<<<<<< HEAD
from utils import cfg_classes

log = logging.getLogger(__name__)

=======
>>>>>>> 4752380 (refactor config import and minimal info passing)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int = 1000):
        super().__init__()
        # Taken from https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Residual connection + pos encoding
        return self.dropout(x + self.pos_encoding[:, : x.size(1), :])  # type: ignore


# THIS CLASS IS NOT USED ANYWHERE CURRENTLY, could maybe be removed but somewhat relevant
# to the current model implementation so for now it can stay
class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout_p: float,
        linear_map: bool = False,
        num_embeddings: int = 0,
    ):
        super().__init__()

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=10000
        )
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True,
        )

        if linear_map:
            self.trf_out_to_tokens = nn.Linear(dim_model, num_embeddings)
        else:
            self.trf_out_to_tokens = nn.Identity()

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (batch_size, sequence length, num_tokens)
        trf_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_mask=src_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        out = self.trf_out_to_tokens(trf_out)
        return out

    def get_tgt_mask(self, size: int) -> torch.Tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask


<<<<<<< HEAD
def get_transformer(name: str, cfg: cfg_classes.BaseConfig) -> Transformer:
    if name == "base":
        return Transformer(
            dim_model=cfg.hyper.latent_dim,
            num_heads=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
=======
def get_transformer(hyper_cfg: HyperConfig) -> Transformer:
    if hyper_cfg.model == "transformer":
        return Transformer(
            dim_model=hyper_cfg.latent_dim,
            num_heads=hyper_cfg.latent_dim
            // hyper_cfg.transformer.num_heads_latent_dimension_div,
            num_encoder_layers=hyper_cfg.transformer.num_enc_layers,
            num_decoder_layers=hyper_cfg.transformer.num_dec_layers,
>>>>>>> a07db98 (implement pre-trained load, refactor model creation)
            dropout_p=0.1,
            linear_map=hyper_cfg.transformer.linear_map,
            num_embeddings=hyper_cfg.vqvae.num_embeddings,
        )
    else:
        raise ValueError(f"Transformer model not implemented: {hyper_cfg.model}")
