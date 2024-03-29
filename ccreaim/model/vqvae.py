import logging

import torch
from torch import nn
from torch.nn import functional as F

<<<<<<< HEAD:model/vqvae.py
from model import ae
from utils import cfg_classes
=======
from . import ae
>>>>>>> a1c4262 (package restructure):ccreaim/model/vqvae.py

"""
Reference: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
"""

log = logging.getLogger(__name__)


class VQVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, vq: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.vq = vq
        self.decoder = decoder

    def forward(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.encoder(data)
        quantized_latents, _ = self.vq(e)
        quantized_latents_res = e + (quantized_latents - e).detach()
        d = self.decoder(quantized_latents_res)
        return d, e, quantized_latents

    def encode(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.encoder(data)
        quantized_latents, vq_inds = self.vq(e)
        return quantized_latents, vq_inds


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, reset_patience: float):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
        self.reset_patience = reset_patience
        self.register_buffer(
            "usage", torch.full((self.K,), 1.0 / self.K), persistent=False
        )

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = latents.transpose(1, 2).contiguous()  # [B x D x S] -> [B x S x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.cdist(latents, self.embedding.weight, p=2)  # [B x S x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=-1)  # [B x S]
        # Quantize the latents
        quantized_latents = self.embedding(encoding_inds)
        if self.training and self.reset_patience > 0:
            # Update usage and reset unused latents
            self.update_and_reset(encoding_inds, latents)
        # [B x D x S]
        return quantized_latents.transpose(1, 2).contiguous(), encoding_inds

    def update_and_reset(self, ids: torch.Tensor, latents: torch.Tensor) -> None:
        flat_latents = latents.view(-1, self.D)
        num_latent_vectors = flat_latents.size(0)
        self.usage = self.usage * 0.5
        unique_ids, counts = torch.unique(ids, return_counts=True)
        i = torch.arange(0, len(unique_ids))
        self.usage[unique_ids] = self.usage[unique_ids[i]] + 0.5 * (
            counts[i] / num_latent_vectors
        )
        reset_mask = self.usage < (1.0 / (self.K * self.reset_patience))
        reset_num = torch.count_nonzero(reset_mask).item()
        if reset_num > 0:
            if reset_num <= num_latent_vectors:
                random_ids = torch.randperm(num_latent_vectors)
                random_ids = random_ids[:reset_num]
            else:
                random_ids = torch.randint(
                    low=0, high=num_latent_vectors, size=(reset_num,)
                )
            with torch.no_grad():
                self.embedding.weight.data[reset_mask] = flat_latents[random_ids]
            self.usage[reset_mask] = 1.0 / self.K

    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


def _create_vqvae(hyper_cfg: HyperConfig) -> VQVAE:
    encoder = ae.Encoder(hyper_cfg.seq_len, hyper_cfg.latent_dim)
    decoder = ae.Decoder(
        hyper_cfg.seq_len, hyper_cfg.latent_dim, encoder.output_lengths
    )
    reparam = VectorQuantizer(
        hyper_cfg.vqvae.num_embeddings,
        hyper_cfg.latent_dim,
        hyper_cfg.vqvae.reset_patience,
    )
    return VQVAE(encoder, decoder, reparam)


def _create_res_vqvae(hyper_cfg: HyperConfig) -> VQVAE:
    encoder = ae.get_res_encoder(hyper_cfg)
    decoder = ae.get_res_decoder(hyper_cfg)
    reparam = VectorQuantizer(
        hyper_cfg.vqvae.num_embeddings,
        hyper_cfg.latent_dim,
        hyper_cfg.vqvae.reset_patience,
    )
    return VQVAE(encoder, decoder, reparam)


<<<<<<< HEAD
def get_vqvae(name: str, cfg: cfg_classes.BaseConfig):
    if name == "base":
<<<<<<< HEAD
<<<<<<< HEAD
        return _create_vqvae(cfg.hyper.seq_len, cfg.hyper.latent_dim)
=======
        return _create_vqvae(
            hyper_cfg.seq_len, hyper_cfg.latent_dim, hyper_cfg.vqvae.num_embeddings
        )
<<<<<<< HEAD
>>>>>>> 49078d8 (vqvae implementation, config refactor)
=======
=======
        return _create_vqvae(hyper_cfg)
>>>>>>> 652e702 (add vq coodbook reset implementation)
    elif name == "res-vqvae":
=======
def get_vqvae(hyper_cfg: HyperConfig) -> VQVAE:
    if hyper_cfg.model == "vq-vae":
        return _create_vqvae(hyper_cfg)
    elif hyper_cfg.model == "res-vqvae":
>>>>>>> a07db98 (implement pre-trained load, refactor model creation)
        return _create_res_vqvae(hyper_cfg)
>>>>>>> 5116081 (Jukebox implementation of VQ-VAE with just autoencoder)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(hyper_cfg.model))
