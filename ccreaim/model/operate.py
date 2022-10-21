import logging
from typing import Union

import torch
from torch.nn import functional as F

from ..utils import util
from ..utils.cfg_classes import HyperConfig
from . import decoder_only, transformer

log = logging.getLogger(__name__)


def step(
    model: torch.nn.Module,
    batch: Union[tuple[torch.Tensor, str], tuple[torch.Tensor, str, torch.Tensor]],
    device: torch.device,
    hyper_cfg: HyperConfig,
    batchnum: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    info: dict[str, float] = {}
    if "bank-classifier" in hyper_cfg.model:
        features, inds, _ = batch
        features = features.to(device)
        inds = inds.long().to(device)

        # Insert the zero-token, remove the last token
        tgt = torch.cat(
            (
                torch.zeros_like(features[:, 0:1, :], device=features.device),
                features[:, :-1, :],
            ),
            dim=1,
        )

        # Create the causal mask
        tgt_mask = util.get_tgt_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(device)
        pred = model(tgt, tgt_mask=tgt_mask)

        # Reshape the transformer output and indices for the loss function,
        # so that each correct index matches the corresponding logit output from
        # the transformer
        pred = pred.view(-1, hyper_cfg.transformer.vocab_size)
        inds = inds.view(-1)
        trf_auto = F.cross_entropy(pred, inds)

        loss = trf_auto
    else:
        raise ValueError(f"Model type {hyper_cfg.model} is not defined!")

    return loss, pred, info


def get_model_init_function(hyper_cfg: HyperConfig):
    # Model init function mapping
<<<<<<< HEAD
    if hyper_cfg.model == "ae":
<<<<<<< HEAD
        get_model = lambda: ae.get_autoencoder(
            "base", hyper_cfg.seq_len, hyper_cfg.latent_dim
        )
    elif hyper_cfg.model == "res-ae":
        get_model = lambda: ae.get_autoencoder(
            "res-ae", hyper_cfg.seq_len, hyper_cfg.latent_dim
        )
    elif hyper_cfg.model == "vae":
        get_model = lambda: vae.get_vae("base", hyper_cfg.seq_len, hyper_cfg.latent_dim)
    elif hyper_cfg.model == "vq-vae":
<<<<<<< HEAD
        get_model = lambda: vqvae.get_vqvae(
            "base", hyper_cfg.seq_len, hyper_cfg.latent_dim
        )
=======
        get_model = lambda: vqvae.get_vqvae("base", hyper_cfg)
    elif hyper_cfg.model == "res-vq-vae":
        get_model = lambda: vqvae.get_vqvae("res-vqvae", hyper_cfg)
>>>>>>> 5116081 (Jukebox implementation of VQ-VAE with just autoencoder)
    elif hyper_cfg.model == "transformer":
        get_model = lambda: transformer.get_transformer("base", hyper_cfg.latent_dim)
=======
        get_model = lambda: ae.get_autoencoder(hyper_cfg)
    elif hyper_cfg.model == "res-ae":
        get_model = lambda: ae.get_autoencoder(hyper_cfg)
    elif hyper_cfg.model == "vae":
        get_model = lambda: vae.get_vae(hyper_cfg)
    elif hyper_cfg.model == "res-vae":
        get_model = lambda: vae.get_vae(hyper_cfg)
    elif hyper_cfg.model == "vq-vae":
        get_model = lambda: vqvae.get_vqvae(hyper_cfg)
    elif hyper_cfg.model == "res-vqvae":
        get_model = lambda: vqvae.get_vqvae(hyper_cfg)
    elif hyper_cfg.model == "transformer":
        get_model = lambda: transformer.get_transformer(hyper_cfg)
<<<<<<< HEAD
>>>>>>> a07db98 (implement pre-trained load, refactor model creation)
=======
    elif hyper_cfg.model == "transformer-decoder-only":
        get_model = lambda: decoder_only.get_decoder(hyper_cfg)
>>>>>>> d99644d (Work-in-progress for decoder-only transformer with vqvae)
    elif hyper_cfg.model == "e2e":
        get_model = lambda: e2e.get_e2e(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked":
        get_model = lambda: e2e_chunked.get_e2e_chunked(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked_res-ae":
        get_model = lambda: e2e_chunked.get_e2e_chunked(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked_res-vqvae":
        get_model = lambda: e2e_chunked.get_e2e_chunked(hyper_cfg)
    elif hyper_cfg.model == "bank-classifier":
=======
    if hyper_cfg.model == "bank-classifier":
>>>>>>> 7c70b3e (Removed a lot of nonrelevant code and did a bunch of commenting)
        get_model = lambda: decoder_only.get_decoder(hyper_cfg)
    else:
        raise ValueError(f"Model type {hyper_cfg.model} is not defined!")
    return get_model
