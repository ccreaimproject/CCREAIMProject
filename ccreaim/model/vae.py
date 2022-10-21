import torch
from torch import nn
from torch.nn import functional as F

<<<<<<< HEAD:model/vae.py
from model import ae
from utils import cfg_classes
=======
from . import ae
>>>>>>> a1c4262 (package restructure):ccreaim/model/vae.py


class VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reparam: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.reparam = reparam
        self.decoder = decoder

    def forward(self, data: torch.Tensor):
        e = self.encoder(data)
        z, mu, sigma = self.reparam(e)
        d = self.decoder(z)
        return d, mu, sigma


class Reparametrization(nn.Module):
    def __init__(self, in_out_len: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_out_len = in_out_len
        self.fc_mu = nn.Linear(self.in_out_len * latent_dim, self.latent_dim)
        self.fc_sig = nn.Linear(self.in_out_len * latent_dim, self.latent_dim)
        self.fc_z = nn.Linear(self.latent_dim, self.in_out_len * latent_dim)

    def forward(self, data: torch.Tensor):
        flat_data = data.flatten(start_dim=1)
        mu = self.fc_mu(flat_data)
        log_sigma = self.fc_sig(flat_data)
        std = torch.exp(0.5 * log_sigma)
        # sample = torch.normal(mu, std)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        fc_z_out = self.fc_z(sample)
        z = fc_z_out.view(-1, self.latent_dim, self.in_out_len)
        # z = self.fc_z(sample).view(-1, self.latent_dim, self.in_out_len)
        # raise Exception(f"Data shape: {data.shape}\nLatent_dim: {self.latent_dim}\nFlat data shape: {flat_data.shape}\nIn out len: {self.in_out_len}\nSample shape: {sample.shape}\nfc_z_out_shape: {fc_z_out.shape}\nz shape: {z.shape}\n{self}")
        return z, mu, std


def _create_vae(hyper_cfg: HyperConfig) -> VAE:
    encoder = ae.Encoder(hyper_cfg.seq_len, hyper_cfg.latent_dim)
    decoder = ae.Decoder(
        hyper_cfg.seq_len, hyper_cfg.latent_dim, encoder.output_lengths
    )
    reparam = Reparametrization(encoder.output_lengths[-1], hyper_cfg.latent_dim)
    return VAE(encoder, decoder, reparam)


<<<<<<< HEAD
<<<<<<< HEAD
def _create_res_vae(cfg: cfg_classes.BaseConfig):
    encoder = ae.get_res_encoder(cfg)
    decoder = ae.get_res_decoder(cfg)
    assert cfg.hyper.res_ae.levels == 1, "Res-VAE with multiple levels not supported"
    encoder_out_seq_len = cfg.hyper.seq_len // (
        cfg.hyper.res_ae.strides_t[0] ** cfg.hyper.res_ae.downs_t[0]
    )
    reparam = Reparametrization(encoder_out_seq_len, cfg.hyper.latent_dim)
    return ResVAE(encoder, decoder, reparam)
=======
def _create_res_vae(hyper_cfg: HyperConfig):
=======
def _create_res_vae(hyper_cfg: HyperConfig) -> VAE:
>>>>>>> a07db98 (implement pre-trained load, refactor model creation)
    encoder = ae.get_res_encoder(hyper_cfg)
    decoder = ae.get_res_decoder(hyper_cfg)
    assert hyper_cfg.res_ae.levels == 1, "Res-VAE with multiple levels not supported"
    encoder_out_seq_len = ae.res_encoder_output_seq_length(hyper_cfg)
    reparam = Reparametrization(encoder_out_seq_len, hyper_cfg.latent_dim)
    return VAE(encoder, decoder, reparam)
>>>>>>> e05277c (Divide res-ae implementation into multilevel/single-level versions)


<<<<<<< HEAD
def get_vae(name: str, cfg: cfg_classes.BaseConfig):
    if name == "base":
        return _create_vae(cfg.hyper.seq_len, cfg.hyper.latent_dim)
    elif name == "res-vae":
        return _create_res_vae(cfg)
=======
def get_vae(hyper_cfg: HyperConfig) -> VAE:
    if hyper_cfg.model == "vae":
        return _create_vae(hyper_cfg)
    elif hyper_cfg.model == "res-vae":
        return _create_res_vae(hyper_cfg)
>>>>>>> a07db98 (implement pre-trained load, refactor model creation)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(hyper_cfg.model))
