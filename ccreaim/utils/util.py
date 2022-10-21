import io
import re
import logging
import math
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

from ..model import decoder_only, transformer
from .cfg_classes import BaseConfig, HyperConfig

log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)


"""
Both the chop_sample and chop_dataset utilities are pretty much unused with the bank-classifier implementation.
They could be useful for something in the future however
"""


def chop_sample(sample: torch.Tensor, sample_length: int) -> list[torch.Tensor]:
    if len(sample.size()) != 1:
        sample = torch.mean(sample, 0)
    assert len(sample.size()) == 1, "Sample is not 1 dimensional" + str(sample.size())
    chopped_samples_list: list[torch.Tensor] = []
    n_chops = len(sample) // sample_length
    for s in range(n_chops):
        chopped_samples_list.append(sample[s * sample_length : (s + 1) * sample_length])
    remainder = sample[n_chops * sample_length :]
    if remainder.size(0) > 0:
        chopped_samples_list.append(remainder)
    assert sum([len(chopped_sample) for chopped_sample in chopped_samples_list]) == len(
        sample
    ), f"Chopping did not maintain total sample length ({len(sample)})."
    return chopped_samples_list


def chop_dataset(in_root: str, out_tar_file_path: str, ext: str, sample_length: int):
    samples_paths = get_sample_path_list(Path(in_root), ext)
    with tarfile.open(out_tar_file_path, "a") as out_tar:
        for pth in samples_paths:
            try:
                full_sample, sample_rate = torchaudio.load(str(pth), format=ext)  # type: ignore
            except RuntimeError as e:
                log.warn(f"Could not open file, with error: {e}")
                continue

            try:
                chopped_samples = chop_sample(full_sample.squeeze(), sample_length)
                for i, cs in enumerate(chopped_samples):
                    out_name = str(pth.stem) + f"_{i:03d}" + ".wav"
                    with io.BytesIO() as buffer:
                        try:
                            torchaudio.save(  # type: ignore
                                buffer,
                                cs.unsqueeze(0),
                                sample_rate,
                                encoding="PCM_F",
                                bits_per_sample=32,
                                format="wav",
                            )
                            buffer.seek(0)  # go to the beginning for reading the buffer
                            out_info = tarfile.TarInfo(name=out_name)
                            out_info.size = buffer.getbuffer().nbytes
                            out_tar.addfile(tarinfo=out_info, fileobj=buffer)
                        except Exception as e:
                            log.error(e)
            except:
                print(f"Couldn't produce samples from path {pth}")


def save_to_tar(
    out_tar: tarfile.TarFile,
    data: dict[str, torch.Tensor],
    data_name: str,
):

    with io.BytesIO() as buffer:
        try:
            torch.save(data, buffer)
            buffer.seek(0)  # go to the beginning for reading the buffer
            out_info = tarfile.TarInfo(name=data_name)
            out_info.size = buffer.getbuffer().nbytes
            out_tar.addfile(tarinfo=out_info, fileobj=buffer)
        except Exception as e:
            log.error(e)


def save_model_prediction(model_name: str, pred: torch.Tensor, save_path: Path) -> None:
    try:
        if model_name == "transformer":
            torch.save(pred, save_path)
        else:
            torchaudio.save(  # type: ignore
                save_path, pred, 16000, encoding="PCM_F", bits_per_sample=32
            )
    except Exception as e:
        log.error(e)
        
        
# Returns files in ascending order w.r.t. the integer values that appear in the filenames of the training set files
def extract_numbers(filename, path):
    pattern = r'{}/sample_(\d+)_context_(\d+)(?:_aug(\d+))?\.pt'.format(path)
    match = re.match(pattern, str(filename))
    if match:
        id, j, k = match.groups()
        id, j = map(int, (id, j))
        k = int(k) if k else 0
        return (id, j, k)
    else:
        raise ValueError(f'Invalid filename format: {str(filename)}')
        

# Returns a sorted list of all the files in data_root that have extension ext
def get_sample_path_list(data_root: Path, ext: str = "mp3") -> list[Path]:
    print(data_root, len(list(data_root.rglob(f"*.{ext}"))))
    files = list(data_root.rglob(f"*.{ext}"))
    return sorted(files, key=lambda fn: extract_numbers(fn, data_root))


# Returns a sorted list of all the files in data_root that have extension ext
def get_sample_path_list_orig(data_root: Path, ext: str = "mp3") -> list[Path]:
    print(data_root, len(list(data_root.rglob(f"*.{ext}"))))
    return sorted(list(data_root.rglob(f"*.{ext}")))


# Returns the path to the directory where a model is exported to/imported from according
# to configuration in cfg, as well as the base name of the model.
def get_model_path(cfg: BaseConfig):
    exp_path = Path(cfg.logging.model_checkpoints)
    model_name = f"{cfg.hyper.model}_seqlen-{cfg.hyper.seq_len}_bs-{cfg.hyper.batch_size}_lr-{cfg.hyper.learning_rate}_seed-{cfg.hyper.seed}"
    return exp_path, model_name


def load_pre_trained_transformer(
    hyper_cfg: HyperConfig, trf: transformer.Transformer
) -> transformer.Transformer:
    checkpoint = torch.load(hyper_cfg.pre_trained_transformer_path, map_location="cpu")
    pretrained_state_dict = checkpoint["model_state_dict"]
    hyper_cfg_schema = OmegaConf.structured(HyperConfig)
    conf = OmegaConf.create(checkpoint["hyper_config"])
    pretrained_hyper_cfg = OmegaConf.merge(hyper_cfg_schema, conf)

    if (
        hyper_cfg.latent_dim == pretrained_hyper_cfg.latent_dim
        and hyper_cfg.vqvae.num_embeddings == pretrained_hyper_cfg.vqvae.num_embeddings
        and hyper_cfg.transformer.num_heads_latent_dimension_div
        == pretrained_hyper_cfg.transformer.num_heads_latent_dimension_div
        and hyper_cfg.transformer.num_enc_layers
        == pretrained_hyper_cfg.transformer.num_enc_layers
        and hyper_cfg.transformer.num_dec_layers
        == pretrained_hyper_cfg.transformer.num_dec_layers
        and hyper_cfg.transformer.linear_map
        == pretrained_hyper_cfg.transformer.linear_map
    ):
        trf.load_state_dict(pretrained_state_dict)
        log.info(
            f"Loaded Transformer weights from {hyper_cfg.pre_trained_transformer_path}"
        )
        return trf
    else:
        raise ValueError(
            f"Pre-trained config is not matching current config:\n"
            "\t\t\t\t\tCurrent config\t---\tPre-trained config\n"
            "latent_dim:\t\t\t\t\t"
            f"{hyper_cfg.latent_dim}"
            "\t---\t"
            f"{pretrained_hyper_cfg.latent_dim}\n"
            "vqvae.num_embeddings:\t\t\t\t"
            f"{hyper_cfg.vqvae.num_embeddings}"
            "\t---\t"
            f"{pretrained_hyper_cfg.vqvae.num_embeddings}\n"
            "transformer.num_heads_latent_dimension_div: \t"
            f"{hyper_cfg.transformer.num_heads_latent_dimension_div}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_heads_latent_dimension_div} \n"
            "transformer.num_enc_layers: \t\t\t"
            f"{hyper_cfg.transformer.num_enc_layers}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_enc_layers}\n"
            "transformer.num_dec_layers: \t\t\t"
            f"{hyper_cfg.transformer.num_dec_layers}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_dec_layers}\n"
            "transformer.linear_map:\t\t\t\t"
            f"{hyper_cfg.transformer.linear_map}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.linear_map}\n"
        )


def load_pre_trained_decoder_only(
    hyper_cfg: HyperConfig, trf: decoder_only.CachedDecoderOnly
) -> decoder_only.CachedDecoderOnly:
    checkpoint = torch.load(hyper_cfg.pre_trained_decoder_only_path, map_location="cpu")
    pretrained_state_dict = checkpoint["model_state_dict"]
    hyper_cfg_schema = OmegaConf.structured(HyperConfig)
    conf = OmegaConf.create(checkpoint["hyper_config"])
    pretrained_hyper_cfg = OmegaConf.merge(hyper_cfg_schema, conf)

    if (
        hyper_cfg.latent_dim == pretrained_hyper_cfg.latent_dim
        and hyper_cfg.vqvae.num_embeddings == pretrained_hyper_cfg.vqvae.num_embeddings
        and hyper_cfg.transformer.num_heads_latent_dimension_div
        == pretrained_hyper_cfg.transformer.num_heads_latent_dimension_div
        and hyper_cfg.transformer.num_dec_layers
        == pretrained_hyper_cfg.transformer.num_dec_layers
        and hyper_cfg.transformer.linear_map
        == pretrained_hyper_cfg.transformer.linear_map
    ):
        trf.load_state_dict(pretrained_state_dict)
        log.info(
            f"Loaded Decoder-only weights from {hyper_cfg.pre_trained_transformer_path}"
        )
        return trf
    else:
        raise ValueError(
            f"Pre-trained config is not matching current config:\n"
            "\t\t\t\t\tCurrent config\t---\tPre-trained config\n"
            "latent_dim:\t\t\t\t\t"
            f"{hyper_cfg.latent_dim}"
            "\t---\t"
            f"{pretrained_hyper_cfg.latent_dim}\n"
            "vqvae.num_embeddings:\t\t\t\t"
            f"{hyper_cfg.vqvae.num_embeddings}"
            "\t---\t"
            f"{pretrained_hyper_cfg.vqvae.num_embeddings}\n"
            "transformer.num_heads_latent_dimension_div: \t"
            f"{hyper_cfg.transformer.num_heads_latent_dimension_div}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_heads_latent_dimension_div} \n"
            "transformer.num_dec_layers: \t\t\t"
            f"{hyper_cfg.transformer.num_dec_layers}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_dec_layers}\n"
            "transformer.linear_map:\t\t\t\t"
            f"{hyper_cfg.transformer.linear_map}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.linear_map}\n"
        )

<<<<<<< HEAD
<<<<<<< HEAD

# Spectral loss
class STFTValues:
    def __init__(self, n_bins: int, hop_length: int, window_size: int):
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.window_size = window_size


def norm(x: torch.Tensor):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def spec(seq: torch.Tensor, stft_val: STFTValues):
    return torch.norm(
        torch.stft(
            seq,
            stft_val.n_bins,
            stft_val.hop_length,
            win_length=stft_val.window_size,
            window=torch.hann_window(stft_val.window_size, device=seq.device),
        ),
        p=2,
        dim=-1,
    )


def spectral_loss(
    seq: torch.Tensor, pred: torch.Tensor, spectral_loss_cfg: SpectralLossConfig
) -> torch.Tensor:
    stft_val = STFTValues(
        spectral_loss_cfg.stft_bins[0],
        spectral_loss_cfg.stft_hop_length[0],
        spectral_loss_cfg.stft_window_size[0],
    )
    spec_in = spec(seq.float().squeeze(), stft_val)
    spec_out = spec(pred.float().squeeze(), stft_val)
    return norm(spec_in - spec_out)


def multispectral_loss(
    seq: torch.Tensor, pred: torch.Tensor, spectral_loss_cfg: SpectralLossConfig
) -> torch.Tensor:
    losses = torch.zeros(*seq.size()[:-1], device=seq.device)
    if losses.ndim == 1:
        losses = losses.unsqueeze(-1)
        seq = seq.unsqueeze(1)
        pred = pred.unsqueeze(1)
    args = (
        spectral_loss_cfg.stft_bins,
        spectral_loss_cfg.stft_hop_length,
        spectral_loss_cfg.stft_window_size,
    )
    for n_bins, hop_length, window_size in zip(*args):
        stft_val = STFTValues(n_bins, hop_length, window_size)
        for i in range(losses.size(-1)):
            spec_in = spec(seq[:, i], stft_val)
            spec_out = spec(pred[:, i], stft_val)
            losses[:, i] = norm(spec_in - spec_out)
    return losses
<<<<<<< HEAD
<<<<<<< HEAD


def step(
    model: torch.nn.Module,
    batch: Union[tuple[torch.Tensor, str], tuple[torch.Tensor, str, torch.Tensor]],
    device: torch.device,
    cfg: cfg_classes.BaseConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    info: dict[str, float] = {}
    if isinstance(model, transformer.Transformer):
        seq, _ = batch
        seq = seq.to(device)
        src = seq[:, :-1, :]
        tgt = seq[:, 1:, :]
        tgt_mask = model.get_tgt_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(device)
        pred = model(src, tgt, tgt_mask)
        loss = F.mse_loss(pred, tgt)
    elif isinstance(model, e2e_chunked.E2EChunked):
        seq, _, pad_mask = batch
        seq = seq.to(device)
        pad_mask = pad_mask.to(device)
        pred = model(seq, pad_mask)
        tgt = seq[:, 1:, :]
        tgt_pad_mask = pad_mask[:, 1:]
        mse = F.mse_loss(pred, tgt, reduction="none")
        mse[tgt_pad_mask] = 0
        mse = mse.mean()
        spec_weight = cfg.hyper.spectral_loss.weight
        multi_spec = multispectral_loss(tgt, pred, cfg)
        multi_spec[tgt_pad_mask] = 0
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_spectral": spec_weight * multi_spec.item(),
            }
        )
        loss = mse + spec_weight * multi_spec
    elif isinstance(model, vae.VAE) or isinstance(model, vae.ResVAE):
        seq, _ = batch
        seq = seq.to(device)
        pred, mu, sigma = model(seq)
        mse = F.mse_loss(pred, seq)
        kld_weight = cfg.hyper.kld_loss.weight
        kld = -0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()
        spec_weight = cfg.hyper.spectral_loss.weight
        multi_spec = multispectral_loss(seq, pred, cfg)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_kld": float((kld_weight * kld).item()),
                "loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
<<<<<<< HEAD
        loss = mse + kld_weight * kld + spec_weight * multi_spec
=======
        loss = mse + model.kld_weight * kld + spec_weight * multi_spec

>>>>>>> 1d6844d (e2e chunked implementation, chopping bugfix)
    else:
        seq, _ = batch
        seq = seq.to(device)
        pred = model(seq)
        mse = F.mse_loss(pred, seq)
        spec_weight = cfg.hyper.spectral_loss.weight
        multi_spec = multispectral_loss(seq, pred, cfg)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
        loss = mse + spec_weight * multi_spec
    return loss, pred, info
=======
>>>>>>> 120e2cd (step function to separate file, audio tools added)
=======


def spectral_convergence(
    seq: torch.Tensor,
    pred: torch.Tensor,
    spectral_loss_cfg: SpectralLossConfig,
    epsilon: float = 2e-3,
) -> torch.Tensor:
    stft_val = STFTValues(
        spectral_loss_cfg.stft_bins[0],
        spectral_loss_cfg.stft_hop_length[0],
        spectral_loss_cfg.stft_window_size[0],
    )
    spec_in = spec(seq.float().squeeze(), stft_val)
    spec_out = spec(pred.float().squeeze(), stft_val)
    gt_norm = norm(spec_in)
    residual_norm = norm(spec_in - spec_out)
    mask = (gt_norm > epsilon).float()
    return (residual_norm * mask) / torch.clamp(gt_norm, min=epsilon)


def log_magnitude_loss(
    seq: torch.Tensor,
    pred: torch.Tensor,
    spectral_loss_cfg: SpectralLossConfig,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    stft_val = STFTValues(
        spectral_loss_cfg.stft_bins[0],
        spectral_loss_cfg.stft_hop_length[0],
        spectral_loss_cfg.stft_window_size[0],
    )
    spec_in = torch.log(spec(seq.float().squeeze(), stft_val) + epsilon)
    spec_out = torch.log(spec(pred.float().squeeze(), stft_val) + epsilon)
    return torch.abs(spec_in - spec_out)
<<<<<<< HEAD
>>>>>>> c2e5d33 (spectral losses extended)
=======


=======
>>>>>>> 7c70b3e (Removed a lot of nonrelevant code and did a bunch of commenting)
=======

>>>>>>> 07cd83c (Ran pre-commit to fix formatting)
def get_tgt_mask(size: int) -> torch.Tensor:
    # Generates a square matrix where each row allows one more word to be seen
    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
    return mask
>>>>>>> a9374fa (decoder only added from 8bit branch)
