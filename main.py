import logging

import hydra
import torch
import torch.utils.data
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

from ccreaim.model import operate
from ccreaim.process.cross_validation import cross_validation
from ccreaim.process.test import test
from ccreaim.utils import cfg_classes, dataset, util

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)


class LogJobReturnCallback(Callback):
    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: cfg_classes.BaseConfig, job_return: JobReturn, **kwargs
    ) -> None:
        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Succeeded with return value: {job_return.return_value}")
        elif job_return.status == JobStatus.FAILED:
            self.log.error("", exc_info=job_return._return_value)
        else:
            self.log.error("Status unknown. This should never happen.")


@hydra.main(version_base=None, config_path="cfg", config_name="base")
def main(cfg: cfg_classes.BaseConfig):
    """The main entry point to the training loop/testing

    Args:
        cfg (BaseConfig): The config object provided by Hydra

    Raises:
        ValueError: if misconfiguration
    """
    log.info(OmegaConf.to_yaml(cfg))

    util.set_seed(cfg.hyper.seed)

<<<<<<< HEAD
<<<<<<< HEAD
    # Fetch the model:
    # if training initialize a new model, if testing load an existing trained one
    if cfg.train:
<<<<<<< HEAD
        if cfg.hyper.model == "ae":
<<<<<<< HEAD
            model = ae.get_autoencoder("base")
        elif cfg.hyper.model == "vae":
<<<<<<< HEAD
=======
        if cfg.model == "ae":
            model = ae.get_autoencoder("base", cfg.seq_length)
        elif cfg.model == "vae":
>>>>>>> e5fb617 (Parametrize Autoencoder input/output length)
            model = None
=======
            model = vae.get_vae("base", cfg.hyper.seq_len)
>>>>>>> 0086cdc (vae fix and smoke test)
=======
            model = ae.get_autoencoder("base", cfg.hyper.seq_len, cfg.hyper.latent_dim)
        elif cfg.hyper.model == "vae":
            model = vae.get_vae("base", cfg.hyper.seq_len, cfg.hyper.latent_dim)
>>>>>>> 3fbde05 (latent dim added as cfg, fix cfg to wandb)
        elif cfg.hyper.model == "vq-vae":
            model = vqvae.get_vqvae("base", cfg.hyper.seq_len, cfg.hyper.latent_dim)
        elif cfg.hyper.model == "transformer":
            model = transformer.get_transformer("base", cfg.hyper.latent_dim)
        elif cfg.hyper.model == "end-to-end":
            model = end_to_end.get_end_to_end(
                "base_ae", cfg.hyper.seq_len, 10, cfg.hyper.latent_dim
            )
        else:
            raise ValueError(f"Model type {cfg.hyper.model} is not defined!")
    else:
        checkpoint = torch.load(cfg.logging.load_model_path, map_location="cpu")
        model = checkpoint["model"]

=======
>>>>>>> 3dc3db0 (cross validation, process added)
    # Get the dataset, use audio data for any non-transformer model,
    # feature data for transformers
    if cfg.hyper.model != "transformer":
        data_root_sample_len = Path(cfg.data.data_root) / Path(
            "chopped_" + str(cfg.hyper.seq_len)
        )
        if not data_root_sample_len.exists():
            log.info(
                "Creating new chopped dataset with sample length: "
                + str(data_root_sample_len)
            )
            data_root_sample_len.mkdir()
            util.chop_dataset(
                cfg.data.original_data_root,
                str(data_root_sample_len),
                "mp3",
                cfg.hyper.seq_len,
            )
        # Sound dataset. Return name if testing
        data = dataset.AudioDataset(data_root_sample_len, cfg.hyper.seq_len)

    else:
        data_root = Path(cfg.data.data_root)
        if not data_root.exists():
            raise ValueError("Data folder does not exist: " + cfg.data.data_root)
        # Feature dataset
        data = dataset.FeatureDataset(data_root)

=======
>>>>>>> 8b2fcbb (tar dataset loading and copying to tmp)
    # Use gpu if available, move the model to device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tmp_data_root = dataset.prepare_dataset_on_tmp(data_tar=cfg.data.data_tar)

    # Get the dataset, use audio data for any non-transformer model,
    # feature data for transformers
    if "transformer" in cfg.hyper.model:
        # Feature dataset
        data = dataset.FeatureDataset(tmp_data_root)
    elif "bank-classifier" in cfg.hyper.model:
        # "Bank dataset"
        data = dataset.BankTransformerDataset(tmp_data_root)
    else:
        # Sound dataset
        data = dataset.AudioDataset(tmp_data_root, cfg.hyper.seq_len)

    # Train/test
    if cfg.process.train:
        cross_validation(data, device, cfg)
    else:
        # Fetch the model:
<<<<<<< HEAD
        # testing load an existing trained ones
        if (
            cfg.logging.load_model_path is None
            and cfg.hyper.pre_trained_ae_path is None
            and cfg.hyper.pre_trained_vqvae_path is None
            and cfg.hyper.pre_trained_transformer_path is None
        ):
            raise ValueError("No trained model path specified for testing.")

        if cfg.logging.load_model_path is not None:
            checkpoint = torch.load(cfg.logging.load_model_path, map_location="cpu")
            model_state_dict = checkpoint["model_state_dict"]
            hyper_cfg_schema = OmegaConf.structured(cfg_classes.HyperConfig)
            conf = OmegaConf.create(checkpoint["hyper_config"])
            cfg.hyper = OmegaConf.merge(hyper_cfg_schema, conf)
=======
        # testing load an existing trained one
        checkpoint = torch.load(cfg.logging.load_model_path, map_location="cpu")
        model_state_dict = checkpoint["model_state_dict"]
        hyper_cfg_schema = OmegaConf.structured(cfg_classes.HyperConfig)
        conf = OmegaConf.create(checkpoint["hyper_config"])
        cfg.hyper = OmegaConf.merge(hyper_cfg_schema, conf)
        log.info(
            f"Loading model with the following cfg.hyper:\n{OmegaConf.to_yaml(cfg.hyper)}"
        )
>>>>>>> 4c3631a (Fix issues with separated transformer training)
        get_model = operate.get_model_init_function(cfg.hyper)
        model = get_model()
        if cfg.logging.load_model_path is not None:
            model.load_state_dict(model_state_dict)
            log.info(f"Loaded model weights from {cfg.logging.load_model_path}")
        model = model.to(device)

        # Make a dataloader
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=cfg.hyper.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.resources.num_workers,
        )
        log.info(f"VALIDATION STARTED")
        test(model, dataloader, device, cfg)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=cfg_classes.BaseConfig)
    main()
