from torch import nn

<<<<<<< HEAD:model/ae.py
from model.resnet import Resnet1D
from utils import cfg_classes, util
=======
from ..utils import util
from .resnet import Resnet1D
>>>>>>> a1c4262 (package restructure):ccreaim/model/ae.py

"""
Semi-hardcoded AutoEncoder implementation

Possible TODOs:
- Make more hardcoded models
AND/OR
- configurable model structures
"""

# Returns an nn.Conv1d layer according to given parameters, with proper padding calculated for
# output_length to be ceil(input_length/stride)
def create_conv1d_layer(
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    input_length: int,
):
    padding, length_out = util.conf_same_padding_calc(input_length, stride, kernel_size)
    return (
        nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding
        ),
        length_out,
    )


class Encoder(nn.Module):
    def __init__(self, seq_length: int, latent_dim: int):
        super().__init__()
        # The negative slope coefficient for leaky ReLU
        leaky_relu_alpha = 0.2

        # Record the output lengths of the layers for decoder
        self.output_lengths = []

        # First layer
        self.conv1, len_out = create_conv1d_layer(
            input_channels=1,
            output_channels=64,
            kernel_size=7,
            stride=1,
            input_length=seq_length,
        )
        self.b_norm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Second layer
        self.conv2, len_out = create_conv1d_layer(
            input_channels=64,
            output_channels=128,
            kernel_size=5,
            stride=2,
            input_length=len_out,
        )
        self.b_norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Third layer
        self.conv3, len_out = create_conv1d_layer(
            input_channels=128,
            output_channels=256,
            kernel_size=9,
            stride=4,
            input_length=len_out,
        )
        self.b_norm3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Fourth layer
        self.conv4, len_out = create_conv1d_layer(
            input_channels=256,
            output_channels=512,
            kernel_size=9,
            stride=4,
            input_length=len_out,
        )
        self.b_norm4 = nn.BatchNorm1d(512)
        self.relu4 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Fifth layer
        self.conv5, len_out = create_conv1d_layer(
            input_channels=512,
            output_channels=1024,
            kernel_size=9,
            stride=4,
            input_length=len_out,
        )
        self.relu5 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Final layer
        self.conv6, len_out = create_conv1d_layer(
            input_channels=1024,
            output_channels=latent_dim,
            kernel_size=5,
            stride=1,
            input_length=len_out,
        )
        self.output_lengths.append(len_out)

    def forward(self, data):

        data = self.conv1(data)
        data = self.b_norm1(data)
        data = self.relu1(data)

        data = self.conv2(data)
        data = self.b_norm2(data)
        data = self.relu2(data)

        data = self.conv3(data)
        data = self.b_norm3(data)
        data = self.relu3(data)

        data = self.conv4(data)
        data = self.b_norm4(data)
        data = self.relu4(data)

        data = self.conv5(data)
        data = self.relu5(data)

        data = self.conv6(data)

        return data


# Creates an nn.ConvTranspose1d layer according to given parameters, where a the correct
# padding and output_padding is used for a given input_length => output_length mapping
def create_convtranspose1d_layer(
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    input_length: int,
    output_length: int,
):
    padding, output_padding = util.conf_same_padding_calc_t(
        input_length, output_length, stride, kernel_size
    )
    return nn.ConvTranspose1d(
        input_channels,
        output_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )


class Decoder(nn.Module):
    def __init__(self, seq_length: int, latent_dim: int, output_lengths: list[int]):
        super().__init__()

        # The negative slope coefficient for leaky ReLU
        leaky_relu_alpha = 0.2

        # Output lengths for getting correct paddings
        # that reflect the encoder's sequence lengths
        len_outputs = output_lengths.copy()

        # First layer
        len_in = len_outputs.pop()
        len_out = len_outputs.pop()
        self.conv6 = create_convtranspose1d_layer(
            input_channels=latent_dim,
            output_channels=1024,
            kernel_size=5,
            stride=1,
            input_length=len_in,
            output_length=len_out,
        )
        self.relu5 = nn.LeakyReLU(leaky_relu_alpha)

        # Second layer
        len_in = len_out
        len_out = len_outputs.pop()
        self.conv5 = create_convtranspose1d_layer(
            input_channels=1024,
            output_channels=512,
            kernel_size=9,
            stride=4,
            input_length=len_in,
            output_length=len_out,
        )
        self.relu4 = nn.LeakyReLU(leaky_relu_alpha)

        # Third layer
        len_in = len_out
        len_out = len_outputs.pop()
        self.conv4 = create_convtranspose1d_layer(
            input_channels=512,
            output_channels=256,
            kernel_size=9,
            stride=4,
            input_length=len_in,
            output_length=len_out,
        )
        self.relu3 = nn.LeakyReLU(leaky_relu_alpha)

        # Fourth layer
        len_in = len_out
        len_out = len_outputs.pop()
        self.conv3 = create_convtranspose1d_layer(
            input_channels=256,
            output_channels=128,
            kernel_size=9,
            stride=4,
            input_length=len_in,
            output_length=len_out,
        )
        self.relu2 = nn.LeakyReLU(leaky_relu_alpha)

        # Fifth layer
        len_in = len_out
        len_out = len_outputs.pop()
        self.relu1 = nn.LeakyReLU(leaky_relu_alpha)
        self.conv2 = create_convtranspose1d_layer(
            input_channels=128,
            output_channels=64,
            kernel_size=5,
            stride=2,
            input_length=len_in,
            output_length=len_out,
        )

        # Final layer
        len_in = len_out
        len_out = seq_length
        self.conv1 = create_convtranspose1d_layer(
            input_channels=64,
            output_channels=1,
            kernel_size=7,
            stride=1,
            input_length=len_in,
            output_length=len_out,
        )

    def forward(self, data):

        data = self.conv6(data)
        data = self.relu5(data)

        data = self.conv5(data)
        data = self.relu4(data)

        data = self.conv4(data)
        data = self.relu3(data)

        data = self.conv3(data)
        data = self.relu2(data)

        data = self.conv2(data)
        data = self.relu1(data)

        data = self.conv1(data)

        return data


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.net = nn.Sequential(self.encoder, self.decoder)

    def forward(self, input_data):
        return self.net(input_data)

    def encode(self, input_data):
        return self.encoder(input_data)


# Jukebox imitating ResNet-based AE:


def assert_shape(x, exp_shape):
    # This could move to util/be removed from productions version
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"


class EncoderConvBlock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        res_scale=False,
    ):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(
                        input_emb_width if i == 0 else width,
                        width,
                        filter_t,
                        stride_t,
                        pad_t,
                    ),
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        res_scale,
                    ),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBlock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        res_scale=False,
    ):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        res_scale=res_scale,
                    ),
                    nn.ConvTranspose1d(
                        width,
                        input_emb_width if i == (down_t - 1) else width,
                        filter_t,
                        stride_t,
                        pad_t,
                    ),
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# Multilevel AE:s from jukebox implementation
class MultiLevelResEncoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        levels,
        downs_t,
        strides_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(
            input_emb_width if level == 0 else output_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            **block_kwargs_copy,
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t**down_t)
            # ssert_shape(x, (N, emb, T))
            xs.append(x)

        return xs


class MultiLevelResDecoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        levels,
        downs_t,
        strides_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: DecoderConvBlock(
            output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(
            list(zip(list(range(self.levels)), self.downs_t, self.strides_t))
        )
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t**down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x


# Single level AE-components for simplicity/avoiding encoder list output
class ResEncoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.down_t = down_t
        self.stride_t = stride_t
        block_kwargs_copy = dict(**block_kwargs)
        self.encoder_block = EncoderConvBlock(
            input_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            **block_kwargs_copy,
        )

    def forward(self, x):
        x = self.encoder_block(x)
        return x


class ResDecoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.down_t = down_t
        self.stride_t = stride_t
        self.decoder_block = DecoderConvBlock(
            output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs
        )
        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, x):
        x = self.decoder_block(x)
        x = self.out(x)
        return x


# Returns latent dimension, resnet block configurations and the whole ResAeConfig object
def _res_ae_configs(
    cfg: cfg_classes.BaseConfig,
) -> tuple[int, dict[int], cfg_classes.ResAeConfig]:
    res_ae_config = cfg.hyper.res_ae
    assert (
        len(res_ae_config.downs_t)
        == len(res_ae_config.strides_t)
        == res_ae_config.levels
    ), "Mismatch in res_ae levels configurations"
    block_kwargs = dict(
        width=res_ae_config.block_width,
        depth=res_ae_config.block_depth,
        m_conv=res_ae_config.block_m_conv,
        dilation_cycle=res_ae_config.block_dilation_cycle,
        dilation_growth_rate=res_ae_config.block_dilation_growth_rate,
    )
    return cfg.hyper.latent_dim, block_kwargs, res_ae_config


<<<<<<< HEAD
def get_res_encoder(cfg: cfg_classes.BaseConfig) -> ResEncoder:
    latent_dim, block_kwargs, res_ae_cfg = _res_ae_configs(cfg)
=======
# For single-level res-encoders
def res_encoder_output_seq_length(hyper_cfg: HyperConfig) -> int:
    res_ae_cfg = hyper_cfg.res_ae
    assert (
        res_ae_cfg.levels == 1
    ), f"Method only supported for single-level but number of levels was {res_ae_cfg.levels}"
    return hyper_cfg.seq_len // (res_ae_cfg.strides_t[0] ** res_ae_cfg.downs_t[0])


def get_res_encoder(hyper_cfg: HyperConfig) -> ResEncoder:
    latent_dim, block_kwargs, res_ae_cfg = _res_ae_configs(hyper_cfg)
    assert (
        res_ae_cfg.levels == 1
    ), f"Method only supported for single-level but number of levels was {res_ae_cfg.levels}"
>>>>>>> e05277c (Divide res-ae implementation into multilevel/single-level versions)
    return ResEncoder(
        res_ae_cfg.input_emb_width,
        latent_dim,
        res_ae_cfg.downs_t[0],
        res_ae_cfg.strides_t[0],
        **block_kwargs,
    )


<<<<<<< HEAD
def get_res_decoder(cfg: cfg_classes.BaseConfig) -> ResDecoder:
    latent_dim, block_kwargs, res_ae_cfg = _res_ae_configs(cfg)
=======
def get_res_decoder(hyper_cfg: HyperConfig) -> ResDecoder:
    latent_dim, block_kwargs, res_ae_cfg = _res_ae_configs(hyper_cfg)
    assert (
        res_ae_cfg.levels == 1
    ), f"Method only supported for single-level but number of levels was {res_ae_cfg.levels}"
>>>>>>> e05277c (Divide res-ae implementation into multilevel/single-level versions)
    return ResDecoder(
        res_ae_cfg.input_emb_width,
        latent_dim,
        res_ae_cfg.downs_t[0],
        res_ae_cfg.strides_t[0],
        **block_kwargs,
    )


<<<<<<< HEAD
def _create_res_autoencoder(cfg: cfg_classes.BaseConfig):
    encoder = get_res_encoder(cfg)
    decoder = get_res_decoder(cfg)
=======
def _create_res_autoencoder(hyper_cfg: HyperConfig) -> AutoEncoder:
    encoder = get_res_encoder(hyper_cfg)
    decoder = get_res_decoder(hyper_cfg)
>>>>>>> a07db98 (implement pre-trained load, refactor model creation)
    return AutoEncoder(encoder, decoder)


def _create_autoencoder(hyper_cfg: HyperConfig) -> AutoEncoder:
    encoder = Encoder(hyper_cfg.seq_len, hyper_cfg.latent_dim)
    decoder = Decoder(hyper_cfg.seq_len, hyper_cfg.latent_dim, encoder.output_lengths)
    return AutoEncoder(encoder, decoder)


<<<<<<< HEAD
def get_autoencoder(name: str, cfg: cfg_classes.BaseConfig):
    if name == "base":
        return _create_autoencoder(cfg.hyper.seq_len, cfg.hyper.latent_dim)
    elif name == "res-ae":
        return _create_res_autoencoder(cfg)
=======
def get_autoencoder(hyper_cfg: HyperConfig) -> AutoEncoder:
    if hyper_cfg.model == "ae":
        return _create_autoencoder(hyper_cfg)
    elif hyper_cfg.model == "res-ae":
        return _create_res_autoencoder(hyper_cfg)
>>>>>>> a07db98 (implement pre-trained load, refactor model creation)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(hyper_cfg.model))
