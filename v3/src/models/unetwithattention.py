"""
Defines a U-Net with attention model for precipitation downscaling.
This is similar to the se_unetwithattention but without SE blocks.

Updated: Replaced the final torch.clamp with a Softplus activation to allow for strictly
non-negative outputs while preserving gradient flow near zero.
"""

import torch
import torch.nn as nn
from src.constants import UNET_DEPTH, MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS, TILE_SIZE

class AttentionBlock(nn.Module):
    """
    Attention Block that selectively highlights relevant encoder features.
    """
    def __init__(self, in_channels, gating_channels):
        super(AttentionBlock, self).__init__()
        self.theta_x = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(2, 2))
        self.phi_g = nn.Conv2d(in_channels=gating_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.psi = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, gating):
        theta_x = self.theta_x(x)
        gating = self.phi_g(gating)
        add = self.relu(theta_x + gating)
        psi = self.psi(add)
        sigmoid_psi = self.sigmoid(psi)
        upsample_psi = self.upsample(sigmoid_psi)
        y = upsample_psi * x
        return y

class ResConvBlock(nn.Module):
    """
    Residual convolutional block maintaining spatial dimensions.
    Includes dropout for regularization.
    """
    def __init__(self, in_channels, out_channels, shape=(TILE_SIZE, TILE_SIZE), dropout_rate=0.0):
        super(ResConvBlock, self).__init__()
        self.resconvblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]),
            nn.ReLU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.LayerNorm([out_channels, shape[0], shape[1]])
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = self.resconvblock(x)
        x = x + x_shortcut
        x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    U-Net with attention gates at skip connections.
    Designed for precipitation downscaling from coarse to fine resolution grids.

    Updated: Removed final clamp and replaced with Softplus to avoid negative outputs
    without hard-clamping gradients.
    """
    def __init__(self,
                 in_channels=MODEL_INPUT_CHANNELS,
                 out_channels=MODEL_OUTPUT_CHANNELS,
                 depth=UNET_DEPTH,
                 tile_size=TILE_SIZE):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.tile_size = tile_size

        base_h, base_w = tile_size, tile_size
        enc_channels = [64 * (2 ** i) for i in range(self.depth)]
        bridge_channels = enc_channels[-1] * 2

        enc_shapes = []
        for i in range(self.depth):
            enc_shapes.append((base_h // (2**i), base_w // (2**i)))
        bridge_shape = (base_h // (2**self.depth), base_w // (2**self.depth))

        # Encoder blocks
        self.encoders = nn.ModuleList()
        prev_channels = self.in_channels
        for i in range(self.depth):
            self.encoders.append(ResConvBlock(prev_channels, enc_channels[i], shape=enc_shapes[i]))
            prev_channels = enc_channels[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bridge = ResConvBlock(enc_channels[-1], bridge_channels, shape=bridge_shape)

        def get_dropout_for_layer(layer_index):
            if layer_index == 0:
                return 0.1
            elif layer_index == 1:
                return 0.3
            elif layer_index == 2:
                return 0.3
            else:
                return 0.5

        # Decoder with attention
        self.upconvs = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_dec_channels = bridge_channels
        for i in range(self.depth):
            enc_ch = enc_channels[self.depth - 1 - i]
            self.upconvs.append(nn.ConvTranspose2d(prev_dec_channels, enc_ch, kernel_size=2, stride=2))
            self.attn_blocks.append(AttentionBlock(in_channels=enc_ch, gating_channels=prev_dec_channels))
            dec_shape = enc_shapes[self.depth - 1 - i]
            dropout_rate = get_dropout_for_layer(i)
            self.decoders.append(ResConvBlock(2 * enc_ch, enc_ch, shape=dec_shape, dropout_rate=dropout_rate))
            prev_dec_channels = enc_ch

        self.final_conv = nn.Conv2d(enc_channels[0], self.out_channels, kernel_size=1)
        self.final_activation = nn.Softplus()

    def forward(self, x):
        # Encoder forward pass
        enc_results = []
        out = x
        for i, enc in enumerate(self.encoders):
            out = enc(out)
            enc_results.append(out)
            if i < self.depth - 1:
                out = self.pool(out)

        out = self.pool(out)
        bridge_out = self.bridge(out)

        # Decoder with attention gating
        dec_out = bridge_out
        for i in range(self.depth):
            up = self.upconvs[i](dec_out)
            enc_feat = enc_results[self.depth - 1 - i]
            gating = dec_out if i > 0 else bridge_out
            gating_attn = self.attn_blocks[i](enc_feat, gating)
            merged = torch.cat([up, gating_attn], dim=1)
            dec_out = self.decoders[i](merged)

        final = self.final_conv(dec_out)
        final = self.final_activation(final)  # Softplus activation for non-negative output

        return final