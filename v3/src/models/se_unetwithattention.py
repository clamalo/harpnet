"""
Defines a Squeeze-and-Excitation U-Net with attention mechanisms.
This model extends the original U-Net with attention by adding SE blocks for channel recalibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.constants import UNET_DEPTH, MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS, MODEL_OUTPUT_SHAPE

class AttentionBlock(nn.Module):
    """
    Attention Block for selectively gating encoder features before combining with decoder features.
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

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise feature recalibration.
    Enhances informative features and suppresses less useful ones.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Compute channel-wise statistics and rescale features
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResConvBlock(nn.Module):
    """
    Residual convolutional block with LayerNorm and SEBlock.
    Maintains spatial dimensions and includes dropout.
    """
    def __init__(self, in_channels, out_channels, shape=(64,64), dropout_rate=0.0):
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
        self.se_block = SEBlock(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = self.resconvblock(x)
        x = x + x_shortcut
        x = self.se_block(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    Squeeze-and-Excitation U-Net with Attention.
    Incorporates SE blocks in residual convolutional layers and uses attention on skip connections.
    Suitable for precipitation downscaling.
    """
    def __init__(self,
                 in_channels=MODEL_INPUT_CHANNELS,
                 out_channels=MODEL_OUTPUT_CHANNELS,
                 output_shape=MODEL_OUTPUT_SHAPE,
                 depth=UNET_DEPTH):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.depth = depth

        # Define encoder channel sizes
        enc_channels = [64 * (2 ** i) for i in range(self.depth)]
        bridge_channels = enc_channels[-1] * 2

        base_h, base_w = self.output_shape
        enc_shapes = []
        for i in range(self.depth):
            enc_shapes.append((base_h // (2**i), base_w // (2**i)))
        bridge_shape = (base_h // (2**self.depth), base_w // (2**self.depth))

        # Encoders
        self.encoders = nn.ModuleList()
        prev_channels = self.in_channels
        for i in range(self.depth):
            self.encoders.append(ResConvBlock(prev_channels, enc_channels[i], shape=enc_shapes[i]))
            prev_channels = enc_channels[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bridge = ResConvBlock(enc_channels[-1], bridge_channels, shape=bridge_shape)

        # Determine dropout rates for decoder layers
        def get_dropout_for_layer(layer_index):
            if layer_index == 0:
                return 0.1
            elif layer_index == 1:
                return 0.3
            elif layer_index == 2:
                return 0.3
            else:
                return 0.5

        # Decoders with attention
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
        final = torch.clamp(final, min=0)
        return final