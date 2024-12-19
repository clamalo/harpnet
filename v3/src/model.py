import torch
import torch.nn as nn
import torch.nn.functional as F
from src.constants import UNET_DEPTH, MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS, MODEL_OUTPUT_SHAPE

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(AttentionBlock, self).__init__()
        self.theta_x = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(2, 2))
        self.phi_g = nn.Conv2d(in_channels=gating_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.psi = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Restore bilinear interpolation
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
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = self.resconvblock(x)
        x = x + x_shortcut
        x = self.dropout(x)
        return x

class UNetWithAttention(nn.Module):
    def __init__(self):
        super(UNetWithAttention, self).__init__()

        depth = UNET_DEPTH
        output_shape = MODEL_OUTPUT_SHAPE
        in_channels = MODEL_INPUT_CHANNELS
        out_channels = MODEL_OUTPUT_CHANNELS

        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        self.depth = depth
        self.output_shape = output_shape

        # Base number of channels
        base_channels = 64

        # Compute shapes at each encoder level
        shapes = [output_shape]
        for _ in range(1, depth):
            shapes.append((shapes[-1][0] // 2, shapes[-1][1] // 2))
        bridge_shape = (shapes[-1][0] // 2, shapes[-1][1] // 2)

        # Encoder channels
        enc_out_channels = [base_channels * (2 ** i) for i in range(depth)]
        bridge_out_channels = enc_out_channels[-1] * 2

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth):
            self.encoders.append(ResConvBlock(prev_channels, enc_out_channels[i], shape=shapes[i]))
            prev_channels = enc_out_channels[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bridge = ResConvBlock(enc_out_channels[-1], bridge_out_channels, shape=bridge_shape)

        self.upconvs = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        dec_in_channels = bridge_out_channels
        gating_channels = bridge_out_channels
        for i in reversed(range(depth)):
            self.upconvs.append(nn.ConvTranspose2d(dec_in_channels, enc_out_channels[i], kernel_size=2, stride=2))
            self.attn_blocks.append(AttentionBlock(enc_out_channels[i], gating_channels))

            dec_shape = shapes[i]
            dropout_rate = 0.0
            if self.depth == 5:
                if i == 4: # dec5
                    dropout_rate = 0.5
                elif i == 3: # dec4
                    dropout_rate = 0.5
                elif i == 2: # dec3
                    dropout_rate = 0.3
                elif i == 1: # dec2
                    dropout_rate = 0.3
                elif i == 0: # dec1
                    dropout_rate = 0.1

            self.decoders.append(ResConvBlock(enc_out_channels[i]*2, enc_out_channels[i], shape=dec_shape, dropout_rate=dropout_rate))

            dec_in_channels = enc_out_channels[i]
            gating_channels = dec_in_channels

        self.final_conv = nn.Conv2d(enc_out_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        enc_features = []
        out = x
        for i in range(self.depth):
            out = self.encoders[i](out)
            enc_features.append(out)
            if i < self.depth - 1:
                out = self.pool(out)

        out = self.pool(out)
        bridge = self.bridge(out)

        dec_out = bridge
        for i in range(self.depth):
            upconv = self.upconvs[i]
            attn_block = self.attn_blocks[i]
            decoder = self.decoders[i]

            enc_feat = enc_features[self.depth - 1 - i]

            gating = dec_out if i > 0 else bridge
            gating_feat = attn_block(enc_feat, gating)

            up = upconv(dec_out)
            cat_feat = torch.cat([up, gating_feat], dim=1)
            dec_out = decoder(cat_feat)

        final = self.final_conv(dec_out)
        final = torch.clamp(final, min=0)
        return final