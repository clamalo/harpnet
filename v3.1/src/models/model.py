import torch
import torch.nn as nn
from src.config import Config
from src.models.blocks import AttentionBlock, ResConvBlock

class UNetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()

        depth = Config.UNET_DEPTH
        output_shape = Config.MODEL_OUTPUT_SHAPE
        in_channels = Config.MODEL_INPUT_CHANNELS
        out_channels = Config.MODEL_OUTPUT_CHANNELS

        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        self.depth = depth
        self.output_shape = output_shape

        base_channels = 64

        shapes = [output_shape]
        for _ in range(1, depth):
            shapes.append((shapes[-1][0] // 2, shapes[-1][1] // 2))
        bridge_shape = (shapes[-1][0] // 2, shapes[-1][1] // 2)

        enc_out_channels = [base_channels * (2 ** i) for i in range(depth)]
        bridge_out_channels = enc_out_channels[-1] * 2

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth):
            self.encoders.append(ResConvBlock(prev_channels, enc_out_channels[i], shape=shapes[i]))
            prev_channels = enc_out_channels[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

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
            dropout_rate = self._get_dropout_rate_for_depth(i, depth)
            self.decoders.append(ResConvBlock(
                enc_out_channels[i]*2, enc_out_channels[i], shape=dec_shape, dropout_rate=dropout_rate
            ))

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

    @staticmethod
    def _get_dropout_rate_for_depth(i, depth):
        # Example dropout schedule
        # This can be made configurable.
        if depth == 5:
            if i == 4: 
                return 0.5
            elif i == 3:
                return 0.5
            elif i == 2:
                return 0.3
            elif i == 1:
                return 0.3
            elif i == 0:
                return 0.1
        return 0.0