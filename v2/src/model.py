import torch
import torch.nn as nn
import torch.nn.functional as F
from src.constants import UNET_DEPTH


class AttentionBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels, output_shape=(64, 64), depth=5):
        super(UNetWithAttention, self).__init__()

        # Validate depth
        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        self.depth = depth
        self.output_shape = output_shape

        # Base number of channels at the first encoder level
        base_channels = 64

        # Compute shapes at each encoder level
        # Each encoder level halves the spatial dimension
        shapes = [output_shape]
        for _ in range(1, depth):
            shapes.append((shapes[-1][0] // 2, shapes[-1][1] // 2))
        # Shape at the bridge (one more halving)
        bridge_shape = (shapes[-1][0] // 2, shapes[-1][1] // 2)

        # Encoder channels: start from base_channels and double each time
        # enc_out_channels[i] is the output channels of the i-th encoder layer (1-based index)
        enc_out_channels = [base_channels * (2 ** (i)) for i in range(depth)]
        # The bridge doubles the last encoder's output channels
        bridge_out_channels = enc_out_channels[-1] * 2

        # Build encoders
        # enc_in_channels for the first block is the input channels of the model
        # subsequent enc_in_channels matches the previous enc_out_channels
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth):
            self.encoders.append(ResConvBlock(prev_channels, enc_out_channels[i], shape=shapes[i]))
            prev_channels = enc_out_channels[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bridge = ResConvBlock(enc_out_channels[-1], bridge_out_channels, shape=bridge_shape)

        # Build attention blocks, upconvs, and decoders
        # Decoder pattern:
        #   For each level i (from bottom to top):
        #       upconv in: bridge_out_channels or dec_out_channels at lower level
        #       upconv out: enc_out_channels[i-1] (mirror of enc side)
        #       attn_block: AttnBlock(enc_out_channels[i-1], gating_channels from next deeper level)
        #       dec block: from concatenation of upconv out + attended enc out to enc_out_channels[i-1]
        #
        # For the gating channels of the attention:
        # - The deepest decoder block uses the bridge as gating (bridge_out_channels)
        # - Subsequent dec blocks use the previously decoded output as gating.
        #
        # dec_out_channels[i] will match enc_out_channels[i], ensuring symmetry.

        self.upconvs = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Construct decoder in reverse order
        # At each decoder step:
        #   upconv in_channels = previous dec out or bridge out
        #   upconv out_channels = enc_out_channels[i]
        # dec in_channels = enc_out_channels[i] * 2 (after concatenation)
        # dec out_channels = enc_out_channels[i]
        #
        # For attention blocks:
        #   gating_channels for the topmost decoder = bridge_out_channels
        #   for the next: it's the dec_out_channels of the previously computed decoder
        #
        # We'll store these in reverse since dec top level corresponds to enc_depth.
        dec_in_channels = bridge_out_channels
        gating_channels = bridge_out_channels
        for i in reversed(range(depth)):
            # Upconv:
            self.upconvs.append(nn.ConvTranspose2d(dec_in_channels, enc_out_channels[i], kernel_size=2, stride=2))
            # Attn Block:
            self.attn_blocks.append(AttentionBlock(enc_out_channels[i], gating_channels))
            # Decoder block:
            # After concat: channels = enc_out_channels[i] (from upconv) + enc_out_channels[i] (from attn-blocked skip)
            dec_shape = shapes[i]  # shape is the same as enc_i
            dropout_rate = 0.0
            # Original code had some dropout at certain depths. We can mimic that pattern:
            # depth 5 pattern: dec5, dec4 had 0.5; dec3 had 0.3; dec2 had 0.3; dec1 had 0.1
            # Let's apply a simple heuristic: apply dropout based on depth from the bridge
            # This ensures backward compatibility for depth=5.
            # For depth=5, order: dec5->0.5, dec4->0.5, dec3->0.3, dec2->0.3, dec1->0.1
            # We'll replicate this pattern proportionally:
            if self.depth == 5:
                # Hardcode original pattern
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

            # Prepare for next iteration
            dec_in_channels = enc_out_channels[i]  # the output of this dec block
            gating_channels = dec_in_channels      # used in the next attn block

        # Final conv
        self.final_conv = nn.Conv2d(enc_out_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # If input has shape [B, H, W], unsqueeze to [B, C=1, H, W]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Interpolate x to output_shape if needed
        interpolated_x = nn.functional.interpolate(x, size=self.output_shape, mode='nearest')

        # Encoder forward
        enc_features = []
        out = interpolated_x
        for i in range(self.depth):
            out = self.encoders[i](out)
            enc_features.append(out)
            if i < self.depth - 1:
                out = self.pool(out)

        # Bridge
        out = self.pool(out)
        bridge = self.bridge(out)

        # Decoder forward
        # We built dec/attn/upconv in reverse order, so we need to traverse them in the correct order:
        # The topmost decoder block corresponds to enc_features[-1]
        dec_out = bridge
        for i in range(self.depth):
            # i-th decoder step (from top):
            # Indexing from last appended: self.upconvs, self.attn_blocks, self.decoders
            upconv = self.upconvs[i]
            attn_block = self.attn_blocks[i]
            decoder = self.decoders[i]

            # Get the corresponding encoder feature (reverse order)
            enc_feat = enc_features[self.depth - 1 - i]

            # Attention
            gating = dec_out if i > 0 else bridge  # For the top layer i=0, gating=bridge, otherwise gating=prev dec_out
            gating_feat = attn_block(enc_feat, gating)

            # Upconv
            up = upconv(dec_out)

            # Concat
            cat_feat = torch.cat([up, gating_feat], dim=1)

            # Decode
            dec_out = decoder(cat_feat)

        final = self.final_conv(dec_out)
        final = torch.clamp(final, min=0)
        return final.squeeze(1)

if __name__ == '__main__':
    # Test the model
    model = UNetWithAttention(1, 1, output_shape=(64, 64), depth=UNET_DEPTH)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    print(out.shape)  # Expected [2, 64, 64] (after squeeze)