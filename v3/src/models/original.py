"""
Defines a U-Net style model with attention for precipitation downscaling.
Uses residual convolution blocks and attention gates in skip connections.
"""

import torch
import torch.nn as nn
from src.constants import UNET_DEPTH, MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS, TILE_SIZE

class AttentionBlock(nn.Module):
    """
    Gated attention block that emphasizes relevant encoder features before concatenation.
    """
    def __init__(self, in_channels, gating_channels):
        super(AttentionBlock, self).__init__()
        # --- MAIN LAYERS ---
        self.theta_x = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(2, 2))
        self.phi_g = nn.Conv2d(in_channels=gating_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.psi = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        # --- ACTIVATIONS / UPSAMPLING ---
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, gating):
        # --- Compute attention map ---
        theta_x = self.theta_x(x)
        gating = self.phi_g(gating)
        add = self.relu(theta_x + gating)
        psi = self.psi(add)
        sigmoid_psi = self.sigmoid(psi)
        upsample_psi = self.upsample(sigmoid_psi)
        # --- Apply attention ---
        y = upsample_psi * x
        return y


class ResConvBlock(nn.Module):
    """
    Residual block with two convolutions, optional dropout, and a shortcut path.
    Maintains spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, shape=(TILE_SIZE, TILE_SIZE), dropout_rate=0.0):
        super(ResConvBlock, self).__init__()
        # --- MAIN CONV BLOCK ---
        self.resconvblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]),
            nn.ReLU()
        )
        # --- SHORTCUT CONV ---
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.LayerNorm([out_channels, shape[0], shape[1]])
        )
        # --- DROPOUT ---
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = self.resconvblock(x)
        x = x + x_shortcut
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    A U-Net model with attention-based skip connections and residual convolution blocks.
    The final output is activated by Softplus for non-negative predictions.
    """
    def __init__(self, in_channels=MODEL_INPUT_CHANNELS, out_channels=MODEL_OUTPUT_CHANNELS, tile_size=TILE_SIZE):
        super(Model, self).__init__()
        self.tile_size = tile_size

        # --- ENCODER ---
        self.enc1 = ResConvBlock(in_channels, 64, (tile_size, tile_size))
        self.enc2 = ResConvBlock(64, 128, (tile_size // 2, tile_size // 2))
        self.enc3 = ResConvBlock(128, 256, (tile_size // 4, tile_size // 4))
        self.enc4 = ResConvBlock(256, 512, (tile_size // 8, tile_size // 8))
        self.enc5 = ResConvBlock(512, 1024, (tile_size // 16, tile_size // 16))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BRIDGE ---
        self.bridge = ResConvBlock(1024, 2048, (tile_size // 32, tile_size // 32))

        # --- ATTENTION BLOCKS ---
        self.attn_block5 = AttentionBlock(1024, 2048)
        self.attn_block4 = AttentionBlock(512, 1024)
        self.attn_block3 = AttentionBlock(256, 512)
        self.attn_block2 = AttentionBlock(128, 256)
        self.attn_block1 = AttentionBlock(64, 128)

        # --- DECODER (UPSAMPLING) ---
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # --- DECODER RESBLOCKS ---
        self.dec5 = ResConvBlock(2048, 1024, (tile_size // 16, tile_size // 16), dropout_rate=0.5)
        self.dec4 = ResConvBlock(1024, 512, (tile_size // 8, tile_size // 8), dropout_rate=0.5)
        self.dec3 = ResConvBlock(512, 256, (tile_size // 4, tile_size // 4), dropout_rate=0.3)
        self.dec2 = ResConvBlock(256, 128, (tile_size // 2, tile_size // 2), dropout_rate=0.3)
        self.dec1 = ResConvBlock(128, 64, (tile_size, tile_size), dropout_rate=0.1)

        # --- FINAL CONV & ACTIVATION ---
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_activation = nn.Softplus()

    def forward(self, x):
        # --- ENCODER PATH ---
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # --- BRIDGE ---
        bridge = self.bridge(self.pool(enc5))
        
        # --- DECODER WITH ATTENTION ---
        gating5 = self.attn_block5(enc5, bridge)
        up5 = self.upconv5(bridge)
        up5 = torch.cat([up5, gating5], dim=1)
        dec5 = self.dec5(up5)

        gating4 = self.attn_block4(enc4, dec5)
        up4 = self.upconv4(dec5)
        up4 = torch.cat([up4, gating4], dim=1)
        dec4 = self.dec4(up4)

        gating3 = self.attn_block3(enc3, dec4)
        up3 = self.upconv3(dec4)
        up3 = torch.cat([up3, gating3], dim=1)
        dec3 = self.dec3(up3)

        gating2 = self.attn_block2(enc2, dec3)
        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, gating2], dim=1)
        dec2 = self.dec2(up2)

        gating1 = self.attn_block1(enc1, dec2)
        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, gating1], dim=1)
        dec1 = self.dec1(up1)

        # --- FINAL PREDICTION ---
        final = self.final_conv(dec1)
        final = self.final_activation(final)
        return final