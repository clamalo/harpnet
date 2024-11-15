import torch
import torch.nn as nn
import torch.nn.functional as F


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
    

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
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
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shape=(64,64), dropout_rate=0.0):
        super(ResConvBlock, self).__init__()
        self.resconvblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]),
            nn.ReLU())
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]))
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = self.resconvblock(x)
        x = x + x_shortcut
        x = self.dropout(x)
        return x
    
    
class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape=(64, 64)):
        super(UNetWithAttention, self).__init__()

        self.enc1 = ResConvBlock(in_channels, 64, (64,64))
        self.se1 = SEBlock(64)
        self.enc2 = ResConvBlock(64, 128, (32,32))
        self.se2 = SEBlock(128)
        self.enc3 = ResConvBlock(128, 256, (16,16))
        self.se3 = SEBlock(256)
        self.enc4 = ResConvBlock(256, 512, (8,8))
        self.se4 = SEBlock(512)
        self.enc5 = ResConvBlock(512, 1024, (4,4))
        self.se5 = SEBlock(1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bridge = ResConvBlock(1024, 2048, (2,2))

        self.attn_block5 = AttentionBlock(1024, 2048)
        self.attn_block4 = AttentionBlock(512, 1024)
        self.attn_block3 = AttentionBlock(256, 512)
        self.attn_block2 = AttentionBlock(128, 256)
        self.attn_block1 = AttentionBlock(64, 128)

        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.dec5 = ResConvBlock(2048, 1024, (4,4), dropout_rate=0.5)
        self.dec4 = ResConvBlock(1024, 512, (8,8), dropout_rate=0.5)
        self.dec3 = ResConvBlock(512, 256, (16,16), dropout_rate=0.3)
        self.dec2 = ResConvBlock(256, 128, (32,32), dropout_rate=0.3)
        self.dec1 = ResConvBlock(128, 64, (64,64), dropout_rate=0.1)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.output_shape = output_shape

    def forward(self, x):#, elevation):

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        output_shape = self.output_shape
        interpolated_x = nn.functional.interpolate(x, size=output_shape, mode='nearest')#, align_corners=True)

        enc1 = self.enc1(interpolated_x)
        enc1 = self.se1(enc1)
        enc2 = self.enc2(self.pool(enc1))
        enc2 = self.se2(enc2)
        enc3 = self.enc3(self.pool(enc2))
        enc3 = self.se3(enc3)
        enc4 = self.enc4(self.pool(enc3))
        enc4 = self.se4(enc4)
        enc5 = self.enc5(self.pool(enc4))
        enc5 = self.se5(enc5)

        bridge = self.bridge(self.pool(enc5))
        
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

        final = self.final_conv(dec1)
        
        final = torch.clamp(final, min=0)

        return final.squeeze(1)
    

if __name__ == "__main__":
    model = UNetWithAttention(1, 1, output_shape=(64,64)).to('mps')

    x = torch.randn(32, 64, 64).to('mps')
    y = model(x)
    print(y.shape)