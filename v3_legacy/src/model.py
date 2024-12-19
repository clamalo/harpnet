import torch
import torch.nn as nn

class SelfAttention2D(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention2D, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C' x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = self.softmax(energy)  # B x N x N
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

class AttentionBlockWithSelfAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int=None):
        super(AttentionBlockWithSelfAttention, self).__init__()
        if F_int is None:
            F_int = F_l // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.self_attention = SelfAttention2D(F_int)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        combined = self.relu(g1 + x1)
        attended = self.self_attention(combined)
        psi = self.psi(attended)
        return x * psi

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
        self.enc2 = ResConvBlock(64, 128, (32,32))
        self.enc3 = ResConvBlock(128, 256, (16,16))
        self.enc4 = ResConvBlock(256, 512, (8,8))
        self.enc5 = ResConvBlock(512, 1024, (4,4))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bridge = ResConvBlock(1024, 2048, (2,2))

        # Updated F_g and F_l to match actual channels
        self.attn_block5 = AttentionBlockWithSelfAttention(F_g=1024, F_l=1024)
        self.attn_block4 = AttentionBlockWithSelfAttention(F_g=512, F_l=512)
        self.attn_block3 = AttentionBlockWithSelfAttention(F_g=256, F_l=256)
        self.attn_block2 = AttentionBlockWithSelfAttention(F_g=128, F_l=128)
        self.attn_block1 = AttentionBlockWithSelfAttention(F_g=64, F_l=64)

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

    def forward(self, x):

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        output_shape = self.output_shape
        interpolated_x = nn.functional.interpolate(x, size=output_shape, mode='nearest')

        # Encoder
        enc1 = self.enc1(interpolated_x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Bridge
        bridge = self.bridge(self.pool(enc5))
        
        # Decoder with Attention Blocks
        up5 = self.upconv5(bridge)
        gating5 = self.attn_block5(up5, enc5)
        up5 = torch.cat([up5, gating5], dim=1)
        dec5 = self.dec5(up5)

        up4 = self.upconv4(dec5)
        gating4 = self.attn_block4(up4, enc4)
        up4 = torch.cat([up4, gating4], dim=1)
        dec4 = self.dec4(up4)

        up3 = self.upconv3(dec4)
        gating3 = self.attn_block3(up3, enc3)
        up3 = torch.cat([up3, gating3], dim=1)
        dec3 = self.dec3(up3)

        up2 = self.upconv2(dec3)
        gating2 = self.attn_block2(up2, enc2)
        up2 = torch.cat([up2, gating2], dim=1)
        dec2 = self.dec2(up2)

        up1 = self.upconv1(dec2)
        gating1 = self.attn_block1(up1, enc1)
        up1 = torch.cat([up1, gating1], dim=1)
        dec1 = self.dec1(up1)

        final = self.final_conv(dec1)
        final = torch.clamp(final, min=0)

        return final.squeeze(1)


if __name__ == '__main__':
    model = UNetWithAttention(1, 1)
    print(model)

    x = torch.randn(16, 1, 64, 64)
    y = model(x)
    print(y.shape)
