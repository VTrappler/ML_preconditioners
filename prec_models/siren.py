from torch import nn
import pytorch.lightning as pl


class SineActivation(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenModifier:
    def __init__(self, dim, c=6.0, w0_initial=30.0, w0=1.0):
        self.dim = dim
        self.c = c
        self.w0_initial = w0_initial
        self.w0 = w0

    def initializer(self, module, is_first=False):
        w_std = (1 / self.dim) if is_first else (math.sqrt(self.c / self.dim) / self.w0)
        try:
            torch.nn.init.uniform_(module.weight.data, -w_std, w_std)
        except AttributeError:
            pass
        try:
            torch.nn.init.uniform_(module.bias.data, -w_std, w_std)
        except AttributeError:
            pass

    def __call__(self, module):
        self.initializer(module, is_first=False)

    def first_initializer(self, module):
        self.initializer(module, is_first=True)

    def activation(self, is_first=False):
        return SineActivation(self.w0_initial) if is_first else SineActivation(self.w0)


class UNet_SIREN(pl.LightningModule):
    def __init__(self, rank: int):
        super().__init__()
        self.siren = SirenModifier(64 * 64 * 3)
        self.rank = rank
        self.first_activation = self.siren.activation(is_first=True)
        self.activation = self.siren.activation()
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.1)
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # (channels x h x w)
        # input: 3x64x64
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 64x64x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 64x64x64
        self.norm1 = nn.LayerNorm([64, 64, 64])
        self.enc_layers1 = nn.Sequential(
            self.e11,
            self.norm1,
            self.first_activation,  # CHANGED from classic UNet
            self.dropout,
            self.e12,
            self.norm1,
            self.activation,
            self.dropout,
        )
        self.pool1 = self.pooling  # output: 64x32x32

        # input: 64x32x32
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 128x32x32
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 128x32x32

        self.norm2 = nn.LayerNorm([128, 32, 32])
        self.enc_layers2 = nn.Sequential(
            self.e21,
            self.norm2,
            self.activation,
            self.dropout,
            self.e22,
            self.norm2,
            self.activation,
            self.dropout,
        )
        self.pool2 = self.pooling  # output: 128x16x16

        # input: 128x16x16
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # output: 256x16x16
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 256x16x16
        self.norm3 = nn.LayerNorm([256, 16, 16])
        self.enc_layers3 = nn.Sequential(
            self.e31,
            self.norm3,
            self.activation,
            self.dropout,
            self.e32,
            self.norm3,
            self.activation,
            self.dropout,
        )
        self.pool3 = self.pooling  # output: 256x8x8

        # input: 256x8x8
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # output: 512x8x8
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output: 512x8x8
        self.enc_layers4 = nn.Sequential(
            self.e41, self.activation, self.e42, self.activation
        )
        self.pool4 = self.pooling  # output: 512x4x4

        # input: 512x4x4
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output:1024x4x4
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # output: 1024x4x4
        self.enc_layers5 = nn.Sequential(
            self.e51,
            self.activation,
            self.dropout,
            self.e52,
            self.activation,
            self.dropout,
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_layers1 = nn.Sequential(
            self.d11,
            self.activation,
            self.dropout,
            self.d12,
            self.activation,
            self.dropout,
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_layers2 = nn.Sequential(
            self.d21,
            self.activation,
            self.dropout,
            self.d22,
            self.activation,
            self.dropout,
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_layers3 = nn.Sequential(
            self.d31,
            self.activation,
            self.dropout,
            self.d32,
            self.activation,
            self.dropout,
        )

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 3 * rank, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(3 * rank, 3 * rank, kernel_size=3, padding=1)
        self.dec_layers4 = nn.Sequential(
            self.d41,
            self.activation,
            self.dropout,
            self.d42,
            self.activation,
            self.dropout,
        )

        # Output layer
        self.final_dense_layer = nn.Linear(3 * 64**2, controlsize + 1)
        # self.final_dense_layer2 = nn.Linear(3 * 64**2, controlsize + 1)
        # self.final_layers = nn.Sequential(
        #     self.final_dense_layer,
        #     self.activation,
        #     self.final_dense_layer2,
        # )
        self.sigmoid = ParamSigmoid(0, 12)
        print("Custom SIREN weights initializations")
        self.apply(self.siren.initializer)
        self.siren.first_initializer(self.e11)

        total_size = 0
        for i in self.parameters():
            total_size += i.element_size() * i.nelement()
        print("BasicUNet:", total_size / 1e9, "Gb")

    def forward_unet(self, x):
        # Encoder
        xe12 = self.enc_layers1(x)
        xp1 = self.pool1(xe12)
        xe22 = self.enc_layers2(xp1)
        xp2 = self.pool2(xe22)
        xe32 = self.enc_layers3(xp2)
        xp3 = self.pool3(xe32)
        xe42 = self.enc_layers4(xp3)
        xp4 = self.pool4(xe42)
        xe52 = self.enc_layers5(xp4)

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd12 = self.dec_layers1(xu11)

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd22 = self.dec_layers2(xu22)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd32 = self.dec_layers3(xu33)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd42 = self.dec_layers4(xu44)
        # Output layer
        return xd42

    def forward(self, x):
        x = self.forward_unet(x)
        x = x.reshape(-1, int(self.rank), 3 * 64**2)
        outputs = self.final_dense_layer(x).transpose(-1, -2)
        vecs, vals = outputs[:, :-1, :], outputs[:, -1, :]
        vals = self.sigmoid(vals)
        return torch.concat([vecs, vals[:, None, :]], axis=1)


class SW_UNet_SIREN(SW_UNet):
    def __init__(self, state_dimension, rank, config):
        self.controlsize = 64**2 + 2 * 63 * 64
        super().__init__(state_dimension, rank, config)
        self.layers = UNet_SIREN(rank)
