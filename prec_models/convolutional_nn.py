import torch


class PeriodicConv1DBlock(torch.nn.Module):
    def __init__(
        self,
        n_in,
        n_channels: int,
        kernel_size: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        padding_mode = "circular"
        layerslist = [
            torch.nn.Conv1d(
                1,
                n_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            torch.nn.BatchNorm1d(n_channels),
            torch.nn.LeakyReLU(),
        ]
        for _ in range(n_layers):
            layerslist.extend(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        padding_mode=padding_mode,
                    ),
                    torch.nn.BatchNorm1d(n_channels),
                    torch.nn.LeakyReLU(),
                ]
            )
        self.layers_vec = torch.nn.Sequential(*layerslist)

    def forward(self, x):
        return self.layers_vec(x)


class ConvLayersSVD(torch.nn.Module):
    def __init__(self, state_dimension, n_latent, kernel_size, n_layers) -> None:
        super().__init__()
        self.state_dimension = state_dimension
        self.n_latent = n_latent
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.layers_vec = PeriodicConv1DBlock(
            self.state_dimension,
            n_channels=self.n_latent,
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
        )
        self.layers_singval = torch.nn.Sequential(
            PeriodicConv1DBlock(
                self.state_dimension,
                n_channels=self.n_latent,
                kernel_size=self.kernel_size,
                n_layers=self.n_layers,
            ),
            torch.nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = torch.atleast_2d(x)
        x = x.view(len(x), 1, -1)
        n_batch = len(x)
        vectors = torch.nn.functional.normalize(self.layers_vec(x), dim=-1)
        # print(vectors.shape)
        singvals = self.layers_singval(x)
        # print(singvals.shape)
        return torch.concat(
            (vectors, singvals.view(n_batch, self.n_latent, 1)), -1
        ).transpose(1, 2)
