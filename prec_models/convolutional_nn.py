from typing import List
import torch


class ParamSigmoid(torch.nn.Module):
    def __init__(
        self, a: float, b: float, val_for_half: float = 0.0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b
        self.val_for_half = val_for_half
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x - self.val_for_half) * (self.b - self.a) + self.a

    def __call__(self, x):
        return self.forward(x)


class ParallelConv1DDilations(torch.nn.Module):
    def __init__(
        self, n_in, dilation_list: List = [1, 2, 4], kernel_size: int = 3, skip=True
    ):
        super().__init__()
        self.n_in = n_in
        self.padding_mode = "circular"
        self.dilation_list = dilation_list
        self.kernel_size = kernel_size
        self.conv_layers_dilations = []
        self.skip_connection = skip
        for dilation in self.dilation_list:
            self.conv_layers_dilations.append(
                torch.nn.Conv1d(
                    1,
                    1,
                    self.kernel_size,
                    padding=(self.kernel_size // 2) * dilation,
                    dilation=dilation,
                    padding_mode=self.padding_mode,
                    bias=False,
                )
            )

    def forward(self, x):
        forw = [conv_layer.forward(x) for conv_layer in self.conv_layers_dilations]
        forw.append(x)
        return torch.concatenate(forw, dim=1)


class PeriodicConv1DBlock(torch.nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels: int,
        kernel_size: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        padding_mode = "circular"
        layerslist = [
            torch.nn.Conv1d(
                n_channels_in,
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
        self.dilation_list = [1, 2, 4]
        self.dilations_layers = ParallelConv1DDilations(
            n_in=self.state_dimension,
            dilation_list=self.dilation_list,
            kernel_size=self.kernel_size,
        )
        self.convlayers_vec = PeriodicConv1DBlock(
            n_channels_in=len(self.dilation_list) + 1,  # skip
            n_channels=self.n_latent,
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
        )

        self.mlp_block = torch.nn.Sequential(
            torch.nn.Linear(
                (self.n_latent) * self.state_dimension,  # for skip connection
                self.n_latent * self.state_dimension,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                self.n_latent * self.state_dimension,
                self.n_latent * self.state_dimension,
            ),
        )

        self.param_sig = ParamSigmoid(0, 10, 0)
        self.mlp_singval = torch.nn.Sequential(
            torch.nn.Linear(
                (self.n_latent) * self.state_dimension,  # for skip connection
                self.n_latent * self.state_dimension,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                self.n_latent * self.state_dimension,
                self.n_latent,
            ),
            self.param_sig,
        )

        # self.convlayers_singval = torch.nn.Sequential(
        #     PeriodicConv1DBlock(
        #         self.state_dimension,
        #         n_channels=self.n_latent,
        #         kernel_size=self.kernel_size,
        #         n_layers=self.n_layers,
        #     ),
        #     torch.nn.Flatten(1),
        #     torch.nn.Linear(
        #         self.n_latent * self.state_dimension,
        #         self.n_latent * self.state_dimension,
        #     ),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(
        #         self.n_latent * self.state_dimension,
        #         self.n_latent,
        #     ),
        #     self.param_sig
        #     # torch.nn.AdaptiveAvgPool1d(1),
        # )

    def forward(self, x):
        x = torch.atleast_2d(x)
        x = x.view(len(x), 1, -1)  # dim x: nbatch * state_dimension
        n_batch = len(x)
        # print(f"{x.shape=}")
        dilalayers = self.dilations_layers(
            x
        )  # nbatch * (n_latent + 1) * state_dimension
        # print(f"{dilalayers.shape=}")

        convlayers = self.convlayers_vec(
            dilalayers
        )  # dim: nbatch * n_latent * state_dimension
        # print(f"{convlayers.shape=}")
        flat_vecs = self.mlp_block(torch.flatten(convlayers, 1))
        vectors = torch.nn.functional.normalize(
            flat_vecs.view(n_batch, self.n_latent, self.state_dimension), dim=-1
        )
        # print(vectors.shape)
        # singvals = self.convlayers_singval(x)
        singvals = self.mlp_singval(torch.flatten(convlayers, 1))
        # print(singvals.shape)
        return torch.concat(
            (vectors, singvals.view(n_batch, self.n_latent, 1)), -1
        ).transpose(1, 2)
