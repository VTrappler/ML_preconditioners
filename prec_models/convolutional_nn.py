import torch

def construct_conv1D(n_in, n_c, kernel_size):
    padding_mode = 'circular'

    layers_vec = torch.nn.Sequential(
        torch.nn.Conv1d(
            1, n_c, kernel_size, padding=kernel_size // 2, padding_mode=padding_mode
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(
            n_c, n_c, kernel_size, padding=kernel_size // 2, padding_mode=padding_mode
        ),        
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(
            n_c, n_c, kernel_size, padding=kernel_size // 2, padding_mode=padding_mode
        )
    )
    return layers_vec

def construct_conv1D_singularvalue(n_in, n_c, kernel_size):
    padding_mode = 'circular'
    layers_vec = torch.nn.Sequential(
        torch.nn.Conv1d(
            1, n_c, kernel_size, padding=kernel_size // 2, padding_mode=padding_mode
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(
            n_c, n_c, kernel_size, padding=kernel_size // 2, padding_mode=padding_mode
        ),
        torch.nn.AdaptiveAvgPool1d(1)
)
    return layers_vec

class ConvLayersSVD(torch.nn.Module):
    def __init__(self, state_dimension, n_latent, kernel_size) -> None:
        super().__init__()
        self.state_dimension = state_dimension
        self.n_latent = n_latent
        self.kernel_size = kernel_size
        self.layers_vec = construct_conv1D(self.state_dimension, self.n_latent, kernel_size=self.kernel_size)
        self.layers_sing = construct_conv1D_singularvalue(self.state_dimension, self.n_latent, kernel_size=self.kernel_size)
    
    def forward(self, x):
        x = torch.atleast_2d(x)
        x = x.view(len(x), 1, -1)
        n_batch = len(x)
        vectors = torch.nn.functional.normalize(self.layers_vec(x), dim=-1)
        # print(vectors.shape)
        singvals = self.layers_sing(x)
        # print(singvals.shape)
        return torch.concat((vectors, singvals.view(n_batch, self.n_latent, 1)), -1).transpose(1, 2)