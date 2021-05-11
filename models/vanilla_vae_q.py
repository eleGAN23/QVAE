import torch
from ..utils.base import *
from ..utils.base import BaseVAE
from ..utils.quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)
from torch import nn
from torch.nn import functional as F


class QuaternionVanillaVAE(BaseVAE):
    '''Main QVAE class. '''

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(QuaternionVanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder with quaternion convolutions
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    QuaternionConv(in_channels, out_channels=h_dim,
                                kernel_size=3, stride=2, padding=[1, 1], dilatation=[1, 1]),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = QuaternionLinear(hidden_dims[-1]*4, 4*latent_dim)
        self.fc_var = QuaternionLinear(hidden_dims[-1]*4, 4*latent_dim)


        # Build Decoder with transposed quaternion convolutions
        modules = []
        self.decoder_input = QuaternionLinear(4*latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    QuaternionTransposeConv(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            QuaternionTransposeConv(hidden_dims[-1], hidden_dims[-1],
                                               kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.LeakyReLU(),
                            QuaternionConv(hidden_dims[-1], out_channels=4,
                                      kernel_size=3, stride=1, padding=1),
                            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        '''Encode input images in latent space and compute mean and variance.
            Return list containing mean and variance.'''

        # Input encoding
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Mean and variance computation
        mu = self.fc_mu(result)
        mu = mu.view(mu.size()[0], 4, self.latent_dim)
        log_var = self.fc_var(result)
        log_var = log_var.view(log_var.size()[0], 4, self.latent_dim)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''Decode back latent vector into images.
            Return reconstructed images.'''

        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        '''Perform reparameterization trick. Adjust sample eps N(0,1) to z N(mu, logvar).
            Return latent vector z.'''

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        '''Complete forward training pass.'''

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z = z.view(z.size()[0], -1)
        return  [self.decode(z), input, mu, log_var]

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        '''Sample num_samples from latent space eps N(0,1) and return the corresponding images.'''

        z = torch.randn(num_samples,
                        self.latent_dim*4)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        '''Reconstruct given test image.'''

        return self.forward(x)[0]
