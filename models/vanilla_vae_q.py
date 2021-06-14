import torch
from ..utils.base import *
from ..utils.base import BaseVAE
from ..utils.quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)
from torch import nn
from torch.nn import functional as F

#### plain VAE with QuaternionConvolutional layers
# Model for the paper: 'A Quaternion-Valued Variational Autoencoder',
# Eleonora Grassucci, Danilo Comminiello, Aurelio Uncini,
# Published at the IEEE ICASSP 2021, June 6-11, 2021.

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

#### plain VAE with QuaternionLinear layers
# Model for the paper: 'An Information-Theoretic Perspective on Proper Quaternion Variational Autoencoders',
# Eleonora Grassucci, Danilo Comminiello, Aurelio Uncini,
# Submitted to Entropy on May 30, 2021.

class QuaternionFCVAE2(BaseVAE):
    """Main QVAE class."""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        n: int = 100,
        **kwargs
    ):
        super(QuaternionFCVAE2, self).__init__()

        self.latent_dim = latent_dim
        self.n = n
        # self.hidden_dims = hidden_dims

        # Build Encoder with quaternion fc layers
        self.encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            QuaternionLinear(32, 32),
            nn.ReLU(),
            QuaternionLinear(32, 64),
            nn.ReLU(),
            QuaternionLinear(64, 128),
            nn.ReLU(),
            QuaternionLinear(128, 256),
            nn.ReLU(),
            QuaternionLinear(256, 512),
            nn.ReLU(),
        )

        self.fc_mu = QuaternionLinear(512, latent_dim * 4)
        self.fc_var = QuaternionLinear(512, latent_dim * 4)

        # Build Decoder with transposed quaternion convolutions
        # hidden_dims.reverse()
        self.decoder_input = QuaternionLinear(latent_dim * 4, 512)
        self.decoder = nn.Sequential(
            QuaternionLinear(512, 256),
            nn.ReLU(),
            QuaternionLinear(256, 128),
            nn.ReLU(),
            QuaternionLinear(128, 64),
            nn.ReLU(),
            QuaternionLinear(64, 32),
            nn.ReLU()
        )

        self.final_layer = nn.Sequential(QuaternionLinear(32, 32), nn.Linear(32, 4))

    def encode(self, input: Tensor):
        """Encode input images in latent space and compute mean and variance.
        Return list containing mean and variance."""
        # Input encoding
        result = self.encoder(input)
        # Mean and variance computation
        mu = self.fc_mu(result)
        mu = mu.view(mu.size(0), 4, self.latent_dim)
        log_var = self.fc_var(result)
        log_var = log_var.view(log_var.size(0), 4, self.latent_dim)
        return [mu, log_var]

    def decode(self, z: Tensor):
        """Decode back latent vector into images.
        Return reconstructed images."""
        result = self.decoder_input(z)
        # result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """Perform reparameterization trick. Adjust sample eps N(0,1) to z N(mu, logvar).
        Return latent vector z."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        """Complete forward training pass."""
        s = input.size()
        input = input.view(-1, s[1])

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z = z.view(z.size(0), -1)
        return [self.decode(z), input, mu, log_var]

    def sample(self, num_samples: int, current_device: int, **kwargs):
        """Sample num_samples from latent space eps N(0,1) and return the corresponding signal."""

        z = torch.randn((num_samples, self.latent_dim * 4))
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        """Reconstruct given test image."""

        return self.forward(x)[0]
