import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple
import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim, input_dims, layout="THWC", use_attention=False, num_heads=8):
        super(Encoder, self).__init__()

        self.layout = layout
        self.use_attention = use_attention
        self.num_heads = num_heads

        if self.layout == "THWC":
            self.encoder_input_dims = input_dims[:3]
        else:  # CTHW
            self.encoder_input_dims = input_dims[1:]

        modules = []
        for i, h_dim in enumerate(hidden_dims):
            # Changement ici : on met la stride temporelle à 2 pour les deux dernières couches
            if i >= len(hidden_dims) - 2:
                stride = (2, 2, 2)
            else:
                stride = (1, 2, 2)
            
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.attention = None

        # Calculate flattened size
        self.encoder_out_dims = self.calculate_encoder_output_dims(self.encoder_input_dims, len(hidden_dims))
        self.flattened_size = hidden_dims[-1] * np.prod(self.encoder_out_dims)

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)

    def calculate_encoder_output_dims(self, input_dims, num_layers):
        t, h, w = input_dims
        for i in range(num_layers):
            # Changement ici : on réduit t deux fois au lieu d'une
            if i >= num_layers - 2:
                t = (t - 1) // 2 + 1
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        return (t, h, w)

    def init_attention(self):
        embed_dim = self.flattened_size
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.num_heads)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        if self.layout == "THWC":
            input = input.permute(0, 4, 1, 2, 3)

        result = self.encoder(input)
        if self.use_attention:
            if self.attention is None:
                self.init_attention()
            result = result.permute(2, 0, 1, 3, 4).contiguous()
            result = result.view(self.encoder_out_dims[0], result.size(1), -1)
            result, _ = self.attention(result, result, result)
            result = result.view(result.size(1), -1)

        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, input_dims, output_dims, layout="THWC"):
        super(Decoder, self).__init__()
        
        self.layout = layout
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims

        if self.layout == "THWC":
            self.out_channels = output_dims[3]
            self.encoder_input_dims = input_dims[:3]
            self.decoder_output_dims = output_dims[:3]
        else:  # CTHW
            self.out_channels = output_dims[0]
            self.encoder_input_dims = input_dims[1:]
            self.decoder_output_dims = output_dims[1:]

        # Calculate the encoder output dimensions
        self.encoder_out_dims = self.calculate_encoder_output_dims(self.encoder_input_dims, len(hidden_dims))
        
        # Calculate flattened size
        self.flattened_size = hidden_dims[-1] * np.prod(self.encoder_out_dims)

        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        # Calculate required upsampling factors
        self.upsample_factors = [t // s for s, t in zip(self.encoder_out_dims, self.output_dims)]

        modules = []
        reversed_hidden_dims = hidden_dims[::-1]

        current_dims = list(self.encoder_out_dims)
        for i in range(len(reversed_hidden_dims) - 1):
            # Calculate stride and output padding for this layer
            stride = tuple(min(2, factor) for factor in self.upsample_factors)
            output_padding = tuple((factor + 1) % 2 for factor in self.upsample_factors)
            
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(reversed_hidden_dims[i],
                                       reversed_hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       output_padding=output_padding),
                    nn.BatchNorm3d(reversed_hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
            
            # Update current dimensions and remaining upsample factors
            current_dims = [cd * s for cd, s in zip(current_dims, stride)]
            self.upsample_factors = [(f + 1) // 2 for f in self.upsample_factors]

        self.decoder = nn.Sequential(*modules)

        # Final convolution to get the correct number of channels
        self.final_conv = nn.Conv3d(reversed_hidden_dims[-1], self.out_channels, kernel_size=3, padding=1)

        # Calculate the dimensions after upsampling
        self.upsampled_dims = current_dims

    def calculate_encoder_output_dims(self, input_dims, num_layers):
        t, h, w = input_dims
        for i in range(num_layers):
            if i >= num_layers - 2:
                t = (t - 1) // 2 + 1
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        return (t, h, w)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], *self.encoder_out_dims)
        result = self.decoder(result)
        result = self.final_conv(result)

        # Crop the result to the desired output dimensions
        result = result[:, :,
                        :self.output_dims[0], 
                        :self.output_dims[1], 
                        :self.output_dims[2]]

        if self.layout == "THWC":
            result = result.permute(0, 2, 3, 4, 1)

        return result

# The rest of the BetaVAE3D class remains the same


class BetaVAE3D(nn.Module):
    def __init__(self,
                 input_dims: Tuple[int, int, int, int],
                 output_dims: Tuple[int, int, int, int],
                 latent_dim: int,
                 hidden_dims: List = None,
                 layout: str = "THWC",
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 num_heads: int = 8,
                 use_attention: bool = False,
                 **kwargs) -> None:
        super(BetaVAE3D, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.num_iter = 0
        self.layout = layout

        if self.layout == "THWC":
            in_channels = input_dims[3]
        else:  # CTHW
            in_channels = input_dims[0]

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder = Encoder(in_channels, hidden_dims, latent_dim, input_dims, layout, use_attention, num_heads)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dims, output_dims, layout)


    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        result = self.decoder(z)
        return [result, input, mu, log_var]

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        pred = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        pred_loss = F.mse_loss(pred, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':
            loss = pred_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = pred_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'prediction_Loss': pred_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decoder(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]

if __name__ == '__main__':
    # Test the BetaVAE3D model
    x = torch.randn(1, 6, 64, 64, 6)  # (B, T, H, W, C)
    model = BetaVAE3D(layout = "THWC",latent_dim=10, input_dims=(6, 64, 64, 6), output_dims = (4, 20, 20, 1))

    output = model(x)
    print(output[2].shape)