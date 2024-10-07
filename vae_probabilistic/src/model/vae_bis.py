import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Tuple

class Encoder3D(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dims, input_dims, layout="THWC"):
        super(Encoder3D, self).__init__()
        
        self.layout = layout
        if self.layout == "THWC":
            self.encoder_input_dims = input_dims[:3]
        else:  # CTHW
            self.encoder_input_dims = input_dims[1:]

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened size
        self.encoder_out_dims = self.calculate_encoder_output_dims(self.encoder_input_dims, len(hidden_dims))
        self.flattened_size = hidden_dims[-1] * np.prod(self.encoder_out_dims)

        # 3D latent space
        self.fc_mu = nn.Linear(self.flattened_size, np.prod(latent_dims))
        self.fc_var = nn.Linear(self.flattened_size, np.prod(latent_dims))
        self.latent_dims = latent_dims

    def calculate_encoder_output_dims(self, input_dims, num_layers):
        t, h, w = input_dims
        for _ in range(num_layers):
            t = (t - 1) // 2 + 1
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        return (t, h, w)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        if self.layout == "THWC":
            input = input.permute(0, 4, 1, 2, 3)

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result).view(-1, *self.latent_dims)
        log_var = self.fc_var(result).view(-1, *self.latent_dims)

        return [mu, log_var]

class Decoder3D(nn.Module):
    def __init__(self, latent_dims, hidden_dims, input_dims, output_dims, layout="THWC"):
        super(Decoder3D, self).__init__()
        
        self.layout = layout
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims

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

        self.decoder_input = nn.Linear(np.prod(latent_dims), self.flattened_size)

        modules = []
        reversed_hidden_dims = hidden_dims[::-1]

        for i in range(len(reversed_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(reversed_hidden_dims[i],
                                       reversed_hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm3d(reversed_hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_conv = nn.Conv3d(reversed_hidden_dims[-1], self.out_channels, kernel_size=3, padding=1)

    def calculate_encoder_output_dims(self, input_dims, num_layers):
        t, h, w = input_dims
        for _ in range(num_layers):
            t = (t - 1) // 2 + 1
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        return (t, h, w)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), -1)  # Flatten the 3D latent space
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], *self.encoder_out_dims)
        result = self.decoder(result)
        result = self.final_conv(result)

        result = F.adaptive_avg_pool3d(result, self.decoder_output_dims)

        if self.layout == "THWC":
            result = result.permute(0, 2, 3, 4, 1)

        return result

class BetaVAE3DLatent(nn.Module):
    def __init__(self,
                 input_dims: Tuple[int, int, int, int],
                 output_dims: Tuple[int, int, int, int],
                 latent_dims: Tuple[int, int, int],
                 hidden_dims: List = None,
                 layout: str = "THWC",
                 beta: int = 1,
                 **kwargs) -> None:
        super(BetaVAE3DLatent, self).__init__()

        self.latent_dims = latent_dims
        self.beta = beta
        self.layout = layout

        if self.layout == "THWC":
            in_channels = input_dims[3]
        else:  # CTHW
            in_channels = input_dims[0]

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder = Encoder3D(in_channels, hidden_dims, latent_dims, input_dims, layout)
        self.decoder = Decoder3D(latent_dims, hidden_dims, input_dims, output_dims, layout)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> List[torch.Tensor]:
        input, target = input.float(), target.float()
        mu, log_var = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        result = self.decoder(z)
        return [result, target, mu, log_var]

    def custom_extreme_loss(self, y, y_pred):
        # Compute the custom loss Le
        positive_extreme = F.mse_loss(torch.exp(y_pred), torch.exp(y))
        negative_extreme = F.mse_loss(torch.exp(-y_pred), torch.exp(-y))
        return 0.5 * positive_extreme +0.5 * negative_extreme

    def loss_function(self, *args, **kwargs) -> dict:
        pred = args[0]
        target = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        pred_loss = F.mse_loss(pred, target)
        extreme_loss = self.custom_extreme_loss(target, pred)
        pred_loss = 0.5*extreme_loss + 0.5*pred_loss

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=[1,2,3]), dim=0)

        loss = pred_loss + self.beta * kld_weight * kld_loss

        return {'loss': loss, 'prediction_Loss': pred_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, *self.latent_dims)
        z = z.to(current_device)
        samples = self.decoder(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]

# Test the model
if __name__ == '__main__':
    x = torch.randn(1, 6, 111, 360, 6)  # (B, T, H, W, C)
    y = torch.randn(1, 4, 1, 5, 1)  # (B, T, H, W, C)
    latent_dims = (10, 10, 10)  # 3D latent space
    model = BetaVAE3D(layout="THWC", latent_dims=latent_dims, input_dims=(6, 111, 360, 6), output_dims=(4, 1, 5, 1))

    output = model(x, y)
    print("Output shape:", output[0].shape)
    print("Mu shape:", output[2].shape)
    print("Log_var shape:", output[3].shape)