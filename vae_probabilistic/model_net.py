import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalVAE3D(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim, sequence_length):
        super(SpatioTemporalVAE3D, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.ReLU()
        )
        
        self.attention = nn.MultiheadAttention(64 * 8 * 8, num_heads=4)
        
        self.fc_mu = nn.Linear(64 * 8 * 8 * sequence_length, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8 * sequence_length, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 64 * 8 * 8 * sequence_length)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, input_channels, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1)),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(self.sequence_length, x.size(1), -1)
        
        # Apply attention
        x, _ = self.attention(x, x, x)
        
        x = x.view(x.size(1), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 64, self.sequence_length, 8, 8)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Fonction de perte
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Exemple d'utilisation
input_channels = 1  # par exemple, pour des cartes de température
hidden_dim = 256
latent_dim = 32
sequence_length = 10
batch_size = 16
H, W = 64, 64  # dimensions spatiales

model = SpatioTemporalVAE3D(input_channels, hidden_dim, latent_dim, sequence_length)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Données factices pour l'exemple
x = torch.randn(batch_size, input_channels, sequence_length, H, W)

# Une étape d'entraînement
model.train()
optimizer.zero_grad()
recon_x, mu, logvar = model(x)
loss = vae_loss(recon_x, x, mu, logvar)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")