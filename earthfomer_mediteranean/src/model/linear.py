import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
import torch.distributions as dist

class LinearEra(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim):
        super(LinearEra, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        # Two outputs: shape (alpha) and rate (beta) parameters for the Gamma distribution
        self.alpha_layer = nn.Linear(hidden_dim, 1)
        self.beta_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        alpha = F.softplus(self.alpha_layer(x)) + 1e-6  # Ensuring positive values for alpha
        beta = F.softplus(self.beta_layer(x)) + 1e-6   # Ensuring positive values for beta
        return alpha, beta
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        alpha, beta = self.forward(x)
        gamma_dist = dist.Gamma(alpha, beta)
        loss = -gamma_dist.log_prob(y).mean()  # Negative log likelihood
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        alpha, beta = self.forward(x)
        gamma_dist = dist.Gamma(alpha, beta)
        loss = -gamma_dist.log_prob(y).mean()  # Negative log likelihood
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        alpha, beta = self.forward(x)
        gamma_dist = dist.Gamma(alpha, beta)
        loss = -gamma_dist.log_prob(y).mean()  # Negative log likelihood
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
