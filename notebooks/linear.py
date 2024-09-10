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

if __name__ == "__main__":
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import pytorch_lightning as pl

        # Generate some random example data
        batch_size = 32
        input_dim = 10
        hidden_dim = 5

        # Example inputs and targets
        x = torch.randn((batch_size, input_dim))  # Random input data
        y = torch.randn((batch_size))             # Random target data

        # Create a DataLoader for training
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the model
        model = LinearEra(input_dim=input_dim, hidden_dim=hidden_dim)

        # Initialize PyTorch Lightning Trainer
        trainer = pl.Trainer(max_epochs=5)

        # Train the model
        trainer.fit(model, train_dataloaders=train_loader)