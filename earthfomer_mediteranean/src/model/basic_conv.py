import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
import torch.distributions as dist
from data.dataset import DatasetEra
from utils.scaler import DataScaler
from utils.temporal_aggregator import TemporalAggregatorFactory
import wandb
from pytorch_lightning.loggers import WandbLogger
from copy import deepcopy


class Conv3DEra(pl.LightningModule):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super(Conv3DEra, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_dim * 28 * 28, hidden_dim)
        self.alpha_layer = nn.Linear(hidden_dim, out_dim)
        self.beta_layer = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # Reorder dimensions to [batch, channels, temporal, height, width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Reorder to [batch, temporal, channels, height, width]
        x = x.view(x.size(0), x.size(1), -1)  # Flatten [batch, temporal, channels * height * width]
        x = F.relu(self.fc1(x))
        alpha = F.softplus(self.alpha_layer(x)) + 1e-6  # Ensuring positive values for alpha
        beta = F.softplus(self.beta_layer(x)) + 1e-6   # Ensuring positive values for beta
        return alpha, beta
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        alpha, beta = self.forward(x)
        gamma_dist = dist.Gamma(alpha, beta)
        loss = -gamma_dist.log_prob(y).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        alpha, beta = self.forward(x)
        gamma_dist = dist.Gamma(alpha, beta)
        loss = -gamma_dist.log_prob(y).mean()  
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        alpha, beta = self.forward(x)
        gamma_dist = dist.Gamma(alpha, beta)
        loss = -gamma_dist.log_prob(y).mean()  
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    import pytorch_lightning as pl

    TRAINING_YEARS = list(range(1940, 1950))
    VAL_YEARS = list(range(1990, 2000))
    TEST_YEARS = list(range(2005, 2015))

    data_dirs = {'mediteranean': {'tp':"/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/PR/PR_era5_MED_1degr_19400101_20240229.nc"},
                 'north_hemisphere': {}}

    wandb_config = {
        'dataset': {
            'variables_nh': [],
            'variables_med': ['tp'],
            'target_variable': 'tp',
            'relevant_years': TRAINING_YEARS,
            'relevant_months': [10,11,12,1,2,3],
            'land_sea_mask': '/scistor/ivm/shn051/extreme_events_forecasting/primary_code/data/ERA5_land_sea_mask_1deg.nc',
            'spatial_resolution': 1,
            'predict_sea_land': True,
        },
        'scaler': {
            'mode': 'standardize'
        },
        'temporal_aggregator': {
            'stack_number_input': 3,
            'lead_time_number': 3,
            'resolution_input': 7,
            'resolution_output': 7,
            'scaling_years': TRAINING_YEARS,
            'scaling_months': [10,11,12,1,2,3], 
            'gap': 1,
        }
    }

    batch_size = 8
    hidden_dim = 16
    C = 1
    out_dim = 1
    
    wandb.init(project='linear_era', config=wandb_config)

    # Initialize dataset and dataloaders
    scaler = DataScaler(wandb_config['scaler'])
    temp_aggregator_factory = TemporalAggregatorFactory(wandb_config['temporal_aggregator'], scaler)

    # Create train, val, and test datasets
    train_config = deepcopy(wandb_config)
    train_config['dataset']['relevant_years'] = TRAINING_YEARS
    train_dataset = DatasetEra(train_config, data_dirs, temp_aggregator_factory)

    val_config = deepcopy(wandb_config)
    val_config['dataset']['relevant_years'] = VAL_YEARS
    val_dataset = DatasetEra(val_config, data_dirs, temp_aggregator_factory)

    test_config = deepcopy(wandb_config)
    test_config['dataset']['relevant_years'] = TEST_YEARS
    test_dataset = DatasetEra(test_config, data_dirs, temp_aggregator_factory)

    # Dataloaders
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = Conv3DEra(in_channels=1, hidden_dim=16, out_dim=1)

    # Logger
    wandb_logger = WandbLogger(project='linear_era', config=wandb_config)

    # Train
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Test
    trainer.test(model, dataloaders=test_dataloader)

    wandb.finish()