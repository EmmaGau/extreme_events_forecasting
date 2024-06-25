import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import DatasetEra  # Assurez-vous que le nom du fichier contenant DatasetEra est correct
from model.linear import LinearEra  # Assurez-vous que le nom du fichier contenant LinearEra est correct
from utils.temporal_aggregator import TemporalAggregator, TemporalAggregatorFactory
from utils.statistics import DataStatistics
from utils.scaler import DataScaler
from pytorch_lightning.loggers import WandbLogger


data_dirs = {'mediteranean': {'tp':''},
             'north_hemisphere': {}}

def main():
    # Wandb configuration
    wandb_config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'max_epochs': 10,
        'variables_nh': ['var1', 'var2'],  # Exemple de variables
        'variables_med': ['var3', 'var4'],
        'data_dirs': '/path/to/data',  # Exemple de chemin vers les données
        # Ajoutez d'autres configurations si nécessaire
    }
     # Initialize wandb
    wandb.init(project='linear_era', config=wandb_config)

    # Initialize dataset and dataloaders
    scaler = DataScaler(wandb_config['scaler'])
    temp_aggregator_factory = TemporalAggregatorFactory(wandb_config['temporal_aggregator'], scaler)
    

    train_dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory)
    print("len dataset", train_dataset.__len__())

    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    sample = next(iter(dataloader))

    # Initialize the model
    model = LinearEra()

    # Define Trainer
    trainer = pl.Trainer(
        max_epochs=wandb_config['max_epochs'],
        logger=pl.loggers.WandbLogger()  # This automatically logs to wandb
    )

    # Train the model
    trainer.fit(model, train_dataloader)

    # Save the model checkpoint
    trainer.save_checkpoint("linear_era_model.ckpt")

    # Finish the wandb run
    wandb.finish()
