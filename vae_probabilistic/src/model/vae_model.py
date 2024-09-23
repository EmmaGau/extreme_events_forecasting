import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from typing import Dict, Any
import sys 
import argparse
import time
from copy import deepcopy
from model.vae_net import BetaVAE3D  # Import your BetaVAE3D model
from copy import deepcopy
from shutil import copyfile
# Ajoutez le répertoire parent de 'data' au sys.path
sys.path.append(os.path.abspath("/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src"))

# Maintenant vous pouvez importer le module
from data.dataset import DatasetEra
from utils.statistics import DataScaler
from utils.temporal_aggregator import TemporalAggregatorFactory

# Obtenir le chemin absolu du répertoire contenant le script en cours
_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_parent_dir = os.path.dirname(os.path.dirname(_curr_dir))

exps_dir = os.path.join(_parent_dir, 'experiments')

os.makedirs(exps_dir, exist_ok=True)

class VAE3DLightningModule(pl.LightningModule):
    def __init__(self, config_file_path, save_dir, input_dims, output_dims):
        super().__init__()
        oc = OmegaConf.load(open(config_file_path))
        self.config = OmegaConf.to_object(oc)
        self.save_dir = save_dir

        self.save_hyperparameters()

        self.model = BetaVAE3D(
            input_dims=input_dims,
            output_dims=output_dims,
            latent_dim=self.config['model']['latent_dim'],
            hidden_dims=self.config['model']['hidden_dims'],
            layout=self.config['model']['layout'],
            beta=self.config['model']['beta'],
            gamma=self.config['model']['gamma'],
            max_capacity=self.config['model']['max_capacity'],
            Capacity_max_iter=self.config['model']['Capacity_max_iter'],
            loss_type=self.config['model']['loss_type'],
            num_heads=self.config['model']['num_heads'],
            use_attention=self.config['model']['use_attention']
        )
        self.configure_save(config_file_path)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
                
    def set_trainer_kwargs(self,config: Dict[str, Any], devices ) -> Dict[str, Any]:
        """
        Set up trainer kwargs including callbacks for checkpoint saving.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary.
        
        Returns:
            Dict[str, Any]: Trainer kwargs dictionary.
        """
        # Define callbacks
        callbacks = []
        
        # ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.save_dir, "checkpoints", "loss"),
            filename='{epoch:02d}-{val_loss:.2f}',
            save_top_k= config['optim']['save_top_k'],
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        
        # LearningRateMonitor callback
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # Set up WandbLogger
        logger = WandbLogger(project=config['logging']['project_name'])
        
        trainer_kwargs = {
            "max_epochs": config['optim']['max_epochs'],
            "devices": devices,  # Number of devices (int or list of ints)
            "accelerator": "gpu" if isinstance(devices, (int, list)) else "cpu",
            "logger": logger,
            "callbacks": callbacks,
            "log_every_n_steps": config['logging']['log_every_n_steps'],
        }
        
        return trainer_kwargs


    @staticmethod
    def get_dataloaders(dataset_cfg: dict,batch_size: int, val_years: list, test_years: list):
        
        train_config = deepcopy(dataset_cfg)
        val_config = deepcopy(dataset_cfg)
        test_config = deepcopy(dataset_cfg)
        
        train_config['dataset']['relevant_years'] = dataset_cfg['dataset']['relevant_years']
        val_config['dataset']['relevant_years'] = val_years
        test_config['dataset']['relevant_years'] = test_years
        
        data_dirs = dataset_cfg['data_dirs']
        
        scaler = DataScaler(dataset_cfg['scaler'])
        temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])
        train_dataset = DatasetEra(train_config, data_dirs, temp_aggregator_factory, scaler)
        val_dataset = DatasetEra(val_config, data_dirs, temp_aggregator_factory, scaler)
        test_dataset = DatasetEra(test_config, data_dirs, temp_aggregator_factory, scaler)
        print("len train_dataset", len(train_dataset))
        print("len val_dataset", len(val_dataset))
        print("len test_dataset", len(test_dataset))
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_dataloader, val_dataloader, test_dataloader
                
    def forward(self, input, target ):
        return self.model(input,target)

    def training_step(self, batch, batch_idx):
        input, target, *_ = batch
        pred, target, mu, log_var = self(input, target)
        loss_dict = self.model.loss_function(pred,target, mu, log_var, M_N=1.0)
        loss = loss_dict['loss']
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_prediction_loss', loss_dict['prediction_Loss'], on_step=True, on_epoch=True, logger=True)
        self.log('train_kld_loss', loss_dict['KLD'], on_step=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input, target, *_ = batch 
        pred,target, mu, log_var = self(input, target)
        loss_dict = self.model.loss_function(pred,target, mu, log_var, M_N=1.0)
        loss = loss_dict['loss']
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_prediction_loss', loss_dict['prediction_Loss'], on_step=False, on_epoch=True, logger=True)
        self.log('val_kld_loss', loss_dict['KLD'], on_step=False, on_epoch=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        input, target, *_ = batch
        input.shape
        pred, target, mu, log_var = self(input, target)
        loss_dict = self.model.loss_function(pred,target, mu, log_var, M_N=1.0)
        loss = loss_dict['loss']
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_prediction_loss', loss_dict['prediction_Loss'], on_step=False, on_epoch=True, logger=True)
        self.log('test_kld_loss', loss_dict['KLD'], on_step=False, on_epoch=True, logger=True)
        
        return loss


    # TODO : configure save hyperparameters change the decay paramrenters 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['optim']['lr'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BetaVAE3DLightningModule")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parent_parser

def default_save_name():
    now = time.strftime("%Y%m%d_%H%M%S")
    return f"VAE_{now}"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default=default_save_name(), type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on ICAR-ENSO-2021.')
    return parser

def cli_main():
    parser = get_parser()
    args = parser.parse_args()

    if args.cfg is None:
        args.cfg = "/home/egauillard/extreme_events_forecasting/vae_probabilistic/src/configs/default_config.yaml"
    
    oc_from_file = OmegaConf.load(open(args.cfg, "r"))
    dataset_cfg = OmegaConf.to_object(oc_from_file.data)
    batch_size = oc_from_file.optim.batch_size
    max_epochs = oc_from_file.optim.max_epochs
    seed = oc_from_file.optim.seed
        
    pl.seed_everything(seed, workers=True)

    # Initialize dataset and data loader
    val_years, test_years = dataset_cfg["dataset"]["val_years"], dataset_cfg["dataset"]["test_years"]
    train_dl, val_dl, test_dl = VAE3DLightningModule.get_dataloaders(dataset_cfg, batch_size, val_years, test_years) 
    sample = next(iter(train_dl))
    input_shape = list(sample[0].shape)[1:]
    output_shape = list(sample[1].shape)[1:]
    print("input_shape", input_shape)
    print("output_shape", output_shape)

    # Initialize model
    pl_module = VAE3DLightningModule(args.cfg, args.save,input_shape, output_shape)

    # Get trainer kwargs
    trainer_kwargs = pl_module.set_trainer_kwargs(oc_from_file, args.gpus)

    # Initialize Trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Train the pl_module
    trainer.fit(pl_module, train_dl, val_dl)

    # Test the pl_module
    trainer.test(pl_module, test_dl)

if __name__ == '__main__':
    cli_main()