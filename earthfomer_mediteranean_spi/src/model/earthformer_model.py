import warnings
from typing import Sequence
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from model.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.enso.enso_dataloader import ENSOLightningDataModule, NINO_WINDOW_T
from copy import deepcopy
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data.dataset import DatasetEra
from torch.utils.data import DataLoader
from utils.statistics import DataScaler
from utils.temporal_aggregator import TemporalAggregatorFactory
import time
from torchmetrics import R2Score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")
pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
pytorch_state_dict_name = "earthformer_icarenso2021.pt"

VAL_YEARS = [2006,2015]
TEST_YEARS = [2016, 2024]

class CuboidERAModule(pl.LightningModule):
    def __init__(self,
                 total_num_steps: int,
                 config_file_path: str = "config.yaml",
                 save_dir: str = None,
                 input_shape: Sequence[int] = None,
                 output_shape: Sequence[int]= None,):
        super(CuboidERAModule, self).__init__()
        oc = OmegaConf.load(open(config_file_path))
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        self.input_shape = input_shape if input_shape is not None else model_cfg["input_shape"]
        self.output_shape = output_shape if output_shape is not None else model_cfg["target_shape"]
        self.season_float = model_cfg["season_float"] if "season_float" in model_cfg else False
        
        self.torch_nn_module = CuboidTransformerModel(
            season_float = self.season_float,
            gaussian = False,
            input_shape= self.input_shape,
            target_shape= self.output_shape,
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=model_cfg["self_pattern"] if isinstance(model_cfg["self_pattern"], str) else OmegaConf.to_container(model_cfg["self_pattern"]),
            dec_self_attn_patterns=model_cfg["cross_self_pattern"] if isinstance(model_cfg["cross_self_pattern"], str) else OmegaConf.to_container(model_cfg["cross_self_pattern"]),
            dec_cross_attn_patterns=model_cfg["cross_pattern"] if isinstance(model_cfg["cross_pattern"], str) else OmegaConf.to_container(model_cfg["cross_pattern"]),
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],

             # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers= model_cfg.get("initial_downsample_stack_conv_num_layers", 3),
            initial_downsample_stack_conv_dim_list= model_cfg.get("initial_downsample_stack_conv_dim_list", [16, 64, 128]),
            initial_downsample_stack_conv_downscale_list= model_cfg.get("initial_downsample_stack_conv_downscale_list", [3, 2, 2]),
            initial_downsample_stack_conv_num_conv_list= model_cfg.get("initial_downsample_stack_conv_num_conv_list", [2, 2, 2]),
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],  
        )

        self.total_num_steps = total_num_steps
        self.save_hyperparameters(oc)
        self.oc = oc
        self.dataset_config = oc.data
        self.layout = oc.layout.layout
        self.channel_axis = self.layout.find("C")
        print("channel axis", self.channel_axis)
        self.batch_axis = self.layout.find("N")
        print("input_shape",input_shape)
        print("output_shape", output_shape)
        self.channels = self.input_shape[self.channel_axis-1]
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_r2 = R2Score(num_outputs=self.output_shape[self.channel_axis-1], multioutput='uniform_average')
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        for i in range(self.output_shape[self.channel_axis-1]):
            setattr(self, f'valid_r2_var_{i}', R2Score())

        for i in range(self.output_shape[self.channel_axis-1]):
            setattr(self, f'valid_mae_var_{i}', torchmetrics.MeanAbsoluteError())

        self.configure_save(cfg_file_path=config_file_path)

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
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)
   
    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        # Checkpoint callback for valid_loss_epoch
        checkpoint_callback_loss = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints", "loss"),
            filename="model-loss-{epoch:03d}_{valid_loss_epoch:.2f}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )

        # Checkpoint callback for valid_skill_score
        checkpoint_callback_skill = ModelCheckpoint(
            monitor="valid_skill_score",
            dirpath=os.path.join(self.save_dir, "checkpoints", "skill"),
            filename="model-skill-{epoch:03d}-{valid_skill_score:.2f}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="max",
        )

        # Collect callbacks from kwargs and add our checkpoint callbacks
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback_loss, checkpoint_callback_skill]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
        logger += [tb_logger, csv_logger, wandb_logger]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            # strategy="ddp",
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_dataloaders(dataset_cfg: dict, total_batch_size: int, micro_batch_size: int, 
                        VAL_YEARS, TEST_YEARS):
        
        train_config = deepcopy(dataset_cfg)
        val_config = deepcopy(dataset_cfg)
        test_config = deepcopy(dataset_cfg)
        
        train_config['dataset']['relevant_years'] = dataset_cfg['dataset']['relevant_years']
        val_config['dataset']['relevant_years'] = VAL_YEARS
        test_config['dataset']['relevant_years'] = TEST_YEARS

        data_dirs = dataset_cfg['data_dirs']
        
        scaler = DataScaler(dataset_cfg['scaler'])
        temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])
        train_dataset = DatasetEra(train_config, data_dirs, temp_aggregator_factory, scaler)
        val_dataset = DatasetEra(val_config, data_dirs, temp_aggregator_factory, scaler)
        test_dataset = DatasetEra(test_config, data_dirs, temp_aggregator_factory, scaler)
        print("len train_dataset", len(train_dataset))
        print("len val_dataset", len(val_dataset))
        print("len test_dataset", len(test_dataset))
        
        train_dataloader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=micro_batch_size, shuffle=False, num_workers=4)
        
        return train_dataloader, val_dataloader, test_dataloader

    def forward(self, batch):
        input, target, season_float, year_float, clim_tensor, *_ = batch
        input = input.float() # (N, in_len, lat, lon, nb_target)
        target = target.float()# (N, in_len, lat, lon, nb_target)
        clim_tensor = clim_tensor.float()
        season_float = season_float.float()
        year_float = year_float.float()
        pred_seq = self.torch_nn_module(input, season_float, year_float)
        print(pred_seq.shape, target.shape)
        loss = F.mse_loss(pred_seq, target)
        return pred_seq, loss, input, target, clim_tensor

    def training_step(self, batch, batch_idx):
        pred_seq, loss, input, target, clim_tensor = self(batch)
        mse_per_var = F.mse_loss(pred_seq, target, reduction='none').mean(dim=(0, 1, 2, 3))
        
        for i, mse in enumerate(mse_per_var):
            self.log(f'train_loss_var_{i}', mse, on_step=False, on_epoch=True)

        # Calcul du skill score par rapport à la climatologie
        mse_climatology = F.mse_loss(clim_tensor, target)
        skill_score = 1 - (loss / mse_climatology)
        self.log('train_skill_score', skill_score, on_step=False, on_epoch=True)

        micro_batch_size = input.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if self.local_rank == 0:
            self.save_vis_step_end(
                data_idx=data_idx,
                context_seq=input.detach().float().cpu().numpy(),
                target_seq=target.detach().float().cpu().numpy(),
                pred_seq=pred_seq.detach().float().cpu().numpy(),
                mode="train", )
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch[0].shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            pred_seq, loss, in_seq, target_seq, clim_tensor = self(batch)
            mae_per_var = torch.abs(pred_seq - target_seq).mean(dim=(0, 1, 2, 3))
            for i, mae in enumerate(mae_per_var):
                getattr(self, f'valid_mae_var_{i}')(pred_seq[..., i], target_seq[..., i])
    
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    mode="val", )
            self.valid_mse(pred_seq, target_seq)
            self.valid_mae(pred_seq, target_seq)
            self.valid_r2(pred_seq.view(-1, self.output_shape[self.channel_axis-1]), 
              target_seq.view(-1, self.output_shape[self.channel_axis-1]))
            for i in range(self.output_shape[self.channel_axis-1]):
                getattr(self, f'valid_r2_var_{i}')(pred_seq[..., i].view(-1, 1), target_seq[..., i].view(-1, 1))

            # Calcul du skill score par rapport à la climatologie
            mse_model = F.mse_loss(pred_seq, target_seq)
            mse_climatology = F.mse_loss(clim_tensor, target_seq)
            skill_score = 1 - (mse_model / mse_climatology)
            num_variables = pred_seq.shape[-1]
            for i in range(num_variables):
                mse_model_i = F.mse_loss(pred_seq[..., i], target_seq[..., i])
                mse_climatology_i = F.mse_loss(clim_tensor[..., i], target_seq[..., i])
                skill_score_i = 1 - (mse_model_i / mse_climatology_i)
                self.log(f'valid_skill_score_var_{i}', skill_score_i, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('valid_skill_score', skill_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {'loss': loss}

    def on_validation_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        valid_r2 = self.valid_r2.compute()
        valid_loss = self.trainer.callback_metrics['valid_loss']
        
        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('valid_r2_epoch', valid_r2, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_epoch=True, sync_dist=True)

        for i in range(self.output_shape[self.channel_axis-1]):
            r2_var = getattr(self, f'valid_r2_var_{i}').compute()
            self.log(f'valid_r2_epoch_var_{i}', r2_var, prog_bar=True, on_epoch=True, sync_dist=True)
            getattr(self, f'valid_r2_var_{i}').reset()

        for i in range(self.output_shape[self.channel_axis-1]):
            mae_var = getattr(self, f'valid_mae_var_{i}').compute()
            self.log(f'valid_mae_epoch_var_{i}', mae_var, prog_bar=True, on_epoch=True, sync_dist=True)
            getattr(self, f'valid_mae_var_{i}').reset()
        
        self.valid_mae.reset()
        self.valid_mse.reset()
        self.valid_r2.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch[0].shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            pred_seq, loss, in_seq, target_seq, clim_tensor = self(batch)
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    mode="test", )

            self.test_mse(pred_seq, target_seq)
            self.test_mae(pred_seq, target_seq)

            # Calcul du skill score par rapport à la climatologie
            mse_model = F.mse_loss(pred_seq, target_seq)
            mse_climatology = F.mse_loss(clim_tensor, target_seq)
            skill_score = 1 - (mse_model / mse_climatology)
            num_variables = pred_seq.shape[-1]
            for i in range(num_variables):
                mse_model_i = F.mse_loss(pred_seq[..., i], target_seq[..., i])
                mse_climatology_i = F.mse_loss(clim_tensor[..., i], target_seq[..., i])
                skill_score_i = 1 - (mse_model_i / mse_climatology_i)
                self.log(f'test_skill_score_var_{i}', skill_score_i, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('test_skill_score', skill_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
    def on_test_epoch_end(self, outputs=None):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()

        self.log('test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()


    def save_vis_step_end(
            self,
            data_idx: int,
            context_seq: np.ndarray,
            target_seq: np.ndarray,
            pred_seq: np.ndarray,
            mode: str = "train",
            prefix: str = ""):
        r"""
        Parameters
        ----------
        data_idx
        context_seq, target_seq, pred_seq:   np.ndarray
            layout should not include batch
        mode:   str
        """
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if data_idx in example_data_idx_list:
                # TODO: add visualization
                pass

def default_save_name():
    now = time.strftime("%Y%m%d_%H%M%S")
    return f"earthformer_era_{now}"

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

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.cfg is None:
        args.cfg = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean_spi/src/configs/earthformer_default.yaml"
    
    oc_from_file = OmegaConf.load(open(args.cfg, "r"))
    dataset_cfg = OmegaConf.to_object(oc_from_file.data)
    total_batch_size = oc_from_file.optim.total_batch_size
    micro_batch_size = oc_from_file.optim.micro_batch_size
    max_epochs = oc_from_file.optim.max_epochs
    seed = oc_from_file.optim.seed
        
    seed_everything(seed, workers=True)

    train_dl, val_dl, test_dl = CuboidERAModule.get_dataloaders(dataset_cfg, total_batch_size, micro_batch_size, VAL_YEARS, TEST_YEARS)
    sample = next(iter(train_dl))
    input_shape = list(sample[0].shape)[1:]
    output_shape = list(sample[1].shape)[1:]
    print("input_shape", input_shape)
    print("output_shape", output_shape)

    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)

    total_num_steps = CuboidERAModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=  len(train_dl.dataset),
        total_batch_size=total_batch_size,
    )
    pl_module = CuboidERAModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        config_file_path=args.cfg,
        input_shape = input_shape,
        output_shape = output_shape)

    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)

    checkpoint_callback_loss = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(pl_module.save_dir, "checkpoints", "loss"),
            filename="model-loss-{epoch:03d}",
            save_top_k=oc_from_file.optim.save_top_k,
            save_last=True,
            mode="min",
        )

    # Train the model
    trainer.fit(model=pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Load the best checkpoint
    best_checkpoint_path = checkpoint_callback_loss.best_model_path
    if best_checkpoint_path:
        pl_module = CuboidERAModule.load_from_checkpoint(best_checkpoint_path)

    # Run the test process
    trainer.test(model=pl_module, dataloaders=test_dl)

if __name__ == "__main__":
    main()