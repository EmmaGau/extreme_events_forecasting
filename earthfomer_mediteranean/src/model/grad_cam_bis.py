from pytorch_grad_cam import GradCAM
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from earthformer_model import CuboidERAModule
from omegaconf import OmegaConf
import os 
from torch.utils.data import DataLoader
from utils.statistics import DataScaler
from utils.temporal_aggregator import TemporalAggregatorFactory
from data.dataset import DatasetEra




def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == "__main__":
    # Charger le module et la configuration
    checkpoint_path = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_003448_every_fine_15/checkpoints/loss/last.ckpt"
    exp_dir = checkpoint_path.split('/checkpoints/')[0]
    print(f"Experiment directory: {exp_dir}")
    config_path = os.path.join(exp_dir, 'cfg.yaml')
    oc_from_file = OmegaConf.load(config_path) 
    dataset_cfg = OmegaConf.to_object(oc_from_file.data)
    dataset_cfg['dataset']['relevant_years'] = [2016, 2017]
    total_batch_size = oc_from_file.optim.total_batch_size
    micro_batch_size = oc_from_file.optim.micro_batch_size

    # Obtenir un batch de données
    scaler = DataScaler(dataset_cfg['scaler'])
    data_dirs = dataset_cfg['data_dirs']
    temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])
    test_dataset = DatasetEra(dataset_cfg, data_dirs , temp_aggregator_factory, scaler)
    test_dl = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    batch = next(iter(test_dl))
    input_shape = batch[0].shape[1:]
    output_shape = batch[1].shape[1:]
    pl_module = CuboidERAModule(
            total_num_steps=1,
            config_file_path=config_path,
            save_dir="temp",
            input_shape=input_shape,
            output_shape=output_shape
        )
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    pl_module.load_state_dict(state_dict['state_dict'])
    print("module loaded")

    smap = pl_module.compute_saliency_map(batch)
    print(smap)

    # Définir les couches cibles
    target_layers = [
        pl_module.torch_nn_module.encoder.blocks[0][0].attn_l[-1]]

    grad_module = GradCAM(pl_module, target_layers)
    cams = grad_module(batch[0], target_category=None)