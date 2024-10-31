from pytorch_grad_cam import GradCAM
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from model.earthformer_model import CuboidERAModule
from omegaconf import OmegaConf
import os 
from torch.utils.data import DataLoader
from data.temporal_aggregator import TemporalAggregatorFactory
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
    checkpoint_path = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241010_113122/checkpoints/loss/model-loss-epoch=004_valid_loss_epoch=1.10.ckpt"
    exp_dir = checkpoint_path.split('/checkpoints/')[0]
    print(f"Experiment directory: {exp_dir}")
    config_path = os.path.join(exp_dir, 'cfg.yaml')
    oc_from_file = OmegaConf.load(config_path) 
    dataset_cfg = OmegaConf.to_object(oc_from_file.data)
    dataset_cfg['dataset']['relevant_years'] = [2016, 2017]
    total_batch_size = oc_from_file.optim.total_batch_size
    micro_batch_size = oc_from_file.optim.micro_batch_size

    # Obtenir un batch de données
    data_dirs = dataset_cfg['data_dirs']
    temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])
    test_dataset = DatasetEra(dataset_cfg, data_dirs , temp_aggregator_factory)
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
    grad_cam_dir = os.path.join(exp_dir, 'grad_cam')
    os.makedirs(grad_cam_dir, exist_ok=True)
    
    # Save saliency map
    np.save(os.path.join(grad_cam_dir, 'saliency_map.npy'), smap.cpu().numpy())

    # Plot and save saliency map
    plt.figure(figsize=(10, 8))
    plt.imshow(smap.cpu().numpy().mean(axis=0), cmap='viridis')
    plt.colorbar()
    plt.title('Saliency Map')
    plt.savefig(os.path.join(grad_cam_dir, 'saliency_map.png'))
    plt.close()

    # Define target layers
    target_layers = [
        pl_module.torch_nn_module.encoder.blocks[0][0].attn_l[-1]]

    grad_module = GradCAM(pl_module, target_layers)
    cams = grad_module(batch[0], target_category=None)

    # Save GradCAM results
    for i, cam in enumerate(cams):
        np.save(os.path.join(grad_cam_dir, f'gradcam_layer_{i}.npy'), cam.cpu().numpy())
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cam.cpu().numpy().mean(axis=0), cmap='jet')
        plt.colorbar()
        plt.title(f'GradCAM Layer {i}')
        plt.savefig(os.path.join(grad_cam_dir, f'gradcam_layer_{i}.png'))
        plt.close()

    print(f"Saliency map and GradCAM results saved in {grad_cam_dir}")