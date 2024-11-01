import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from model.earthformer_model import CuboidERAModule
from omegaconf import OmegaConf
import os 
from torch.utils.data import DataLoader
from utils.statistics import DataScaler
from data.temporal_aggregator import TemporalAggregatorFactory
from data.dataset import DatasetEra

VAL_YEARS = [2006,2015]
TEST_YEARS = [2016, 2024]


class GradCAMLightning:
    def __init__(self, pl_module, target_layers):
        self.pl_module = pl_module
        self.target_layers = target_layers
        self.activations = {layer: None for layer in target_layers}
        self.gradients = {layer: None for layer in target_layers}

        for layer in target_layers:
            layer.register_forward_hook(self.save_activation(layer))
            layer.register_full_backward_hook(self.save_gradient(layer))

    def save_activation(self, layer):
        def hook(module, input, output):
            self.activations[layer] = output
        return hook

    def save_gradient(self, layer):
        def hook(module, grad_input, grad_output):
            self.gradients[layer] = grad_output[0]
        return hook

    def generate_cam(self, batch):
        self.pl_module.eval()
        pred_seq, loss, input, target, clim_tensor = self.pl_module(batch)
        self.pl_module.zero_grad()
        loss.backward()

        cams = {}

        for layer in self.target_layers:
            activations = self.activations[layer][0] # (B,T,H,W,C)
            gradients = self.gradients[layer]
            print(f"Layer: {layer.__class__.__name__}")
            print(f"Gradient shape: {gradients.shape}")


            print(f"Layer: {layer.__class__.__name__}")
            print(f"Gradient shape: {gradients.shape}")
            print(f"Activation shape: {activations.shape}")

            # Calculer l'importance globale
            weights = torch.mean(gradients, dim=(1, 2, 3))
            cam = torch.sum(weights.unsqueeze(1).unsqueeze(2).unsqueeze(3) * activations, dim=0)
            cam = F.relu(cam)
            cam = cam / cam.max()

            # Interpoler la cam pour correspondre aux dimensions de l'entrée
            cam_resized = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                                        size=(input.shape[1], input.shape[2], input.shape[3]),
                                        mode='trilinear',
                                        align_corners=False).squeeze()

            cams[layer] = cam_resized

        return cams, pred_seq, input, target

def visualize_cams(cams, input, pred_seq, save_path):
    num_layers = len(cams)
    fig, axes = plt.subplots(num_layers, 4, figsize=(20, 5*num_layers))
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (layer, cam) in enumerate(cams.items()):
        # Importance temporelle
        temporal_importance = torch.mean(cam, dim=(1, 2))
        axes[idx, 0].plot(temporal_importance.detach().cpu().numpy())
        axes[idx, 0].set_title(f"Layer {idx+1} Temporal Importance")
        axes[idx, 0].set_xlabel("Time steps")
        axes[idx, 0].set_ylabel("Importance")
        
        # Importance spatiale (moyenne sur le temps et les canaux)
        spatial_importance = torch.mean(cam, dim=(0, 3))
        im = axes[idx, 1].imshow(spatial_importance.detach().cpu().numpy(), cmap='hot')
        axes[idx, 1].set_title(f"Layer {idx+1} Spatial Importance")
        plt.colorbar(im, ax=axes[idx, 1])
        
        # Importance des canaux
        channel_importance = torch.mean(cam, dim=(0, 1, 2))
        axes[idx, 2].bar(range(len(channel_importance)), channel_importance.detach().cpu().numpy())
        axes[idx, 2].set_title(f"Layer {idx+1} Channel Importance")
        axes[idx, 2].set_xlabel("Channels")
        axes[idx, 2].set_ylabel("Importance")
        
        # Visualiser l'entrée ou la prédiction
        if idx == 0:
            input_slice = input[0, 0, :, :, 0].detach().cpu().numpy()
            im = axes[idx, 3].imshow(input_slice, cmap='viridis')
            axes[idx, 3].set_title("Input (first timestep, first channel)")
        elif idx == 1:
            pred_slice = pred_seq[0, 0, :, :, 0].detach().cpu().numpy()
            im = axes[idx, 3].imshow(pred_slice, cmap='viridis')
            axes[idx, 3].set_title("Prediction (first timestep, first channel)")
        plt.colorbar(im, ax=axes[idx, 3])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")


if __name__ == "__main__":
    # Charger le module et la configuration
    checkpoint_path = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_114852_every_coarse_16/checkpoints/skill/model-skill-epoch=031-valid_skill_score=0.03.ckpt"
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

    # Définir les couches cibles
    target_layers = [
        pl_module.torch_nn_module.encoder.blocks[0][0].attn_l[-1],
        pl_module.torch_nn_module.encoder.blocks[-1][0].attn_l[-1],
        pl_module.torch_nn_module.decoder.self_blocks[0][0].attn_l[-1]
    ]

    grad_cam = GradCAMLightning(pl_module, target_layers)

    # Générer les CAMs
    cams, pred_seq, input, target = grad_cam.generate_cam(batch)

    # Visualisation
    save_dir = os.path.join(exp_dir, 'grad_cam_results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'grad_cam_visualization.png')
    visualize_cams(cams, input, pred_seq, save_path)