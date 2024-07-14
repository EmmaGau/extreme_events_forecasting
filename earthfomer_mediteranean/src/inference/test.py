import os
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
from omegaconf import OmegaConf
from  model.earthformer_model import CuboidERAModule
from data.dataset import DatasetEra
from utils.scaler import DataScaler
from utils.temporal_aggregator import TemporalAggregatorFactory
import pandas as pd

def load_model(checkpoint_path, config_path):
    # Load configuration
    oc = OmegaConf.load(config_path)
    
    # Initialize the model
    model = CuboidERAModule(
        total_num_steps=1,  # This value doesn't matter for inference
        config_file_path=config_path,
        save_dir="temp"  # This directory won't be used for inference
    )
    
    # Load the state dict
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'])
    
    return model, oc

def inference_and_plot(model, test_dataloader, save_folder, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Processing batches"):
            batch = [i.to(device) for i in batch]
            inputs, targets = batch
            
            
            predictions, _, _, _ = model(batch)
            
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(targets.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)


    # Create DataArrays
    lats = np.linspace(30, 45, all_predictions.shape[2])  # Adjust as needed
    lons = np.linspace(-10, 40, all_predictions.shape[3])  # Adjust as needed
    times = pd.date_range(start='2010-01-01', periods=all_predictions.shape[1], freq='D')  # Adjust as needed
    print([range(all_predictions.shape[0]), times, lats, lons])

    coords_predictions = {
    'sample': np.arange(all_predictions.shape[0]),
    'time': times,
    'lat': lats,
    'lon': lons
}

    coords_ground_truth = {
        'sample': np.arange(all_ground_truth.shape[0]),
        'time': times,
        'lat': lats,
        'lon': lons
    }
    print(all_predictions.shape)
    print(coords_predictions)
    da_predictions = xr.DataArray(
        all_predictions.squeeze(),
        coords=coords_predictions,
        dims=['sample', 'time', 'lat', 'lon'])

    da_ground_truth = xr.DataArray(
        all_ground_truth.squeeze(),
        coords=coords_ground_truth,
        dims=['sample', 'time', 'lat', 'lon']
    )

    # Plotting and saving
    os.makedirs(save_folder, exist_ok=True)

    for i in tqdm(range(all_predictions.shape[0]), desc="Saving plots"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot prediction
        da_predictions[i, 0].plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='viridis')
        ax1.coastlines()
        ax1.set_title('Prediction')

        # Plot ground truth
        da_ground_truth[i, 0].plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='viridis')
        ax2.coastlines()
        ax2.set_title('Ground Truth')

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'comparison_plot_{i}.png'))
        plt.close()

if __name__ == "__main__":
    # Paths
    checkpoint_path = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240710_165244/checkpoints/model-epoch=024.ckpt"
    config_path = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240710_165244/cfg.yaml"
    save_folder = "results"

    # Load model and configuration
    model, oc = load_model(checkpoint_path, config_path)

    # Get test dataloader
    dataset_cfg = OmegaConf.to_object(oc.data)
    test_config = dataset_cfg.copy()
    test_config['dataset']['relevant_years'] = [2010, 2011]  # Your TEST_YEARS
    data_dirs = dataset_cfg['data_dirs']
    scaler = DataScaler(dataset_cfg['scaler'])
    temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])
    test_dataset = DatasetEra(test_config, data_dirs, temp_aggregator_factory, scaler)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=oc.optim.micro_batch_size, shuffle=False, num_workers=4)

    # Run inference and save plots
    inference_and_plot(model, test_dataloader, save_folder, scaler)