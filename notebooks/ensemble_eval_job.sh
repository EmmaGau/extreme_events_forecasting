#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=9000

#SBATCH --output=/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/outputs/output-%j.out
#SBATCH --error=/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/outputs/error-%j.err


#!/bin/bash

# Chemins absolus
HOME_DIR="/gpfs/home3/egauillard"
PROJECT_PATH="$HOME_DIR/extreme_events_forecasting/primary_code"
ENV_PATH="$HOME_DIR/extreme_events_forecasting/extreme_events_env"

# Chargez les modules
module load 2023
module load CUDA-Samples/12.1-GCC-12.3.0-CUDA-12.1.1

# Activez l'environnement virtuel
source $ENV_PATH/bin/activate

# Exportez PYTHONPATH
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"

# Allez dans le répertoire du projet
cd $PROJECT_PATH

# Débogage
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Python version: $(python --version)"
echo "Python executable: $(which python)"

# Exécutez le script
python -c "import sys; print(sys.path)"

python ensemble_eval.py --predfiles /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240814_100758_2005_every_fine/inference_plots/all_predictions.nc  /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_002103_every_fine_14/inference_plots/all_predictions.nc  /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_003448_every_fine_15/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_145406_every_fine_19/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_164613_every_fine_20/inference_plots/all_predictions.nc
python ensemble_eval.py --predfiles /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240818_171920_every_coarse_2005_10/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240820_142751_every_coarse_2005_twh/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_114852_every_coarse_16/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_125058_every_coarse_17/inference_plots/all_predictions.nc
python ensemble_eval.py --predfiles /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240814_102829_2005_coarse_input_fine_target_45/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_120806_coarse_input_fine_target_42/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_122224_coarse_input_fine_target_0/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_123711_coarse_input_fine_target_10/inference_plots/all_predictions.nc /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_132714_coarse_input_fine_target_17/inference_plots/all_predictions.nc