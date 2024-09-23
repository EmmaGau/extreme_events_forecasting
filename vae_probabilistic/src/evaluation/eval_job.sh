#!/bin/bash
#SBATCH --job-name=vae_eval
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=9000

#SBATCH --output=/home/egauillard/extreme_events_forecasting/vae_probabilistic/outputs/output-%j.out
#SBATCH --error=/home/egauillard/extreme_events_forecasting/vae_probabilistic/outputs/error-%j.err

#!/bin/bash

# Chemins absolus
HOME_DIR="/gpfs/home3/egauillard"
PROJECT_PATH="$HOME_DIR/extreme_events_forecasting/vae_probabilistic"
ENV_PATH="$HOME_DIR/extreme_events_forecasting/extreme_events_env"

# Chargez les modules
module load 2023
module load CUDA-Samples/12.1-GCC-12.3.0-CUDA-12.1.1

# Activez l'environnement virtuel
source $ENV_PATH/bin/activate

# Exportez PYTHONPATH
export PYTHONPATH="$PROJECT_PATH/src:$PYTHONPATH"

# Allez dans le répertoire du projet
cd $PROJECT_PATH/src

# Débogage
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Python version: $(python --version)"
echo "Python executable: $(which python)"

# Exécutez le script
python -c "import sys; print(sys.path)"
# python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240923_151732/checkpoints/loss/epoch=19-val_loss=27.20.ckpt
# python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240923_151732/checkpoints/loss/epoch=20-val_loss=21.92.ckpt
# python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240923_135523/checkpoints/loss/epoch=29-val_loss=24.31.ckpt
python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240923_151801/checkpoints/loss/epoch=20-val_loss=21.92.ckpt
python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240920_194025/checkpoints/loss/epoch=20-val_loss=21.11.ckpt
python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240919_180714/checkpoints/loss/epoch=10-val_loss=9.21.ckpt
python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240920_190320/checkpoints/loss/epoch=24-val_loss=16.62.ckpt