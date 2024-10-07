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
# python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240923_151801/checkpoints/loss/epoch=20-val_loss=21.92.ckpt
# python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240920_194025/checkpoints/loss/epoch=20-val_loss=21.11.ckpt
# python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240919_180714/checkpoints/loss/epoch=10-val_loss=9.21.ckpt
# python evaluation/eval.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240920_190320/checkpoints/loss/epoch=24-val_loss=16.62.ckpt

# python evaluation/main.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240919_174408_ld4000_beta10_gamma1000_every_coarse/checkpoints/loss/epoch=17-val_loss=30.40.ckpt
# python evaluation/main.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240919_180714_ld4000_beta10_gamma1000_every_coarse/checkpoints/loss/epoch=10-val_loss=9.21.ckpt
# python evaluation/main.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240920_190320_ld9000_beta10_gamma1000_every_coarse/checkpoints/loss/epoch=24-val_loss=16.62.ckpt
# python evaluation/main.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240920_190320_ld9000_beta10_gamma1000_every_coarse/checkpoints/loss/epoch=59-val_loss=20.73.ckpt

# python evaluation/main.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240920_194025_ld9000_beta7_gamma1000_every_coarse/checkpoints/loss/epoch=20-val_loss=21.11.ckpt
# python evaluation/main.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240923_110503_ld9000_beta7_gamma_1000_every_coarse/checkpoints/loss/epoch=09-val_loss=59.63.ckpt
# python evaluation/main.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240923_135523_ld9000_beta7_gamma1000_coarse_input_fine_target/checkpoints/loss/epoch=29-val_loss=24.31.ckpt
# python evaluation/main.py --checkpoint_path /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240923_151732_ld9000_beta7_gamma1000_every_coarse_out10/checkpoints/loss/epoch=20-val_loss=21.92.ckpt
# python evaluation/main.py --checkpoint_path  /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240924_133700_ld4000_beta4_gamma500_every_coarse_in_len9/checkpoints/loss/epoch=12-val_loss=5.44.ckpt
# python evaluation/main.py --checkpoint_path  /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240925_112910_ld4000_beta4_gamma500_every_coarse_only_med_input/checkpoints/epoch=29-val_prediction_loss=0.79-val_kld_loss=0.37.ckpt
# python evaluation/main.py --checkpoint_path  /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20241001_164646/checkpoints/epoch=72-val_prediction_loss=1.14-val_kld_loss=10.38.ckpt
python evaluation/main.py --checkpoint_path  /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20241002_175159/checkpoints/epoch=46-val_prediction_loss=141513.03-val_kld_loss=122.71.ckpt
pyhton evaluation/main.py --checkpoint_path  /home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20241002_172453/checkpoints/epoch=99-val_prediction_loss=141510.98-val_kld_loss=10.00.ckpt