#!/bin/bash
#SBATCH --job-name=results
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
PROJECT_PATH="$HOME_DIR/extreme_events_forecasting/earthfomer_mediteranean"
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

python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241012_212441_s_fine_tp_in8_gap7_25/checkpoints/loss/last.ckpt
python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241011_150519_s_fine_t2m_in8_gap7/checkpoints/skill/model-skill-epoch=021-valid_skill_score=0.04.ckpt