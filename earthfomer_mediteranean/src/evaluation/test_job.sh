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
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240812_171252_to_coarse_t_fine_s/checkpoints/skill/model-skill-epoch=022-valid_skill_score=0.05.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240812_172634_tp_coarse_input_fine_target/checkpoints/loss/model-loss-epoch=083_valid_loss_epoch=0.90.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240813_172430_2005_coarse_t_fine_s/checkpoints/loss/model-loss-epoch=004_valid_loss_epoch=1.07.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240814_100758_2005_every_fine/checkpoints/loss/model-loss-epoch=002_valid_loss_epoch=1.14.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240814_102829_2005_coarse_input_fine_target_45/checkpoints/loss/model-loss-epoch=061_valid_loss_epoch=1.13.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240814_173430_2005_coarse_input-fine_target_res4/checkpoints/loss/model-loss-epoch=069_valid_loss_epoch=1.18.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_115432_coarse_input_fine_target_res1/checkpoints/loss/model-loss-epoch=010_valid_loss_epoch=1.25.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_120806_coarse_input_fine_target_42/checkpoints/loss/model-loss-epoch=051_valid_loss_epoch=1.13.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_122224_coarse_input_fine_target_0/checkpoints/loss/model-loss-epoch=018_valid_loss_epoch=1.13.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_123711_coarse_input_fine_target_10/checkpoints/loss/model-loss-epoch=033_valid_loss_epoch=1.13.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240816_122007_coarse_input_fine_target_out1/checkpoints/skill/model-skill-epoch=026-valid_skill_score=0.00.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240818_171920_every_coarse_2005_10/checkpoints/loss/model-loss-epoch=012_valid_loss_epoch=0.99.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240820_142751_every_coarse_2005_twh/checkpoints/skill/model-skill-epoch=010-valid_skill_score=0.02.ckpt
# pyhton evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240820_163610_every_coarse_t2m/checkpoints/skill/model-skill-epoch=017-valid_skill_score=0.04.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240820_172912_coarse_t_fine_s/checkpoints/skill/model-skill-epoch=001-valid_skill_score=-0.00.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240812_165347_tp_every_coarse/checkpoints/skill/model-skill-epoch=021-valid_skill_score=-0.03.ckpt
# pyhton evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240812_171128_tp_every_fine/checkpoints/skill/model-skill-epoch=049-valid_skill_score=0.05.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_114852/checkpoints/loss/model-loss-epoch=031_valid_loss_epoch=0.98.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_125058/checkpoints/skill/model-skill-epoch=025-valid_skill_score=0.02.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_132714/checkpoints/loss/model-loss-epoch=020_valid_loss_epoch=1.13.ckpt

# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240813_172430_2005_coarse_t_fine_s/checkpoints/skill/model-skill-epoch=004-valid_skill_score=-0.03.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240814_100758_2005_every_fine/checkpoints/skill/last.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240814_102829_2005_coarse_input_fine_target_45/checkpoints/loss/model-loss-epoch=061_valid_loss_epoch=1.13.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240814_173430_2005_coarse_input-fine_target_res4/checkpoints/skill/model-skill-epoch=059-valid_skill_score=0.00.ckpt 
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_115432_coarse_input_fine_target_res1/checkpoints/skill/model-skill-epoch=010-valid_skill_score=0.01.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_120806_coarse_input_fine_target_42/checkpoints/skill/model-skill-epoch=014-valid_skill_score=0.01.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_122224_coarse_input_fine_target_0/checkpoints/skill/model-skill-epoch=028-valid_skill_score=0.00.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240815_123711_coarse_input_fine_target_10/checkpoints/skill/model-skill-epoch=008-valid_skill_score=0.01.ckpt
# pyhton evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240816_122007_coarse_input_fine_target_out1/checkpoints/skill/model-skill-epoch=026-valid_skill_score=0.00.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240818_171920_every_coarse_2005_10/checkpoints/skill/model-skill-epoch=012-valid_skill_score=0.03.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240820_142751_every_coarse_2005_twh/checkpoints/skill/model-skill-epoch=010-valid_skill_score=0.02.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240820_163610_every_coarse_t2m/checkpoints/skill/model-skill-epoch=022-valid_skill_score=0.05.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240820_172912_coarse_t_fine_s/checkpoints/skill/last.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_002103_every_fine_14/checkpoints/loss/last.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_003448_every_fine_15/checkpoints/skill/last.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_114852_every_coarse_16/checkpoints/skill/model-skill-epoch=031-valid_skill_score=0.03.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_125058_every_coarse_17/checkpoints/skill/model-skill-epoch=025-valid_skill_score=0.02.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_132714_coarse_input_fine_target_17/checkpoints/skill/model-skill-epoch=031-valid_skill_score=0.01.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_145406_every_fine_19/checkpoints/skill/last.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_164613_every_fine_20/checkpoints/skill/last.ckpt

# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240911_172909_fine_target_coarse_truth_lead_time_14/checkpoints/loss/model-loss-epoch=013_valid_loss_epoch=1.14.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240911_141950_every_coarse_lead_tim14/checkpoints/skill/model-skill-epoch=001-valid_skill_score=0.02.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240912_174031_every_coarse-lead_time_20/checkpoints/skill/model-skill-epoch=003-valid_skill_score=0.02.ckpt

# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240913_171523_every_coarse_lead_time_20_out1/checkpoints/skill/model-skill-epoch=020-valid_skill_score=0.01.ckpt
# python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240913_172122_every_coarse_lead_time_20_14/checkpoints/skill/model-skill-epoch=028-valid_skill_score=0.01.ckpt

python evaluation/test.py --checkpoint_path /home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_003448_every_fine_15/checkpoints/loss/last.ckpt