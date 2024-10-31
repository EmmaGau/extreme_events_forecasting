from eval import Evaluation
from ensemble_eval import EnsembleEvaluation
import argparse
import os
import sys
from omegaconf import OmegaConf

 """This file is used to run the evaluation of the model. It will run the deterministic evaluation and the ensemble evaluation."""

sys.path.append(os.path.abspath("/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src"))
# Maintenant vous pouvez importer le module
from data.dataset import DatasetEra
from data.temporal_aggregator import TemporalAggregatorFactory

def main(checkpoint_path):
    exp_dir = checkpoint_path.split('/checkpoints/')[0]
    config_path = os.path.join(exp_dir, 'cfg.yaml')

    oc = OmegaConf.load(config_path)
    dataset_cfg = OmegaConf.to_object(oc.data)
    test_config = dataset_cfg.copy()

    # use exactly the same parameters for the training but change the relevant years
    test_config['dataset']['relevant_years'] = [2016, 2024]
    lead_time = dataset_cfg['temporal_aggregator']['out_len']
    spatial_out_res = test_config['dataset']['out_spatial_resolution']
    temporal_out_res = dataset_cfg['temporal_aggregator']['resolution_output']

    data_dirs = dataset_cfg['data_dirs']

    # do not overlap the samples of the test set 
    dataset_cfg['temporal_aggregator']['gap'] = temporal_out_res * lead_time
    temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])

    test_dataset = DatasetEra(test_config, data_dirs, temp_aggregator_factory)

    # save the whole dataset to get a climatology for the percentiles to compute RPSS 

    # run inference and deterministic plots
    eval = Evaluation(checkpoint_path, config_path, test_dataset)
    eval.run_evaluation()

    path_dic = eval.get_save_paths()
    path_dic["truth_era"] = f"/home/egauillard/data/tp_1940_2023_{spatial_out_res}deg_{temporal_out_res}res_winter.nc"

    # run ensemble evaluation
    evaluator = EnsembleEvaluation(path_dic["ensemble_pred"], path_dic["truth"], path_dic["climatology"], path_dic["climatology_std"], path_dic["truth_era"], "tp", path_dic["save_folder"])
    evaluator.calculate_scores()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    main(checkpoint_path)
