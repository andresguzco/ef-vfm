import glob
import json
import os
import pickle
import random

import numpy as np
from ef_vfm.metrics import TabMetrics
from ef_vfm.modules.main_modules import UniModMLP
from ef_vfm.models.flow_model import ExpVFM
from ef_vfm.trainer import Trainer
import src
import torch

from torch.utils.data import DataLoader
import argparse
import warnings

import wandb


from utils_train import EFVFMDataset

warnings.filterwarnings('ignore')


def main(args):
    device = args.device

    ## Disable scientific numerical format
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    ## Get data info
    dataname = args.dataname
    data_dir = f'data/{dataname}'
    info_path = f'data/{dataname}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    ## Set up flags
    is_dcr = 'dcr' in dataname

    ## Set experiment name
    exp_name = args.exp_name
    assert args.exp_name is not None, "Experiment name must be provided"
    
    ## Load configs
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/ef_vfm_configs.toml'
    raw_config = src.load_config(config_path)
    
    print(f"{args.mode.capitalize()} Mode is Enabled")
    num_samples_to_generate = None
    ckpt_path = None
    if args.mode == 'train':
        print("NEW training is started")
    elif args.mode == 'test':
        num_samples_to_generate = args.num_samples_to_generate
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            ckpt_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name}"
            ckpt_path_arr = glob.glob(f"{ckpt_parent_path}/best_ema_model*")
            assert ckpt_path_arr, f"Cannot not infer ckpt_path from {ckpt_parent_path}, please make sure that you first train a model before testing!"
            ckpt_path = ckpt_path_arr[0]
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                cached_raw_config = pickle.load(f)
                print(f"Found cached config at {config_path}")
        raw_config = cached_raw_config
    
    
    ## Creat model_save and result paths
    model_save_path, result_save_path = None, None
    if args.mode == 'train':
        model_save_path = 'debug/ckpt' if args.debug else f'{curr_dir}/ckpt/{dataname}/{exp_name}'
        result_save_path = model_save_path.replace('ckpt', 'result')  #i.e., f'{curr_dir}/results/{dataname}/{exp_name}'
    elif args.mode == 'test':
        if args.report:
            result_save_path = f"eval/report_runs/{exp_name}/{dataname}"
        else:
            result_save_path = os.path.dirname(ckpt_path).replace('ckpt', 'result')    # infer the exp_name from the ckpt_name
    raw_config['model_save_path'] = model_save_path
    raw_config['result_save_path'] = result_save_path
    if model_save_path is not None:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    if result_save_path is not None:
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
    
    ## Make everything determinstic if needed
    raw_config['deterministic'] = args.deterministic
    if args.deterministic:
        print("DETERMINISTIC MODE is enabled!!!")
        ## Set global random seeds
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        ## Ensure deterministic CUDA operations
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    ## Set debug mode parameters
    if args.debug:  # fast eval for DEBUG mode
        raw_config['train']['main']['check_val_every'] = 2
        raw_config['train']['main']['batch_size'] = 4096
        raw_config['sample']['batch_size'] = 10000

    ## Load training data
    batch_size = raw_config['train']['main']['batch_size']

    train_data = EFVFMDataset(dataname, data_dir, info, isTrain=True, dequant_dist=raw_config['data']['dequant_dist'], int_dequant_factor=raw_config['data']['int_dequant_factor'])
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories
    
    val_data = EFVFMDataset(dataname, data_dir, info, isTrain=False, dequant_dist=raw_config['data']['dequant_dist'], int_dequant_factor=raw_config['data']['int_dequant_factor'])

    ## Load Metrics
    real_data_path = f'synthetic/{dataname}/real.csv'
    test_data_path = f'synthetic/{dataname}/test.csv'
    val_data_path = f'synthetic/{dataname}/val.csv'
    if not os.path.exists(val_data_path):
        print(f"{args.dataname} does not have its validation set. During MLE evaluation, a validation set will be splitted from the training set!")
        val_data_path = None
    if args.mode == 'train':
        metric_list = ["density"]
    else:
        if is_dcr:
            metric_list = ["dcr"]
        else:
            metric_list = [
                "density", 
                "mle", 
                "c2st",
            ]
    metrics = TabMetrics(real_data_path, test_data_path, val_data_path, info, device, metric_list=metric_list)
    
    ## Load the module and models
    raw_config['unimodmlp_params']['d_numerical'] = d_numerical
    raw_config['unimodmlp_params']['categories'] = (categories+1).tolist()  # add one for the mask category
    model = UniModMLP(**raw_config['unimodmlp_params'])
    model.to(device)
    
    flow_model = ExpVFM(
        num_classes=categories,
        num_numerical_features=d_numerical,
        vf_fn=model,
        device=device,
    )
    num_params = sum(p.numel() for p in flow_model.parameters())
    print("The number of parameters = ", num_params)
    flow_model.to(device)
    flow_model.train()

    ## Print the configs
    printed_configs = json.dumps(raw_config, default=lambda x: int(x) if isinstance(x, np.int64) else x, indent=4)
    print(f"The config of the current run is : \n {printed_configs}")
    
    ## Enable Wandb
    project_name = f"XVFM_{dataname}"
    raw_config['project_name'] = project_name
    logger = wandb.init(
        project=raw_config['project_name'], 
        name=exp_name,
        config=raw_config,
        mode='disabled' if args.debug or args.no_wandb else 'online',
    )

    ## Load Trainer
    sample_batch_size = raw_config['sample']['batch_size']
    trainer = Trainer(
        flow_model,
        train_loader,
        train_data,
        val_data,
        metrics,
        logger,
        **raw_config['train']['main'],
        sample_batch_size=sample_batch_size,
        num_samples_to_generate=num_samples_to_generate,
        model_save_path=raw_config['model_save_path'],
        result_save_path=raw_config['result_save_path'],
        device=device,
        ckpt_path=ckpt_path,
    )
    if args.mode == 'test':
        if args.report:
            if  is_dcr:
                trainer.report_test_dcr(args.num_runs)
            else:
                trainer.report_test(args.num_runs)
        else:
            trainer.test()
    else:
        ## Save config
        config_save_path = raw_config['model_save_path']
        with open (os.path.join(config_save_path, 'config.pkl'), 'wb') as f:
            pickle.dump(raw_config, f)
        trainer.run_loop()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabbyFlow')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'