import glob
import numpy as np
import pandas as pd
import os 
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

pd.options.mode.chained_assignment = None

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str)
parser.add_argument('--exp_name', type=str, default=None)


args = parser.parse_args()

def evaluate_quality(real_path, syn_path, info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)


    ''' Special treatment for default dataset and CoDi model '''

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx
        
    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')
        

    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    num_syn_data_np = num_syn_data.to_numpy()

    # cat_syn_data_np = np.array
    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)


    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    le_real_data = pd.DataFrame(np.concatenate((num_real_data_np, cat_real_data_oh), axis = 1)).astype(float)
    le_real_num = pd.DataFrame(num_real_data_np).astype(float)
    le_real_cat = pd.DataFrame(cat_real_data_oh).astype(float)


    le_syn_data = pd.DataFrame(np.concatenate((num_syn_data_np, cat_syn_data_oh), axis = 1)).astype(float)
    le_syn_num = pd.DataFrame(num_syn_data_np).astype(float)
    le_syn_cat = pd.DataFrame(cat_syn_data_oh).astype(float)
    
     # Check for nan
    if le_syn_data.isnull().values.any():
        nan_coordinate = np.isnan(le_syn_data.to_numpy()).nonzero()
        nan_row = np.unique(nan_coordinate[0])
        print(f"Synthetic data contains NaN at row {nan_row}: ")
        print(le_syn_data.iloc[nan_row])
        return None, None
        

    np.set_printoptions(precision=4)

    result = []

    print('=========== All Features ===========')
    print('Data shape: ', le_syn_data.shape)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }  # use the naive implementation of AlphaPrecision
    qual_score = np.mean(list(qual_res.values()))

    print('alpha precision: {:.6f}, beta recall: {:.6f}'.format(qual_res['delta_precision_alpha_naive'], qual_res['delta_coverage_beta_naive'] ))

    Alpha_Precision_all = qual_res['delta_precision_alpha_naive']
    Beta_Recall_all = qual_res['delta_coverage_beta_naive']

    return Alpha_Precision_all, Beta_Recall_all

if __name__ == '__main__':
    exp_name = args.exp_name
    assert exp_name is not None, "Experiment name must be provided"
    dataname = args.dataname
    data_dir = f'data/{dataname}' 
    info_path = f'{data_dir}/info.json'
    real_path = f'synthetic/{dataname}/real.csv'
    
    sample_dir = f"eval/report_runs/{exp_name}/{dataname}/all_samples"
    sample_paths = glob.glob(os.path.join(sample_dir, "*.csv"))
    print(f"{len(sample_paths )} samples loaded from {sample_dir}")

    alphas, betas = [], []
    for syn_path in sample_paths:
        alpha_precision, beta_recall = evaluate_quality(real_path, syn_path, info_path)
        if (alpha_precision is None) or (beta_recall is None):
            continue
        alphas.append(alpha_precision)
        betas.append(beta_recall)

    alphas = np.array(alphas)
    betas = np.array(betas)
    alpha_percent = alphas * 100
    beta_percent = betas * 100
    
    quality = pd.DataFrame({
        'alpha': alpha_percent, 
        'beta': beta_percent
    })
    avg = quality.mean(axis=0).round(2)
    std = quality.std(axis=0).round(2)
    quality_avg_std = pd.concat([avg, std], axis=1, ignore_index=True)
    quality_avg_std.columns = ["avg", "std"]
    quality_avg_std.index = ["alpha", "beta"]
    
    save_dir = os.path.dirname(sample_dir)
    quality.to_csv(os.path.join(save_dir, "quality.csv"), index=True)
    avg_std = pd.read_csv(os.path.join(save_dir, "avg_std.csv"), index_col=0)
    avg_std = pd.concat([avg_std, quality_avg_std])
    print(avg_std)
    avg_std.to_csv(os.path.join(save_dir, "avg_std.csv"), index=True)
