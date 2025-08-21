# %%
import numpy as np
import pandas as pd
import torch
import os 

import json

# Metrics
from sdmetrics.visualization import get_column_plot

import plotly.io as pio
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import argparse

def main(args):
    dataname = args.dataname        
    sample_file_name = args.sample_file_name

    syn_path = f'synthetic/{dataname}/{sample_file_name}'
    real_path = f'synthetic/{dataname}/real.csv'

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    print((real_data[:2]))

    data_dir = f'data/{dataname}' 
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    big_img = plot_density(syn_data, real_data, info)

    save_dir =  f"eval/density_graphs/{dataname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, sample_file_name.replace('.csv', '.png'))
    big_img.save(save_path)
    print(f"Saved density graph to {save_path}")

def plot_density(syn_data, real_data, info, num_per_row=3):
    column_names = info['column_names']
    num_cat = len(column_names)
    num_col = num_per_row
    num_row = (num_cat-1)//num_col+1

    imgs = []
    for i, col in tqdm(enumerate(column_names), total = len(column_names)):
        # plot_type = 'bar' if i in info['cat_col_idx'] else 'distplot'
        plot_type = 'bar' if info['metadata']['columns'][str(i)]['sdtype'] == 'categorical' else 'distplot'
        if plot_type == 'distplot' and (syn_data[col][0] == syn_data[col]).all():     # to tackle a very weird bug 
        # If the continuous data all aggregate at a single value, get_column_plot() cannot plot a density curve for it.
        # So, we perturb one entry of the cont data by a small amount
            print(f"\n ALERT: the generated samples column_{i} with name '{col}' all has the same value of {syn_data[col][0]} \n")
            syn_data[col][0] += 1e-5
        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=syn_data,
            column_name=col,
            plot_type=plot_type
        )
        
        img_bytes = pio.to_image(fig, format='png')
        img = Image.open(BytesIO(img_bytes))
        imgs.append(img)
        
    width, height = imgs[0].size
    big_img = Image.new('RGB', (width * num_col, height * num_row))
    for i, img in enumerate(imgs):
        coordinate = (i%num_col * width, i//num_col * height)
        big_img.paste(img, coordinate)
    return big_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--sample_file_name', type=str, default='tabsyn.csv')

    args = parser.parse_args()
    
    main(args)