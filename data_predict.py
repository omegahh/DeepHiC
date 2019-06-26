import os, sys
import time
import argparse
import multiprocessing
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scipy.sparse import coo_matrix

import models.deephic as deephic

from utils.io import spreadM, together

from all_parser import *


def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    dataset = TensorDataset(inputs, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes

get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))
def filename_parser(filename):
    info_str = filename.split('.')[0].split('_')[2:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
    return chunk, stride, bound, scale

def deephic_predictor(deephic_loader, ckpt_file, scale, res_num, device):
    deepmodel = deephic.Generator(scale_factor=scale, in_channel=1, resblock_num=res_num).to(device)
    if not os.path.isfile(ckpt_file):
        ckpt_file = f'save/{ckpt_file}'
    deepmodel.load_state_dict(torch.load(ckpt_file))
    print(f'Loading DeepHiC checkpoint file from "{ckpt_file}"')
    result_data = []
    result_inds = []
    deepmodel.eval()
    with torch.no_grad():
        for batch in tqdm(deephic_loader, desc='DeepHiC Predicting: '):
            lr, inds = batch
            lr = lr.to(device)
            out = deepmodel(lr)
            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    deep_hics = together(result_data, result_inds, tag='Reconstructing: ')
    return deep_hics

def save_data(deep_hic, compact, size, file):
    deephic = spreadM(deep_hic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, deephic=deephic, compact=compact)
    print('Saving file:', file)

if __name__ == '__main__':
    args = data_predict_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    low_res = args.low_res
    ckpt_file = args.checkpoint
    res_num = args.resblock
    cuda = args.cuda
    print('WARNING: Predict process needs large memory, thus ensure that your machine have ~150G memory.')
    if multiprocessing.cpu_count() > 23:
        pool_num = 23
    else:
        exit()

    in_dir = os.path.join(root_dir, 'data')
    out_dir = os.path.join(root_dir, 'predict', cell_line)
    mkdir(out_dir)

    files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]
    deephic_file = [f for f in files if f.find(cell_line.lower()+'.npz') >= 0][0]

    chunk, stride, bound, scale = filename_parser(deephic_file)

    device = torch.device(f'cuda:{cuda}' if (torch.cuda.is_available() and cuda>-1 and cuda<torch.cuda.device_count()) else 'cpu')
    print(f'Using device: {device}')
    
    start = time.time()
    print(f'Loading data[DeepHiC]: {deephic_file}')
    deephic_data = np.load(os.path.join(in_dir, deephic_file), allow_pickle=True)
    deephic_loader = dataloader(deephic_data)
    
    indices, compacts, sizes = data_info(deephic_data)
    deep_hics = deephic_predictor(deephic_loader, ckpt_file, scale, res_num, device)
    
    def save_data_n(key):
        file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
        save_data(deep_hics[key], compacts[key], sizes[key], file)

    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
    for key in compacts.keys():
        pool.apply_async(save_data_n, (key,))
    pool.close()
    pool.join()
    print(f'All data saved. Running cost is {(time.time()-start)/60:.1f} min.')