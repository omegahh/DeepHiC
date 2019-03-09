import os, sys
import time
import argparse
import multiprocessing
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

project_dir = '..'
sys.path.append(project_dir)
from utilities import mkdir
from models.hicplus import ConvNet
from models.deephic import Generator
from utils.io import spreadM, together

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

def rebuild(data, indices):
    """Rebuild hic matrices from hicplus data, return a dict of matrices."""
    div_hic = data['data']
    hics = together(div_hic, indices, corp=6, tag='HiC[orig]')
    return hics

get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))
def filename_parser(deephic_file):
    info_str = deephic_file.split('.')[0].split('_')[-5:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
    return chunk, stride, bound, scale

def save_data(hic, plushic, deephic, compact, size, file):
    hic = spreadM(hic, compact, size)
    plushic = spreadM(plushic, compact, size)
    deephic = spreadM(deephic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=hic, hicplus=plushic, deephic=deephic, compact=compact)
    print('Saving file:', file)

print('WARNING: Predicting process needs large memory, thus please ensure that your machine have ~100G memory.')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if multiprocessing.cpu_count() > 20:
    pool_num = 20
else:
    exit()

parser = argparse.ArgumentParser(description='Arguments for Predicting data')
parser.add_argument('-c', dest='cell_type', help='cell type will be read', required=True)
parser.add_argument('-ckpt', dest='checkpoint', help='Checkpoint of DeepHiC')
parser.add_argument('-res', dest='resblock', help='The number of Resblock layers[default:5]', default=5, type=int)

args = parser.parse_args(sys.argv[1:])
cell_type = args.cell_type
ckpt_file = args.checkpoint
tag = args.model
res_num = args.resblock

start = time.time()

in_dir = '/data/MouseHiC/divided')
out_dir = os.path.join('/data/MouseHiC/predict', cell_type)
mkdir(out_dir)

files = [f for f in os.listdir(in_dir) if f.find(cell_type) >= 0]
deephic_file = [f for f in files if f.find('deephic') >= 0][0]
hicplus_file = [f for f in files if f.find('hicplus') >= 0][0]

chunk, stride, bound, scale = filename_parser(deephic_file)
print(f'Arguments parsed: chunk{chunk}, stride{stride}, bound{bound}, scale{scale}')


deepmodel = Generator(scale_factor=scale, in_channel=1, resblock_num=res_num).to(device)
if not os.path.isfile(ckpt_file):
    ckpt_file = os.path.join(project_dir, f'save/{ckpt_file}')
deepmodel.load_state_dict(torch.load(ckpt_file))
print(f'Loading DeepHiC checkpoint file from "{os.path.basename(ckpt_file)}"')

hicplus = ConvNet(40, 28).to(device)
hicplus_ckpt = os.path.join(project_dir, 'save/pytorch_model_12000.pytorch')
hicplus.load_state_dict(torch.load(hicplus_ckpt))

deephic_data = np.load(os.path.join(in_dir, deephic_file))
hicplus_data = np.load(os.path.join(in_dir, hicplus_file))
print(f'Loading data[DeepHiC]: {deephic_file}')
print(f'Loading data[HiCPlus]: {hicplus_file}')

indices, compacts, sizes = data_info(hicplus_data)
hics = rebuild(hicplus_data, indices) # rebuild matrices by hicplus data

deephic_loader = dataloader(deephic_data)
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
deep_hics = together(result_data, result_inds, tag='HiC[deep]')

hicplus_loader = dataloader(hicplus_data)
result_data_plus = []
result_inds_plus = []
hicplus.eval()
with torch.no_grad():
    for batch in tqdm(hicplus_loader, desc='HiCPlus Predicting: '):
        imgs, inds = batch
        imgs = imgs.to(device)
        out = hicplus(imgs)
        result_data_plus.append(out.to('cpu').numpy())
        result_inds_plus.append(inds.numpy())
result_data_plus = np.concatenate(result_data_plus, axis=0)
result_inds_plus = np.concatenate(result_inds_plus, axis=0)
plus_hics = together(result_data_plus, result_inds_plus, tag='HiC[plus]')

def save_data_n(key):
    file = os.path.join(out_dir, f'predict_chr{key}_40kb.npz')
    save_data(hics[key], plus_hics[key], deep_hics[key], compacts[key], sizes[key], file)

pool = multiprocessing.Pool(processes=pool_num)
print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
for key in compacts.keys():
    pool.apply_async(save_data_n, (key,))
pool.close()
pool.join()
print(f'All data saved. Running cost is {(time.time()-start)/60:.1f} min.')
