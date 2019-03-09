import os, sys
import time
import argparse
import multiprocessing
import torch
import numpy as np

project_dir = '..'
sys.path.append(project_dir)
from utilities import fdir, mkdir
from utils.io import compactM, divide, pooling

get_number = lambda x: int(x[x.find('chr')+3:x.rfind('_')])

def hicplus_generator(hic_file, chunk=40, stride=28, max_reads=255):
    """According to Yue's Code, arguments are fixed."""
    n = get_number(os.path.basename(hic_file))
    hic_data = np.load(hic_file)
    compact_idx = hic_data['compact']
    full_size = hic_data['hic'].shape[0]
    # Compacting
    hic = compactM(hic_data['hic'], compact_idx) * 16
    # Clamping
    hic = np.minimum(max_reads, hic)
    # Deviding
    div_hic, div_inds = divide(hic, n, verbose=True)
    return n, div_hic, div_inds, compact_idx, full_size

def deephic_generator(hic_file, scale=1, pool_type='max', chunk=40, stride=40, bound=201, max_reads=255):
    n = get_number(os.path.basename(hic_file))
    hic_data = np.load(hic_file)
    compact_idx = hic_data['compact']
    full_size = hic_data['hic'].shape[0]
    # Compacting
    hic = compactM(hic_data['hic'], compact_idx)
    # Clamping
    hic = np.minimum(max_reads, hic)
    # Normalizing
    hic = hic / np.max(hic)
    # Deviding and Pooling
    div_hic, div_inds = divide(hic, n, chunk, stride, bound, verbose=True)
    div_hic = pooling(div_hic, scale, pool_type=pool_type, verbose=False).numpy() # default is max_pool2d
    return n, div_hic, div_inds, compact_idx, full_size

parser = argparse.ArgumentParser(description='Arguments for Generating data')
parser.add_argument('-i', dest='in_dir', help='data folder for input', required=True)
parser.add_argument('-r', dest='resolution', default='40kb', help='data resolution[40kb]')

parser.add_argument('-chunk', dest='chunk', default=40, type=int, help='Attn: only for deephic[40]')
parser.add_argument('-stride', dest='stride', default=40, type=int, help='Attn: only for deephic[40]')
parser.add_argument('-bound', dest='bound', default=201, type=int, help='Attn: only for deephic[201]')
parser.add_argument('-type', dest='pool_type', default='max', choices=['max','avg'], help='Pooling type[max]')
parser.add_argument('-scale', dest='scale', type=int, help='Pooling scale', required=True)

args = parser.parse_args(sys.argv[1:])

in_dir = (args.in_dir).rstrip('/')
resolution = args.resolution

chunk = args.chunk
stride = args.stride
bound = args.bound
pool_type = args.pool_type
scale = args.scale

postfix = in_dir.rsplit(sep='/', maxsplit=1)[-1]
pool_str = 'nonpool' if scale == 1 else f'{pool_type}pool{scale}'
print(f'Going to read {resolution} data, then deviding these matrices with {pool_str}')

pool_num = 20 if multiprocessing.cpu_count() > 20 else multiprocessing.cpu_count()

files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.find(resolution) >= 0]

out_dir = os.path.join(in_dir.rsplit(sep='/', maxsplit=2)[0], 'divided')
mkdir(out_dir)

start = time.time()
pool = multiprocessing.Pool(processes=pool_num)
print(f'Start a multiprocess pool with process_num = {pool_num} for generating DeepHiC data')
results = []
for file in files:
    kwargs = {'scale': scale, 'pool_type': pool_type, 'chunk': chunk, 'stride': stride, 'bound': bound}
    res = pool.apply_async(deephic_generator, (file,), kwargs)
    results.append(res)
pool.close()
pool.join()
print(f'All DeepHiC data generated. Running cost is {(time.time()-start)/60:.1f} min.')

# return: n, div_hhic, div_inds, compact_idx, full_size
data = np.concatenate([r.get()[1] for r in results])
inds = np.concatenate([r.get()[2] for r in results])
compacts = {r.get()[0]: r.get()[3] for r in results}
sizes = {r.get()[0]: r.get()[4] for r in results}

filename = f'{postfix}_c{chunk}_s{stride}_b{bound}_{pool_str}_deephics.npz'
deephic_file = os.path.join(out_dir, filename)
np.savez_compressed(deephic_file, data=data, inds=inds, compacts=compacts, sizes=sizes)
print('Saving file:', deephic_file)

start = time.time()
pool = multiprocessing.Pool(processes=pool_num)
print(f'Start a new multiprocess pool with process_num = {pool_num} for generating HiCPlus data')
results = []
for file in files:
    res = pool.apply_async(hicplus_generator, (file,))
    results.append(res)
pool.close()
pool.join()
print(f'All HiCPlus data generated. Running cost is {(time.time()-start)/60:.1f} min.')

# return: n, div_hhic, div_inds, compact_idx, full_size
data = np.concatenate([r.get()[1] for r in results])
inds = np.concatenate([r.get()[2] for r in results])
compacts = {r.get()[0]: r.get()[3] for r in results}
sizes = {r.get()[0]: r.get()[4] for r in results}

filename = f'{postfix}_c{chunk}_s{stride}_b{bound}_hicplus.npz'
hicplus_file = os.path.join(out_dir, filename)
np.savez_compressed(hicplus_file, data=data, inds=inds, compacts=compacts, sizes=sizes)
print('Saving file:', hicplus_file)