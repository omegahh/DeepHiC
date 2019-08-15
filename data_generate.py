import os, sys
import time
import argparse
import multiprocessing
import torch
import numpy as np
from utils.io import compactM, divide, pooling
from all_parser import *

def deephic_divider(n, high_file, down_file, scale=1, pool_type='max', chunk=40, stride=40, bound=201, lr_cutoff=100, hr_cutoff=255):
    hic_data = np.load(high_file)
    down_data = np.load(down_file)
    compact_idx = hic_data['compact']
    full_size = hic_data['hic'].shape[0]
    # Compacting
    hic = compactM(hic_data['hic'], compact_idx)
    down_hic = compactM(down_data['hic'], compact_idx)
    # Clamping
    hic = np.minimum(hr_cutoff, hic)
    down_hic = np.minimum(lr_cutoff, down_hic)
    # Rescaling
    hic = hic / np.max(hic)
    down_hic = down_hic / lr_cutoff
    # Deviding and Pooling (pooling is not performed actually)
    div_dhic, div_inds = divide(down_hic, n, chunk, stride, bound)
    div_dhic = pooling(div_dhic, scale, pool_type=pool_type, verbose=False).numpy()
    div_hhic, _ = divide(hic, n, chunk, stride, bound, verbose=True)
    return n, div_dhic, div_hhic, div_inds, compact_idx, full_size

if __name__ == '__main__':
    args = data_divider_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    # Note: cutoff for low-sequencing data differ according to the count values' distribution
    # we recommand a value smaller than the 99.5 percentile
    # For Rao's Hi-C, we used 100 for 1/16 downsampled data, 80 for 1/25 downsampled data, 
    # 50 for 1/50 downsampled data and 25 for 1/100 downsampled data.
    lr_cutoff = args.lr_cutoff
    dataset = args.dataset

    chunk = args.chunk
    stride = args.stride
    bound = args.bound
    scale = args.scale
    pool_type = args.pool_type

    chr_list = set_dict[dataset]
    postfix = cell_line.lower() if dataset == 'all' else dataset
    pool_str = 'nonpool' if scale == 1 else f'{pool_type}pool{scale}'
    print(f'Going to read {high_res} and {low_res} data, then deviding matrices with {pool_str}')

    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    data_dir = os.path.join(root_dir, 'mat', cell_line)
    out_dir = os.path.join(root_dir, 'data')
    mkdir(out_dir)

    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with processes = {pool_num} for generating DeepHiC data')
    results = []
    for n in chr_list:
        high_file = os.path.join(data_dir, f'chr{n}_{high_res}.npz')
        down_file = os.path.join(data_dir, f'chr{n}_{low_res}.npz')
        kwargs = {'scale':scale, 'pool_type':pool_type, 'chunk':chunk, 'stride':stride, 'bound':bound, 'lr_cutoff': lr_cutoff}
        res = pool.apply_async(deephic_divider, (n, high_file, down_file,), kwargs)
        results.append(res)
    pool.close()
    pool.join()
    print(f'All DeepHiC data generated. Running cost is {(time.time()-start)/60:.1f} min.')

    # return: n, div_dhic, div_hhic, div_inds, compact_idx, full_size
    data = np.concatenate([r.get()[1] for r in results])
    target = np.concatenate([r.get()[2] for r in results])
    inds = np.concatenate([r.get()[3] for r in results])
    compacts = {r.get()[0]: r.get()[4] for r in results}
    sizes = {r.get()[0]: r.get()[5] for r in results}

    filename = f'deephic_{high_res}{low_res}_c{chunk}_s{stride}_b{bound}_{pool_str}_{postfix}.npz'
    deephic_file = os.path.join(out_dir, filename)
    np.savez_compressed(deephic_file, data=data, target=target, inds=inds, compacts=compacts, sizes=sizes)
    print('Saving file:', deephic_file)
