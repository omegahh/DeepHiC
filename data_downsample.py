import os, sys
import time
import argparse
import multiprocessing
import numpy as np
from utils.io import compactM, spreadM, downsampling
from all_parser import *

def downsample(in_file, low_res, ratio):
    data = np.load(in_file)
    hic = data['hic']
    compact_idx = data['compact']
    down_hic = downsampling(hic, ratio)
    chr_name = os.path.basename(in_file).split('_')[0]
    out_file = os.path.join(os.path.dirname(in_file), f'{chr_name}_{low_res}.npz')
    np.savez_compressed(out_file, hic=down_hic, compact=compact_idx, ratio=ratio)
    print('Saving file:', out_file)

if __name__ == '__main__':
    args = data_down_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    ratio = args.ratio

    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    data_dir = os.path.join(root_dir, 'mat', cell_line)
    in_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.find(high_res) >= 0]

    print(f'Generating {low_res} files from {high_res} files by {ratio}x downsampling.')
    start = time.time()
    print(f'Start a multiprocess pool with process_num = {pool_num}')
    pool = multiprocessing.Pool(pool_num)
    for file in in_files:
        pool.apply_async(downsample, (file, low_res, ratio))
    pool.close()
    pool.join()
    print(f'All downsampling processes done. Running cost is {(time.time()-start)/60:.1f} min.')

