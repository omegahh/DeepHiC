import os, sys
import time
import argparse
import multiprocessing
import numpy as np

from utils.io import readcoo2mat
from all_parser import *

def read_data(data_file, norm_file, out_dir, resolution):
    filename = os.path.basename(data_file).split('.')[0] + '.npz'
    out_file = os.path.join(out_dir, filename)
    try:
        HiC, idx = readcoo2mat(data_file, norm_file, resolution)
    except:
        print(f'Abnormal file: {norm_file}')
    np.savez_compressed(out_file, hic=HiC, compact=idx)
    print('Saving file:', out_file)

if __name__ == '__main__':
    args = data_read_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    resolution = args.high_res
    map_quality = args.map_quality
    postfix = [args.norm_file, 'RAWobserved']

    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    raw_dir = os.path.join(root_dir, 'raw', cell_line)

    norm_files = []
    data_files = []
    for root, dirs, files in os.walk(raw_dir):
        if len(files) > 0:
            if (resolution in root) and (map_quality in root):
                for f in files:
                    if (f.endswith(postfix[0])):
                        norm_files.append(os.path.join(root, f))
                    elif (f.endswith(postfix[1])):
                        data_files.append(os.path.join(root, f))

    out_dir = os.path.join(root_dir, 'mat', cell_line)
    mkdir(out_dir)
    print(f'Start reading data, there are {len(norm_files)} files ({resolution}).')
    print(f'Output directory: {out_dir}')

    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num={pool_num} for reading raw data')
    for data_fn, norm_fn in zip(data_files, norm_files):
        pool.apply_async(read_data, (data_fn, norm_fn, out_dir, res_map[resolution]))
    pool.close()
    pool.join()
    print(f'All reading processes done. Running cost is {(time.time()-start)/60:.1f} min.')