import os, sys
import gzip
import time
import argparse
import multiprocessing
import numpy as np
from utilities import fdir, mkdir

project_dir = '/home/omega/Codes/Notebooks/model_deephic'
sys.path.append(project_dir)
from input_pfithic import *
from all_parser import *

if __name__ == '__main__':
    args = pfithic_input_parser().parse_args(sys.argv[1:])
    cell_type = args.cell_line
    low_res = args.low_res
    high_res = args.high_res # default is 10kb
    bounds = (args.lowerbound, args.upperbound) # default is (1, 110]
    
    pool_num = 20 if multiprocessing.cpu_count() > 20 else multiprocessing.cpu_count()

    in_dir  = os.path.join('/data/MouseHiC/predict', cell_type)
    out_dir = os.path.join(f'/data/MouseHiC/results/{cell_type}/pfithic_input')
    mkdir(out_dir)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.find(low_res) >= 0]

    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num = {pool_num} for generating pfithic inputs')
    for file in files:
        pool.apply_async(pfithic_scheme, (file, out_dir, res_map[high_res], bounds,))
    pool.close()
    pool.join()
    print(f'All processing done. Running cost is {(time.time()-start)/60:.1f} min')