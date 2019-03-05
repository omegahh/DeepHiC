import os, sys
import gzip
import time
import argparse
import multiprocessing
import numpy as np
from functools import partial
from all_parser import *

def locus_cvter(i, resolution):
    return i * resolution + resolution // 2

def fithic_input(countsfile, fragsfile, mat, chrn, resolution, compact, bounds):
    locus = partial(locus_cvter, resolution=resolution)
    nrows, ncols = mat.shape
    lb, ub = bounds

    counts_f = gzip.open(countsfile, 'wt')
    frags_f  = gzip.open(fragsfile, 'wt')
    print(f'Writing to: {countsfile} and {os.path.basename(fragsfile)} with bounds ({lb}, {ub}]')

    maps = np.zeros(nrows, dtype=np.int)
    maps[compact] = 1
    frag_lines = '\n'.join([f'{chrn}\t0\t{locus(i)}\t{maps[i]}\t0' for i in range(nrows)])
    frags_f.write(frag_lines)
    frags_f.close()

    lower = lambda x: min(ncols, x+lb+1)
    upper = lambda x: min(ncols, x+ub+1)
    contact_lines = '\n'.join([f'{chrn}\t{locus(i)}\t{chrn}\t{locus(j)}\t{mat[i,j]}' for i in range(nrows) for j in range(lower(i), upper(i))])
    counts_f.write(contact_lines)
    counts_f.close()

def pfithic_scheme(file, out_dir, resolution, bounds, multiple=255):
    n = chr_num_str(os.path.basename(file))
    data = np.load(file)
    compact = data['compact']
    keys = list(data.keys())
    keys.remove('compact')
    for key in keys:
        mat = data[key]
        if key == 'deephic':
            mat = mat * multiple
        mat = mat.astype(int)
        countsfile = os.path.join(out_dir, f'chr{n}_{key}_count.gz')
        fragsfile  = os.path.join(out_dir, f'chr{n}_{key}_frags.gz')
        fithic_input(countsfile, fragsfile, mat, n, resolution, compact, bounds)

if __name__ == '__main__':
    args = pfithic_input_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    low_res = args.low_res
    high_res = args.high_res # default is 10kb
    bounds = (args.lowerbound, args.upperbound) # default is (1, 110]

    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    in_dir  = os.path.join('data/predict', cell_line)
    out_dir = os.path.join(f'data/results/{cell_line}/{low_res}/pfithic_input')
    mkdir(out_dir)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.find(low_res) >= 0]

    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with processes = {pool_num} for generating pfithic inputs')
    for file in files:
        pool.apply_async(pfithic_scheme, (file, out_dir, res_map[high_res], bounds,))
    pool.close()
    pool.join()
    print(f'All processing done. Running cost is {(time.time()-start)/60:.1f} min')
