import os, sys
import time
import argparse
import multiprocessing
import numpy as np
from utilities import fdir, mkdir
from scipy.sparse import coo_matrix

def read_data(file, out_dir):
    filename = os.path.basename(file).split('_')[0] + '_40kb.npz'
    out_file = os.path.join(out_dir, filename)
    try:
        data = np.loadtxt(file).astype(int)
        print(f'Successfully read: {file}')
    except:
        print(f'Abnormal file: {file}')
    coo_data = coo_matrix(data)
    col, row = np.unique(coo_data.col), np.unique(coo_data.row)
    idx = np.array(list(set.union(set(col), set(row))))
    np.savez_compressed(out_file, hic=data, compact=idx)
    print(f'Writing to {out_file}')
    
parser = argparse.ArgumentParser(description='Arguments for Reading data')
parser.add_argument('-i', dest='in_dir', help='data folder for input', required=True)
parser.add_argument('-o', dest='out', help='data folder for output', required=True)

args = parser.parse_args(sys.argv[1:])

in_dir = (args.in_dir).rstrip('/')
outname = args.out
pattern = 'chr'

out_dir = os.path.join(in_dir.rsplit(sep='/', maxsplit=1)[0], outname)
mkdir(out_dir)

pool_num = 20 if multiprocessing.cpu_count() > 20 else multiprocessing.cpu_count()

files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.find(pattern) >= 0]

print(f'Start reading process, there are {len(files)} files need to be read and written.')
print('Writing to directory', out_dir)

start = time.time()
pool = multiprocessing.Pool(processes=pool_num)
print(f'Start a multiprocess pool with process_num = {pool_num} for reading raw data')
for file in files:
    pool.apply_async(read_data, (file, out_dir,))
pool.close()
pool.join()
print(f'All data saved. Running cost is {(time.time()-start)/60:.1f} min.')