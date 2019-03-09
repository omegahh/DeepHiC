import os, sys
import time
import math
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_dir = '..'
sys.path.append(project_dir)
from utils.io import compactM
from utils.viz import hic_heatmap
from utils.corr import diagcorr

from utilities import mkdir

get_number = lambda x: int(x[x.find('chr')+3:x.rfind('_')])


################################ Save Heatmap ################################
def save_heatmap(out_dir, hic, plushic, deephic, n, tag='raw', chunk=200, stride=80, resolution=10):
    draw, y_labels = [], []
    titles = ['expriment', 'hicplus', 'deephic']
    total_range = (hic.shape[0] - chunk)//stride
    count = 1
    for i in range(total_range):
        start, end = stride * i, stride * i + chunk
        start_pos = f'{start*resolution/1000}Mb' if (start*resolution) >= 1000 else f'{start*resolution}kb'
        end_pos = f'{end*resolution/1000}Mb' if (end*resolution) >= 1000 else f'{end*resolution}kb'
        y_labels.append(f'chr{n}: {start_pos} - {end_pos}')
        draw.extend([hic[start:end, start:end], plushic[start:end, start:end], deephic[start:end, start:end]])
        if ((i+1) % 15) == 0 or i == total_range-1:
            file = os.path.join(out_dir, f'chr{n}_{tag}_part{count}.svg')
            hic_heatmap(draw, dediag=0, ncols=3, titles=titles, y_labels=y_labels, file=file)
            draw, y_labels = [], []
            count += 1

def plot_heatmap(file, base_dir, tag='raw'):
    chr_num = get_number(os.path.basename(file))
    out_dir = os.path.join(base_dir, f'heatmap/chr{chr_num}')
    mkdir(out_dir)
    chr_num = get_number(os.path.basename(file))
    data = np.load(file)
    print('Reading', file)
    hic = data['hic']
    plushic = data['hicplus']
    deephic = data['deephic']
    save_heatmap(out_dir, hic, plushic, deephic, chr_num, tag=tag)
################################ Save Heatmap ################################

################################ Mark Loops #################################
def read_signicants(fithic_dir, n, pass_num):
    sigfile_hic = os.path.join(fithic_dir, f'chr{n}/hic.pass{pass_num}_spline.significant.gz')
    sig_hic = pd.read_csv(sigfile_hic, sep='\t', compression='gzip')
    sigfile_sr = os.path.join(fithic_dir, f'chr{n}/deephic.pass{pass_num}_spline.significant.gz')
    sig_sr = pd.read_csv(sigfile_sr, sep='\t', compression='gzip')
    sigfile_plus = os.path.join(fithic_dir, f'chr{n}/hicplus.pass{pass_num}_spline.significant.gz')
    sig_plus = pd.read_csv(sigfile_plus, sep='\t', compression='gzip')
    return sig_hic, sig_sr, sig_plus

def getpoints(sig_df, thres):
    loci2posi = lambda x: (x-5_000)//10_000
    desired = sig_df[sig_df.q_vals<thres]
    coordx = desired.locus1.apply(loci2posi).values
    coordy = desired.locus2.apply(loci2posi).values
    return coordx, coordy

def markpoints(mat, coordx, coordy):
    mat = mat.astype(np.float)
    for i, j in zip(coordx, coordy):
        mat[i, j] = np.NaN
    return mat

def plot_loops(file, fithic_dir, base_dir, pass_num):
    n = get_number(os.path.basename(file))
    out_dir = os.path.join(base_dir, f'heatmap/chr{n}')
    mkdir(out_dir)
    data = np.load(file)
    print('Reading', file)
    hic = data['hic']
    plushic = data['hicplus']
    deephic = data['deephic']
    sig_hic, sig_sr, sig_plus = read_signicants(fithic_dir, n, pass_num)
    x_posi, y_posi = getpoints(sig_hic, thres=np.percentile(sig_hic.q_vals.values, 1))
    hic = markpoints(hic, x_posi, y_posi)
    x_posi, y_posi = getpoints(sig_plus, thres=np.percentile(sig_plus.q_vals.values, 1))
    plushic = markpoints(plushic, x_posi, y_posi)
    x_posi, y_posi = getpoints(sig_sr, thres=np.percentile(sig_sr.q_vals.values, 1))
    deephic = markpoints(deephic, x_posi, y_posi)
    save_heatmap(out_dir, hic, plushic, deephic, n, tag='fithicloops')
################################ Mark Loops #################################

pool_num = 20 if multiprocessing.cpu_count() > 20 else multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description='Arguments for FitHiC analysis')
parser.add_argument('-c', dest='cell_type', help='cell data folder for input', required=True)
parser.add_argument('-fp', dest='file_pattern', help='Pattern for desired files', default='40kb', required=False)
parser.add_argument('--raw', action='store_true', dest='raw_trigger', required=False)
parser.add_argument('--loops', action='store_true', dest='loops_trigger', required=False)

args = parser.parse_args(sys.argv[1:])
pattern = args.file_pattern
cell_type = args.cell_type

in_dir = os.path.join('/data/MouseHiC/predict', cell_type)
files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.find(pattern) >= 0]

base_dir = os.path.join('/data/MouseHiC/results/{cell_type}/visual')
mkdir(base_dir)

if args.raw_trigger:
    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Ploting raw heatmap, Start a multiprocess pool with process_num = {pool_num}')
    for file in files:
        pool.apply_async(plot_heatmap, (file, base_dir,), {'tag': 'raw'})
    pool.close()
    pool.join()
    print(f'All raw heatmap ploting processes done. Running cost is {(time.time()-start)/60:.1f} min.')


if args.loops_trigger:
    fithic_dir = os.path.join('/data/MouseHiC/results/{cell_type}/pfithic_output')
    passNo = 2
    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Ploting Loops, Start a multiprocess pool with process_num = {pool_num}')
    for file in files:
        pool.apply_async(plot_loops, (file, fithic_dir, base_dir, passNo,))
    pool.close()
    pool.join()
    print(f'All correlation ploting processes done. Running cost is {(time.time()-start)/60:.1f} min.')