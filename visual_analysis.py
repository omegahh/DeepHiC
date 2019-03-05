import os, sys
import time
import math
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from utils.io import compactM
from utils.viz import hic_heatmap
from utils.corr import diagcorr

from all_parser import *

################################ PSNR & SSIM #################################
from utils.ssim import ssim
from math import log10

def ssim_psnr(file, stride=100):
    chrn = chr_num_str(os.path.basename(file))
    data = np.load(file)
    print('Reading', file)
    hic = compactM(data['hic'], data['compact'])
    downhic = compactM(data['downhic'], data['compact'])
    plushic = compactM(data['hicplus'], data['compact'])
    deephic = compactM(data['deephic'], data['compact'])
    hic = hic / np.max(hic)
    downhic = downhic / np.max(downhic)
    plushic = plushic / np.max(plushic)
    all_length = hic.shape[0]
    parts_num = all_length // stride
    down_ssims = np.zeros(parts_num-1)
    plus_ssims = np.zeros(parts_num-1)
    deep_ssims = np.zeros(parts_num-1)
    down_psnrs = np.zeros(parts_num-1)
    plus_psnrs = np.zeros(parts_num-1)
    deep_psnrs = np.zeros(parts_num-1)
    for i in range(parts_num-1):
        s = i * stride
        e = (i+1) * stride
        down_ssims[i] = ssim(hic[s:e,s:e], downhic[s:e,s:e]).item()
        plus_ssims[i] = ssim(hic[s:e,s:e], plushic[s:e,s:e]).item()
        deep_ssims[i] = ssim(hic[s:e,s:e], deephic[s:e,s:e]).item()
        down_psnrs[i] = 10 * log10(1/((hic[s:e,s:e] - downhic[s:e,s:e]) ** 2).mean())
        plus_psnrs[i] = 10 * log10(1/((hic[s:e,s:e] - plushic[s:e,s:e]) ** 2).mean())
        deep_psnrs[i] = 10 * log10(1/((hic[s:e,s:e] - deephic[s:e,s:e]) ** 2).mean())
    ssim_down = down_ssims.mean()
    ssim_plus = plus_ssims.mean()
    ssim_deep = deep_ssims.mean()
    psnr_down = down_psnrs.mean()
    psnr_plus = plus_psnrs.mean()
    psnr_deep = deep_psnrs.mean()
    return chrn, ssim_down, ssim_plus, ssim_deep, psnr_down, psnr_plus, psnr_deep
################################ PSNR & SSIM #################################

################################ Plot Correlation ############################
def plot_diagcorr(file, out_dir, cell_line, k=5, shift=101, chunk=(0, 2000)):
    """k means minimal genomic distance"""
    s, e = chunk
    chrn = chr_num_str(os.path.basename(file))
    data = np.load(file)
    print('Reading', file)
    hic = compactM(data['hic'], data['compact'])
    downhic = compactM(data['downhic'], data['compact'])
    plushic = compactM(data['hicplus'], data['compact'])
    deephic = compactM(data['deephic'], data['compact'])
    # pearson correlation
    pr_down, _ = diagcorr(hic[s:e, s:e], downhic[s:e, s:e], max_shift=shift, percentile=99)
    pr_plus, _ = diagcorr(hic[s:e, s:e], plushic[s:e, s:e], max_shift=shift, percentile=99)
    pr_deep, _ = diagcorr(hic[s:e, s:e], deephic[s:e, s:e], max_shift=shift, percentile=99)
    # spearman correlation
    sr_down, _ = diagcorr(hic[s:e, s:e], downhic[s:e, s:e], max_shift=shift, percentile=99, rtype='spearman')
    sr_plus, _ = diagcorr(hic[s:e, s:e], plushic[s:e, s:e], max_shift=shift, percentile=99, rtype='spearman')
    sr_deep, _ = diagcorr(hic[s:e, s:e], deephic[s:e, s:e], max_shift=shift, percentile=99, rtype='spearman')
    # start plotting
    fig = plt.figure(figsize = [12, 4])
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(range(k, shift), pr_down[k:], label='vs. Downsampled', color='C5', linestyle='--')
    ax.plot(range(k, shift), pr_plus[k:], label='vs. HiCPlus', color='C0')
    ax.plot(range(k, shift), pr_deep[k:], label='vs. DeepHiC', color='C1')
    ax.set(title=f'Pearson correlation in {cell_line} [chr{chrn}]', xlabel='Genomic distance (10kb)', ylabel='Pearson correlation')
    ax.legend(loc='lower left')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(range(k, shift), sr_down[k:], label='vs. Downsampled', color='C5', linestyle='--')
    ax.plot(range(k, shift), sr_plus[k:], label='vs. HiCPlus', color='C0')
    ax.plot(range(k, shift), sr_deep[k:], label='vs. DeepHiC', color='C1')
    ax.set(title=f'Spearman correlation in {cell_line} [chr{chrn}]', xlabel='Genomic distance (10kb)', ylabel='Spearman correlation')
    ax.legend(loc='lower left')
    print(f"Plot to {os.path.join(out_dir, f'chr{chrn}_correlation.')}svg and .eps")
    fig.savefig(os.path.join(out_dir, f'chr{chrn}_correlation.svg'), format='svg')
    fig.savefig(os.path.join(out_dir, f'chr{chrn}_correlation.eps'), format='eps')
    # store results
    data_dir = os.path.join(out_dir, 'data')
    mkdir(data_dir)
    all_result = np.column_stack((range(k, shift), pr_down[k:], pr_plus[k:], pr_deep[k:], sr_down[k:], sr_plus[k:], sr_deep[k:]))
    np.savetxt(os.path.join(data_dir, f'chr{chrn}_correlation.csv'), all_result, fmt='%.6f', delimiter='\t', header='range\tpr_down\tpr_plus\tpr_deep\tsr_down\tsr_plus\tsr_deep')
    return pr_down[k:], pr_plus[k:], pr_deep[k:], sr_down[k:], sr_plus[k:], sr_deep[k:]
################################ Plot Correlation ############################

################################ Save Heatmap ################################
def save_heatmap(out_dir, hic, down, plus, deep, n, tag='raw', chunk=100, stride=80, resolution=10):
    draw, y_labels = [], []
    titles = ['Experimental', 'Downsampled', 'HiCPlus', 'DeepHiC']
    total_range = (hic.shape[0] - chunk)//stride
    count = 1
    for i in range(total_range):
        start, end = stride * i, stride * i + chunk
        start_pos = f'{start*resolution/1000}Mb' if (start*resolution) >= 1000 else f'{start*resolution}kb'
        end_pos = f'{end*resolution/1000}Mb' if (end*resolution) >= 1000 else f'{end*resolution}kb'
        y_labels.append(f'chr{n}: {start_pos} - {end_pos}')
        draw.extend([hic[start:end, start:end], down[start:end, start:end], plus[start:end, start:end], deep[start:end, start:end]])
        if ((i+1) % 15) == 0 or i == total_range-1:
            file = os.path.join(out_dir, f'{tag}_chr{n}_part{count:02d}.svg')
            hic_heatmap(draw, dediag=0, ncols=4, titles=titles, y_labels=y_labels, file=file)
            draw, y_labels = [], []
            count += 1

def plot_heatmap(file, base_dir, tag='raw'):
    chrn = chr_num_str(os.path.basename(file))
    out_dir = os.path.join(base_dir, f'heatmap/chr{chrn}')
    mkdir(out_dir)
    data = np.load(file)
    print('Reading', file)
    hic = data['hic']
    downhic = data['downhic']
    plushic = data['hicplus']
    deephic = data['deephic']
    save_heatmap(out_dir, hic, downhic, plushic, deephic, chrn, tag=tag)
################################ Save Heatmap ################################

################################ Plot Coverage ###############################
def read_signicants(fithic_dir, n, pass_num):
    print(f'Reading {fithic_dir}/chr{n}/***.pass{pass_num}_sig.gz')
    sigfile_hic = os.path.join(fithic_dir, f'chr{n}/hic.pass{pass_num}_spline.significant.gz')
    sig_hic = pd.read_csv(sigfile_hic, sep='\t', compression='gzip')

    sigfile_down = os.path.join(fithic_dir, f'chr{n}/downhic.pass{pass_num}_spline.significant.gz')
    sig_down = pd.read_csv(sigfile_down, sep='\t', compression='gzip')

    sigfile_plus = os.path.join(fithic_dir, f'chr{n}/hicplus.pass{pass_num}_spline.significant.gz')
    sig_plus = pd.read_csv(sigfile_plus, sep='\t', compression='gzip')

    sigfile_deep = os.path.join(fithic_dir, f'chr{n}/deephic.pass{pass_num}_spline.significant.gz')
    sig_deep = pd.read_csv(sigfile_deep, sep='\t', compression='gzip')
    return sig_hic, sig_down, sig_plus, sig_deep

def coverage(orig, down, plus, deep, stride=100):
    orig_qval_rank = np.argsort(orig)
    down_qval_rank = np.argsort(down)
    plus_qval_rank = np.argsort(plus)
    deep_qval_rank = np.argsort(deep)
    down_cov = []
    plus_cov = []
    deep_cov = []
    length = len(orig)
    topTicks = np.array(list(range(100, 20000, stride)))
    for i in topTicks:
        orig_set = set(orig_qval_rank[:i])
        down_set = set(down_qval_rank[:i])
        plus_set = set(plus_qval_rank[:i])
        deep_set = set(deep_qval_rank[:i])
        down_cov.append(len(orig_set.intersection(down_set))/len(orig_set.union(down_set)))
        plus_cov.append(len(orig_set.intersection(plus_set))/len(orig_set.union(plus_set)))
        deep_cov.append(len(orig_set.intersection(deep_set))/len(orig_set.union(deep_set)))
    return topTicks//stride, down_cov, plus_cov, deep_cov

def fdr_bins(q_values, minFDR, maxFDR, increment):
    qvalTicks = np.arange(minFDR, maxFDR+increment, increment)
    significantTicks = np.zeros(len(qvalTicks))
    qvalBins = -np.ones(len(q_values))
    for i, q in enumerate(q_values):
        if math.isnan(q): q = 1
        qvalBins[i] = int(math.floor(q/increment))
    return qvalBins.astype(int)

def coverage_fdr(orig, down, plus, deep, increment=1e-3):
    minFDR, maxFDR = 0, 0.05
    fdrTicks = np.arange(minFDR+increment, maxFDR+increment, increment)
    orig_qvalBins = fdr_bins(orig, minFDR, maxFDR, increment)
    down_qvalBins = fdr_bins(down, minFDR, maxFDR, increment)
    plus_qvalBins = fdr_bins(plus, minFDR, maxFDR, increment)
    deep_qvalBins = fdr_bins(deep, minFDR, maxFDR, increment)
    down_cov = []
    plus_cov = []
    deep_cov = []
    for i in range(len(fdrTicks)):
        orig_set = set(np.where(orig_qvalBins <= i)[0])
        down_set = set(np.where(down_qvalBins <= i)[0])
        plus_set = set(np.where(plus_qvalBins <= i)[0])
        deep_set = set(np.where(deep_qvalBins <= i)[0])
        down_cov.append(len(orig_set.intersection(down_set))/len(orig_set.union(down_set)))
        plus_cov.append(len(orig_set.intersection(plus_set))/len(orig_set.union(plus_set)))
        deep_cov.append(len(orig_set.intersection(deep_set))/len(orig_set.union(deep_set)))
    return fdrTicks, down_cov, plus_cov, deep_cov

def fdr(q_values, minFDR, maxFDR, increment):
    qvalTicks = np.arange(minFDR, maxFDR+increment, increment)
    significantTicks = [0 for i in range(len(qvalTicks))]
    qvalBins = [-1 for i in range(len(q_values))]
    for i, q in enumerate(q_values):
        if math.isnan(q): q = 1 # make sure NaNs are set to 1
        qvalBins[i] = int(math.floor(q/increment))
    for b in qvalBins:
        if b >= len(qvalTicks):
            continue
        significantTicks[b] += 1

    # make it cumulative
    significantTicks = np.cumsum(significantTicks)
    # shift them by 1
    significantTicks = np.array(significantTicks[1:])
    qvalTicks = np.array(qvalTicks[1:])
    return qvalTicks, significantTicks # fdrx, fdry

def plot_qvalues(ax, hic_qval, down_qval, plus_qval, deep_qval, minFDR, maxFDR, increment):
    hic_fdrx, hic_fdry = fdr(hic_qval, minFDR, maxFDR, increment)
    down_fdrx, down_fdry = fdr(down_qval, minFDR, maxFDR, increment)
    plus_fdrx, plus_fdry = fdr(plus_qval, minFDR, maxFDR, increment)
    deep_fdrx, deep_fdry = fdr(deep_qval, minFDR, maxFDR, increment)
    ax.plot(hic_fdrx, hic_fdry, label='Experimental', color='C2')
    ax.plot(down_fdrx, down_fdry, label='Downsampled', color='C5', linestyle='--')
    ax.plot(plus_fdrx, plus_fdry, label='HiCPlus', color='C0')
    ax.plot(deep_fdrx, deep_fdry, label='DeepHiC', color='C1')
    ax.set_yscale('log')
    ax.set_xlabel('FDR threshold')
    ax.set_ylabel('Number of significant contacts (log scale)')
    ax.legend()
    return hic_fdrx, hic_fdry, down_fdry, plus_fdry, deep_fdry

def plot_coverage(fithic_dir, n, pass_num, out_dir, comp_type):
    sig_hic, sig_down, sig_plus, sig_deep = read_signicants(fithic_dir, n, pass_num)

    hic_qval = sig_hic.q_vals.values
    down_qval = sig_down.q_vals.values
    plus_qval = sig_plus.q_vals.values
    deep_qval = sig_deep.q_vals.values
    if comp_type == 'topk':
        xticks, down_cov, plus_cov, deep_cov = coverage(hic_qval, down_qval, plus_qval, deep_qval)
    elif comp_type == 'fdr':
        xticks, down_cov, plus_cov, deep_cov = coverage_fdr(hic_qval, down_qval, plus_qval, deep_qval)
    plt.clf()
    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(xticks, down_cov, label='Downsampled', color='C5', linestyle='--')
    ax.plot(xticks, plus_cov, label='HiCPlus', color='C0')
    ax.plot(xticks, deep_cov, label='DeepHiC', color='C1')
    if comp_type == 'topk':
        ax.set_xlabel('Top K significant points (x$10^2$ scale)')
    elif comp_type == 'fdr':
        ax.set_xlabel('FDR')
    ax.set_ylabel('Jaccard index comp. to raw Hi-C')
    ax.legend(loc='lower right')
    ax = fig.add_subplot(1, 2, 2)
    fdrx, hic_fdry, down_fdry, plus_fdry, deep_fdry = \
                plot_qvalues(ax, hic_qval, down_qval, plus_qval, deep_qval, 0, 0.05, 1e-3)
    print(f"Ploting to {os.path.join(out_dir, f'coverage_chr{n}')}.svg and .eps")
    plt.savefig(os.path.join(out_dir, f'coverage_chr{n}_{comp_type}.svg'), format='svg')
    plt.savefig(os.path.join(out_dir, f'coverage_chr{n}_{comp_type}.eps'), format='eps')
    # saving data
    data_dir = os.path.join(out_dir, 'data')
    mkdir(data_dir)
    result_cov = np.column_stack((xticks, down_cov, plus_cov, deep_cov))
    np.savetxt(os.path.join(data_dir, f'coverage_chr{n}_{comp_type}.csv'), result_cov, fmt='%.6f', delimiter='\t', 
               header='xticks\tdown_cov\tplus_cov\tdeep_cov')
    result_fdr = np.column_stack((fdrx, hic_fdry, down_fdry, plus_fdry, deep_fdry))
    np.savetxt(os.path.join(data_dir, f'fdr_frequency_chr{n}.csv'), result_fdr, fmt='%.6f', delimiter='\t', 
               header='fdrx\thic_fdry\tdown_fdry\tplus_fdry\tdeep_fdry')
    return xticks, down_cov, deep_cov, plus_cov
################################ Plot Coverage ##############################

################################ Mark Loops #################################
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
    n = chr_num_str(os.path.basename(file))
    out_dir = os.path.join(base_dir, f'heatmap/chr{n}')
    mkdir(out_dir)
    data = np.load(file)
    print('Reading', file)
    hic = data['hic']
    downhic = data['downhic']
    plushic = data['hicplus']
    deephic = data['deephic']
    sig_hic, sig_down, sig_plus, sig_deep = read_signicants(fithic_dir, n, pass_num)
    x_posi, y_posi = getpoints(sig_hic, thres=np.percentile(sig_hic.q_vals.values, 1))
    hic = markpoints(hic, x_posi, y_posi)
    x_posi, y_posi = getpoints(sig_down, thres=np.percentile(sig_down.q_vals.values, 1))
    downhic = markpoints(downhic, x_posi, y_posi)
    x_posi, y_posi = getpoints(sig_plus, thres=np.percentile(sig_plus.q_vals.values, 1))
    plushic = markpoints(plushic, x_posi, y_posi)
    x_posi, y_posi = getpoints(sig_deep, thres=np.percentile(sig_deep.q_vals.values, 1))
    deephic = markpoints(deephic, x_posi, y_posi)
    save_heatmap(out_dir, hic, downhic, plushic, deephic, n, tag='fithicloops')
################################ Mark Loops #################################

########################## Significant Correlation ##########################
def jaccard(setA, setB):
    inter_set = setA.intersection(setB)
    union_set = setA.union(setB)
    jac_idx = len(inter_set)/len(union_set) if len(union_set) != 0 else 0
    return jac_idx

def plot_distjac(fithic_dir, out_dir, cell_line, chrn, pass_num, percentage=1):
    sig_hic, sig_down, sig_plus, sig_deep = read_signicants(fithic_dir, chrn, pass_num)
    cutted_len = (len(sig_hic) // 100) * percentage
    sig_hic.sort_values('q_vals', inplace=True)
    sig_down.sort_values('q_vals', inplace=True)
    sig_plus.sort_values('q_vals', inplace=True)
    sig_deep.sort_values('q_vals', inplace=True)
    sig_hic = sig_hic[:cutted_len]
    sig_down = sig_down[:cutted_len]
    sig_plus = sig_plus[:cutted_len]
    sig_deep = sig_deep[:cutted_len]
    dist_grp_orig = sig_hic.groupby('distance')
    dist_grp_down = sig_down.groupby('distance')
    dist_grp_plus = sig_plus.groupby('distance')
    dist_grp_deep = sig_deep.groupby('distance')
    orig_dists = set(dist_grp_orig.count().index.values)
    down_dists = set(dist_grp_down.count().index.values)
    plus_dists = set(dist_grp_plus.count().index.values)
    deep_dists = set(dist_grp_deep.count().index.values)
    all_dists = sorted(list(set.union(orig_dists, down_dists, plus_dists, deep_dists)))
    orig_cnts = np.zeros(len(all_dists))
    down_cnts = np.zeros(len(all_dists))
    plus_cnts = np.zeros(len(all_dists))
    deep_cnts = np.zeros(len(all_dists))
    orig_sets = dict().fromkeys(all_dists)
    down_sets = dict().fromkeys(all_dists)
    plus_sets = dict().fromkeys(all_dists)
    deep_sets = dict().fromkeys(all_dists)
    down_jacs = np.zeros(len(all_dists))
    plus_jacs = np.zeros(len(all_dists))
    deep_jacs = np.zeros(len(all_dists))
    for i, d in enumerate(all_dists):
        orig_cnts[i] = dist_grp_orig.count().loc[d]['chr1'] if d in orig_dists else 0
        down_cnts[i] = dist_grp_down.count().loc[d]['chr1'] if d in down_dists else 0
        plus_cnts[i] = dist_grp_plus.count().loc[d]['chr1'] if d in plus_dists else 0
        deep_cnts[i] = dist_grp_deep.count().loc[d]['chr1'] if d in deep_dists else 0
        orig_sets[d] = set(dist_grp_orig.get_group(d).index.values) if d in orig_dists else set()
        down_sets[d] = set(dist_grp_down.get_group(d).index.values) if d in down_dists else set()
        plus_sets[d] = set(dist_grp_plus.get_group(d).index.values) if d in plus_dists else set()
        deep_sets[d] = set(dist_grp_deep.get_group(d).index.values) if d in deep_dists else set()
        down_jacs[i] = jaccard(orig_sets[d], down_sets[d])
        plus_jacs[i] = jaccard(orig_sets[d], plus_sets[d])
        deep_jacs[i] = jaccard(orig_sets[d], deep_sets[d])
    # start plotting
    x_ticks = np.array(all_dists) // 10_000
    fig = plt.figure(figsize = [12, 4])
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_ticks, down_jacs, label='vs. Downsampled', color='C5', linestyle='--')
    ax.plot(x_ticks, plus_jacs, label='vs. HiCPlus', color='C0')
    ax.plot(x_ticks, deep_jacs, label='vs. DeepHiC', color='C1')
    ax.set(title=f'Jaccard index along with genomic distance in {cell_line} [chr{chrn}]', xlabel='Genomic distance (10kb)', ylabel='Jaccard index')
    ax.legend(loc='lower left')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_ticks, orig_cnts, label='Experimental', color='C2')
    ax.plot(x_ticks, down_cnts, label='Downsampled', color='C5', linestyle='--')
    ax.plot(x_ticks, plus_cnts, label='HiCPlus', color='C0')
    ax.plot(x_ticks, deep_cnts, label='DeepHiC', color='C1')
    ax.set(title=f'# of significant points in {cell_line} [chr{chrn}]', xlabel='Genomic distance (10kb)', ylabel='# of significant points')
    ax.legend(loc='lower left')
    print(f"Plot to {os.path.join(out_dir, f'chr{chrn}_cover_dist.')}svg and .eps")
    fig.savefig(os.path.join(out_dir, f'chr{chrn}_cover_dist.svg'), format='svg')
    fig.savefig(os.path.join(out_dir, f'chr{chrn}_cover_dist.eps'), format='eps')
    # store results
    data_dir = os.path.join(out_dir, 'data')
    mkdir(data_dir)
    all_result = np.column_stack((x_ticks, orig_cnts, down_cnts, plus_cnts, deep_cnts, down_jacs, plus_jacs, deep_jacs))
    np.savetxt(os.path.join(data_dir, f'chr{chrn}_distjacs.csv'), all_result, fmt='%.6f', delimiter='\t', header='distance\torig_cnts\tdown_cnts\tplus_cnts\tdeep_cnts\tdown_jacs\tplus_jacs\tdeep_jacs')
    return x_ticks, orig_cnts, down_cnts, plus_cnts, deep_cnts, down_jacs, plus_jacs, deep_jacs

def sig_matrix(sig_hic, resolution=10_000):
    sig_hic = sig_hic[sig_hic.contactType == 'intraInRange'].reset_index()
    sig_hic = sig_hic[sig_hic.q_vals < 1.0].reset_index()
    row = sig_hic.locus1.values // resolution
    col = sig_hic.locus2.values // resolution
    data = sig_hic.q_vals.values
    nsize = np.max([row, col]) + 1
    sig_mat = coo_matrix((data, (row, col)), shape=(nsize, nsize)).toarray()
    sig_mat[sig_mat == 0] = 1.0
    return -np.log(sig_mat)
    # return sig_mat

def plot_sigcorr(fithic_dir, out_dir, cell_line, chrn, pass_num, k=5, shift=101, chunk=(0, 4000)):
    """k means minimal genomic distance"""
    sig_hic, sig_down, sig_plus, sig_deep = read_signicants(fithic_dir, chrn, pass_num)

    hic = sig_matrix(sig_hic)
    downhic = sig_matrix(sig_down)
    plushic = sig_matrix(sig_plus)
    deephic = sig_matrix(sig_deep)
#     heatmap_dir = os.path.join(out_dir, f'heatmap/')
#     mkdir(heatmap_dir)
#     save_heatmap(heatmap_dir, hic, downhic, plushic, deephic, chrn, tag='significant')
    # return None
    s, e = chunk
    # pearson correlation
    pr_down, _ = diagcorr(hic[s:e, s:e], downhic[s:e, s:e], max_shift=shift, clearmaxmin=True)
    pr_plus, _ = diagcorr(hic[s:e, s:e], plushic[s:e, s:e], max_shift=shift, clearmaxmin=True)
    pr_deep, _ = diagcorr(hic[s:e, s:e], deephic[s:e, s:e], max_shift=shift, clearmaxmin=True)
    # spearman correlation
    sr_down, _ = diagcorr(hic[s:e, s:e], downhic[s:e, s:e], max_shift=shift, clearmaxmin=True, rtype='spearman')
    sr_plus, _ = diagcorr(hic[s:e, s:e], plushic[s:e, s:e], max_shift=shift, clearmaxmin=True, rtype='spearman')
    sr_deep, _ = diagcorr(hic[s:e, s:e], deephic[s:e, s:e], max_shift=shift, clearmaxmin=True, rtype='spearman')
    # start plotting
    fig = plt.figure(figsize = [12, 4])
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(range(k, shift), pr_down[k:], label='vs. Downsampled', color='C5', linestyle='--')
    ax.plot(range(k, shift), pr_plus[k:], label='vs. HiCPlus', color='C0')
    ax.plot(range(k, shift), pr_deep[k:], label='vs. DeepHiC', color='C1')
    ax.set(title=f'Pearson correlation in {cell_line} [chr{chrn}]', xlabel='Genomic distance (10kb)', ylabel='Pearson correlation')
    ax.legend(loc='lower left')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(range(k, shift), sr_down[k:], label='vs. Downsampled', color='C5', linestyle='--')
    ax.plot(range(k, shift), sr_plus[k:], label='vs. HiCPlus', color='C0')
    ax.plot(range(k, shift), sr_deep[k:], label='vs. DeepHiC', color='C1')
    ax.set(title=f'Spearman correlation in {cell_line} [chr{chrn}]', xlabel='Genomic distance (10kb)', ylabel='Spearman correlation')
    ax.legend(loc='lower left')
    print(f"Plot to {os.path.join(out_dir, f'chr{chrn}_correlation.')}svg and .eps")
    fig.savefig(os.path.join(out_dir, f'chr{chrn}_correlation.svg'), format='svg')
    fig.savefig(os.path.join(out_dir, f'chr{chrn}_correlation.eps'), format='eps')
    # store results
    data_dir = os.path.join(out_dir, 'data')
    mkdir(data_dir)
    all_result = np.column_stack((range(k, shift), pr_down[k:], pr_plus[k:], pr_deep[k:], sr_down[k:], sr_plus[k:], sr_deep[k:]))
    np.savetxt(os.path.join(data_dir, f'chr{chrn}_correlation.csv'), all_result, fmt='%.6f', delimiter='\t', header='range\tpr_down\tpr_plus\tpr_deep\tsr_down\tsr_plus\tsr_deep')
    return pr_down[k:], pr_plus[k:], pr_deep[k:], sr_down[k:], sr_plus[k:], sr_deep[k:]
########################## Significant Correlation ##########################


if __name__ == '__main__':
    args = visual_all_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    low_res = args.low_res

    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()
    in_dir = os.path.join('data/predict', cell_line)
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.find(low_res) >= 0]
    files = sorted(files, key=chr_digit)

    base_dir = os.path.join(f'data/results/{cell_line}/{low_res}/visual')
    mkdir(base_dir)

    if args.ssim_trigger:
        out_dir = os.path.join(base_dir, 'ssim')
        mkdir(out_dir)
        start = time.time()
        pool = multiprocessing.Pool(processes=pool_num)
        print(f'Calculate SSIM and PSNR, Start a multiprocess pool with processes = {pool_num}')
        result = []
        for file in files:
            res = pool.apply_async(ssim_psnr, (file,))
            result.append(res)
        pool.close()
        pool.join()
        print(f'All processes done. Running cost is {(time.time()-start)/60:.1f} min.')
        values = [r.get() for r in result]
        chrns, ssim_down, ssim_plus, ssim_deep, psnr_down, psnr_plus, psnr_deep = zip(*values)
        plt.clf()
        fig = plt.figure(figsize = [12, 4])
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(chrns, ssim_down, label='vs. Downsampled', marker='.', color='C5', linestyle='--')
        ax.plot(chrns, ssim_plus, label='vs. HiCPlus', marker='.', color='C0')
        ax.plot(chrns, ssim_deep, label='vs. DeepHiC', marker='.', color='C1')
        ax.set(title=f'SSIM score for all chromosomes in {cell_line}', xlabel='Chromosomes', ylabel='SSIM score')
        ax.set_xticks(['1', '5', '9', '13', '17', '20', 'X'])
        ax.legend()
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(chrns, psnr_down, label='vs. Downsampled', marker='.', color='C5', linestyle='--')
        ax.plot(chrns, psnr_plus, label='vs. HiCPlus', marker='.', color='C0')
        ax.plot(chrns, psnr_deep, label='vs. DeepHiC', marker='.', color='C1')
        # ax.set_ylim(bottom=21)
        ax.set(title=f'PSNR for all chromosomes in {cell_line}', xlabel='Chromosomes', ylabel='PSNR (dB)')
        ax.set_xticks(['1', '5', '9', '13', '17', '20', 'X'])
        # ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.legend()
        fig.savefig(os.path.join(out_dir, 'ssim_psnr_plot.eps'), format='eps')
        fig.savefig(os.path.join(out_dir, 'ssim_psnr_plot.svg'), format='svg')
        # saving data
        all_result = np.column_stack((ssim_down, ssim_plus, ssim_deep, psnr_down, psnr_plus, psnr_deep))
        np.savetxt(os.path.join(out_dir, 'result.csv'), all_result, fmt='%.6f', delimiter='\t', 
                   header='ssim_down\tssim_plus\tssim_deep\tpsnr_down\tpsnr_plus\tpsnr_deep')

    if args.corr_trigger:
        shift = args.shift
        out_dir = os.path.join(base_dir, 'correlation')
        mkdir(out_dir)
        k = 5 # the minimal genome distance
        start = time.time()
        pool = multiprocessing.Pool(processes=pool_num)
        print(f'Ploting correlation, Start a multiprocess pool with process_num = {pool_num}')
        result = []
        for file in files:
            res = pool.apply_async(plot_diagcorr, (file, out_dir, cell_line,), {'shift':shift, 'k':k})
            result.append(res)
        pool.close()
        pool.join()
        values = [r.get() for r in result]
        pr_down, pr_plus, pr_deep, sr_down, sr_plus, sr_deep = zip(*values)
        # pearson correlation
        pr_down = np.vstack(pr_down)
        pr_plus = np.vstack(pr_plus)
        pr_deep = np.vstack(pr_deep)
        down_prm = np.mean(pr_down, axis=0)
        plus_prm = np.mean(pr_plus, axis=0)
        deep_prm = np.mean(pr_deep, axis=0)
        # spearman correlation
        sr_down = np.vstack(sr_down)
        sr_plus = np.vstack(sr_plus)
        sr_deep = np.vstack(sr_deep)
        down_srm = np.mean(sr_down, axis=0)
        plus_srm = np.mean(sr_plus, axis=0)
        deep_srm = np.mean(sr_deep, axis=0)
        # start plotting
        fig = plt.figure(figsize = [12, 4])
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(range(k, shift), down_prm, label='vs. Downsampled', color='C5', linestyle='--')
        ax.plot(range(k, shift), plus_prm, label='vs. HiCPlus', color='C0')
        ax.plot(range(k, shift), deep_prm, label='vs. DeepHiC', color='C1')
        ax.set(title=f'Pearson correlation in {cell_line}', xlabel='Genomic distance (10kb)', ylabel='Pearson correlation')
        ax.legend(loc='lower left')
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(range(k, shift), down_srm, label='vs. Downsampled', color='C5', linestyle='--')
        ax.plot(range(k, shift), plus_srm, label='vs. HiCPlus', color='C0')
        ax.plot(range(k, shift), deep_srm, label='vs. DeepHiC', color='C1')
        ax.set(title=f'Spearman correlation in {cell_line}', xlabel='Genomic distance (10kb)', ylabel='Spearman correlation')
        ax.legend(loc='lower left')
        print('Ploting overall correlation. (svg and eps)')
        fig.savefig(os.path.join(out_dir, 'allchr_correlation.svg'), format='svg')
        fig.savefig(os.path.join(out_dir, 'allchr_correlation.eps'), format='eps')
        # saving data
        all_result = np.column_stack((range(k, shift), down_prm, plus_prm, deep_prm, down_srm, plus_srm, deep_srm))
        np.savetxt(os.path.join(out_dir, f'data/allchr_mean_result.csv'), all_result, delimiter='\t', 
                   fmt='%.6f',  header='range\tdown_prm\tplus_prm\tdeep_prm\tdown_srm\tplus_srm\tdeep_srm')
        print(f'All correlation ploting processes done. Running cost is {(time.time()-start)/60:.1f} min.')

    if args.raw_trigger:
        start = time.time()
        pool = multiprocessing.Pool(processes=pool_num)
        print(f'Ploting raw heatmap, Start a multiprocess pool with processes = {pool_num}')
        for file in files:
            pool.apply_async(plot_heatmap, (file, base_dir,), {'tag': 'raw'})
        pool.close()
        pool.join()
        print(f'All raw heatmap ploting processes done. Running cost is {(time.time()-start)/60:.1f} min.')

    if args.coverage_trigger:
        # FitHi-C coverage
        fithic_dir = os.path.join(f'data/results/{cell_line}/{low_res}/pfithic_output')
        out_dir = os.path.join(f'data/results/{cell_line}/{low_res}/visual/coverage')
        mkdir(out_dir)
        passNo = args.passNo
        comp_type = args.comp_type
        start = time.time()
        pool = multiprocessing.Pool(processes=pool_num)
        print(f'Ploting coverage, Start a multiprocess pool with processes = {pool_num}')
        result = []
        for chrn in (list(range(1, 23)) + ['X']):
            res = pool.apply_async(plot_coverage, (fithic_dir, chrn, passNo, out_dir, comp_type))
            result.append(res)
        pool.close()
        pool.join()
        print(f'All coverage figure saved. Running cost is {(time.time()-start)/60:.1f} min.')
        values = [r.get() for r in result]
        xticks, down_covs, deep_covs, plus_covs = zip(*values)
        xticks = xticks[0]
        down_cov_mean = np.mean(np.vstack(down_covs), axis=0)
        plus_cov_mean = np.mean(np.vstack(plus_covs), axis=0)
        deep_cov_mean = np.mean(np.vstack(deep_covs), axis=0)
        fig = plt.figure(figsize=[12, 4])
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(xticks, down_cov_mean, label='Downsampled', color='C5', linestyle='--')
        ax.plot(xticks, plus_cov_mean, label='HiCPlus', color='C0')
        ax.plot(xticks, deep_cov_mean, label='DeepHiC', color='C1')
        if comp_type == 'topk':
            ax.set_xlabel('Top K significant points (x$10^2$ scale)')
        elif comp_type == 'fdr':
            ax.set_xlabel('FDR')
        ax.set_ylabel('Jaccard index comp. to raw Hi-C')
        ax.legend()
        print(f"Ploting to {os.path.join(out_dir, 'coverage_allchr')}.svg and .eps")
        plt.savefig(os.path.join(out_dir, f'coverage_allchr_{comp_type}.svg'), format='svg')
        plt.savefig(os.path.join(out_dir, f'coverage_allchr_{comp_type}.eps'), format='eps')
        mean_result = np.column_stack((xticks, down_cov_mean, plus_cov_mean, deep_cov_mean))
        np.savetxt(os.path.join(out_dir, f'data/coverage_mean_{comp_type}.csv'), mean_result, fmt='%.6f', delimiter='\t', 
                   header='xticks\tdown_cov_mean\tplus_cov_mean\tdeep_cov_mean')

    if args.loops_trigger:
        fithic_dir = os.path.join(f'data/results/{cell_line}/{low_res}/pfithic_output')
        passNo = args.passNo
        start = time.time()
        pool = multiprocessing.Pool(processes=pool_num)
        print(f'Ploting Loops, Start a multiprocess pool with processes = {pool_num}')
        for file in files:
            pool.apply_async(plot_loops, (file, fithic_dir, base_dir, passNo))
        pool.close()
        pool.join()
        print(f'All loops ploting processes done. Running cost is {(time.time()-start)/60:.1f} min.')

    if args.distjac_trigger:
        fithic_dir = os.path.join(f'data/results/{cell_line}/{low_res}/pfithic_output')
        out_dir = os.path.join(f'data/results/{cell_line}/{low_res}/visual/distjac')
        mkdir(out_dir)
        passNo = args.passNo
        start = time.time()
        pool = multiprocessing.Pool(processes=pool_num)
        print(f'Ploting distance-wise jaccard index, Start a multiprocess pool with processes = {pool_num}')
        result = []
        for chrn in (list(range(1, 23)) + ['X']):
            res = pool.apply_async(plot_distjac, (fithic_dir, out_dir, cell_line, chrn, passNo,))
            result.append(res)
        pool.close()
        pool.join()
        values = [r.get() for r in result]
        x_ticks, orig_cnts, down_cnts, plus_cnts, deep_cnts, down_jacs, plus_jacs, deep_jacs = zip(*values)
        all_ticks = sorted(list(set.intersection(*(map(set, x_ticks)))))
        orig_cnt_sum = np.zeros(len(all_ticks))
        down_cnt_sum = np.zeros(len(all_ticks))
        plus_cnt_sum = np.zeros(len(all_ticks))
        deep_cnt_sum = np.zeros(len(all_ticks))
        down_jac_ave = np.zeros(len(all_ticks))
        plus_jac_ave = np.zeros(len(all_ticks))
        deep_jac_ave = np.zeros(len(all_ticks))
        for idx, xi in enumerate(all_ticks):
            for i in range(len(x_ticks)):
                j = np.where(x_ticks[i] == xi)[0][0]
                orig_cnt_sum[idx] += orig_cnts[i][j]
                down_cnt_sum[idx] += down_cnts[i][j]
                plus_cnt_sum[idx] += plus_cnts[i][j]
                deep_cnt_sum[idx] += deep_cnts[i][j]
                down_jac_ave[idx] += down_jacs[i][j]
                plus_jac_ave[idx] += plus_jacs[i][j]
                deep_jac_ave[idx] += deep_jacs[i][j]
            down_jac_ave[idx] /= len(x_ticks)
            plus_jac_ave[idx] /= len(x_ticks)
            deep_jac_ave[idx] /= len(x_ticks)
        # start plotting
        fig = plt.figure(figsize = [12, 4])
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(all_ticks, down_jac_ave, label='vs. Downsampled', color='C5', linestyle='--')
        ax.plot(all_ticks, plus_jac_ave, label='vs. HiCPlus', color='C0')
        ax.plot(all_ticks, deep_jac_ave, label='vs. DeepHiC', color='C1')
        ax.set(title=f'Jaccard index along with genomic distance in {cell_line}', xlabel='Genomic distance (10kb)', ylabel='Jaccard index')
        ax.legend(loc='lower left')
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(all_ticks, orig_cnt_sum, label='Experimental', color='C2')
        ax.plot(all_ticks, down_cnt_sum, label='Downsampled', color='C5', linestyle='--')
        ax.plot(all_ticks, plus_cnt_sum, label='HiCPlus', color='C0')
        ax.plot(all_ticks, deep_cnt_sum, label='DeepHiC', color='C1')
        ax.set(title=f'# of significant points in {cell_line}', xlabel='Genomic distance (10kb)', ylabel='# of significant points')
        ax.legend(loc='lower left')
        print(f"Plot to {os.path.join(out_dir, f'allchr_cover_dist.')}svg and .eps")
        fig.savefig(os.path.join(out_dir, f'allchr_cover_dist.svg'), format='svg')
        fig.savefig(os.path.join(out_dir, f'allchr_cover_dist.eps'), format='eps')
        # store results
        data_dir = os.path.join(out_dir, 'data')
        mkdir(data_dir)
        all_result = np.column_stack((all_ticks, orig_cnt_sum, down_cnt_sum, plus_cnt_sum, deep_cnt_sum, down_jac_ave, plus_jac_ave, deep_jac_ave))
        np.savetxt(os.path.join(data_dir, f'allchr_distjacs.csv'), all_result, fmt='%.6f', delimiter='\t', header='distance\torig_cnts\tdown_cnts\tplus_cnts\tdeep_cnts\tdown_jacs\tplus_jacs\tdeep_jacs')
        print(f'All jaccard indices figure saved. Running cost is {(time.time()-start)/60:.1f} min.')

    if args.sigcorr_trigger:
        fithic_dir = os.path.join(f'data/results/{cell_line}/{low_res}/pfithic_output')
        out_dir = os.path.join(f'data/results/{cell_line}/{low_res}/visual/sig_corr')
        mkdir(out_dir)
        k = 5 # the minimal genome distance
        shift = args.shift
        passNo = args.passNo
        start = time.time()
        pool = multiprocessing.Pool(processes=pool_num)
        print(f'Ploting significant correlation, Start a multiprocess pool with processes = {pool_num}')
        result = []
        kwargs = {'shift':shift, 'k':k}
        for chrn in (list(range(1, 23)) + ['X']):
            res = pool.apply_async(plot_sigcorr, (fithic_dir, out_dir, cell_line, chrn, passNo,), kwargs)
            result.append(res)
        pool.close()
        pool.join()
        print(f'All coverage figure saved. Running cost is {(time.time()-start)/60:.1f} min.')
        values = [r.get() for r in result]
        pr_down, pr_plus, pr_deep, sr_down, sr_plus, sr_deep = zip(*values)
        # pearson correlation
        pr_down = np.vstack(pr_down)
        pr_plus = np.vstack(pr_plus)
        pr_deep = np.vstack(pr_deep)
        down_prm = np.mean(pr_down, axis=0)
        plus_prm = np.mean(pr_plus, axis=0)
        deep_prm = np.mean(pr_deep, axis=0)
        # spearman correlation
        sr_down = np.vstack(sr_down)
        sr_plus = np.vstack(sr_plus)
        sr_deep = np.vstack(sr_deep)
        down_srm = np.mean(sr_down, axis=0)
        plus_srm = np.mean(sr_plus, axis=0)
        deep_srm = np.mean(sr_deep, axis=0)
        # start plotting
        fig = plt.figure(figsize = [12, 4])
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(range(k, shift), down_prm, label='vs. Downsampled', color='C5', linestyle='--')
        ax.plot(range(k, shift), plus_prm, label='vs. HiCPlus', color='C0')
        ax.plot(range(k, shift), deep_prm, label='vs. DeepHiC', color='C1')
        ax.set(title=f'Pearson correlation in {cell_line}', xlabel='Genomic distance (10kb)', ylabel='Pearson correlation')
        ax.legend(loc='lower left')
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(range(k, shift), down_srm, label='vs. Downsampled', color='C5', linestyle='--')
        ax.plot(range(k, shift), plus_srm, label='vs. HiCPlus', color='C0')
        ax.plot(range(k, shift), deep_srm, label='vs. DeepHiC', color='C1')
        ax.set(title=f'Spearman correlation in {cell_line}', xlabel='Genomic distance (10kb)', ylabel='Spearman correlation')
        ax.legend(loc='lower left')
        print('Ploting overall correlation. (svg and eps)')
        fig.savefig(os.path.join(out_dir, 'allchr_correlation.svg'), format='svg')
        fig.savefig(os.path.join(out_dir, 'allchr_correlation.eps'), format='eps')
        # saving data
        all_result = np.column_stack((range(k, shift), down_prm, plus_prm, deep_prm, down_srm, plus_srm, deep_srm))
        np.savetxt(os.path.join(out_dir, f'data/allchr_mean_result.csv'), all_result, delimiter='\t', 
                   fmt='%.6f',  header='range\tdown_prm\tplus_prm\tdeep_prm\tdown_srm\tplus_srm\tdeep_srm')
        print(f'All correlation ploting processes done. Running cost is {(time.time()-start)/60:.1f} min.')
