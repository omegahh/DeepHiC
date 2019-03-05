import os, sys
import bisect
import time, argparse
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.stats import zscore
from scipy.sparse import coo_matrix
from utils.io import compactM
from all_parser import *

chr_number = lambda x: x[x.find('chr')+3:x.rfind('_')]

strip_nan = lambda x: x[np.where(np.isnan(x)^True)[0]]

def zscore_hic(mat):
    mat_coo = coo_matrix(mat)
    mat_data_zs = zscore(mat_coo.data)
    mat = coo_matrix((mat_data_zs, (mat_coo.row, mat_coo.col)), shape=mat.shape).toarray()
    return mat

def bar_score(bar):
    height, width = bar.shape
    result = np.zeros(height//2)
    for j in range(height//2):
        result[j] = np.sum(bar[j,:] - bar[-j-1,:])
    score = np.sum(result)
    return score

def boundary_score(mat, height, width):
    n = mat.shape[0]
    m = max(height, width)
    scores = np.zeros(n) + np.nan
    for i in range(m, n-m):
        bar = mat[i-height:i+height+1, i-width:i+width+1]
        scores[i] = bar_score(bar)
    return scores

def boundary_intensity(mat, chunk):
    h, w = mat.shape
    intensity = np.zeros(h) + np.nan
    for i in range(chunk, h-chunk):
        a = mat[i-chunk:i, i-chunk:i]
        b = mat[i:i+chunk, i:i+chunk]
        c = mat[i-chunk:i-1, i+1:i+chunk]
        A = np.sum(np.triu(a, 1))
        B = np.sum(np.triu(b, 1))
        C = np.sum(c)
        intensity[i] = A+B-C
    return intensity

def insulation_score(mat, offset, chunk):
    h, w = mat.shape
    scores = np.zeros(h) + np.nan
    for i in range(offset+chunk-1, h-offset-chunk):
        lb_i, lb_j = i-offset, i+offset # left bottom
        lt_i, lt_j = lb_i-chunk+1, lb_j # left top point (aka. startpoint)
        rb_i, rb_j = lt_i+chunk, lt_j+chunk
        scores[i] = np.sum(mat[lt_i:rb_i, lt_j:rb_j])
    return scores

def score_delta(scores, width):
    delta = np.zeros_like(scores) + np.nan
    for i in range(width, len(scores)-width):
        delta[i] = np.mean(scores[i+1:i+width]) - np.mean(scores[i-width:i])
    return delta

def zero_points(arr, orient='descent'):
    points = []
    if orient == 'descent':
        for i in range(len(arr)-1):
            if arr[i]>0 and arr[i+1]<0:
                points.append(i)
    if orient == 'ascent':
        for i in range(len(arr)-1):
            if arr[i]<0 and arr[i+1]>0:
                points.append(i)
    return np.array(points)

def min_dist(base_ind, comp_ind):
    dists = [np.min(np.abs(base_ind - i)) for i in comp_ind]
    return np.array(dists)

def count_overlap(base_ind, comp_ind):
    result = []
    for i in range(len(comp_ind)-1):
        comp_left, comp_right = comp_ind[i], comp_ind[i+1]
        loc_left = bisect.bisect_left(base_ind, comp_left)
        loc_right = bisect.bisect_left(base_ind, comp_right)
        cross_num = loc_right - loc_left
        if loc_right==len(base_ind): loc_right = loc_right - 1
        if loc_left>0: loc_left = loc_left - 1
        max_jaccard = 0.
        for loc in range(loc_left, loc_right):
            base_left, base_right = base_ind[loc], base_ind[loc+1]
            intervals = sorted([comp_left, comp_right, base_left, base_right])
            jaccard_index = (intervals[2]-intervals[1])/(intervals[3]-intervals[0])
            max_jaccard = np.max([max_jaccard, jaccard_index])
        result.append([cross_num, max_jaccard])
        if loc_right==len(base_ind): break
    return np.array(result)

def plot_scores(ax, orig, down, plus, deep, xlabel='None', ylabel='None', title='None', 
                start=1400, end=1600, zeros=False):
    ax.plot(range(start, end), orig[start:end], label='Experimental', color='C2')
    ax.plot(range(start, end), down[start:end], label='Downsampled', color='C5')
    ax.plot(range(start, end), plus[start:end], label='HiCplus', color='C0')
    ax.plot(range(start, end), deep[start:end], label='DeepHiC', color='C1')
    if zeros:
        ax.plot(range(start, end), np.zeros(end-start), color='gray', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

def plot_hists(ax, down, plus, deep, bins=None, xlabel='None', ylabel='None', title='None'):
    color = ['C5', 'C0', 'C1']
    label = ['Downsampled', 'HiCPlus', 'DeepHiC']
    ax.hist([down, plus, deep], bins, density=True, histtype='bar', color=color, label=label)
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_title(title, fontsize='x-large')
    ax.legend()

def mark_tad(mat, zero_points, start=1400, end=1600):
    ind_s = bisect.bisect_left(zero_points, start) - 1
    ind_e = bisect.bisect_left(zero_points, end) + 1
    block = zero_points[ind_s:ind_e] - zero_points[ind_s]
    pad_before = start - zero_points[ind_s]
    pad_after = zero_points[ind_e] - end
    aug_mat = np.pad(mat, ((pad_before, pad_after), (pad_before, pad_after)), 'constant')
    for n in range(len(block)-1):
        i = block[n]
        j = block[n+1]
        aug_mat[i, i:j] = np.nan
        aug_mat[i:j, j] = np.nan
    return aug_mat[pad_before:-pad_after, pad_before:-pad_after]

def plot_tad(axs, orig, deep, plus, down, orig_zeros, deep_zeros, plus_zeros, down_zeros, start=1400, end=1600):
    orig_tad = mark_tad(orig, orig_zeros, start, end)
    deep_tad = mark_tad(deep, deep_zeros, start, end)
    plus_tad = mark_tad(plus, plus_zeros, start, end)
    down_tad = mark_tad(down, down_zeros, start, end)
    axs[0].imshow(orig_tad)
    axs[0].set_title('Experimental')
    axs[1].imshow(deep_tad)
    axs[1].set_title('DeepHiC')
    axs[2].imshow(plus_tad)
    axs[2].set_title('HiCplus')
    axs[3].imshow(down_tad)
    axs[3].set_title('Downsampled')

def tad_analysis(file, out_dir, bs_h=10, bs_w=2, bi_size=17, offset=2, is_size=10, delta_width=5, start=1400, end=1600):
    chrn = chr_num_str(os.path.basename(file))
    data = np.load(file)
    print(f'Reading {file}')
    hic = compactM(data['hic'], data['compact'])
    deephic = compactM(data['deephic'], data['compact'])
    plushic = compactM(data['hicplus'], data['compact'])
    downhic = compactM(data['downhic'], data['compact'])
    hic = zscore_hic(hic)
    deephic = zscore_hic(deephic)
    plushic = zscore_hic(plushic)
    downhic = zscore_hic(downhic)
    # calculating boundary score
    orig_bs = boundary_score(hic, bs_h, bs_w)
    down_bs = boundary_score(downhic, bs_h, bs_w)
    plus_bs = boundary_score(plushic, bs_h, bs_w)
    deep_bs = boundary_score(deephic, bs_h, bs_w)
    # calculating boundary intensity
    orig_bi = boundary_intensity(hic, bi_size)
    down_bi = boundary_intensity(downhic, bi_size)
    plus_bi = boundary_intensity(plushic, bi_size)
    deep_bi = boundary_intensity(deephic, bi_size)
    # find zero_points in boundary score
    orig_bs_zero = zero_points(orig_bs)
    down_bs_zero = zero_points(down_bs)
    plus_bs_zero = zero_points(plus_bs)
    deep_bs_zero = zero_points(deep_bs)
    # count minimal distance of segments
    down_bs_dists = min_dist(orig_bs_zero, down_bs_zero)
    plus_bs_dists = min_dist(orig_bs_zero, plus_bs_zero)
    deep_bs_dists = min_dist(orig_bs_zero, deep_bs_zero)
    # count jaccard index of segments
    down_bs_jacrd = count_overlap(orig_bs_zero, down_bs_zero)
    plus_bs_jacrd = count_overlap(orig_bs_zero, plus_bs_zero)
    deep_bs_jacrd = count_overlap(orig_bs_zero, deep_bs_zero)
    # calculation insulation score
    orig_is = insulation_score(hic, offset, is_size)
    down_is = insulation_score(downhic, offset, is_size)
    plus_is = insulation_score(plushic, offset, is_size)
    deep_is = insulation_score(deephic, offset, is_size)
    # calculation the delta of insulation score
    orig_isd = score_delta(orig_is, delta_width)
    down_isd = score_delta(down_is, delta_width)
    plus_isd = score_delta(plus_is, delta_width)
    deep_isd = score_delta(deep_is, delta_width)
    # find zero_points in insulation score
    orig_is_zero = zero_points(orig_isd, orient='ascent')
    down_is_zero = zero_points(down_isd, orient='ascent')
    plus_is_zero = zero_points(plus_isd, orient='ascent')
    deep_is_zero = zero_points(deep_isd, orient='ascent')
    # count minimal distance of segments
    down_is_dists = min_dist(orig_is_zero, down_is_zero)
    plus_is_dists = min_dist(orig_is_zero, plus_is_zero)
    deep_is_dists = min_dist(orig_is_zero, deep_is_zero)
    # count jaccard index of segments
    down_is_jacrd = count_overlap(orig_is_zero, down_is_zero)
    plus_is_jacrd = count_overlap(orig_is_zero, plus_is_zero)
    deep_is_jacrd = count_overlap(orig_is_zero, deep_is_zero)
    # start ploting figures
    fig = plt.figure(figsize=[15, 20], constrained_layout=True)
    gs = GridSpec(nrows=4, ncols=4, figure=fig)
    ax11 = fig.add_subplot(gs[0, :2])
    plot_scores(ax11, orig_bs, down_bs, plus_bs, deep_bs, xlabel='bins', ylabel='boundary score', title='Boundary score', zeros=True)
    ax12 = fig.add_subplot(gs[0, -2:])
    plot_scores(ax12, orig_bi, down_bi, plus_bi, deep_bi, xlabel='bins', ylabel='boundary intensity', title='Boundary intensity')
    ax2 = [fig.add_subplot(gs[1, i]) for i in range(4)]
    plot_tad(ax2, hic[start:end, start:end], deephic[start:end, start:end], plushic[start:end, start:end], downhic[start:end, start:end], orig_bs_zero, deep_bs_zero, plus_bs_zero, down_bs_zero, start, end)
    ax31 = fig.add_subplot(gs[2, :2])
    plot_hists(ax31, down_bs_dists, plus_bs_dists, deep_bs_dists, bins=25, xlabel='segment distance', ylabel='density', title='Distribution of segments distance')
    ax32 = fig.add_subplot(gs[2, 2])
    ax32.bar(['Downsampled', 'HiCplus', 'DeepHiC', 'Experimental'], [len(down_bs_zero), len(plus_bs_zero), len(deep_bs_zero), len(orig_bs_zero)], width=0.5)
    ax32.set_title('Number of zero points')
    ax33 = fig.add_subplot(gs[2, 3])
    ax33.bar(['Downsampled', 'HiCplus', 'DeepHiC'], [np.mean(down_bs_dists), np.mean(plus_bs_dists), np.mean(deep_bs_dists)], width=0.5)
    ax33.set_title('Average distance')
    ax41 = fig.add_subplot(gs[3, :2])
    plot_hists(ax41, down_bs_jacrd[:,0].astype(int), plus_bs_jacrd[:,0].astype(int), deep_bs_jacrd[:,0].astype(int), xlabel='Overlap Type', ylabel='density', title='Distribution of overlap type')
    ax42 = fig.add_subplot(gs[3, -2:])
    plot_hists(ax42, down_bs_jacrd[:,1], plus_bs_jacrd[:,1], deep_bs_jacrd[:,1], bins=10, xlabel='jaccard index', ylabel='density', title='Distribution of jaccard index overall')
    bs_fig = os.path.join(out_dir, f'chr{chrn}_bs_section_{start}to{end}.eps')
    bs_fig_svg = os.path.join(out_dir, f'chr{chrn}_bs_section_{start}to{end}.svg')
    print(f'Ploting fig and saving to {bs_fig}')
    fig.savefig(bs_fig, format='eps')
    fig.savefig(bs_fig_svg, format='svg')
    plt.clf()
    fig = plt.figure(figsize=[15, 20], constrained_layout=True)
    gs = GridSpec(nrows=4, ncols=4, figure=fig)
    ax11 = fig.add_subplot(gs[0, :2])
    plot_scores(ax11, orig_is, down_is, plus_is, deep_is, xlabel='bins', ylabel='insulation score', title='The Insulation Score')
    ax12 = fig.add_subplot(gs[0, -2:])
    plot_scores(ax12, orig_isd, down_isd, plus_isd, deep_isd, xlabel='bins', ylabel='scores\' delta', title='The $\Delta$ of Insulation Score', zeros=True)
    ax2 = [fig.add_subplot(gs[1, i]) for i in range(4)]
    plot_tad(ax2, hic[start:end, start:end], deephic[start:end, start:end], plushic[start:end, start:end], downhic[start:end, start:end], orig_is_zero, deep_is_zero, plus_is_zero, down_is_zero, start, end)
    ax31 = fig.add_subplot(gs[2, :2])
    plot_hists(ax31, down_is_dists, plus_is_dists, deep_is_dists, bins=25, xlabel='segment distance', ylabel='density', title='Distribution of segments distance')
    ax32 = fig.add_subplot(gs[2, 2])
    ax32.bar(['Downsampled', 'HiCplus', 'DeepHiC', 'Experimental'], [len(down_is_zero), len(plus_is_zero), len(deep_is_zero), len(orig_is_zero)], width=0.5)
    ax32.set_title('Number of zero points')
    ax33 = fig.add_subplot(gs[2, 3])
    ax33.bar(['Downsampled', 'HiCplus', 'DeepHiC'], [np.mean(down_is_dists), np.mean(plus_is_dists), np.mean(deep_is_dists)], width=0.5)
    ax33.set_title('Average distance')
    ax41 = fig.add_subplot(gs[3, :2])
    plot_hists(ax41, down_is_jacrd[:,0].astype(int), plus_is_jacrd[:,0].astype(int), deep_is_jacrd[:,0].astype(int), xlabel='Overlap Type', ylabel='density', title='Distribution of overlap type')
    ax42 = fig.add_subplot(gs[3, -2:])
    plot_hists(ax42, down_is_jacrd[:,1], plus_is_jacrd[:,1], deep_is_jacrd[:,1], bins=10, xlabel='jaccard index', ylabel='density', title='Distribution of jaccard index overall')
    is_fig = os.path.join(out_dir, f'chr{chrn}_is_section_{start}to{end}.eps')
    is_fig_svg = os.path.join(out_dir, f'chr{chrn}_is_section_{start}to{end}.svg')
    print(f'Ploting fig and saving to {is_fig}')
    fig.savefig(is_fig, format='eps')
    fig.savefig(is_fig_svg, format='svg')
    # saving boundary score and insulation score
    bscore = np.column_stack([orig_bs, down_bs, plus_bs, deep_bs])
    iscore = np.column_stack([orig_is, down_is, plus_is, deep_is])
    mkdir(os.path.join(out_dir, 'data'))
    bs_file = os.path.join(out_dir, f'data/chr{chrn}_boundary_score_compact.csv')
    is_file = os.path.join(out_dir, f'data/chr{chrn}_insulation_score_compact.csv')
    header = '\t'.join(['origin', 'downsample', 'hicplus', 'deephic'])
    np.savetxt(bs_file, bscore, header=header, delimiter='\t', fmt='%.10f')
    np.savetxt(is_file, iscore, header=header, delimiter='\t', fmt='%.10f')
    # prepare return values
    bs_zeros = (orig_bs_zero, down_bs_zero, plus_bs_zero, deep_bs_zero)
    is_zeros = (orig_is_zero, down_is_zero, plus_is_zero, deep_is_zero)
    bs_dists = (down_bs_dists, plus_bs_dists, deep_bs_dists)
    is_dists = (down_is_dists, plus_is_dists, deep_is_dists)
    bs_jacrd = (down_bs_jacrd, plus_bs_jacrd, deep_bs_jacrd)
    is_jacrd = (down_is_jacrd, plus_is_jacrd, deep_is_jacrd)
    return bs_zeros, bs_dists, bs_jacrd, is_zeros, is_dists, is_jacrd
    
if __name__ == '__main__':
    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(description='Arguments for Visulize TAD Analysis')
    parser.add_argument('-c', dest='cell_line', help='Cell line folder for input', required=True)
    parser.add_argument('-r', dest='resolution', help='Resolution of data [default: 10kb]', default='10_000', type=int)
    parser.add_argument('-fp', dest='file_pattern', help='Pattern for desired files', default='40kb', required=True)

    args = parser.parse_args(sys.argv[1:])

    cell_line = args.cell_line
    pattern = args.file_pattern
    resolution = args.resolution # default is 10kb

    in_dir  = os.path.join('data/predict', cell_line)
    out_dir = os.path.join('data/results/{cell_line}/{pattern}/visual/tad_analysis')
    mkdir(out_dir)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.find(pattern) >= 0]

    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num = {pool_num} for generating pfithic inputs')
    kwargs = {'bs_h':10, 'bs_w':2, 'bi_size':17, 'offset':2, 'is_size':10, 'delta_width':5, 'start':2000, 'end':2270}
    result = []
    for file in files:
        res = pool.apply_async(tad_analysis, (file, out_dir,), kwargs)
        result.append(res)
    pool.close()
    pool.join()
    print(f'All process done.')
    values = [r.get() for r in result]
    bs_zeros, bs_dists, bs_jacrd, is_zeros, is_dists, is_jacrd = zip(*values)
    
    # boundary score part
    orig_bs_zero, down_bs_zero, plus_bs_zero, deep_bs_zero = zip(*bs_zeros)
    down_bs_dists, plus_bs_dists, deep_bs_dists = zip(*bs_dists)
    down_bs_jacrd, plus_bs_jacrd, deep_bs_jacrd = zip(*bs_jacrd)
    # start ploting
    fig = plt.figure(figsize=[15, 10], constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=2, figure=fig)
    ## number of zero points
    ax1 = fig.add_subplot(gs[0, 0])
    orig_len = [len(arr) for arr in orig_bs_zero]
    down_len = [len(arr) for arr in down_bs_zero]
    plus_len = [len(arr) for arr in plus_bs_zero]
    deep_len = [len(arr) for arr in deep_bs_zero]
    ax1.plot(range(1, 24), orig_len, label='Experimental', color='C0')
    ax1.plot(range(1, 24), down_len, label='Downsampled', color='C3')
    ax1.plot(range(1, 24), plus_len, label='HiCplus', color='C2')
    ax1.plot(range(1, 24), deep_len, label='DeepHiC', color='C1')
    ax1.set_xlabel('chromosomes')
    ax1.set_ylabel('NO. of TADs boundaries')
    ax1.set_title('Detected TAD boundaries across all chromosomes')
    ax1.legend()
    ## mean segment distance
    ax2 = fig.add_subplot(gs[0, 1])
    down_mean = [np.mean(arr) for arr in down_bs_dists]
    plus_mean = [np.mean(arr) for arr in plus_bs_dists]
    deep_mean = [np.mean(arr) for arr in deep_bs_dists]
    ax2.plot(range(1, 24), down_mean, label='Downsampled', color='C5')
    ax2.plot(range(1, 24), plus_mean, label='HiCplus', color='C0')
    ax2.plot(range(1, 24), deep_mean, label='DeepHiC', color='C1')
    ax2.set_xlabel('chromosomes')
    ax2.set_ylabel('Average of segment distance')
    ax2.set_title('Average of segment distance across all chromosomes')
    ax2.legend()
    # distribution of segments distance
    ax3 = fig.add_subplot(gs[1, 0])
    down_bs_dists = np.concatenate(down_bs_dists, axis=0)
    plus_bs_dists = np.concatenate(plus_bs_dists, axis=0)
    deep_bs_dists = np.concatenate(deep_bs_dists, axis=0)
    plot_hists(ax3, down_bs_dists, plus_bs_dists, deep_bs_dists, bins=25, xlabel='segment distance', ylabel='density', title='Distribution of segments distance')
    # distribution of jaccard index
    ax4 = fig.add_subplot(gs[1, 1])
    down_bs_jacrd = np.concatenate(down_bs_jacrd, axis=0)
    plus_bs_jacrd = np.concatenate(plus_bs_jacrd, axis=0)
    deep_bs_jacrd = np.concatenate(deep_bs_jacrd, axis=0)
    plot_hists(ax4, down_bs_jacrd[:,1], plus_bs_jacrd[:,1], deep_bs_jacrd[:,1], bins=10, xlabel='jaccard index', ylabel='density', title='Distribution of jaccard index across all chromosomes')
    bs_fig = os.path.join(out_dir, f'allchr_bs_analysis.eps')
    bs_fig_svg = os.path.join(out_dir, f'allchr_bs_analysis.svg')
    print(f'Ploting fig and saving to {bs_fig}')
    fig.savefig(bs_fig, format='eps')
    fig.savefig(bs_fig_svg, format='svg')
    plt.clf()
    # insulation score part
    orig_is_zero, down_is_zero, plus_is_zero, deep_is_zero = zip(*is_zeros)
    down_is_dists, plus_is_dists, deep_is_dists = zip(*is_dists)
    down_is_jacrd, plus_is_jacrd, deep_is_jacrd = zip(*is_jacrd)
    # start ploting
    fig = plt.figure(figsize=[16, 10], constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=2, figure=fig)
    ## number of zero points
    ax1 = fig.add_subplot(gs[0, 0])
    orig_len = [len(arr) for arr in orig_is_zero]
    down_len = [len(arr) for arr in down_is_zero]
    plus_len = [len(arr) for arr in plus_is_zero]
    deep_len = [len(arr) for arr in deep_is_zero]
    ax1.plot(range(1, 24), orig_len, label='Experimental', color='C2')
    ax1.plot(range(1, 24), down_len, label='Downsampled', color='C5')
    ax1.plot(range(1, 24), plus_len, label='HiCplus', color='C0')
    ax1.plot(range(1, 24), deep_len, label='DeepHiC', color='C1')
    ax1.set_xlabel('chromosomes')
    ax1.set_ylabel('NO. of TADs boundaries')
    ax1.set_title('Detected TAD boundaries across all chromosomes')
    ax1.legend()
    ## mean segment distance
    ax2 = fig.add_subplot(gs[0, 1])
    down_mean = [np.mean(arr) for arr in down_is_dists]
    plus_mean = [np.mean(arr) for arr in plus_is_dists]
    deep_mean = [np.mean(arr) for arr in deep_is_dists]
    ax2.plot(range(1, 24), down_mean, label='Downsampled', color='C5')
    ax2.plot(range(1, 24), plus_mean, label='HiCplus', color='C0')
    ax2.plot(range(1, 24), deep_mean, label='DeepHiC', color='C1')
    ax2.set_xlabel('chromosomes')
    ax2.set_ylabel('Average of segment distance')
    ax2.set_title('Average of segment distance across all chromosomes')
    ax2.legend()
    # distribution of segments distance
    ax3 = fig.add_subplot(gs[1, 0])
    down_is_dists = np.concatenate(down_is_dists, axis=0)
    plus_is_dists = np.concatenate(plus_is_dists, axis=0)
    deep_is_dists = np.concatenate(deep_is_dists, axis=0)
    plot_hists(ax3, down_is_dists, plus_is_dists, deep_is_dists, bins=25, xlabel='segment distance', ylabel='density', title='Distribution of segments distance')
    # distribution of jaccard index
    ax4 = fig.add_subplot(gs[1, 1])
    down_is_jacrd = np.concatenate(down_is_jacrd, axis=0)
    plus_is_jacrd = np.concatenate(plus_is_jacrd, axis=0)
    deep_is_jacrd = np.concatenate(deep_is_jacrd, axis=0)
    plot_hists(ax4, down_is_jacrd[:,1], plus_is_jacrd[:,1], deep_is_jacrd[:,1], bins=10, xlabel='jaccard index', ylabel='density', title='Distribution of jaccard index across all chromosomes')
    is_fig = os.path.join(out_dir, f'allchr_is_analysis.eps')
    is_fig_svg = os.path.join(out_dir, f'allchr_is_analysis.svg')
    print(f'Ploting fig and saving to {is_fig}')
    fig.savefig(is_fig, format='eps')
    fig.savefig(is_fig_svg, format='svg')
    print(f'All pipeline done. Running cost is {(time.time()-start)/60:.1f} min')
