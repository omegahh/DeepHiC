import os
import argparse

res_map = {'5kb': 5_000, '10kb': 10_000, '25kb': 25_000, '50kb': 50_000, '100kb': 100_000, '250kb': 250_000, '500kb': 500_000, '1mb': 1_000_000}

set_dict = {'all':   list(range(1, 23)) + ['X'], 
            'train': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
            'valid': [15, 16, 17, 18, 19, 20, 21, 22]}

help_opt = (('--help', '-h'), {
    'action':'help',
    'help':"Print this help message and exit"})

def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

# chr12_10kb.npz, predict_chr13_40kb.npz
def chr_num_str(x):
    start = x.find('chr')
    part = x[start+3:]
    end = part.find('_')
    return part[:end]

def chr_digit(filename):
    chrn = chr_num_str(os.path.basename(filename))
    if chrn == 'X':
        n = 23
    else:
        n = int(chrn)
    return n

def data_read_parser():
    parser = argparse.ArgumentParser(description='Read raw data from Rao\'s Hi-C.', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]', 
                          required=True)
    
    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='High resolution specified[default:10kb]', 
                          default='10kb', choices=res_map.keys())
    misc_args.add_argument('-q', dest='map_quality', help='Mapping quality of raw data[default:MAPQGE30]',  
                          default='MAPQGE30', choices=['MAPQGE30', 'MAPQG0'])
    misc_args.add_argument('-n', dest='norm_file', help='The normalization file for raw data[default:KRnorm]', 
                          default='KRnorm', choices=['KRnorm', 'SQRTVCnorm', 'VCnorm'])
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser

def data_down_parser():
    parser = argparse.ArgumentParser(description='Downsample data from high resolution data', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]', 
                          required=True)
    req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]', 
                          default='10kb', choices=res_map.keys(), required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]', 
                          default='40kb', required=True)
    req_args.add_argument('-r', dest='ratio', help='REQUIRED: The ratio of downsampling[example:16]', 
                          default=16, type=int, required=True)
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser

def data_divider_parser():
    parser = argparse.ArgumentParser(description='Divide data for train and predict', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]', 
                          required=True)
    req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]', 
                          default='10kb', choices=res_map.keys(), required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]', 
                          default='40kb', required=True)
    req_args.add_argument('-s', dest='dataset', help='REQUIRED: Dataset for train/valid/predict(all)', 
                          default='train', choices=['all', 'train', 'valid'], )
    srgan_args = parser.add_argument_group('DeepHiC Arguments')
    srgan_args.add_argument('-chunk', dest='chunk', help='REQUIRED: chunk size for dividing[example:40]', 
                            default=40, type=int, required=True)
    srgan_args.add_argument('-stride', dest='stride', help='REQUIRED: stride for dividing[example:40]', 
                            default=40, type=int, required=True)
    srgan_args.add_argument('-bound', dest='bound', help='REQUIRED: distance boundary interested[example:201]', 
                            default=201, type=int, required=True)
    srgan_args.add_argument('-scale', dest='scale', help='REQUIRED: Downpooling scale[example:1]', 
                            type=int, required=True)
    srgan_args.add_argument('-type', dest='pool_type', help='OPTIONAL: Downpooling type[default:max]',
                            default='max', choices=['max','avg'])
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser

def data_predict_parser():
    parser = argparse.ArgumentParser(description='Predict data with HiCPlus and DeepHiC model', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]', 
                          required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]', 
                          default='40kb', required=True)
    gan_args = parser.add_argument_group('GAN model Arguments')
    gan_args.add_argument('-ckpt', dest='checkpoint', help='REQUIRED: Checkpoint file of DeepHiC model', 
                          required=True)
    gan_args.add_argument('-res', dest='resblock', help='IMPORTANT: The number of Resblock layers[default:5]', 
                          default=5, type=int)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('--cuda', dest='cuda', help='Whether or not using CUDA[default:1]', 
                          default=1, type=int)
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser

def pfithic_input_parser():
    parser = argparse.ArgumentParser(description='Generate pFitHiC inputs for Loops calling', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]', 
                          required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]', 
                          default='40kb', required=True)
    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='OPTIONAL: High resolution specified[default:10kb]', 
                           default='10kb')
    misc_args.add_argument("-L", dest="lowerbound", help="OPTIONAL: lower bound on the intra-chromosomal distance range[default:1]",
                           default=1, type=int, required=False)
    misc_args.add_argument("-U", dest="upperbound", help="OPTIONAL: upper bound on the intra-chromosomal distance range[default:110]",
                           default=110, type=int,required=False)
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser

def pfithic_runner_parser():
    parser = argparse.ArgumentParser(description='Run pFitHiC for Loops calling', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]', 
                          required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]', 
                          default='40kb', required=True)
    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='OPTIONAL: High resolution specified[default:10kb]', 
                           default='10kb')
    misc_args.add_argument("-L", dest="lowerbound", help="OPTIONAL: lower bound on the intra-chromosomal distance range[default:1]", 
                           default=1, type=int, required=False)
    misc_args.add_argument("-U", dest="upperbound", help="OPTIONAL: upper bound on the intra-chromosomal distance range[default:110]", 
                           default=110, type=int,required=False)
    misc_args.add_argument('-p', dest='passNo', help='OPTIONAL: the number of pass of FitHiC result[default:2]', 
                           default=2, type=int, required=False)
    misc_args.add_argument('-b', dest='bins', help='OPTIONAL: the number of equal-occupancy bins for FitHiC [default:100]', 
                           default=100, type=int, required=False)
    misc_args.add_argument("-log", dest="logger", help="OPTINAL: whether to writting to a log file[default:False]", 
                           action="store_true", required=False)
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser

def visual_all_parser():
    parser = argparse.ArgumentParser(description='Basic Visualization Analysis', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]', 
                          required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]', 
                          default='40kb', required=True)
    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('--ssim', dest='ssim_trigger', help='TRIGGER: drawing ssim & psnr scores of all chromosomes', 
                           action='store_true', required=False)
    misc_args.add_argument('--raw', dest='raw_trigger', help='TRIGGER: drawing raw heatmap of all chromosomes', action='store_true', required=False)
    misc_args.add_argument('--distjac', dest='distjac_trigger', help='NEEDFITHIC: drawing coverage of all chromosomes along with genomic distance', action='store_true', required=False)

    corr_args = parser.add_argument_group('Correlation Arguments')
    corr_args.add_argument('--corr', dest='corr_trigger', help='TRIGGER: drawing corelation along with genomic distance', 
                           action='store_true', required=False)
    corr_args.add_argument('--shift', dest='shift', help='OPTIONAL: maximal value of genomic distance', 
                           default=101, type=int, required=False)
    corr_args.add_argument('--sig', dest='sigcorr_trigger', help='NEEDFITHIC: drawing the heatmap of significant', action='store_true', required=False)

    cove_args = parser.add_argument_group('Coverage Arguments')
    cove_args.add_argument('--cover', dest='coverage_trigger', help='NEEDFITHIC: drawing fithic coverage of all chromosomes', 
                           action='store_true', required=False)
    cove_args.add_argument('-p', dest='passNo', help='OPTIONAL: the number of pass of FitHiC result[default:2]', 
                           default=2, type=int, required=False)
    cove_args.add_argument('-t', dest='comp_type', help='OPTIONAL: the x ticks for coverage comparison[default:topk]', 
                           default='topk', choices=['topk', 'fdr'])
    
    loop_args = parser.add_argument_group('Loops Arguments')
    loop_args.add_argument('--loops', dest='loops_trigger', help='NEEDFITHIC: drawing loops by fithic', 
                           action='store_true', required=False)
    
    parser.add_argument(*help_opt[0], **help_opt[1])
    
    return parser

def visual_chiapet_parser():
    parser = argparse.ArgumentParser(description='Draw ChIA-PET ROC analysis', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:K562]', 
                          choices=['K562'], required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: The low resolution predicted from[example:40kb]', 
                          default='40kb', required=True)
    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='OPTIONAL: High resolution specified[default:10kb]', 
                          default='10kb', required=False)
    misc_args.add_argument('-p', dest='passNo', help='OPTIONAL: the number of pass of FitHiC result[default:2]', 
                           default=2, type=int, required=False)
    misc_args.add_argument("-L", dest="lowerbound", help="OPTIONAL: lower bound on the intra-chromosomal distance range[default:5]",
                           default=5, type=int, required=False)
    misc_args.add_argument("-U", dest="upperbound", help="OPTIONAL: upper bound on the intra-chromosomal distance range[default:120]",
                           default=120, type=int,required=False)
    
    parser.add_argument(*help_opt[0], **help_opt[1])
    
    return parser
