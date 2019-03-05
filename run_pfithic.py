import os, sys

from pfithic.runner import main as fithic
from all_parser import *

if __name__ == '__main__':
    args = pfithic_runner_parser().parse_args(sys.argv[1:])
    
    cell_line = args.cell_line
    low_res = args.low_res
    high_res = args.high_res
    lowerbound = args.lowerbound
    upperbound = args.upperbound
    passNo = args.passNo
    bins = args.bins
    logger = args.logger
    
    input_dir = os.path.join(f'data/results/{cell_line}/{low_res}/pfithic_input')
    output_dir = os.path.join(f'data/results/{cell_line}/{low_res}/pfithic_output')
    resolution = res_map[high_res]
    lower_dist = lowerbound * resolution
    upper_dist = upperbound * resolution
    
    arguments = ['-i', input_dir, '-f', input_dir, '-o', output_dir, '-r', str(resolution), '-L', str(lower_dist), '-U', str(upper_dist), f'-p{passNo}', f'-b{bins}']
    if logger: 
        arguments.append('-log')
    print(f"Equally running: python -m pfithic.runner {' '.join(arguments)}")
    fithic(arguments)