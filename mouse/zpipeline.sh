#! /bin/bash

# 8cell  early_2cell  icm  late_2cell  mesc  mii  pn3  pn5  sperm
cell=$1

# python data_divide_mouse.py -i /data/MouseHiC/processed/${cell} -scale 1
python data_predict_mouse.py -c ${cell} -ckpt ../save/g_nopool_b201_r5_swish.pytorch -m swish

python input_pfithic_mouse.py -c ${cell} -lr 40kb -L 1 -U 150
python run_pfithic_mouse.py -c ${cell} -lr 40kb -L 1 -U 150 -b 130 -log

python visual_analysis_mouse.py -c ${cell} --raw --loops
