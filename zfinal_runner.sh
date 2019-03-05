# Reading raw data
python data_aread.py -c GM12878
# Downsampling data
python data_downsample.py -hr 10kb -lr 40kb -r 16 -c GM12878

# Generating trainable/predictable data
python data_generate.py -hr 10kb -lr 40kb -s all -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878

# Predicting data
python data_predict.py -lr 40kb -ckpt save/generator_nonpool_deephic.pytorch -c GM12878

# Running fithic
python input_pfithic.py -lr 40kb -L 2 -U 120 -c GM12878
python run_pfithic.py -lr 40kb -L 2 -U 120 -p2 -b100 -log -c GM12878

# Analysis
python visual_analysis.py -lr 40kb --ssim -c GM12878 
python visual_analysis.py -lr 40kb --corr --shift 150 -c GM12878