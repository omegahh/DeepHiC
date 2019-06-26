
# IMPORTANT: The first three step must have RAW data from Rao's HiC
# storing the `root_dir`/raw as defined in `all_parser.py`
# For example: "/data/RaoHiC/raw/GM12878/10kb_resolution_intrachromosomal"

# Reading raw data
# python data_aread.py -c GM12878

# Downsampling data
# python data_downsample.py -hr 10kb -lr 40kb -r 16 -c GM12878

# Generating trainable/predictable data
# python data_generate.py -hr 10kb -lr 40kb -s all -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878

# Predicting data
python data_predict.py -lr 40kb -ckpt save/generator_nonpool_deephic.pytorch -c GM12878

# Training the model
python train.py