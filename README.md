# DeepHiC in PyTorch

We provide PyTorch implementations for both predicting and training procedures.

## Summary
---

DeepHiC is based on [Generative Adversarial Network](https://arxiv.org/abs/1406.2661), take low-resolution data as conditional inputs for *Generator* Net in GAN. Here we trained DeepHiC for 200 epochs on chromosome 1-14 in GM12878 cell line from [Rao's HiC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525)

![Heatmap of](imgs/principle_of_deephic.png)
> The framework of DeepHiC. With 10kb high-resolution data as ground truth, DeepHiC predicts enhanced output from 1/16 downsampled reads (simulate 40 kb low-resolution data). The structure similarity between enhanced outputs and high-resolution data is 0.89 on average. 

## Galance
---

A galace of the efficience of DeepHiC.

![Enhancements of DeepHiC](imgs/enhancement_of_deephic.png)
> First example from training set, and the last two are extracted from test set.

## Usage
---

Take GM12878 cell line as example.

### Preprocessing

When raw data of Rao's Hi-C store in directory: `data/raw/[foldername_according_to_cell_line]`

1. Reading raw data and storing as numpy arrays.

~~~bash
python data_aread.py -c GM12878
~~~

2. Downsampling to 1/16 reads.

~~~bash
python data_downsample.py -hr 10kb -lr 40kb -r 16 -c GM12878
~~~

3. Generating tranable/predictable data

~~~bash
python data_generate.py -hr 10kb -lr 40kb -s all -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878
~~~

### Traning or Predicting

1. For training

~~~bash
python train.py
~~~

2. For Predicting

~~~bash
python data_predict.py -lr 40kb -ckpt save/generator_nonpool_deephic.pytorch -c GM12878
~~~

### Running Fit-Hi-C

> Here we used a modified Fit-Hi-C program in [here](https://github.com/omegahh/pFitHiC)

1. Preparing data for Fit-Hi-C

~~~bash
python input_pfithic.py -lr 40kb -L 2 -U 120 -c GM12878
~~~

2. Running Fit-Hi-C

~~~bash
python run_pfithic.py -lr 40kb -L 2 -U 120 -p2 -b100 -log -c GM12878
~~~

### Downstream analysis

For example:

~~~bash
# for calculating SSIM and PSNR scores
python visual_analysis.py -lr 40kb --ssim -c GM12878
# for evaluating Pearson and Spearman correlations
python visual_analysis.py -lr 40kb --corr --shift 150 -c GM12878
~~~

## Requirements
---

DeepHiC is written in Python3 with PyTorch framework. It demands python version 3.6+

Other python packages used in this repo:

- Numpy
- Scipy
- Pandas
- Matplotlib
- scikit-learn
- tqdm
- visdom
- pytorch 0.4.1+
