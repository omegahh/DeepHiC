## Preparing Data for DeepHiC webserver

Our webserver [DeepHiC](http://sysomics.com/deephic) could directly enhancing low-resolution data with a few clicks.

Users could convert their own Hi-C data to a .npz file using python for data preparation. It's very easy in two steps.

> **Note**: Data in following instruction is just for showing how to process your data. They cannot be used because the size of `mat` is too small to divide. We offere a real example data on chromosome 22.

> **Note**: Our server is based on Alibaba Cloud Elastic Compute Service with 8GB memory and two Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GHz cpus. For a intra-interaction data on single chromosome, the calculation may cost 3-5 minutes according to the chromosome size in our server. For very huge data, we recommend you use the python code in this repository.

1. Reading your data in python, and asumming that 'mat' is a two-dimensional numpy.ndarray which stores the Hi-C matrix. A one-dimensional array 'compact' is also required. 'compact' is the index of 'mat' where its bins are greater than zero.

> For example, if we have
> 
> ```
> mat = numpy.array([[3, 2, 0, 7, 0, 0, 5, 2], 
>                    [2, 2, 0, 5, 0, 0, 3, 4], 
>                    [0, 0, 0, 0, 0, 0, 0, 0], 
>                    [7, 5, 0, 7, 0, 0, 6, 4], 
>                    [0, 0, 0, 0, 0, 0, 0, 0], 
>                    [0, 0, 0, 0, 0, 0, 0, 0], 
>                    [5, 3, 0, 6, 0, 0, 3, 3], 
>                    [2, 4, 0, 4, 0, 0, 3, 5]])
> ```
> 
> , thus we have
> 
> ```
> compact = numpy.array([0, 1, 3, 6, 7])
> ```
> because the 2-th, 4-th, 5-th bins (0-indexed) are all zeros. 

2. Saving the compressed data for input.

```
numpy.savez_compressed(output_filename, hic=mat, compact=compact)
```

We using a compressed .npz data for upload because it is in small size for uploading to our server. The intra-chromosome Hi-C data on human chromosome 1 is about 20Mb in size. And prediction on it will cost about 4.5 minutes in our Alibaba Cloud Elastic Compute Service (CPU: Xeon(R) CPU E5-2682 v4 @ 2.50GHz).


### Example data

An example data could be downloaded from our webserver (chr22_40kb.npz). It is the 1/16 downsampled raw data on chromosome 22 from GM12878 cell line.
