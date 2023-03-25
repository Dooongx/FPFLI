# FPFLI
This repository contains the source codes of **F**ew **P**hoton **F**luorescence **L**ietime **I**maging (FPFLI) algorithm presented in the paper "*Deep Learning Enhanced Fast Fluorescence Lifetime Imaging with A Few Photons*" by Dong Xiao, Natakorn Sapermsap, Yu Chen, and David Li.

## Environment
- Python >= 3.6
- Pytorch >=1.10
- tqdm, numpy, scipy, matplotlib, mat73

## Evaluation
To test the performance of FPFLI and reproduce the results in the paper, you can use the provided jupyter files in the `./Evaluation` folder:

`Demo_Low_light_FLIM_images.ipynb`

`Demo_microsphere.ipynb`

`Demo_hek293_GNRs.ipynb`

The data and pretrained model parameters of LLE and NIII are included. These files are tested on the desktop computer with Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz and NVIDIA Quadro RTX 5000.


