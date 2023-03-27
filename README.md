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


## Training 
You can train FPFLI for analyzing specific FLIM data according to your applications. 
 The folder `./synthetic data generation`  provides matlab scripts to generate synthetic fluorescence decay histograms and semi-synthetic FLIM images. 
 ### LLE
 For the training of LLE, the synthetic decays can be generated using `Fluorescence_multi_decay_nonhomopp.mat` with predifined liftime range, lifetime component, timing resolution, and photon count. The decays for training are saved in one matlab file, in which *y* denotes decays and *tau_ave* denotes targets. LLE is trained by `Training_LLE.py`  in `./Network training/LLE_training`.
 ### NIII
For the training of NIII, the semi-synthetic FLIM images can be generated by `Data_Generation.m` and `Data_Generation_segmentation.m`. The original HPA dataset can be found at [Human Protein Atlas - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification). NIII is trained by `Training_NIII.py` in `./Network training/NIII_training`. 
