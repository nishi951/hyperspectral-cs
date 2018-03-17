# hyperspectral-cs
Mark Nishimura

EE 367 Final Project, Winter 2018

PyTorch implementation of Choi et. al. 2017 "High-Quality Hyperspectral Reconstruction using a Spectral Prior" at https://github.com/KAIST-VCLAB/deepcassi

### Notebooks
All analysis was conducted in the notebooks.
- `prepare_data.ipynb` - Constructs .csv files that allow PyTorch data manipulation objects to access patches from the data in an efficient manner.
- `autoencoder.ipynb` - Constructs and trains the autoencoder. Saves checkpoints so the model weights can be loaded later.
- `ADMM.ipynb` - The ADMM reconstruction. Loads the validation set and provides several functions for displaying the data.
- `naive_cs.ipynb` - An implementation of our naive DCT-based smooth spectum ADMM algorithm.

### Installation
To train the model, it is recommended that you have a GPU with CUDA and CUDNN installed. For help setting that up, see http://docs.nvidia.com/cuda/index.html 
and
https://developer.nvidia.com/cudnn

Most of the code is available in the jupyter notebooks. To make sure they run correctly, I recommend setting up the conda environment using:

`$ conda env create -f environment.yml`

This command creates an environment called `pytorch91`. Once activated, all the relevant packages should be available.

#### Data download
The hyperspectral-cs/data folder contains a script for downloading the data for the CAVE and KAIST datasets. To use it, run
`$ source download_data.sh`
from the data folder.


