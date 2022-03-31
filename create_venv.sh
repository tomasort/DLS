#!/bin/bash

# activate existed python3 module to get virtualenv
module load python3/intel/3.6.3

# create virtual environment with python3
virtualenv -p python3 .env

# activate virtual environment
source .env/bin/activate

# install pytorch and orther stuff
pip install --upgrade pip
pip install tensorflow-gpu
pip install torch
pip install keras 
pip install torchvision 
pip install jupyter
pip install tqdm 
pip install numpy
pip install bokeh
pip install matplotlib
# uncomment the following line if you have requirements.txt file
# pip install -r requirements.txt

# unload module
module unload python3/intel/3.8.6
