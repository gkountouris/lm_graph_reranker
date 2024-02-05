#!/bin/bash

# Activate the conda environment
source /storage3/gkou/venvs/dragon/bin/activate

# Set the library path
export LD_LIBRARY_PATH=/storage3/gkou/miniconda3/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib:$LD_LIBRARY_PATH

# You can add any other commands you need to run after activation here

