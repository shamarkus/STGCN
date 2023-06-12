#!/bin/bash

!pip install torch_geometric

KIMORE="1CsN2ctbT5EeBOcdfhCEYZGIjsJOcVf_y"
INCPRMD="1dr9mjtmDfpJRwxWu99-5W45oe9siSYST"
CORPRMD="12eg1BkBYGc8JTWf1lBAAqOiEPks6ys8O"

DESTINATION_KIMORE=../data/raw/kimore.zip
DESTINATION_CORPRMD=../data/raw/corprmd.zip
DESTINATION_INCPRMD=../data/raw/incprmd.zip

# gdown "https://drive.google.com/uc?export=download&id=$KIMORE" -O $DESTINATION_KIMORE
gdown "https://drive.google.com/uc?export=download&id=$INCPRMD" -O $DESTINATION_INCPRMD
gdown "https://drive.google.com/uc?export=download&id=$CORPRMD" -O $DESTINATION_CORPRMD

