#!/bin/bash
#SBATCH -t 1:0:0
#SBATCH --mem=12G
#SBATCH -c 1
python3 processPRMD_absDARI.py
