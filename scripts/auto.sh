#!/bin/bash

# Directories to be created
directories=(
    "/data/raw"
    "/data/processed"
    "/src/utils"
    "/src/models"
    "/src/datasets"
    "/src/training"
    "/scripts"
    "/models"
    "/logs"
)

# Base directory
base_dir=~/STGCN

# Create directories
for dir in "${directories[@]}"; do
  mkdir -p "${base_dir}${dir}"
done

# Create Python files with the directory structure
touch "${base_dir}/src/utils/__init__.py"
touch "${base_dir}/src/utils/unzip_data.py"
touch "${base_dir}/src/utils/process_data.py"
touch "${base_dir}/src/models/__init__.py"
touch "${base_dir}/src/models/model.py"
touch "${base_dir}/src/datasets/__init__.py"
touch "${base_dir}/src/datasets/custom_dataset.py"
touch "${base_dir}/src/training/__init__.py"
touch "${base_dir}/src/training/train_utils.py"
touch "${base_dir}/src/run.py"
touch "${base_dir}/src/config.py"
touch "${base_dir}/scripts/run_training.sh"
touch "${base_dir}/README.md"

