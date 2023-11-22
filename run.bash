#!/bin/bash

# Define your list of transformations
transformations=(
    "log" 
    "sqrt"
)

# Set the model name
model_name="xcit_nano_12_p16_224"

# Initialize an incrementing index
index=1

python setup.py install

# run old model
CUDA_VISIBLE_DEVICES=0 python solutions/v10/solution.py --index "#3" --model_name "$model_name"

# # Loop through the list and call the Python program
# for transformation in "${transformations[@]}"; do
#     CUDA_VISIBLE_DEVICES=0 python solutions/v12/solution.py --index "#$index" --model_name "$model_name" --label_transform "$transformation"
#     ((index++))
# done
