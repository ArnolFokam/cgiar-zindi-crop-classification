#!/bin/bash

# Define the table
TABLE=("15 efficientnet_b0 102"
       "15 convnext_base.fb_in1k 32"
       "15 convnextv2_tiny.fcmae_ft_in22k_in1k_384 48"
       "15 convnextv2_base.fcmae_ft_in22k_in1k 32 32"
       "20 efficientnet_b0 102"
       "20 convnextv2_tiny.fcmae_ft_in22k_in1k_384 48"
       "25 efficientnet_b0 102"
       "25 convnextv2_tiny.fcmae_ft_in22k_in1k_384 48")

# Initialize subfolder index
SUBFOLDER_INDEX=1

# Loop through each row
for row in "${TABLE[@]}"; do
    # Split the row into an array
    IFS=' ' read -ra ARGS <<< "$row"

    # Call your Python script with the specified arguments
    echo python main.py \
        --epochs "${ARGS[0]}" \
        --model_name "${ARGS[1]}" \
        --train_batch_size "${ARGS[2]}" \
        --subfolder \"#${SUBFOLDER_INDEX}\" \

    # Increment subfolder index
    ((SUBFOLDER_INDEX++))
done