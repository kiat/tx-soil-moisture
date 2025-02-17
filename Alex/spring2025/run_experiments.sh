#!/bin/bash

# Define offsets to test
offsets=(24 72 168)

# echo "python version: $(python3 --version)"

# Other parameters
window_size=168
epochs=10
model_path="model.keras"
patience=3

# Loop over each offset
for offset in "${offsets[@]}"; do
    echo "Running model with offset: $offset"
    python3 main.py --window_size $window_size --offset $offset --epochs $epochs --model_path "${model_path}_offset_${offset}" --patience $patience
done

# echo "All runs completed!"