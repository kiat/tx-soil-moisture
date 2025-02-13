#!/bin/bash

# Define offsets to test
offsets=(24 36 168)

# Other parameters
window_size=168
epochs=10
model_path="model.keras"
patience=3

# Loop over each offset
for offset in "${offsets[@]}"; do
    echo "Running model with offset: $offset"
    python main.py --window_size $window_size --offset $offset --epochs $epochs --model_path "${model_path}_offset_${offset}" --patience $patience
done

echo "All runs completed!"
