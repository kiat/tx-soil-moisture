#!/bin/bash

# Define offsets to test
offsets=(24 72 168)

# Other parameters
window_size=168
epochs=10
patience=3

# Default features (modify or pass as argument)
features="SWC_20,T_20,Ppt,Tair,Wx,Wy"

# Loop over each offset
for offset in "${offsets[@]}"; do
    echo "Running model with offset: $offset and features: $features"
    
    # Convert features to a valid filename format (replace commas with underscores)
    feature_str=$(echo "$features" | tr ',' '_')
    
    # Define model path with features and offset
    model_path="model_ws${window_size}_offset${offset}_${feature_str}.keras"
    
    # Run Python script with the given parameters
    python3 main.py --window_size $window_size --offset $offset --epochs $epochs --patience $patience --features "$features"
done

echo "All runs completed!"
