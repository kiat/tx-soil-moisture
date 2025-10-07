#!/bin/bash

# Array of configurations as JSON-like strings
configurations=(
    '{"features": "SWC_5", "input_steps": 24, "output_steps": 6, "num_stations": 6}'
    # '{"features": "SWC_5", "input_steps": 48, "output_steps": 12, "num_stations": 6}'
    # '{"features": "SWC_5", "input_steps": 168, "output_steps": 24, "num_stations": 6}'  # 7*24 is 168
    # '{"features": "SWC_5", "input_steps": 168, "output_steps": 48, "num_stations": 6}'
    # '{"features": "SWC_10", "input_steps": 24, "output_steps": 1, "num_stations": 6}'
    # '{"features": "SWC_10", "input_steps": 24, "output_steps": 6, "num_stations": 6}'
    # '{"features": "SWC_10", "input_steps": 48, "output_steps": 12, "num_stations": 6}'
    # '{"features": "SWC_10", "input_steps": 168, "output_steps": 24, "num_stations": 6}'
    # '{"features": "SWC_10", "input_steps": 168, "output_steps": 48, "num_stations": 6}'
    # '{"features": "SWC_20", "input_steps": 24, "output_steps": 1, "num_stations": 6}'
    # '{"features": "SWC_20", "input_steps": 24, "output_steps": 6, "num_stations": 6}'
    # '{"features": "SWC_20", "input_steps": 48, "output_steps": 12, "num_stations": 6}'
    # '{"features": "SWC_20", "input_steps": 168, "output_steps": 24, "num_stations": 6}'
    # '{"features": "SWC_20", "input_steps": 168, "output_steps": 48, "num_stations": 6}'
)

# Loop through each configuration and run the command
for config in "${configurations[@]}"; do
    # Skip commented-out lines
    if [[ $config == \#* || -z $config ]]; then
        continue
    fi
    
    # Extract parameters from each JSON-like configuration string
    features=$(echo "$config" | grep -oP '(?<=features": ")[^"]*')
    input_steps=$(echo "$config" | grep -oP '(?<=input_steps": )[^,]*')
    output_steps=$(echo "$config" | grep -oP '(?<=output_steps": )[^,]*')
    num_stations=$(echo "$config" | grep -oP '(?<=num_stations": )[^}]*')

    # Run the Python script with the extracted parameters
    echo "Running: python3 model_comparison.py --features $features --input_steps $input_steps --output_steps $output_steps --num_stations $num_stations"
    python3 model_comparison.py --features "$features" --input_steps "$input_steps" --output_steps "$output_steps" --num_stations "$num_stations"
done