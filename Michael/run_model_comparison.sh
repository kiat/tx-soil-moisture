#!/bin/bash

# Array of configurations as JSON-like strings
configurations=(
    '{"features": "SWC_5", "input_steps": 24, "output_steps": 6}'
    '{"features": "SWC_5", "input_steps": 48, "output_steps": 12}'
    '{"features": "SWC_5", "input_steps": 168, "output_steps": 24}'  # 7*24 is 168
    '{"features": "SWC_5", "input_steps": 168, "output_steps": 48}'
    '{"features": "SWC_10", "input_steps": 24, "output_steps": 1}'
    '{"features": "SWC_10", "input_steps": 24, "output_steps": 6}'
    '{"features": "SWC_10", "input_steps": 48, "output_steps": 12}'
    '{"features": "SWC_10", "input_steps": 168, "output_steps": 24}'
    '{"features": "SWC_10", "input_steps": 168, "output_steps": 48}'
    '{"features": "SWC_20", "input_steps": 24, "output_steps": 1}'
    '{"features": "SWC_20", "input_steps": 24, "output_steps": 6}'
    '{"features": "SWC_20", "input_steps": 48, "output_steps": 12}'
    '{"features": "SWC_20", "input_steps": 168, "output_steps": 24}'
    '{"features": "SWC_20", "input_steps": 168, "output_steps": 48}'
)

# Loop through each configuration and run the command
for config in "${configurations[@]}"; do
    # Extract parameters from each JSON-like configuration string
    features=$(echo $config | grep -oP '(?<=features": ")[^"]*')
    input_steps=$(echo $config | grep -oP '(?<=input_steps": )[^,]*')
    output_steps=$(echo $config | grep -oP '(?<=output_steps": )[^}]*')
    
    # Run the Python script with the extracted parameters
    echo "Running: python3 model_comparison.py --features $features --input_steps $input_steps --output_steps $output_steps"
    python3 model_comparison.py --features "$features" --input_steps "$input_steps" --output_steps "$output_steps"
done