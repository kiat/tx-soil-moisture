#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p satellite/LSTM_results

# Define all possible parameters
stations=("Station1" "Station2" "Station3" "Station4" "Station5" "Station6")
targets=("Sat_SM_AMSR" "Sat_SM_SMAP")
model_types=("lstm" "autoregressive")
steps_range=($(seq 1 7))  # Steps from 1 through 7

# Function to generate combinations of stations
get_station_combinations() {
    local n=$1
    local stations=("${@:2}")
    local combinations=()
    
    # Generate combinations using Python
    combinations=$(python -c "
from itertools import combinations
stations = ['${stations[@]}']
for combo in combinations(stations, $n):
    print(' '.join(combo))
")
    echo "$combinations"
}

# Iterate through all possible combinations
for target in "${targets[@]}"; do
    for model_type in "${model_types[@]}"; do
        for steps in "${steps_range[@]}"; do
            # Generate all possible station combinations (1 through 6 stations)
            for ((num_stations=1; num_stations<=${#stations[@]}; num_stations++)); do
                while IFS= read -r station_combo; do
                    if [ ! -z "$station_combo" ]; then
                        echo "Running analysis for stations: $station_combo, target: $target, model: $model_type, steps: $steps"
                        python satellite/Autoregressive_LSTM.py \
                            --stations $station_combo \
                            --target "$target" \
                            --model_type "$model_type" \
                            --steps "$steps"
                    fi
                done < <(get_station_combinations $num_stations "${stations[@]}")
            done
        done
    done
done

echo "All analyses complete. Results are in satellite/LSTM_results/" 