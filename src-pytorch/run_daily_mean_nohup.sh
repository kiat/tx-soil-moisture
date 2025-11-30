#!/usr/bin/env bash

# Launch daily-mean training runs (1D, 3D, 7D horizons) with multiple feature combos.
# Each run executes under nohup so it can continue after the shell exits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

declare -A FEATURE_SETS=(
  ["soil_only"]="SWC_5,SWC_10,SWC_20,SWC_50"
  ["soil_plus_met"]="SWC_5,SWC_10,SWC_20,SWC_50,Ppt,Ppt_24h_sum,Ppt_RainFlag,Tair,RH,Srad,HoursSinceRain"
  ["soil_plus_met_wind"]="SWC_5,SWC_10,SWC_20,SWC_50,Ppt,Ppt_24h_sum,Ppt_RainFlag,Tair,RH,Srad,HoursSinceRain,Wx,Wy,Windspeed,Winddirection"
)

declare -A HORIZON_MAP=(
  ["24"]="1d"
  ["72"]="3d"
  ["168"]="7d"
)

WINDOW_SIZE=168
EPOCHS=50
PATIENCE=5
MODELS="LSTM,BiLSTM,RNN,CNN,AttentionLSTM,attentiononly,transformer,multiheadlstm"

for OFFSET in 24 72 168; do
  TAG="${HORIZON_MAP[$OFFSET]}"
  for FEATURE_KEY in "${!FEATURE_SETS[@]}"; do
    FEATURES="${FEATURE_SETS[$FEATURE_KEY]}"
    LOG_FILE="logs/nohup_daily_${FEATURE_KEY}_${TAG}.txt"
    echo "Starting ${FEATURE_KEY} ${TAG} run -> ${LOG_FILE}"
    nohup python3 main.py \
      --window_size "${WINDOW_SIZE}" \
      --offset "${OFFSET}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --label_type daily_mean \
      --agg_hours 24 \
      --samples_per_hour 1 \
      --features "${FEATURES}" \
      --model_names "${MODELS}" \
      > "${LOG_FILE}" 2>&1
    echo "Completed ${FEATURE_KEY} ${TAG} run."
  done
done

echo "All runs completed. Logs available in ./logs/."

