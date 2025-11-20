#!/bin/bash
# Experiments predicting SWC_20 with varying predictor sets

MODELS="LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,attentiononly,transformer,multiheadlstm,baseline"
PREDICT="SWC_20"

# 24 Hours Offset (Window=48)
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predictors "SWC_20" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predictors "SWC_20,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predictors "SWC_20,Ppt" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predictors "SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predictors "SWC_20,Ppt,Tair,Wx,Wy" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predictors "SWC_20,Ppt,Tair,Wx,Wy,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair,Wx,Wy,Srad,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"

# 48 Hours Offset (Window=48)
python3 main.py --window_size 48 --offset 48 --epochs 100 --patience 30 --predictors "SWC_20" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 48 --epochs 100 --patience 30 --predictors "SWC_20,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 48 --epochs 100 --patience 30 --predictors "SWC_20,Ppt" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 48 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 48 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair,Wx,Wy" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 48 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair,Wx,Wy,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 48 --epochs 100 --patience 30 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 48 --epochs 100 --patience 30 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair,Wx,Wy,Srad,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"

# 72 Hours Offset (Window=48)
python3 main.py --window_size 48 --offset 72 --epochs 100 --patience 30 --predictors "SWC_20" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 72 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 72 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair,Wx,Wy,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 72 --epochs 100 --patience 30 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair,Wx,Wy,Srad,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"

# 168 Hours Offset (Window=48)
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predictors "SWC_20" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predictors "SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predictors "SWC_20,Ppt,Tair,Wx,Wy,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair,Wx,Wy,Srad,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"

# 72 Hours Offset (Window=72)
python3 main.py --window_size 72 --offset 72 --epochs 100 --patience 30 --predictors "SWC_20" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 72 --offset 72 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 72 --offset 72 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair,Wx,Wy,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 72 --offset 72 --epochs 100 --patience 30 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair,Wx,Wy,Srad,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"

# 168 Hours Offset (Window=72)
python3 main.py --window_size 72 --offset 168 --epochs 100 --patience 30 --predictors "SWC_20" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 72 --offset 168 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 72 --offset 168 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair,Wx,Wy,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 72 --offset 168 --epochs 100 --patience 30 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair,Wx,Wy,Srad,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"

# 168 Hours Offset (Window=168)
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 30 --predictors "SWC_20" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 30 --predictors "SWC_20,Ppt,Tair,Wx,Wy,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 30 --predictors "SWC_5,SWC_10,SWC_20,Ppt,Tair,Wx,Wy,Srad,YearSin,YearCos" --predict_features "$PREDICT" --model_names "$MODELS"

