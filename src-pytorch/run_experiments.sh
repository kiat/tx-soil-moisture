# All models: LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline,movingaverage

# echo "Computing 24 Hours Offset"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predict_features "SWC_10" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predict_features "SWC_10,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 24 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair,Wx,Wy,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"

# echo "Computing 48 Hours Offset"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 20 --predict_features "SWC_10" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 20 --predict_features "SWC_10,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair,Wx,Wy,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"

# echo "Computing 168 Hours Offset"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predict_features "SWC_10" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predict_features "SWC_10,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"
python3 main.py --window_size 48 --offset 168 --epochs 100 --patience 20 --predict_features "SWC_10,Ppt,Tair,Wx,Wy,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline,attentiononly,transformer,multiheadlstm,baseline"

# Testing New predict_features (LSTM only)
python3 main.py --window_size 48 --offset 168 --epochs 10 --patience 20 --predict_features "SWC_10,Ppt" --model_names LSTM
python3 main.py --window_size 48 --offset 168 --epochs 10 --patience 20 --predict_features "SWC_10,Ppt_log" --model_names LSTM
python3 main.py --window_size 48 --offset 168 --epochs 10 --patience 20 --predict_features "SWC_10,Ppt_RainFlag" --model_names LSTM
python3 main.py --window_size 48 --offset 168 --epochs 10 --patience 20 --predict_features "SWC_10,Ppt_24h_sum" --model_names LSTM
python3 main.py --window_size 48 --offset 168 --epochs 10 --patience 20 --predict_features "SWC_10,HoursSinceRain" --model_names LSTM
