python3 main.py --window_size 168 --offset 24 --epochs 100 --patience 3 --features "SWC_10" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 24 --epochs 100 --patience 3 --features "SWC_10,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 24 --epochs 100 --patience 3 --features "SWC_10,Ppt" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 24 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 24 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 24 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair,Wx,Wy,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"

echo "Computing 48 Hours Offset"

python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 3 --features "SWC_10" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 3 --features "SWC_10,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 3 --features "SWC_10,Ppt" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 48 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair,Wx,Wy,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"

echo "Computing 168 Hours Offset"

python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 3 --features "SWC_10" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 3 --features "SWC_10,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 3 --features "SWC_10,Ppt" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"
python3 main.py --window_size 168 --offset 168 --epochs 100 --patience 3 --features "SWC_10,Ppt,Tair,Wx,Wy,YearSin,YearCos" --model_names "LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"


