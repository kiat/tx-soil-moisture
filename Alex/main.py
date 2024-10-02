from helper import load_data, preprocess_data, generate_batches, compile_and_fit, plot_single_pred, lstm_model, bi_lstm_model, linear_model, autoregressive_model

# parameters to stay in main.py
TARGET_COL = "SWC_5"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
WINDOW_SIZE = 24 * 7
SHIFT_AMT = 10
PAT = 3
MAX_EPOCHS = 25
BATCH_SIZE = 128

# load and preprocess data

dfs = load_data('../datasets/Simulate_Cleaned_Merged/Station1_simulated_cleaned_merged_data.csv')
cur_df = dfs["Station1"]

X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(cur_df, TARGET_COL, TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT)

train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=BATCH_SIZE)
val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=BATCH_SIZE)
test_dataset, test_steps = generate_batches(X_test, y_test, batch_size=BATCH_SIZE)

# train and plot models
history_bilstm = compile_and_fit(bi_lstm_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="biLSTM", patience=PAT, max_epochs=MAX_EPOCHS)
# plot_single_pred(bi_lstm_model, 'BiLSTM', test_dataset, test_steps, y_test, batch_size=BATCH_SIZE)

history_linear = compile_and_fit(linear_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="linear", patience=PAT, max_epochs=MAX_EPOCHS)
# plot_single_pred(linear_model, 'linear', test_dataset, test_steps, y_test, batch_size=BATCH_SIZE)

history_ar = compile_and_fit(autoregressive_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="autoregressive", patience=PAT, max_epochs=MAX_EPOCHS)
# plot_single_pred(autoregressive_model, 'autoregressive', test_dataset, test_steps, y_test, batch_size=BATCH_SIZE)




