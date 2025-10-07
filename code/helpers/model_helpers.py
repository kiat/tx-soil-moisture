from helpers.data_helpers import retrieve_data, create_sets
import json
import tensorflow as tf
import pandas as pd

#batches the train and test sets, fits the model and saves the model and loss_by_epoch
def train_model(model_dict, patience, folder, max_epochs, window_size, shift_amt, target_idx, batch_size):
    train_df, val_df = retrieve_data()
    train_dataset, train_steps = create_sets(train_df, window_size, shift_amt, target_idx, batch_size)
    val_dataset, val_steps = create_sets(val_df, window_size, shift_amt, target_idx, batch_size)

    for key in model_dict.keys():
        save_path = folder + '/' + key + '.keras'
        compile_and_fit(model_dict[key], max_epochs, patience, train_dataset, train_steps, val_dataset, val_steps, batch_size)
        save_model(model_dict[key], save_path)

    


def save_model(model, save_path):
    loss_by_epoch = model.history.history
    model.save(save_path)
    json.dump(loss_by_epoch, open(save_path + "_loss_by_epoch", 'w'))


def compile_and_fit(model, max_epochs, patience, train_dataset, train_steps, val_dataset, val_steps, batch_size):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
    
    model.fit(train_dataset,
            epochs=max_epochs,
            callbacks=[early_stopping],
            validation_data=val_dataset,
            validation_steps=val_steps,
            shuffle=False,
            batch_size=batch_size,
            steps_per_epoch=train_steps)
    return model


#uses the model to generate predictions
def get_predictions(model, y_test, test_dataset, batch_size, test_steps, window_size, shift_amt):
    forecast = model.predict(test_dataset, batch_size=batch_size, steps=test_steps)
    if len(forecast.shape) == 3:
        forecast = forecast[:, 0, 0]
    elif len(forecast.shape) == 2:
        forecast = forecast[:, 0]

    #y_true = y_test[:len(forecast)]
    #shift_bl = y_test[window_size + shift_amt:]
    #shift_bl = shift_bl[:len(forecast)]
    return forecast #, y_true, shift_bl

#stores predictions, true values and shift baseline
def store_preds(y_pred, y_true, shift, model_name, folder):

    try:
        results = pd.read_csv(folder + "/results.csv")
        min_val = len(results["y_true"])
        results[model_name] = y_pred[:min_val]
        results.to_csv(folder + "/results.csv", index = False)
    except:
        shift_bl = y_true[shift:]
        min_val = min([len(y_pred), len(y_true), len(shift_bl)])
        data = {
            model_name: y_pred[:min_val],
            "y_true": y_true[:min_val],
            "shift_bl": shift_bl[:min_val]
        }
        results = pd.DataFrame(data = data)
        results.to_csv(folder + "/results.csv", index = False)

#creates batches for test set, loads in the model and makes/stores predictions
def evaluate_model(save_path, folder, model_name, window_size, shift_amt, target_idx, batch_size):
    test_df = retrieve_data(test = True)
    test_set, test_steps, X_test, y_test = create_sets(test_df, window_size, shift_amt, target_idx, batch_size, True)
    model = tf.keras.models.load_model(save_path)
    y_pred = get_predictions(model, y_test, test_set, batch_size, test_steps, window_size, shift_amt)
    store_preds(y_pred, y_test, window_size + shift_amt, model_name, folder)


def evaluate_all(folder, model_dict, window_size, shift_amt, target_idx, batch_size):
    for key in model_dict.keys():
        save_path = folder + '/' + key + '.keras'
        print(save_path)
        print(key)
        evaluate_model(save_path, folder, key, window_size, shift_amt, target_idx, batch_size)
    

