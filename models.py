from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from functions import wait_intervals, window, model_errors
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

import xgboost as xgb

import pickle

def model_lstm(X_train_t, y_train_t, X_test_t, y_test_t, train = True, plot = True):
    if (train == True):
        timesteps = X_train_t[0].shape[0]
        features = X_train_t[1].shape[1]

        #* Define the model architecture
        model = Sequential()
        model.add(LSTM(units=80, activation='linear', input_shape=(timesteps, features)))
        model.add(Dense(units=5, activation='linear'))
        model.add(Dense(units=5, activation='linear'))
        model.add(Dense(units=5, activation='linear'))


        #* Compile the model
        model.compile(optimizer='Adam', loss='mean_squared_error')

        #* Train the model
        model.fit(X_train_t, y_train_t, epochs=1500, batch_size=200, verbose=1)
        model.save("lstm_model.h5")

    try:
        #~ Load the saved model
        loaded_model = load_model('lstm_model.h5')

        print("LSTM Model")
            #*Make predictions
        y_pred_LSTM = loaded_model.predict(X_train_t)

            #* print Errors
        print("LSTM Model -- Train Data")
        model_errors(y_train_t, np.round(y_pred_LSTM))

        print("LSTM Model -- Test Data")
        y_pred_LSTM_s = loaded_model.predict(X_test_t)

        model_errors(y_test_t, np.round(y_pred_LSTM_s))

        actual = [y_test_t[i][0] for i in range(len(y_test_t))]
        pred = [y_pred_LSTM_s[i][0] for i in range(len(y_pred_LSTM_s))]

        if(plot == True):
            # Plotting
            plt.bar(np.round(actual), label='Actual')
            plt.bar(np.round(pred), label='Predicted')
            plt.xlabel('Data Point')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Actual vs Predicted Data LSTM')
            plt.show()
        
        return np.round(pred)
    
    except:
        print("No saved model found")




def model_ann(X_train_scaled, y_train, X_test_scaled, y_test, train = True, plot = True):

    if (train == True):
        # Define the ANN model architecture
        model_ann = keras.Sequential([
            layers.Dense(80, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)),
            layers.Dense(10, activation='sigmoid'),
            layers.Dense(1)  # Output layer with 1 neuron for regression
        ])

        # Compile the model
        model_ann.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model_ann.fit(X_train_scaled, y_train, epochs=2000, batch_size=300, verbose=1)

        model_ann.save("ann_model.h5")

    try:
        #~ Load the saved model
        model_ann = load_model('ann_model.h5')

        print("ann Model")
        #^ Predict the target values for the train set

        y_train_ann = model_ann.predict(X_train_scaled)

        #^ Predict the target values for the test set
        y_pred_ann = model_ann.predict(X_test_scaled)

    
        print("\n") 
        print("ann Model -- Train Data")
        model_errors(y_train, np.round(y_train_ann))
        print("\n")
        print("ann Model -- Test Data")
        model_errors(y_test, np.round(y_pred_ann))
        
        if(plot == True):
            # Plotting
            plt.bar(np.round(y_test.reset_index(drop=True)), label='Actual')
            plt.bar(np.round(y_pred_ann), label='Predicted')
            plt.xlabel('Data Point')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Actual vs Predicted Data ANN')
            plt.show()
        
        return np.round(y_pred_ann)

    except:
        print("No saved Model Found")


def model_xgboost(X_train_scaled, y_train, X_test_scaled, y_test, train = True, plot = True):
    """
    XGBOOST
    """
    #^ Create a DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    if(train == True):
        #^ Set the parameters for XGBoost
        params = {
            'max_depth': 7,
            'eta': 0.4,
            'objective': 'reg:squarederror'
        }

        #^ Train the XGBoost model
        num_rounds = 50000
        model_xgb = xgb.train(params, dtrain, num_rounds)

        pickle.dump(model_xgb, open("xgboost_model.pickle", "wb"))

    try:

        model_xgb = pickle.load(open("xgboost_model.pickle", "rb"))

        #^ Predict the target values for the train set
        y_train_xgb = model_xgb.predict(dtrain)

        #^ Predict the target values for the test set
        y_pred_xgb = model_xgb.predict(dtest)

        print("XGBoost")
        print("XGBoost -- Train Data")
        model_errors(y_train, np.round(y_train_xgb))
        print("\n")
        print("XGBoost -- Test Data")
        model_errors(y_test, np.round(y_pred_xgb))

        if(plot == True):
            # Plotting
            plt.bar(np.round(y_test.reset_index(drop=True)), label='Actual')
            plt.bar(np.round(y_pred_xgb), label='Predicted')
            plt.xlabel('Data Point')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Actual vs Predicted Data XGBoost')
            plt.show()

        return np.round(y_pred_xgb)

    except:
        print("No save model found")
