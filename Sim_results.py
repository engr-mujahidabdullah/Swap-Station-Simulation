#%% lib and functions
import pandas as pd
import numpy as np
import random as rn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import random
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import matplotlib.pyplot as plt


#%%
#^ set Seed
np.random.seed(1234)
rn.seed(1234)
random.set_seed(1234)

#^ run script on single core to get reproduceable results
os.environ['TF_NUM_THREADS'] = '1'

#%%

def wait_intervals(l_str: str) -> list:
    time = []
    for i in range(len(l_str)):
        if(l_str[i] == "I"):
            for j in range(i, len(l_str)):
                if(l_str[j] == "O"):
                    time.append((j - i))
                    l_str.replace(l_str[j],"0")
                    break
    return(time)

def window(X_df, win_s = 5):
    X_df = np.array(X_df)
    X = []
    for i in range(len(X_df) - win_s):
        row_ = [a for a in X_df[i:i+win_s]]
        X.append(row_)

    return np.array(X)

def model_errors(actual, predic):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    #*Calculate the MSE
    mse = mean_squared_error(actual, predic)
    print("MSE:", mse)

    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    # Calculate MAE
    mae = mean_absolute_error(actual, predic)
    print('MAE:', mae)

    # Calculare R2
    r2 = r2_score(actual, predic)
    print("R-squared (R2): {:.4f}".format(r2))

def process_test(win):
    df_ = pd.read_excel('Test.xlsx')

    # Merge all values in column 'xyz' as a single string
    merged_string = ' '.join(df_['Event'].astype(str).tolist())
    merged_string = merged_string.replace(" ","")

    wait_time = wait_intervals(merged_string)

    # add the array as a new column, only where df['vehicle'] == 1
    df_.loc[df_['Vehicles'] == 1, 'wait_time'] = wait_time

    # Replace True and False values with 1 and 0, respectively
    df_['Peek'] = df_['Peek'].replace({True: 1, False: 0})
    df_['Event'] = df_['Event'].replace({"I": "IN", "O": "OUT", 0: "NO"})


    event_d = pd.get_dummies(df_['Event'])
    df_ = pd.concat([df_, event_d], axis=1)

    #Replace NaN with 0
    df_ = df_.fillna(0)

    # Columns to exclude from scaling
    exclude_cols = ['Time', 'Event', 'column']

    # Get numeric columns to scale
    numeric_cols = [col for col in df_.columns if col not in exclude_cols]

    # Scale numeric columns using StandartScaler
    scaler = MinMaxScaler()
    df_[numeric_cols] = scaler.fit_transform(df_[numeric_cols])

    #Select predictors and Response from dataset
    X_test = df_[['Time_conversion','Peek', 'Vehicles', 'Queue_A', 'To Serve', 'IN', 'NO', 'OUT',
        'Stock Remaining', 'Slot_1_SoC', 'Slot_2_SoC', 'Slot_3_SoC',
        'Slot_4_SoC']]

    Y_test = df_['wait_time']

    if(win == True):
        X_test = window(X_test)
        Y_test = window(Y_test)

    return X_test, Y_test 


#%%
#* Read the Excel file
df = pd.read_excel('SIM2000.xlsx')

#* Merge all values in column 'xyz' as a single string
merged_string = ''.join(df['Event'].astype(str).tolist())
merged_string = merged_string.replace(" ","")

wait_time = wait_intervals(merged_string)

#* add the array as a new column, only where df['vehicle'] == 1
df.loc[df['Vehicles'] == 1, 'wait_time'] = wait_time

#* Replace True and False values with 1 and 0, respectively
df['Peek'] = df['Peek'].replace({True: 1, False: 0})
df['Event'] = df['Event'].replace({"I": "IN", "O": "OUT", 0: "NO"})

#* Categorical event into indecator variable
event_d = pd.get_dummies(df['Event'])
df = pd.concat([df, event_d], axis=1)

#*Replace NaN with 0
df = df.fillna(0)


#%% Standardize 
"""
Standardize data and drop Time, Event and column due to its str nature 


#^ Columns to exclude from scaling
exclude_cols = ['Time', 'Event', 'column']

#^ Get numeric columns to scale
numeric_cols = [col for col in df.columns if col not in exclude_cols]

#^ Scale numeric columns using StandartScaler
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
"""

#%% 

"""
Select predictors and Response from dataset
"""
#? Predictor Variables
X_df = df[['Time_conversion','Peek', 'Vehicles', 'Queue_A', 'To Serve', 'IN', 'NO', 'OUT',
       'Stock Remaining', 'Slot_1_SoC', 'Slot_2_SoC', 'Slot_3_SoC',
       'Slot_4_SoC']]

#? Response Variable
y_df = df['wait_time']

#! Scale X_df
scaler_xd = MinMaxScaler()
X_df = scaler_xd.fit_transform(X_df)


#%%

#* Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.1, shuffle=False)



#%%
"""
Convert data into windowed i.e. tensor
"""
X_train_t = window(X_train)
y_train_t = window(y_train)
X_test_t = window(X_test)
y_test_t = window(y_test)



#%%
"""
LSTM Implementation
"""

timesteps = X_train_t[0].shape[0]
features = X_train_t[1].shape[1]

#* Define the model architecture
model = Sequential()
model.add(LSTM(units=100, activation='linear', input_shape=(timesteps, features)))
model.add(Dense(units=5, activation='linear'))
model.add(Dense(units=5, activation='linear'))
model.add(Dense(units=5, activation='linear'))


#* Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error')

#* Train the model
model.fit(X_train_t, y_train_t, epochs=1000, batch_size=150)
model.save("best_model.h5")


#%%
#~ Load the saved model
loaded_model = load_model('best_model.h5')

#%%
#*Make predictions
y_pred_LSTM = loaded_model.predict(X_train_t)

#* print Errors
model_errors(y_train_t, np.round(y_pred_LSTM))

y_pred_LSTM_s = loaded_model.predict(X_test_t)

model_errors(y_test_t, np.round(y_pred_LSTM_s))

actual = [y_test_t[i][0] for i in range(len(y_test_t))]
pred = [y_pred_LSTM_s[i][0] for i in range(len(y_pred_LSTM_s))]

# %%
plt.plot(np.round(actual))
plt.plot(np.round(pred))

plt.show()
# %%