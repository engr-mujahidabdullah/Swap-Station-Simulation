#%%

import pandas as pd
import numpy as np

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

def window(X_df, win_s = 3):
    X_df = np.array(X_df)
    X = []
    for i in range(len(X_df) - win_s):
        row_ = [a for a in X_df[i:i+win_s]]
        X.append(row_)
 #       label = X_df[i:i+win_s]
 #       Y.append(label)
    return np.array(X)


#%%
# Read the Excel file
df = pd.read_excel('SIM2000.xlsx')

# Merge all values in column 'xyz' as a single string
merged_string = ' '.join(df['Event'].astype(str).tolist())
merged_string = merged_string.replace(" ","")

wait_time = wait_intervals(merged_string)

# add the array as a new column, only where df['vehicle'] == 1
df.loc[df['Vehicles'] == 1, 'wait_time'] = wait_time

# Replace True and False values with 1 and 0, respectively
df['Peek'] = df['Peek'].replace({True: 1, False: 0})
df['Event'] = df['Event'].replace({"I": "IN", "O": "OUT", 0: "NO"})


event_d = pd.get_dummies(df['Event'])
print(event_d.columns)
df = pd.concat([df, event_d], axis=1)

#Replace NaN with 0
df = df.fillna(0)

print(df.head())

#%%
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object and fit it to the data
scaler = StandardScaler()
scaler.fit(df.drop(['Time', 'Event', 'column'], axis=1))

# Use the scaler to transform the data
data_std = scaler.transform(df.drop(['Time', 'Event', 'column'], axis=1))


#%%
#Select predictors and Response from dataset
X_df = df[['Peek', 'Vehicles', 'Queue_A', 'To Serve', 'IN', 'NO', 'OUT',
       'Stock Remaining', 'Slot_1_SoC', 'Slot_2_SoC', 'Slot_3_SoC',
       'Slot_4_SoC']]

#X_df = X_df.astype(float)

y_df = df['wait_time']



#%%
from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, shuffle=False)


X_train_f = window(X_train, 3)
y_train_f = window(y_train, 3)
X_test_f = window(X_test, 3)
y_test_f = window(y_test, 3)

timesteps = X_train_f[0].shape[0]
features = X_train_f[1].shape[1]


#%%
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Define the model architecture
model = Sequential()
model.add(LSTM(units=200, input_shape=(timesteps, features)))
model.add(Dense(units=3))

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_f, y_train_f, epochs=50, batch_size=32)

#%%
# Make predictions
y_pred = model.predict(X_test_f)

y_pred_f = np.ravel(y_pred)
y_test_f = np.ravel(y_test_f)


#%%

from sklearn.metrics import mean_squared_error

# Calculate the accuracy
mse = mean_squared_error(y_test_f, y_pred_f)

# Print the result
print("MSE:", mse)


# %%
