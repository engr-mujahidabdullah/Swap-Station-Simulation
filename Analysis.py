#%% lib and functions
import pandas as pd
import numpy as np
import random as rn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import random
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
from functions import wait_intervals, window, model_errors

#%% Reproducible
#^ set Seed
np.random.seed(1234)
rn.seed(1234)
random.set_seed(1234)

#^ run script on single core to get reproduceable results
os.environ['TF_NUM_THREADS'] = '1'


#%%
#* Read the Excel file
df = pd.read_excel('sim_38.xlsx')

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

"""
Select predictors and Response from dataset
"""

#df = df[df['Queue_A'] != 0]

#? Predictor Variables
X_df = df[['Peek', 'Vehicles', 'Queue_B',
       'Stock Remaining', 'Slot_1_SoC', 'Slot_2_SoC', 'Slot_3_SoC',
       'Slot_4_SoC']]

#? Response Variable
y_df = df['wait_time']

#%%

from scipy.stats import ttest_ind, ttest_rel

# Assuming you have a numerical variable 'numerical_var' and a categorical variable 'categorical_var'

group1 = df['wait_time'][df['Peek'] == 1]
group2 = df['wait_time'][df['Peek'] == 0]

# Perform independent t-test
t_statistic, p_value = ttest_ind(group1, group2)

# Print the results
print("Peek")
print("T-statistic:", t_statistic)
print("p-value:", p_value)


# Assuming you have two independent samples x1 and x2 [Paired ttest]
t_statistic, p_value = ttest_rel(df['wait_time'], df['Queue_B'])

# Print the results
print("Queue_B")
print("T-statistic:", t_statistic)
print("p-value:", p_value)

# Assuming you have two independent samples x1 and x2 [Paired ttest]
t_statistic, p_value = ttest_rel(df['wait_time'], df['Stock Remaining'])

# Print the results
print("Stock Remaining")
print("T-statistic:", t_statistic)
print("p-value:", p_value)

# Assuming you have two independent samples x1 and x2 [Paired ttest]
t_statistic, p_value = ttest_rel(df['wait_time'], df['Vehicles'])

# Print the results
print("Vehicles")
print("T-statistic:", t_statistic)
print("p-value:", p_value)

# Assuming you have two independent samples x1 and x2 [Paired ttest]
t_statistic, p_value = ttest_rel(df['wait_time'], df['Slot_1_SoC'])

# Print the results
print("Slot_1_SoC")
print("T-statistic:", t_statistic)
print("p-value:", p_value)

# Assuming you have two independent samples x1 and x2 [Paired ttest]
t_statistic, p_value = ttest_rel(df['wait_time'], df['Slot_2_SoC'])

# Print the results
print("Slot_2_SoC")
print("T-statistic:", t_statistic)
print("p-value:", p_value)

# Assuming you have two independent samples x1 and x2 [Paired ttest]
t_statistic, p_value = ttest_rel(df['wait_time'], df['Slot_3_SoC'])

# Print the results
print("Slot_3_SoC")
print("T-statistic:", t_statistic)
print("p-value:", p_value)

# Assuming you have two independent samples x1 and x2 [Paired ttest]
t_statistic, p_value = ttest_ind(df['wait_time'], df['Slot_4_SoC'])

# Print the results
print("Slot_4_SoC")
print("T-statistic:", t_statistic)
print("p-value:", p_value)

df.describe().to_excel("description.xlsx", index=True)

#y_df.describe()



#%%
#* Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.1, shuffle=False)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
"""
Convert data into windowed i.e. tensor
"""
X_train_t = window(X_train_scaled)
y_train_t = window(y_train)
X_test_t = window(X_test_scaled)
y_test_t = window(y_test)

#%%
"""
LSTM Implementation
"""

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
model.fit(X_train_t, y_train_t, epochs=1500, batch_size=200)
model.save("lstm_model.h5")

#%%
#~ Load the saved model
loaded_model = load_model('lstm_model.h5')

#%%
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

# %%
#plt.plot(np.round(actual))
#plt.plot(np.round(pred))

# Plotting
plt.plot(np.round(actual), label='Actual')
plt.plot(np.round(pred), label='Predicted')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.title('Actual vs Predicted Data LSTM')
plt.show()




#%%
# Define the ANN model architecture
model_ann = keras.Sequential([
    layers.Dense(80, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(80, activation='sigmoid'),
    layers.Dense(80, activation='sigmoid'),
    layers.Dense(1)  # Output layer with 1 neuron for regression
])

# Compile the model
model_ann.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_ann.fit(X_train_scaled, y_train, epochs=1500, batch_size=200, verbose=1)

model_ann.save("ann_model.h5")


#%%
#~ Load the saved model
model_ann = load_model('ann_model.h5')

print("ann Model")
#^ Predict the target values for the train set

y_train_ann = model_ann.predict(X_train_scaled)

#^ Predict the target values for the test set
y_pred_ann = model_ann.predict(X_test_scaled)

#%%
print("\n") 
print("ann Model -- Train Data")
model_errors(y_train, np.round(y_train_ann))
print("\n")
print("ann Model -- Test Data")
model_errors(y_test, np.round(y_pred_ann))

#%%

# Plotting
plt.plot(np.round(y_test), label='Actual')
plt.plot(np.round(y_pred_ann), label='Predicted')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.title('Actual vs Predicted Data ANN')
plt.show()



# %%
"""
XGBOOST
"""
#^ Create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

#^ Set the parameters for XGBoost
params = {
    'max_depth': 7,
    'eta': 0.4,
    'objective': 'reg:squarederror'
}

#^ Train the XGBoost model
num_rounds = 50000
model_xgb = xgb.train(params, dtrain, num_rounds)


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


#%%
# Plotting
plt.plot(np.round(y_test), label='Actual')
plt.plot(np.round(y_pred_xgb), label='Predicted')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()
plt.title('Actual vs Predicted Data XGBoost')
plt.show()
# %%
