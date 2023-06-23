
#%% lib and functions
import pandas as pd
import numpy as np

import random as rn
from tensorflow import random

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import os

from functions import wait_intervals, window, model_errors
from models import model_ann, model_lstm, model_xgboost

import matplotlib.pyplot as plt


#%% Reproducible
#^ set Seed
np.random.seed(1234)
rn.seed(1234)
random.set_seed(1234)

#^ run script on single core to get reproduceable results
os.environ['TF_NUM_THREADS'] = '1'

path = 'sim_030.xlsx'

#%%
#* Read the Excel file
df = pd.read_excel(path)

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

#%%

vehicles_sums = df.groupby(df.index // (20*24))['Vehicles'].sum()

soc_sums = df.groupby(df.index // (20*24))['total_Delta_SoC'].sum()

plt.plot(vehicles_sums)
plt.show()
plt.plot(soc_sums)
plt.show()

#%%

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
xgb_pred = pd.DataFrame(model_xgboost(X_train_scaled, y_train, X_test_scaled, y_test, train = False, plot = False)).rename(columns = {0:"XGB"})


# %%
lstm_pred = pd.DataFrame(model_lstm(X_train_t, y_train_t, X_test_t, y_test_t, train = False, plot = False)).rename(columns = {0:"LSTM"})

# %%
ann_pred = pd.DataFrame(model_ann(X_train_scaled, y_train, X_test_scaled, y_test, train = True, plot = False)).rename(columns = {0:"ANN"})

# %%
#y_test = pd.DataFrame(y_test).remove_index(drop = True)

results = pd.concat([pd.DataFrame(y_test.values).rename(columns = {0:"Test"}), ann_pred, lstm_pred, xgb_pred], axis = 1)
results = results.reset_index()

# %%
"""
plt.bar(results['index'], results['Test'], label='TEST')
plt.bar(results['index'], results['ANN'], label='ANN')
plt.bar(results['index'], results['LSTM'], label='LSTM')
plt.bar(results['index'], results['XGB'], label='XGB')

plt.xlabel('Interval No.')
plt.ylabel('Waiting Interval')
plt.title('Multiple Variables')
plt.legend()

plt.show()
"""

# %%

# Calculate cumulative sums in intervals of 20 rows
test_sums = results.groupby(results.index // 20)['Test'].sum()
ann_sums = results.groupby(results.index // 20)['ANN'].sum()
lstm_sums = results.groupby(results.index // 20)['LSTM'].sum()
xgb_sums = results.groupby(results.index // 20)['XGB'].sum()

df_20 = pd.concat([test_sums, ann_sums, lstm_sums, xgb_sums], axis =1)

df_20 = df_20.reset_index()

# %%

plt.bar(df_20['index'], df_20['Test'], label='TEST')
plt.bar(df_20['index'], df_20['ANN'], label='ANN')
plt.bar(df_20['index'], df_20['LSTM'], label='LSTM')
plt.bar(df_20['index'], df_20['XGB'], label='XGB')

plt.xlabel('Interval No.')
plt.ylabel('Waiting Interval')
plt.title('Multiple Variables')
plt.legend()

plt.show()

# %%
# Plot the grouped bar graph
ax = df_20[['Test', 'ANN', 'LSTM', 'XGB']][10:30].plot(kind='bar')
ax.set_ylabel('Value')
#ax.set_xlabel('Category')
ax.legend(title='Variables')

plt.show()

# %%

non_zero = df_20[df_20["Test"] != 0]

print("\nANN")
model_errors(non_zero['Test'], non_zero['ANN'], mape = True)

print("\nXGB")
model_errors(non_zero['Test'], non_zero['XGB'], mape = True)

print("\nLSTM")
model_errors(non_zero['Test'], non_zero['LSTM'], mape = True)
# %%
