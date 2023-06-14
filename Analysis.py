#%% lib and functions
import pandas as pd
import numpy as np

import random as rn
from tensorflow import random

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import os

from functions import wait_intervals, window, model_errors
from models import model_ann, model_lstm, model_xgboost, plott

import matplotlib.pyplot as plt
#%% Reproducible
#^ set Seed
np.random.seed(1234)
rn.seed(1234)
random.set_seed(1234)

#^ run script on single core to get reproduceable results
os.environ['TF_NUM_THREADS'] = '1'

path = 'sim_30.xlsx'

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
model_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
# %%
