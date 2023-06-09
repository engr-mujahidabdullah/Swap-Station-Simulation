import pandas as pd
import numpy as np
import random as rn
from sklearn.preprocessing import MinMaxScaler

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

def data_insights(actual, predic):

    predic = [item for sublist in predic for item in sublist]
    #difference = actual - predic

    # Print the difference
    #print("Difference Mean: ", np.mean(difference))
    #print("Difference Max: ", np.max(difference))
    #print("Difference Min: ", np.min(difference))
    #print("Difference Median: ", np.median(difference))

    return pd.DataFrame(predic)
    #return pd.concat([actual,pd.DataFrame(predic),pd.DataFrame(difference)], axis=1)

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

