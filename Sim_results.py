import pandas as pd

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

# Read the Excel file
df = pd.read_excel('SIM2000.xlsx')

#Replace NaN with 0
df = df.fillna(0)

# Merge all values in column 'xyz' as a single string
merged_string = ' '.join(df['Event'].astype(str).tolist())
merged_string = merged_string.replace(" ","")


wait_time = wait_intervals(merged_string)

# add the array as a new column, only where df['vehicle'] == 1
df.loc[df['Vehicles'] == 1, 'wait_time'] = wait_time

# save the dataframe to an .xlsx file
df.to_excel('SIM2000w.xlsx', index=False)
