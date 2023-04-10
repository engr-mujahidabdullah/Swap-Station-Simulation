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
df = pd.read_excel('Day_1.xlsx')

#Replace NaN with 0
df = df.fillna(0)

# Merge all values in column 'xyz' as a single string
merged_string = ' '.join(df['Event'].astype(str).tolist())
merged_string = merged_string.replace(" ","")


wait_time = wait_intervals(merged_string)

# add the array as a new column, only where df['vehicle'] == 1
df.loc[df['Vehicles'] == 1, 'wait_time'] = wait_time

# save the dataframe to an .xlsx file
df.to_excel('Day_1.xlsx', index=False)


"""
if __name__ == "__main__":
    import numpy as np
    event = "0IO00000IO000000IIOOI0000000000000O00000000000000000000000000000000IOIO000000000000000IO000IO0000000000000I0OIO0000000000I000II0O00O000000000I0O0O0000I0I0000000III00000OO000000000O000000O0I000000000000000I0O0O0000O00000000000000000000000000000000000000000IO0000000000000000IO000000000000IO0000000IO00IO000000I00O00000000I00000IO000IIO000000O00O0000I00000I00000O000O00000I00O00000IO0000000000000000000IO00000000000000000000000IO000IIOO0000000000000000000000000000000IIOO000000"
    time = []
    cont = [0,0,0]
    for i in range(len(event)):
        if(event[i] == "R"):
            cont[0] = cont[0] + 1
            time.append(1)
        #if(event[i] == "0"):
        #    pass
        if(event[i] == "I"):
            cont[1] = cont[1] + 1
            for j in range(i, len(event)):
                if(event[j] == "O"):
                    cont[2] = cont[2] + 1
                    time.append((j - i))
                    event.replace(event[j],"0")
                    break
        #if(event[i] == "O"):
        #    pass
    print(len(event))
    print(cont)
    print(time)
    print(len(time))
    print(np.mean(time))
"""