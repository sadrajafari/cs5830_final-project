import pandas as pd
import numpy as np
import pickle


def getData():
    print("Getting Data...")
    df = pd.read_csv("combine.csv", index_col=0)
    className = "flareType"
    df[className] = 0
    df.loc[df['BFLARE'] > 0, className] = 1
    df.loc[df['CFLARE'] > 0, className] = 2
    df.loc[df['MFLARE'] > 0, className] = 3
    df.loc[df['XFLARE'] > 0, className] = 4
    print("Data Obtained.")
    # feature_columns = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH',
    #     'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45',
    #     'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY',
    #     'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE']
    feature_columns = ['TOTUSJH', 'TOTBSQ', 'TOTUSJZ', 'USFLUX', 'TOTFZ', 'R_VALUE']
    class_columns = [className]
    print("Standardizing Data...")
    for feature in feature_columns:
        df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
    print("Data Standardized.")
    # Only consider data from 2014
    print("Taking only 2014 data...")
    df["year"] = pd.to_datetime(df["Timestamp"]).dt.year
    df = df.drop([
        'MFLARE_LABEL', 
        'BFLARE_LABEL', 
        'CFLARE_LABEL', 
        'XFLARE_LABEL', 
        'XFLARE_LABEL_LOC', 
        'BFLARE_LABEL_LOC', 
        'MFLARE_LABEL_LOC', 
        'CFLARE_LABEL_LOC', 
        'BFLARE_LOC', 
        'XFLARE_LOC',
        'MFLARE_LOC',
        'CFLARE_LOC',
        'XFLARE', 
        'BFLARE',
        'CFLARE',
        'MFLARE',
        'XR_MAX',
    ], axis=1)
    df = df.dropna()
    df = df.reset_index()
    del df["index"]

    data = df[(df.flareType != 0)]
    noFlare = df[(df.flareType == 0)]
    print("Subset of data obtained.")
    to_delete = []
    for col in data.columns:
        if not (feature_columns.__contains__(col) or class_columns.__contains__(col)):
            to_delete.append(col)
    data.drop(columns=to_delete, inplace=True)
    inputs = df[feature_columns]
    outputs = df[class_columns]
    dataPointsPerX = 5 * 24 # 12 mins/point * (5 * 24) points = 1 day of data
    dataPointsPerY = 5 * 12 # Predict next 12 hours 
    y = []
    X = []
    balancedData = pd.concat([data, noFlare.sample(1500)], axis=0)
    balancedData = balancedData.sample(frac=1)
    for idx, _ in balancedData.iterrows():
        x = inputs.loc[idx - dataPointsPerX + 1 : idx].values
        X.append(x)
        y.append(np.array([max(outputs.loc[idx : idx + dataPointsPerY - 1].values)]))
    return np.array(X),np.array(y)

def splitData(X,y, split=0.7):
    print("Splitting Data")
    cutoff = int(split * len(X))
    X_train = X[0:cutoff]
    X_test = X[cutoff:]
    y_train = y[0:cutoff]
    y_test = y[cutoff:]
    print("Data Split")
    return X_train, X_test, y_train, y_test

def persistTrainTestData(X_train, X_test, y_train, y_test, dirName):
    pickleObject(X_train, f"{dirName}/X_train.pck")
    pickleObject(X_test, f"{dirName}/X_test.pck")
    pickleObject(y_train, f"{dirName}/y_train.pck")
    pickleObject(y_test, f"{dirName}/y_test.pck")


def pickleObject(obj, fileName):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    with open(fileName, 'wb') as file:
        for idx in range(0, len(bytes_out), max_bytes):
            file.write(bytes_out[idx:idx+max_bytes])
        file.close()

if __name__=="__main__":
    X,y = getData()
    X_train, X_test, y_train, y_test = splitData(X,y)
    persistTrainTestData(X_train, X_test, y_train, y_test, "balanced_data")