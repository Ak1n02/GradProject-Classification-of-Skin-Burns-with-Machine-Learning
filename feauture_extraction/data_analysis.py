import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_data_set():

    data = pd.read_csv('../Dataset_Test_Eren/Graphs/Dataset_Vector.csv')
    x = data.iloc[:, 1:] # Features
    y = data.iloc[:, 0] # Skin Burn Degrees
    print(x)
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x)

    normalized_data = pd.DataFrame(x_normalized, columns=x.columns)
    normalized_data.insert(0, 'Degrees', y)

    print(normalized_data)


if __name__ == '__main__':
    normalize_data_set()