import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def create_dataset(original_data_set, top_features, start, end, parameters):


    labels = original_data_set.iloc[:, 0]

    # Extract top features using their indices
    selected_features = original_data_set.iloc[:, top_features + 1]  # +1 to skip label column

    # Merge labels with selected features
    new_dataset = pd.concat([labels, selected_features], axis=1)
    new_dataset.to_csv(f'../Dataset_Test_Eren/Graphs/Datasets/{parameters}/New_Dataset_Vector_v2_PCA{start}_PCA{end}_{parameters}.csv', index=False)

def test_features():

    df = pd.read_csv('../Dataset_Test_Eren/Graphs/Datasets/New_Dataset_Vector_v2.csv')

    # Separate labels and features
    labels = df.iloc[:, 0] # first Column (label)
    features = df.iloc[:, 1:] # features

    # Standardize the feature values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=29) # Arrange According to feature
    principal_components = pca.fit_transform(features_scaled)

    # Explained variance ratio (how much variance each PC explains)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)

    for parameter in range(10,16):
        for start in range(1,15):
            for end in range(start + 1, 16):
                components = pca.components_[start:end]
                contribution_sum = np.sum(np.abs(components), axis=0)
                top_features = np.argsort(-contribution_sum)[:parameter]
                print("Top Contributing Features (PC1):", df.columns[1:][top_features])
                create_dataset(df , top_features, start, end, parameter)

if __name__ == '__main__':
    test_features()