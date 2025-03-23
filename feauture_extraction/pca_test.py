import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def test_features():

    df = pd.read_csv('../Dataset_Test_Eren/Graphs/Skewness/Article/General.csv')

    # Separate labels and features
    labels = df.iloc[:, 0] # first Column (label)
    features = df.iloc[:, 1:] # features

    # Standardize the feature values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=2) # Arrange According to feature
    principal_components = pca.fit_transform(features_scaled)

    # Explained variance ratio (how much variance each PC explains)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)

    feature_contributions = np.abs(pca.components_)
    top_features = np.argsort(-feature_contributions[0])[:3]
    print("Top Contributing Features (PC1):", df.columns[1:][top_features])


if __name__ == '__main__':
    test_features()