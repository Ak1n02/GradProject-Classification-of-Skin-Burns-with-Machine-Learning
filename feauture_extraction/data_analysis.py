import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def normalize_data_set():

    data = pd.read_csv('../Dataset_Test_Eren/Graphs/Datasets/Dataset_Vector_TopSelected_3.csv')
    x = data.iloc[:, 1:] # Features
    y = data.iloc[:, 0] # Skin Burn Degrees

    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x)

    normalized_data = pd.DataFrame(x_normalized, columns=x.columns)
    normalized_data.insert(0, 'Degrees', y)

    return normalized_data

def train_model(normalized_data):

    x = normalized_data.iloc[:, 1:] # Features
    y = normalized_data.iloc[:, 0] # Skin Burn Degrees

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    knn_model.fit(x_train, y_train)

    y_predict = knn_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    confusion_m = confusion_matrix(y_test, y_predict)
    print(f'KNN Accuracy: {accuracy:.2f}')
    print(f'Confusion Matrix:\n{confusion_m}')

    # Cross-Validation with 5-folds
    scores = cross_val_score(knn_model, x, y, cv=9)
    print(f'Cross-Validation Scores: {scores}')
    print(f'Mean Cross-Validation Score: {scores.mean():.2f}')


if __name__ == '__main__':
    train_model(normalize_data_set())