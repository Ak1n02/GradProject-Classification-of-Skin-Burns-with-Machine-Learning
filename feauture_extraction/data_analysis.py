import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold


#paths = ['../Dataset_Test_Eren/Graphs/Datasets/4', '../Dataset_Test_Eren/Graphs/Datasets/5', '../Dataset_Test_Eren/Graphs/Datasets/6', '../Dataset_Test_Eren/Graphs/Datasets/7', '../Dataset_Test_Eren/Graphs/Datasets/8', '../Dataset_Test_Eren/Graphs/Datasets/9']
#log_files = ['../Dataset_Test_Eren/Graphs/Logs/knn_log_4.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_5.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_6.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_7.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_8.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_9.txt']


def save_to_log_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")

def normalize_data_set(dataset_path, group_size=12):

        data = pd.read_csv(dataset_path)
        x = data.iloc[:, 1:] # Features
        y = data.iloc[:, 0] # Skin Burn Degrees

        scaler = MinMaxScaler()
        x_normalized = scaler.fit_transform(x)

        normalized_data = pd.DataFrame(x_normalized, columns=x.columns)
        normalized_data.insert(0, 'Degrees', y)

        # Add Group ID
        group_ids = [i // group_size for i in range(len(data))]
        normalized_data['group_id'] = group_ids

        return normalized_data

def train_and_save_model(normalized_data):

    x = normalized_data.iloc[:, 1:-1] # Features
    y = normalized_data.iloc[:, 0] # Skin Burn Degrees
    groups = normalized_data['group_id']

    gkf = GroupKFold(n_splits=5)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

        cm = confusion_matrix(y_test, y_pred)
        print(f"[Fold {fold+1}] Confusion Matrix:\n{cm}")
        print(f"[Fold {fold+1}] Accuracy: {acc:.4f}")

    mean_accuracy = sum(scores) / len(scores)
    print(f"Mean Accuracy: {mean_accuracy:.4f}")

    final_model = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    final_model.fit(x, y)

    model_path = f'../Dataset_Test_Eren/Model/knn_model.pkl'
    joblib.dump(final_model, model_path)

def main(group_size):
    dataset_path = f'../Dataset_Test_Eren/Graphs/Datasets/20/New_Dataset_Vector_v2_PCA7_PCA13_20.csv'
    normalized_data = normalize_data_set(dataset_path, group_size)
    train_and_save_model(normalized_data)

if __name__ == '__main__':
    main(12)