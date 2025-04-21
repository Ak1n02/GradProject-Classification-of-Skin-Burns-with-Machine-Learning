import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  GroupKFold


paths = ['../Dataset_Test_Eren/Graphs/Datasets/10', '../Dataset_Test_Eren/Graphs/Datasets/11', '../Dataset_Test_Eren/Graphs/Datasets/12', '../Dataset_Test_Eren/Graphs/Datasets/13', '../Dataset_Test_Eren/Graphs/Datasets/14', '../Dataset_Test_Eren/Graphs/Datasets/15']
log_files = ['../Dataset_Test_Eren/Graphs/Logs/knn_log_10.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_11.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_12.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_13.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_14.txt', '../Dataset_Test_Eren/Graphs/Logs/knn_log_15.txt']

def save_to_log_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")

def normalize_data_set(group_size=12):

    for index, current_path in enumerate(paths):
        for file in os.listdir(current_path):
            parts = file.split('_')
            start_pca, end_pca, number_parameter = parts[4], parts[5], parts[6]
            save_to_log_file(log_files[index],f"Start PCA: {start_pca}, End PCA: {end_pca}, Number Parameter: {number_parameter}")

            data = pd.read_csv(os.path.join(current_path, file))
            x = data.iloc[:, 1:] # Features
            y = data.iloc[:, 0] # Skin Burn Degrees

            scaler = MinMaxScaler()
            x_normalized = scaler.fit_transform(x)

            normalized_data = pd.DataFrame(x_normalized, columns=x.columns)
            normalized_data.insert(0, 'Degrees', y)

            # Add Group ID
            group_ids = [i // group_size for i in range(len(data))]
            normalized_data['group_id'] = group_ids

            train_model(index, normalized_data)

def train_model(index, normalized_data):

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

#        save_to_log_file(log_files[index], f"[Fold {fold+1}] Accuracy: {acc:.2f}")
#        save_to_log_file(log_files[index], f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    save_to_log_file(log_files[index],f"Mean Accuracy: {sum(scores)/len(scores):.2f}\n")

def main(group_size):
    normalize_data_set(group_size)

if __name__ == '__main__':
    main(12)