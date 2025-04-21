import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.model_selection import cross_val_score

def normalize_data_set(group_size=12):

    data = pd.read_csv('../Dataset_Test_Eren/Graphs/Datasets/New_Dataset_Vector_v2_10_chat.csv')
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

def train_model(normalized_data):

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

        print(f"[Fold {fold+1}] Accuracy: {acc:.2f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    print(f"Mean Accuracy: {sum(scores)/len(scores):.2f}")

if __name__ == '__main__':
    train_model(normalize_data_set(12))