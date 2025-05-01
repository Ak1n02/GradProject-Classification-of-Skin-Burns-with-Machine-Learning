from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import  confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost_model import *
from lightgbm_model import *
from feauture_extraction.data_analysis import normalize_data_set


def train_voting_classifier(normalized_data):

    x = normalized_data.iloc[:, 1:-1]  # Features
    y = normalized_data.iloc[:, 0]     # Skin Burn Degrees
    groups = normalized_data['group_id']

    y = y - 1

    gkf = GroupKFold(n_splits=5)
    scores = []

    knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan') # KNN Classifier
    rf = RandomForestClassifier(n_estimators=250, random_state=42)
    svm = SVC(probability=True) # SVM Classifier

    ensemble = VotingClassifier(
        estimators=[
            ('knn', knn),
            ('rf', rf),
            ('svm', svm),
            ('xgb', train_xgboost_base_model()),
            ('lgb', train_lightgbm_base_model())
        ],
        weights=[1, 2, 1, 1, 2],
        voting='soft'
    )

    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        ensemble.fit(x_train, y_train)

        y_pred = ensemble.predict(x_test)

        y_pred = y_pred + 1
        y_test = y_test + 1

        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

        cm = confusion_matrix(y_test, y_pred)
        print(f"[Fold {fold+1}] Confusion Matrix:\n{cm}")
        print(f"[Fold {fold+1}] Accuracy: {acc:.4f}")

    mean_accuracy = sum(scores) / len(scores)
    print(f"Mean Accuracy (Voting Classifier): {mean_accuracy:.4f}")

def main():
    dataset_p = f'../Dataset_Test_Eren/Graphs/Datasets/20/New_Dataset_Vector_v2_PCA7_PCA13_20.csv'
    normalized_data = normalize_data_set(dataset_p)
    train_voting_classifier(normalized_data)

if __name__ == "__main__":
    main()