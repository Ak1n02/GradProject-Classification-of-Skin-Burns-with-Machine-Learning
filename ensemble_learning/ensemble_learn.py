from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from catboost_model import *
from feauture_extraction.data_analysis import normalize_data_set
from lightgbm_model import *
from xgboost_model import *
from svm_rf import *


def predict_with_2stage_model(x_test, y_pred_main, sub_model):
    final_preds = []
    for i, pred in enumerate(y_pred_main):
        if pred == 2:
            final_preds.append(2)
        elif pred in [0, 1]:
            sub_input = x_test.iloc[i:i+1]
            pred_sub = sub_model.predict(sub_input)
            final_preds.append(int(pred_sub[0]))
    return final_preds

def fit_catboost_model(model, x,y):

    mask = y.isin([0, 1])  # Class 0 & 1 (1. ve 2. derece)
    x_sub = x[mask]
    y_sub = y[mask]

    model.fit(x_sub, y_sub)
    return model

def train_voting_classifier(normalized_data):

    x = normalized_data.iloc[:, 1:-1]  # Features
    y = normalized_data.iloc[:, 0]     # Skin Burn Degrees
    groups = normalized_data['group_id']

    y = y - 1

    gkf = GroupKFold(n_splits=5)
    scores = []

    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean') # KNN Classifier
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
        weights=[1, 2, 2, 2, 3],
        voting='soft'
    )

    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        ensemble.fit(x_train, y_train)

        y_pred_main = ensemble.predict(x_test)
        sub_model = fit_catboost_model(train_catboost_base_model(), x_train, y_train)
        y_pred_final = predict_with_2stage_model(x_test, y_pred_main, sub_model)

        y_test_original = y_test + 1
        y_pred_final_original = np.array(y_pred_final) + 1

        acc = accuracy_score(y_test_original, y_pred_final_original)
        scores.append(acc)

        cm = confusion_matrix(y_test_original, y_pred_final_original)
        print(f"[Fold {fold+1}] Confusion Matrix:\n{cm}")
        print(f"[Fold {fold+1}] Accuracy: {acc:.4f}")

    mean_accuracy = sum(scores) / len(scores)
    print(f"Mean Accuracy (Voting Classifier): {mean_accuracy:.4f}")

def main():
    dataset_p = f'../Dataset_Test_Eren/Graphs/Datasets/27/New_Dataset_Vector_v2_PCA9_PCA11_27.csv'
    normalized_data = normalize_data_set(dataset_p)
    train_voting_classifier(normalized_data)

if __name__ == "__main__":
    main()