import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feauture_extraction.data_analysis import normalize_data_set


def optimize_random_forest_with_optuna(normalized_data):
    x = normalized_data.iloc[:, 1:-1]
    y = normalized_data.iloc[:, 0] - 1
    groups = normalized_data['group_id']

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'random_state': 42
        }

        gkf = GroupKFold(n_splits=5)
        scores = []

        for train_idx, test_idx in gkf.split(x, y, groups):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = RandomForestClassifier(**params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            y_pred += 1
            y_test += 1

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("\n✅ EN İYİ RandomForest PARAMETRELERİ:")
    print(study.best_params)
    print(f"✅ EN İYİ Mean Accuracy (RandomForest): {study.best_value:.4f}")


def optimize_svc_with_optuna(normalized_data):
    x = normalized_data.iloc[:, 1:-1]
    y = normalized_data.iloc[:, 0] - 1
    groups = normalized_data['group_id']

    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        }

        gkf = GroupKFold(n_splits=5)
        scores = []

        for train_idx, test_idx in gkf.split(x, y, groups):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = make_pipeline(
                StandardScaler(),
                SVC(**params, probability=True, random_state=42)
            )

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            y_pred += 1
            y_test += 1

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("\n✅ EN İYİ SVC PARAMETRELERİ:")
    print(study.best_params)
    print(f"✅ EN İYİ Mean Accuracy (SVC): {study.best_value:.4f}")

def train_random_forest_base_model(): # 79.92 öncesi 77.59
    return RandomForestClassifier(
        n_estimators=297,
        max_depth=12,
        min_samples_split=2,
        criterion='log_loss',
        random_state=42
    )

def train_svm_base_model(): # 75.68 öncesi 74.62
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=0.476354015104506,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
    )

def main():
    dataset_p = f'../Dataset_Test_Eren/Graphs/Datasets/27/New_Dataset_Vector_v2_PCA9_PCA11_27.csv'
    normalized_data = normalize_data_set(dataset_p)
    optimize_svc_with_optuna(normalized_data)

if __name__ == "__main__":
    main()

