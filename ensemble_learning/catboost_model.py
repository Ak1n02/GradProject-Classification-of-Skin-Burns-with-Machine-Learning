import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

from feauture_extraction.data_analysis import normalize_data_set


def optimize_catboost_with_optuna(normalized_data):
    x = normalized_data.iloc[:, 1:-1]
    y = normalized_data.iloc[:, 0] - 1  # 1-2-3 → 0-1-2
    groups = normalized_data['group_id']

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 400),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': False,
            'random_state': 42
        }

        gkf = GroupKFold(n_splits=5)
        scores = []

        for train_idx, test_idx in gkf.split(x, y, groups):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Sadece class 0 ve 1 üzerinde eğitiyoruz
            train_mask = y_train.isin([0, 1])
            test_mask = y_test.isin([0, 1])

            x_train_sub = x_train[train_mask]
            y_train_sub = y_train[train_mask]

            x_test_sub = x_test[test_mask]
            y_test_sub = y_test[test_mask]

            model = CatBoostClassifier(**params)
            model.fit(x_train_sub, y_train_sub)

            y_pred = model.predict(x_test_sub)

            # Gerçek classlara döndür (0→1, 1→2)
            y_pred = y_pred + 1
            y_test_sub = y_test_sub + 1

            acc = accuracy_score(y_test_sub, y_pred)
            scores.append(acc)

        mean_accuracy = np.mean(scores)
        return mean_accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("\n✅ EN İYİ CatBoost PARAMETRELERİ:")
    print(study.best_params)
    print(f"✅ EN İYİ Mean Accuracy (CatBoost): {study.best_value:.4f}")

'''
    best_params = study.best_params
    best_params.update({
        'verbose': False,
        'random_state': 42
    })

    # Final modeli tüm class 0-1 için eğit
    mask_all = y.isin([0, 1])
    x_final = x[mask_all]
    y_final = y[mask_all]

    model = CatBoostClassifier(**best_params)
    model.fit(x_final, y_final)

    return model '''

def test_model(normalized_data):

    x = normalized_data.iloc[:, 1:-1]
    y = normalized_data.iloc[:, 0] - 1  # 1-2-3 → 0-1-2
    groups = normalized_data['group_id']

    gkf = GroupKFold(n_splits=5)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Sadece class 0 ve 1 üzerinde eğitiyoruz
        train_mask = y_train.isin([0, 1])
        test_mask = y_test.isin([0, 1])

        x_train_sub = x_train[train_mask]
        y_train_sub = y_train[train_mask]

        x_test_sub = x_test[test_mask]
        y_test_sub = y_test[test_mask]

        model = train_catboost_base_model()
        model.fit(x_train_sub, y_train_sub)

        y_pred = model.predict(x_test_sub)

        # Gerçek classlara döndür (0→1, 1→2)
        y_pred = y_pred + 1
        y_test_sub = y_test_sub + 1

        acc = accuracy_score(y_test_sub, y_pred)
        scores.append(acc)

        cm = confusion_matrix(y_test_sub, y_pred)
        print(f"[Fold {fold+1}] Confusion Matrix:\n{cm}")
        print(f"[Fold {fold+1}] Accuracy: {acc:.4f}")

    mean_accuracy = np.mean(scores)
    print(f"Mean Accuracy (CatBoost): {mean_accuracy:.4f}")


def train_catboost_base_model():
    params = {
        'iterations': 191,
        'depth': 4,
        'learning_rate': 0.1277825270844722,
        'l2_leaf_reg': 7.524517170952342,
        'random_strength': 5.0635920014334825,
        'border_count': 201,
        'verbose': False,
        'random_state': 42
    }

    model = CatBoostClassifier(**params)
    return model

def main():
    dataset_p = f'../Dataset_Test_Eren/Graphs/Datasets/27/New_Dataset_Vector_v2_PCA9_PCA11_27.csv'
    normalized_data = normalize_data_set(dataset_p)
    test_model(normalized_data)

if __name__ == "__main__":
    main()