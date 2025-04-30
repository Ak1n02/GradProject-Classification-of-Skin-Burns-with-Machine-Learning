import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, GridSearchCV
import xgboost as xgb

def train_xgboost_with_tuning(normalized_data):

    x = normalized_data.iloc[:, 1:-1]
    y = normalized_data.iloc[:, 0]
    groups = normalized_data['group_id']

    y = y - 1

    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    # Modeli tanımlıyoruz
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
    )

    # GroupKFold ile CV yapıyoruz
    gkf = GroupKFold(n_splits=5)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=gkf.split(x, y, groups),
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(x, y)

    print("\n✅ EN İYİ PARAMETRELER:")
    print(grid_search.best_params_)
    print(f"✅ EN İYİ Accuracy (CV Mean): {grid_search.best_score_:.4f}")

    best_xgb_model = grid_search.best_estimator_

    return best_xgb_model

def optimize_xgboost_with_optuna(normalized_data):

    x = normalized_data.iloc[:, 1:-1]
    y = normalized_data.iloc[:, 0]
    groups = normalized_data['group_id']

    y = y - 1  


    def objective(trial):
        param = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
        }

        model = xgb.XGBClassifier(**param)

        gkf = GroupKFold(n_splits=5)
        scores = []

        for train_idx, test_idx in gkf.split(x, y, groups):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            y_pred = y_pred + 1
            y_test = y_test + 1

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

        mean_accuracy = sum(scores) / len(scores)

        return mean_accuracy

    # Optuna çalışma
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)  # 50 deneme

    print("\n✅ EN İYİ PARAMETRELER:")
    print(study.best_params)
    print(f"✅ EN İYİ Mean Accuracy: {study.best_value:.4f}")

    # En iyi parametrelerle final modeli döndür
    best_params = study.best_params
    best_params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42
    })

    best_model = xgb.XGBClassifier(**best_params)
    return best_model

def train_xgboost_base_model(): # Ideal for New_Dataset_Vector_v2_PCA7_PCA13_20.csv
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=250,
        subsample=0.7
    )
    return model