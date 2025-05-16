import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, GridSearchCV
import lightgbm as lgb

def optimize_lightgbm_with_optuna(normalized_data):

    x = normalized_data.iloc[:, 1:-1]
    y = normalized_data.iloc[:, 0]
    groups = normalized_data['group_id']

    y = y - 1

    def objective(trial):
        param = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 3,
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
        }

        model = lgb.LGBMClassifier(**param)

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

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("\n✅ EN İYİ LightGBM PARAMETRELERİ:")
    print(study.best_params)
    print(f"✅ EN İYİ Mean Accuracy (LightGBM): {study.best_value:.4f}")

    best_params = study.best_params
    best_params.update({
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': 42
    })

    best_lgb_model = lgb.LGBMClassifier(**best_params)
    return best_lgb_model

def train_lightgbm_with_tuning(normalized_data):

    x = normalized_data.iloc[:, 1:-1]
    y = normalized_data.iloc[:, 0]
    groups = normalized_data['group_id']

    y = y - 1

    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass',
        num_class=3,
        random_state=42,
        verbose = -1
    )

    # GroupKFold ile CV yapıyoruz
    gkf = GroupKFold(n_splits=5)

    grid_search = GridSearchCV(
        estimator=lgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=gkf.split(x, y, groups),
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(x, y)

    print("\n✅ EN İYİ LightGBM PARAMETRELER:")
    print(grid_search.best_params_)
    print(f"✅ EN İYİ Accuracy (CV Mean): {grid_search.best_score_:.4f}")

    best_lgb_model = grid_search.best_estimator_

    return best_lgb_model

def train_lightgbm_base_model():

    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass',
        num_class=3,
        random_state=42,
        n_estimators=246,
        max_depth=3,
        learning_rate=0.035643315221163915,
        subsample=0.8890338275927295,
        colsample_bytree=0.9048865226480963,
        min_child_weight=8,
        reg_alpha=0.3178437027407188,
        reg_lambda=1.7404007794464775,
        verbose=-1
    )
    return lgb_model