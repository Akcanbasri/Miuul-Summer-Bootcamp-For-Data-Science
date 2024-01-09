################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
    RandomizedSearchCV,
    validation_curve,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

warnings.simplefilter(action="ignore", category=Warning)

df = pd.read_csv("diabetes.csv")


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(
    rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)
cv_results["test_accuracy"].mean()
# 0.753896103896104 - başlangıç değeri
cv_results["test_f1"].mean()
# 0.6190701534636385 - başlangıç değeri
cv_results["test_roc_auc"].mean()
# 0.8233960113960114 - başlangıç değeri


rf_params = {
    "max_depth": [5, 8, None],
    "max_features": [3, 5, 7, "auto"],
    "min_samples_split": [2, 5, 8, 15, 20],
    "n_estimators": [100, 200, 500],
}

# cv 10 olabilir
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(
    X, y
)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(
    rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"]
)
cv_results["test_accuracy"].mean()
# 0.753896103896104 - başlangıç değeri
# 0.766848940533151 - hiperparametrelerle
cv_results["test_f1"].mean()
# 0.6190701534636385 - başlangıç değeri
# 0.6447777811143756 - hiperparametrelerle
cv_results["test_roc_auc"].mean()
# 0.8233960113960114 - başlangıç değeri
# 0.8271054131054132 - hiperparametrelerle


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features.columns}
    )
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_final, X)


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=cv,
    )

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")

    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)


val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")


################################################
# GBM
################################################

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

cv_results = cross_validate(
    gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)
cv_results["test_accuracy"].mean()
# 0.7591715474068416
cv_results["test_f1"].mean()
# 0.634
cv_results["test_roc_auc"].mean()
# 0.82548

gbm_params = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 8, 10],
    "n_estimators": [100, 500, 1000],
    "subsample": [1, 0.5, 0.7],
}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(
    X, y
)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(
    **gbm_best_grid.best_params_,
    random_state=17,
).fit(X, y)


cv_results = cross_validate(
    gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)
cv_results["test_accuracy"].mean()
# 0.7800186741363212
cv_results["test_f1"].mean()
# 0.668605747317776
cv_results["test_roc_auc"].mean()
# 0.8257784765897973


################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(
    xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)
cv_results["test_accuracy"].mean()
# 0.75265
cv_results["test_f1"].mean()
# 0.631
cv_results["test_roc_auc"].mean()
# 0.7987

xgboost_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8],
    "n_estimators": [100, 500, 1000],
    "colsample_bytree": [0.7, 1],
}

xgboost_best_grid = GridSearchCV(
    xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True
).fit(X, y)

xgboost_final = xgboost_model.set_params(
    **xgboost_best_grid.best_params_, random_state=17
).fit(X, y)

cv_results = cross_validate(
    xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)
cv_results["test_accuracy"].mean()
# 0.7578643578643579
cv_results["test_f1"].mean()
# 0.6297649135382188
cv_results["test_roc_auc"].mean()
# 0.8145597484276731


################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(
    lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
# 0.7474492827434004
cv_results["test_f1"].mean()
# 0.624110522144179
cv_results["test_roc_auc"].mean()
# 0.7990293501048218

lgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [100, 300, 500, 1000],
    "colsample_bytree": [0.5, 0.7, 1],
}

lgbm_best_grid = GridSearchCV(
    lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True
).fit(X, y)
lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(
    X, y
)

cv_results = cross_validate(
    lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
# 0.7643578643578645
cv_results["test_f1"].mean()
# 0.6372062920577772
cv_results["test_roc_auc"].mean()
# 0.8147491264849755

# Hiperparametre yeni değerlerle
lgbm_params = {
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "n_estimators": [200, 300, 350, 400],
    "colsample_bytree": [0.9, 0.8, 1],
}

lgbm_best_grid = GridSearchCV(
    lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True
).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(
    X, y
)
lgbm_best_grid.best_params_

cv_results = cross_validate(
    lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
# 0.7643833290892115
cv_results["test_f1"].mean()
# 0.6193071162618689
cv_results["test_roc_auc"].mean()
# 0.8227931516422082

# Hiperparametre optimizasyonu sadece n_estimators için.
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

lgbm_best_grid = GridSearchCV(
    lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True
).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(
    X, y
)

cv_results = cross_validate(
    lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
# 0.7643833290892115
cv_results["test_f1"].mean()
# 0.6193071162618689
cv_results["test_roc_auc"].mean()
# 0.8227931516422082


################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(
    catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
# 0.7735251676428148
cv_results["test_f1"].mean()
# 0.6502723851348231
cv_results["test_roc_auc"].mean()
# 0.8378923829489867


catboost_params = {
    "iterations": [200, 500],
    "learning_rate": [0.01, 0.1],
    "depth": [3, 6],
}


catboost_best_grid = GridSearchCV(
    catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True
).fit(X, y)

catboost_final = catboost_model.set_params(
    **catboost_best_grid.best_params_, random_state=17
).fit(X, y)
catboost_best_grid.best_params_

cv_results = cross_validate(
    catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
# 0.7721755368814192
cv_results["test_f1"].mean()
# 0.6322580676028952
cv_results["test_roc_auc"].mean()
# 0.842001397624039


################################################
# Feature Importance
################################################


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features.columns}
    )
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)


################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {
    "max_depth": np.random.randint(5, 50, 10),
    "max_features": [3, 5, 7, "auto", "sqrt"],
    "min_samples_split": np.random.randint(2, 50, 20),
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)],
}

rf_random = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_random_params,
    n_iter=100,  # denenecek parametre sayısı
    cv=3,
    verbose=True,
    random_state=42,
    n_jobs=-1,
)

rf_random.fit(X, y)


rf_random.best_params_


rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(
    X, y
)

cv_results = cross_validate(
    rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
# 0.7761225702402172
cv_results["test_f1"].mean()
0.6476828661140268
cv_results["test_roc_auc"].mean()
# 0.8329608665269042


################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=cv,
    )

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")

    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)


rf_val_params = [
    ["max_depth", [5, 8, 15, 20, 30, None]],
    ["max_features", [3, 5, 7, "auto"]],
    ["min_samples_split", [2, 5, 8, 15, 20]],
    ["n_estimators", [10, 50, 100, 200, 500]],
]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]
