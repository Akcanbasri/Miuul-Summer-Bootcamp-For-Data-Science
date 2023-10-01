################################################
# Decision Tree Classification: CART
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model


# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate,
    validation_curve,
)
from skompiler import skompile
import graphviz

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

warnings.simplefilter(action="ignore", category=Warning)


################################################
# 1. Exploratory Data Analysis
################################################

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

################################################
# 3. Modeling using CART
################################################

diabetes = pd.read_csv("./datasets/diabetes.csv")
df = diabetes.copy()

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# confusian matrix için y_pred değerleri
y_pred = cart_model.predict(X)

# AUC için y_prob değerleri
y_prob = cart_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))

# AUC Score
roc_auc_score(y, y_prob)

#####3############
# 1.Hold Out Yöntemi ile Model Doğrulama
###################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=45
)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

#####################
# CV ile Başarı Değerlendirme
#####################

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(
    cart_model, X, y, cv=5, scoring=["roc_auc", "accuracy", "recall", "precision", "f1"]
)


cv_results["test_accuracy"].mean()
# 0.7058568882098294
cv_results["test_f1"].mean()
# 0.5710621194523633
cv_results["test_roc_auc"].mean()
# 0.6719440950384347
cv_results["test_recall"].mean()
# 0.5598881900768693

################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################

cart_model.get_params()

cart_params = {"max_depth": range(1, 11), "min_samples_split": range(2, 20)}

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Tüm veri setine göre yapılması daha doğru olur.
cart_best_params = GridSearchCV(
    cart_model, cart_params, cv=5, n_jobs=-1, verbose=1, scoring="accuracy"
).fit(X, y)

print(cart_best_params.best_params_)
# for accuracy
# {'max_depth': 5, 'min_samples_split': 4}

# for f1
# {'max_depth': 4, 'min_samples_split': 2}

# for roc_auc
# {'max_depth': 5, 'min_samples_split': 19}


print(cart_best_params.best_score_)
# for accuracy
# 0.7500806383159324

# for f1
# 0.6395752751155839

# for roc_auc
# 0.8020768693221523

random = X.sample(1, random_state=45)

cart_best_params.predict(random)

################################################
# 5. Final Model
################################################

cart_final = DecisionTreeClassifier(
    **cart_best_params.best_params_, random_state=17
).fit(X, y)

cart_final.get_params()

# 2.nd way to create model
cart_final = cart_model.set_params(**cart_best_params.best_params_).fit(X, y)

cv_results = cross_validate(
    cart_final, X, y, cv=5, scoring=["roc_auc", "accuracy", "recall", "precision", "f1"]
)

cv_results["test_accuracy"].mean()
# 0.7058568882098294 before
# 0.7435446906035141 after

cv_results["test_f1"].mean()
# 0.5710621194523633 before
# 0.6045514145623818 after

cv_results["test_roc_auc"].mean()
# 0.6719440950384347 before
# 0.8020768693221523 after

cv_results["test_recall"].mean()
# 0.5598881900768693 before
# 0.5598881900768694 after

################################################
# 6. Feature Importance
################################################

cart_final.feature_importances_


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


plot_importance(cart_final, X, 10, save=True)


################################################
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
################################################




