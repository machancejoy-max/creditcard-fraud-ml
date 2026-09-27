import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE

from src.utils.data_loader import load_data
from src.utils.preprocessing import split_and_scale

def main():
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Hyperparameter tuning
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    search.fit(X_train_res, y_train_res)
    best_model = search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    print("Best Parameters:", search.best_params_)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_proba))

    # Save model + scaler
    joblib.dump(best_model, "src/models/xgb_model.joblib")
    joblib.dump(scaler, "src/models/scaler.joblib")

if __name__ == "__main__":
    main()
