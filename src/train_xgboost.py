import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from src.utils.data_loader import load_data
from src.utils.preprocessing import split_and_scale

def main():
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    joblib.dump(model, "src/models/xgb_model.joblib")
    joblib.dump(scaler, "src/models/scaler.joblib")

if __name__ == "__main__":
    main()
