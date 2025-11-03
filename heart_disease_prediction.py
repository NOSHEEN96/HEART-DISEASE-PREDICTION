# Heart Disease Prediction 
# Author: Nosheen Akhter

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance


DATA_PATH = "heart_disease_data.csv"  
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
ROC_FIG = "roc_curve.png"
CM_FIG  = "confusion_matrix.png"


def ensure_dataframe(X, col_names=None):
    """Ensure X is a pandas DataFrame with column names."""
    if isinstance(X, pd.DataFrame):
        return X
    X = pd.DataFrame(X)
    if col_names is not None and len(col_names) == X.shape[1]:
        X.columns = list(col_names)
    return X

def build_preprocessor():
    """Preprocess numeric (impute+scale) and categorical (impute+onehot)."""
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
       
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_tf, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop"  
    )
    return pre

def cv_model_select(pre, X_train, y_train, random_state=RANDOM_STATE, folds=CV_FOLDS):
    """Compare candidate models via CV AUC, return best (name, pipeline, mean_auc)."""
    candidates = {
        "LogReg": LogisticRegression(max_iter=2000, solver="saga"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "GBDT": GradientBoostingClassifier(random_state=random_state),
        "SVC": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    best_name, best_pipe, best_auc = None, None, -np.inf
    rows = []

   
    X_train = ensure_dataframe(X_train, getattr(X_train, "columns", None))

    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        aucs, accs = [], []

        for tr, va in cv.split(X_train, y_train):
            Xtr = ensure_dataframe(X_train.iloc[tr], X_train.columns)
            Xva = ensure_dataframe(X_train.iloc[va], X_train.columns)
            ytr, yva = y_train.iloc[tr], y_train.iloc[va]

            pipe.fit(Xtr, ytr)

           
            try:
                proba = pipe.predict_proba(Xva)[:, 1]
            except Exception:
                try:
                    sc = pipe.decision_function(Xva)
                    proba = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
                except Exception:
                    proba = pipe.predict(Xva).astype(float)

            aucs.append(roc_auc_score(yva, proba))
            accs.append(accuracy_score(yva, pipe.predict(Xva)))

        mean_auc = float(np.mean(aucs))
        mean_acc = float(np.mean(accs))
        rows.append({"Model": name, "CV_AUC": mean_auc, "CV_Acc": mean_acc})

        if mean_auc > best_auc:
            best_name, best_pipe, best_auc = name, Pipeline([("pre", pre), ("model", est)]), mean_auc

    cv_table = pd.DataFrame(rows).sort_values("CV_AUC", ascending=False).reset_index(drop=True)
    return best_name, best_pipe, best_auc, cv_table

def evaluate_and_plot(pipe, X_test, y_test, roc_path=ROC_FIG, cm_path=CM_FIG):
    """Evaluate on test set, save ROC + Confusion Matrix, return metrics dict."""
    X_test = ensure_dataframe(X_test, getattr(X_test, "columns", None))

    y_pred = pipe.predict(X_test)

    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            sc = pipe.decision_function(X_test)
            y_proba = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
        except Exception:
            y_proba = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve (Test)")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

 
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return {"accuracy": acc, "auc": auc, "cm": cm, "report": report,
            "roc_path": roc_path, "cm_path": cm_path}

def permutation_feature_importance(pipe, X_test, y_test, top_k=20, random_state=RANDOM_STATE):
    """Permutation importance across model types (keeps DataFrame)."""
    X_test = ensure_dataframe(X_test, getattr(X_test, "columns", None))
    r = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=random_state)

    base_cols = list(X_test.columns)
 
    L = min(len(base_cols), len(r.importances_mean))
    imp_df = pd.DataFrame({
        "Feature": base_cols[:L],
        "Importance": r.importances_mean[:L]
    }).sort_values("Importance", ascending=False).head(top_k).reset_index(drop=True)
    return imp_df

def predict_one(pipe, X_cols, features):
    """
    Predict one example.
    - If tuple/list/np.array -> must match CSV column order (X_cols).
    - If dict -> keys are column names.
    Returns (label, prob_of_class_1_or_None)
    """
    if isinstance(features, (list, tuple, np.ndarray)):
        arr = np.asarray(features).reshape(1, -1)
        df_one = pd.DataFrame(arr, columns=X_cols)
    else:
        df_one = pd.DataFrame([features], columns=X_cols)

    try:
        proba = pipe.predict_proba(df_one)[:, 1][0]
    except Exception:
        proba = None
    label = int(pipe.predict(df_one)[0])
    return label, (float(proba) if proba is not None else None)

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns:
        raise ValueError("Your CSV must contain a 'target' column (0/1).")

   
    X = df.drop(columns=["target"])
    y = df["target"]

    X = ensure_dataframe(X, X.columns)  

    pre = build_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    best_name, best_pipe, best_auc, cv_table = cv_model_select(pre, X_train, y_train)
    print("\n=== CV Model Comparison ===")
    print(cv_table)

    best_pipe.fit(ensure_dataframe(X_train, X.columns), y_train)

    metrics = evaluate_and_plot(best_pipe, X_test, y_test)
    print(f"\nBest model: {best_name}")
    print(f"Test Accuracy: {metrics['accuracy']:.3f}")
    print(f"Test ROC AUC: {metrics['auc']:.3f}")
    print("Confusion Matrix:\n", metrics["cm"])
    print("\nClassification Report:\n", metrics["report"])
    print(f"Saved ROC to: {metrics['roc_path']}")
    print(f"Saved Confusion Matrix to: {metrics['cm_path']}")

    
    imp_df = permutation_feature_importance(best_pipe, X_test, y_test, top_k=20)
    print("\n=== Top Features (Permutation Importance) ===")
    print(imp_df.to_string(index=False))

    if X.shape[1] == 13:
        demo_input = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
        label, prob = predict_one(best_pipe, list(X.columns), demo_input)
        print("\nDemo predict_one:")
        print(" Input:", demo_input)
        print(f" Predicted label: {label}, Prob(class=1): {prob}")

if __name__ == "__main__":
    main()
