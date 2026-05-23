"""
QSAR regression for pEC50 prediction using 60 binary fingerprints plus exact mass.

Workflow
--------
1. Read the first worksheet from the input Excel file.
2. Use the predefined Train/Test split stored in the "Set" column.
3. Tune hyperparameters on the training set only with repeated k-fold CV.
4. Refit the best estimator on the full training set.
5. Evaluate each model on the training set and the independent test set.
6. Rank models by final test-set performance and print the summary.
7. Save one final model file.

Models
------
1. PLS
2. ElasticNet
3. SVR (RBF)
4. Kernel Ridge (RBF)
5. Random Forest
6. HistGradientBoosting

Console output
--------------
- Model name
- Best hyperparameters
- Training-set performance
- Independent test-set performance
- Ranking of all models from best to worst

File output
-----------
- One saved model file only (.joblib)
"""

from pathlib import Path
import warnings
import json

import joblib
import numpy as np
import pandas as pd

from scipy.stats import loguniform, randint
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


# =========================
# 1) User settings
# =========================
DATA_PATH = Path("data/model_input.xlsx")
MODEL_OUT_DIR = Path("models")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "pEC50"
SET_COL = "Set"
FEATURE_PREFIX = "fp_"
EXACT_MASS_COL = "ExactMass"
EXPECTED_N_FP = 60

N_SPLITS = 10
N_REPEATS = 10
RANDOM_STATE = 42

# Save mode:
# - "top_ranked": save the best-ranked model based on final test performance
# - or provide one explicit model name, e.g. "SVR_RBF"
SAVE_MODE = "top_ranked"


# =========================
# 2) Utility functions
# =========================
def rmse(y_true, y_pred):
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def resolve_exact_mass_column(df, preferred_col="ExactMass"):
    """
    Return the exact-mass column name.

    The preferred column name is checked first. If it is not found,
    several common aliases are tried automatically.
    """
    candidates = [
        preferred_col,
        "ExactMass",
        "exact_mass",
        "exact mass",
        "Exact_Mass",
        "MonoisotopicMass",
        "Monoisotopic_Mass",
        "Monoisotopic Mass",
        "Monoisotopic mass",
        "Exact mass",
    ]

    seen = set()
    ordered_candidates = []
    for col in candidates:
        if col not in seen:
            ordered_candidates.append(col)
            seen.add(col)

    for col in ordered_candidates:
        if col in df.columns:
            return col

    return None


def format_params(params):
    """Return a stable JSON-like string for hyperparameters."""
    return json.dumps(params, sort_keys=True, default=str)


def ranking_key(row):
    """
    Ranking rule for final model comparison.

    Higher Test_R2 is better.
    If Test_R2 is tied, lower Test_RMSE is better.
    If Test_RMSE is tied, lower Test_MAE is better.
    """
    return (-row["Test_R2"], row["Test_RMSE"], row["Test_MAE"])


# =========================
# 3) Data loading
# =========================
def load_data():
    """
    Load the input dataset from the first worksheet of the Excel file.

    Expected columns:
    - fingerprint columns starting with FEATURE_PREFIX
    - one exact-mass column
    - target column
    - split column with Train/Test labels
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input file was not found: {DATA_PATH.name}. "
            f"Place the file under the configured data directory."
        )

    df = pd.read_excel(DATA_PATH)

    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if len(feature_cols) != EXPECTED_N_FP:
        print(
            f"[WARNING] Detected {len(feature_cols)} fingerprint columns starting with "
            f"'{FEATURE_PREFIX}', while {EXPECTED_N_FP} were expected."
        )

    x_fp = df[feature_cols].astype(float).values

    actual_exact_mass_col = resolve_exact_mass_column(df, preferred_col=EXACT_MASS_COL)
    if actual_exact_mass_col is None:
        raise ValueError(
            "Exact mass column was not found. "
            "Please provide a valid exact-mass column in the input file."
        )

    exact_mass = pd.to_numeric(df[actual_exact_mass_col], errors="coerce").values.astype(float)
    if np.isnan(exact_mass).any():
        bad_n = int(np.isnan(exact_mass).sum())
        raise ValueError(
            f"The exact-mass column contains {bad_n} missing or non-numeric values."
        )

    x = np.column_stack([x_fp, exact_mass])
    y = df[TARGET_COL].astype(float).values

    train_mask = df[SET_COL].astype(str).str.lower().eq("train")
    test_mask = df[SET_COL].astype(str).str.lower().eq("test")

    if train_mask.sum() != 90 or test_mask.sum() != 18:
        raise ValueError(
            f"Split counts mismatch: Train={train_mask.sum()}, Test={test_mask.sum()} "
            f"(expected 90 and 18 for this study dataset)."
        )

    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    return x_train, y_train, x_test, y_test, actual_exact_mass_col


# =========================
# 4) Model definitions
# =========================
def get_models():
    """
    Return model specifications as tuples:
    (name, pipeline, search_type, param_space, n_iter)
    """
    models = []

    # 1) PLS
    pls_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", PLSRegression())
    ])
    pls_param = {
        "model__n_components": list(range(2, 21))
    }
    models.append(("PLS", pls_pipe, "random", pls_param, 30))

    # 2) ElasticNet
    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=50000, random_state=RANDOM_STATE))
    ])
    enet_param = {
        "model__alpha": np.logspace(-4, 1, 20),
        "model__l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    }
    models.append(("ElasticNet", enet_pipe, "grid", enet_param, None))

    # 3) SVR
    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf"))
    ])
    svr_param = {
        "model__C": loguniform(1e-2, 1e2),
        "model__gamma": loguniform(1e-6, 1e-1),
        "model__epsilon": loguniform(1e-2, 1e0),
    }
    models.append(("SVR_RBF", svr_pipe, "random", svr_param, 80))

    # 4) Kernel Ridge
    krr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KernelRidge(kernel="rbf"))
    ])
    krr_param = {
        "model__alpha": np.logspace(-4, 2, 200),
        "model__gamma": np.logspace(-4, 0, 200),
    }
    models.append(("KRR_RBF", krr_pipe, "random", krr_param, 30))

    # 5) Random Forest
    rf_pipe = Pipeline([
        ("model", RandomForestRegressor(random_state=RANDOM_STATE))
    ])
    rf_param = {
        "model__n_estimators": randint(300, 1201),
        "model__max_depth": [None] + list(range(3, 26)),
        "model__min_samples_split": randint(2, 21),
        "model__min_samples_leaf": randint(1, 11),
        "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    }
    models.append(("RF", rf_pipe, "random", rf_param, 30))

    # 6) HistGradientBoosting
    hgb_pipe = Pipeline([
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))
    ])
    hgb_param = {
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 11),
        "model__max_leaf_nodes": randint(15, 256),
        "model__min_samples_leaf": randint(5, 31),
        "model__l2_regularization": loguniform(1e-6, 1e-1),
    }
    models.append(("HistGB", hgb_pipe, "random", hgb_param, 40))

    return models


def build_search(pipe, search_type, param_space, n_iter, cv):
    """Build the hyperparameter search object."""
    if search_type == "grid":
        return GridSearchCV(
            estimator=pipe,
            param_grid=param_space,
            scoring="neg_root_mean_squared_error",
            refit=True,
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        refit=True,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )


# =========================
# 5) Model training and evaluation
# =========================
def evaluate_models(x_train, y_train, x_test, y_test):
    """
    Train all models, tune them on the training set only,
    refit the best estimator, and evaluate train/test performance.
    """
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    results = []
    fitted_models = {}

    for name, pipe, search_type, param_space, n_iter in get_models():
        search = build_search(pipe, search_type, param_space, n_iter, cv)
        search.fit(x_train, y_train)

        best_estimator = search.best_estimator_
        best_params = search.best_params_

        best_estimator.fit(x_train, y_train)

        y_pred_train = np.asarray(best_estimator.predict(x_train)).ravel()
        y_pred_test = np.asarray(best_estimator.predict(x_test)).ravel()

        result = {
            "Model": name,
            "BestParamsDict": best_params,
            "BestParamsText": format_params(best_params),
            "Train_R2": r2_score(y_train, y_pred_train),
            "Train_RMSE": rmse(y_train, y_pred_train),
            "Train_MAE": mean_absolute_error(y_train, y_pred_train),
            "Test_R2": r2_score(y_test, y_pred_test),
            "Test_RMSE": rmse(y_test, y_pred_test),
            "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        }

        results.append(result)
        fitted_models[name] = best_estimator

    results = sorted(results, key=ranking_key)

    return results, fitted_models


# =========================
# 6) Reporting and saving
# =========================
def print_ranked_results(results):
    """Print ranked model results from best to worst."""
    print("=" * 72)
    print("Model ranking based on final independent test-set performance")
    print("Ranking rule: higher Test_R2, then lower Test_RMSE, then lower Test_MAE")
    print("=" * 72)

    for rank, row in enumerate(results, start=1):
        print(f"\nRank {rank}: {row['Model']}")
        print(f"Best hyperparameters: {row['BestParamsText']}")
        print(
            "Training set performance: "
            f"R2={row['Train_R2']:.4f}, "
            f"RMSE={row['Train_RMSE']:.4f}, "
            f"MAE={row['Train_MAE']:.4f}"
        )
        print(
            "Independent test-set performance: "
            f"R2={row['Test_R2']:.4f}, "
            f"RMSE={row['Test_RMSE']:.4f}, "
            f"MAE={row['Test_MAE']:.4f}"
        )


def save_final_model(results, fitted_models):
    """
    Save one final model file.

    If SAVE_MODE == "top_ranked", save the highest-ranked model.
    Otherwise, save the explicitly named model.
    """
    if SAVE_MODE == "top_ranked":
        selected_model_name = results[0]["Model"]
    else:
        selected_model_name = SAVE_MODE
        if selected_model_name not in fitted_models:
            raise ValueError(
                f"Requested model '{selected_model_name}' was not found in fitted models."
            )

    model_filename = f"{selected_model_name}.joblib"
    model_path = MODEL_OUT_DIR / model_filename
    joblib.dump(fitted_models[selected_model_name], model_path)

    print("\n" + "=" * 72)
    print(f"Saved final model file: {model_filename}")
    print("=" * 72)


# =========================
# 7) Main
# =========================
def main():
    x_train, y_train, x_test, y_test, actual_exact_mass_col = load_data()

    print("=" * 72)
    print("Dataset loaded successfully")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Exact-mass column used: {actual_exact_mass_col}")
    print("=" * 72)

    results, fitted_models = evaluate_models(x_train, y_train, x_test, y_test)
    print_ranked_results(results)
    save_final_model(results, fitted_models)


if __name__ == "__main__":
    main()