# -*- coding: utf-8 -*-
"""
QSAR regression (endpoint: pEC50) with 60-bit binary fingerprints + exact mass
Train/Test: pre-defined in the provided Excel file

No information leakage:
- All preprocessing (scaling) and hyperparameter tuning are done using TRAIN ONLY inside CV.
- TEST is used only once for final evaluation.

Models (6):
1) KNN
2) ElasticNet
3) SVR (RBF)
4) Kernel Ridge (RBF)
5) Random Forest
6) HistGradientBoosting

CV (train only): RepeatedKFold
Outputs:
1) model_performance_train_test_with_bootstrapCI.xlsx (+csv)
2) points_for_figures.xlsx
3) Figure_a_observed_vs_predicted_6models.png
4) Figure_b_williams_6models.png
5) deploy_bundle/
   - svr_model.joblib
   - feature_config.json
   - model_meta.json
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

from scipy.stats import loguniform, randint
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


# =========================
# 1) User settings
# =========================
DATA_PATH = r"H:\LXX研究生工作\科研课题\机器学习课题研究\含氮杂环药物模型训练\模型训练\毒性数据库分层划分SVR特征1加EM - 修改2026.04.30.xlsx"
SHEET_NAME = "All_with_split"  # contains Set column (Train/Test)

TARGET_COL = "pEC50"
SET_COL = "Set"                # values: Train / Test
FEATURE_PREFIX = "fp_"         # fingerprint columns start with fp_

# 你现在想把 MW 替换为 exact mass
# 这里优先使用该列名；如果表里不是这个列名，会自动尝试若干常见别名
EXACT_MASS_COL = "ExactMass"

# 最终部署模型固定使用 60 个指纹
EXPECTED_N_FP = 60

# CV setting (TRAIN ONLY)
N_SPLITS = 10
N_REPEATS = 10
RANDOM_STATE = 42

# Bootstrap on test metrics
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42

# Output folder
OUT_DIR = "./qsar_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Plot style / colors
TRAIN_RGB = (167, 166, 166)
TEST_RGB  = (234, 163, 160)
TRAIN_COLOR = tuple([c / 255 for c in TRAIN_RGB])
TEST_COLOR  = tuple([c / 255 for c in TEST_RGB])

POINT_SIZE = 26
POINT_ALPHA = 0.85
LINE_WIDTH = 1.2


# =========================
# 2) Utilities
# =========================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bootstrap_ci_rmse_mae(y_true, y_pred, B=2000, seed=42):
    """
    Bootstrap CI on test metrics:
    resample test indices with replacement, recompute RMSE and MAE using fixed predictions.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)
    rmses = np.empty(B, dtype=float)
    maes = np.empty(B, dtype=float)

    for b in range(B):
        s = rng.choice(idx, size=n, replace=True)
        yt = y_true[s]
        yp = y_pred[s]
        rmses[b] = rmse(yt, yp)
        maes[b] = mean_absolute_error(yt, yp)

    rmse_ci = (np.percentile(rmses, 2.5), np.percentile(rmses, 97.5))
    mae_ci = (np.percentile(maes, 2.5), np.percentile(maes, 97.5))
    return rmse_ci, mae_ci


def format_params(d):
    if d is None:
        return ""
    return "; ".join([f"{k}={v}" for k, v in d.items()])


def set_axes_spines(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def put_model_label(ax, label):
    ax.text(
        0.95, 0.92, label,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=14, fontweight="bold"
    )


def calc_leverage_hat(X_train, X_all):
    """
    Williams plot leverage:
    Use standardized X (fit scaler on train only) + intercept.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xall = scaler.transform(X_all)

    Xtr_aug = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
    Xall_aug = np.column_stack([np.ones(Xall.shape[0]), Xall])

    A = np.linalg.pinv(Xtr_aug.T @ Xtr_aug)
    h = np.sum((Xall_aug @ A) * Xall_aug, axis=1)
    return h


def resolve_exact_mass_column(df, preferred_col="ExactMass"):
    """
    优先用 preferred_col；
    若不存在，则自动尝试常见 exact mass 别名。
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
    for c in candidates:
        if c not in seen:
            ordered_candidates.append(c)
            seen.add(c)

    for col in ordered_candidates:
        if col in df.columns:
            return col

    return None


# =========================
# 3) Load data + split
# =========================
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

# Choose an ID column for point tables (best effort)
if "CAS号" in df.columns:
    sample_id = df["CAS号"].astype(str).values
elif "英文名" in df.columns:
    sample_id = df["英文名"].astype(str).values
else:
    sample_id = df.index.astype(str).values

# Features: 60 fingerprint columns
feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
if len(feature_cols) != EXPECTED_N_FP:
    print(
        f"[WARN] Detected {len(feature_cols)} fingerprint columns starting with '{FEATURE_PREFIX}'. "
        f"Expected {EXPECTED_N_FP}. Please confirm your file."
    )

# Fingerprints matrix
X_fp = df[feature_cols].astype(float).values

# Exact mass as an additional feature
actual_exact_mass_col = resolve_exact_mass_column(df, preferred_col=EXACT_MASS_COL)
if actual_exact_mass_col is None:
    raise ValueError(
        f"Exact mass column not found. Preferred column was '{EXACT_MASS_COL}'. "
        f"Available columns (first 40): {list(df.columns)[:40]}"
    )

exact_mass = pd.to_numeric(df[actual_exact_mass_col], errors="coerce").values.astype(float)

if np.isnan(exact_mass).any():
    bad_n = int(np.isnan(exact_mass).sum())
    raise ValueError(
        f"Exact mass column '{actual_exact_mass_col}' has {bad_n} missing/non-numeric values. "
        f"Please fix them (no blanks/strings)."
    )

# Final input = [fingerprints..., exact mass]
X = np.column_stack([X_fp, exact_mass])

# Target
y = df[TARGET_COL].astype(float).values

# Split indices
train_mask = df[SET_COL].astype(str).str.lower().eq("train")
test_mask  = df[SET_COL].astype(str).str.lower().eq("test")

if train_mask.sum() != 90 or test_mask.sum() != 18:
    raise ValueError(
        f"Split counts mismatch: Train={train_mask.sum()}, Test={test_mask.sum()} (expected 90/20)."
    )

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

id_train = sample_id[train_mask]
id_test = sample_id[test_mask]

# CV splitter (training set only)
cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

# Common scoring
scoring = {
    "R2": "r2",
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
}


# =========================
# 4) Model definitions + search spaces
# =========================
models = []

# 1) PLS (replace KNN)
pls_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", PLSRegression())
])

# n_components must be <= min(n_samples-1, n_features)
# Here n_train=90, n_features≈61 (60 fp + exact mass), so 2..20 is safe and typical.
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

# 3) SVR (RBF)
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

# 4) Kernel Ridge (RBF)
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


# =========================
# 5) Train + evaluate + export point tables
# =========================
results = []
fitted_models = {}

points_xlsx_path = os.path.join(OUT_DIR, "points_for_figures.xlsx")

with pd.ExcelWriter(points_xlsx_path, engine="openpyxl") as points_writer:

    for name, pipe, search_type, param_space, n_iter in models:
        print(f"\n=== {name} ===")

        if search_type == "grid":
            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_space,
                scoring="neg_root_mean_squared_error",
                refit=True,
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )
        else:
            search = RandomizedSearchCV(
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

        # Fit search on TRAIN ONLY
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        best_params = search.best_params_

        # CV performance
        cv_out = cross_validate(
            best_est,
            X_train, y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        cv_r2_mean = np.mean(cv_out["test_R2"])
        cv_r2_sd = np.std(cv_out["test_R2"], ddof=1)

        cv_rmse = -cv_out["test_RMSE"]
        cv_rmse_mean = np.mean(cv_rmse)
        cv_rmse_sd = np.std(cv_rmse, ddof=1)

        cv_mae = -cv_out["test_MAE"]
        cv_mae_mean = np.mean(cv_mae)
        cv_mae_sd = np.std(cv_mae, ddof=1)

        # Fit final on full training set
        best_est.fit(X_train, y_train)

        # Predict train/test
        y_pred_train = np.asarray(best_est.predict(X_train)).ravel()
        y_pred_test  = np.asarray(best_est.predict(X_test)).ravel()

        # Metrics (train)
        train_r2 = r2_score(y_train, y_pred_train)
        train_rmse = rmse(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)

        # Metrics (test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = rmse(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Bootstrap CI
        (rmse_lo, rmse_hi), (mae_lo, mae_hi) = bootstrap_ci_rmse_mae(
            y_true=y_test,
            y_pred=y_pred_test,
            B=BOOTSTRAP_B,
            seed=BOOTSTRAP_SEED
        )

        results.append({
            "Model": name,
            "CV_R2_mean": cv_r2_mean, "CV_R2_sd": cv_r2_sd,
            "CV_RMSE_mean": cv_rmse_mean, "CV_RMSE_sd": cv_rmse_sd,
            "CV_MAE_mean": cv_mae_mean, "CV_MAE_sd": cv_mae_sd,
            "Train_R2": train_r2, "Train_RMSE": train_rmse, "Train_MAE": train_mae,
            "Test_R2": test_r2, "Test_RMSE": test_rmse, "Test_MAE": test_mae,
            "Test_RMSE_CI95_low": rmse_lo, "Test_RMSE_CI95_high": rmse_hi,
            "Test_MAE_CI95_low": mae_lo, "Test_MAE_CI95_high": mae_hi,
            "BestParams": format_params(best_params),
        })

        fitted_models[name] = {
            "estimator": best_est,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test
        }

        # Scatter export
        scatter_df = pd.DataFrame({
            "Set": np.array(["Train"] * len(y_train) + ["Test"] * len(y_test)),
            "ID": np.concatenate([id_train, id_test]),
            "Observed_pEC50": np.concatenate([y_train, y_test]),
            "Predicted_pEC50": np.concatenate([y_pred_train, y_pred_test]),
        })
        sheet_a = f"{name}_scatter"[:31]
        scatter_df.to_excel(points_writer, index=False, sheet_name=sheet_a)

    # Williams leverage
    h_all = calc_leverage_hat(X_train=X_train, X_all=X)
    h_train = h_all[train_mask]
    h_test = h_all[test_mask]

    p = X_train.shape[1]
    n = X_train.shape[0]
    h_star = 3.0 * (p + 1) / n

    const_df = pd.DataFrame({
        "p_features": [p],
        "n_train": [n],
        "h_star": [h_star],
        "StdResidual_threshold": ["±3"],
        "Bootstrap_B": [BOOTSTRAP_B],
        "CV_folds": [N_SPLITS],
        "CV_repeats": [N_REPEATS],
        "Random_state": [RANDOM_STATE],
        "ExactMass_Column_Used": [actual_exact_mass_col],
    })
    const_df.to_excel(points_writer, index=False, sheet_name="Constants")

    for name, _, _, _, _ in models:
        y_pred_train = fitted_models[name]["y_pred_train"]
        y_pred_test = fitted_models[name]["y_pred_test"]

        resid_train = y_train - y_pred_train
        resid_test = y_test - y_pred_test

        s = np.std(resid_train, ddof=1)
        if s == 0:
            s = 1e-12

        std_resid_train = resid_train / s
        std_resid_test = resid_test / s

        williams_df = pd.DataFrame({
            "Set": np.array(["Train"] * len(y_train) + ["Test"] * len(y_test)),
            "ID": np.concatenate([id_train, id_test]),
            "HAT": np.concatenate([h_train, h_test]),
            "Residual": np.concatenate([resid_train, resid_test]),
            "StdResidual": np.concatenate([std_resid_train, std_resid_test]),
        })

        sheet_b = f"{name}_williams"[:31]
        williams_df.to_excel(points_writer, index=False, sheet_name=sheet_b)

print(f"\nSaved point tables Excel: {points_xlsx_path}")


# =========================
# 6) Save metrics table
# =========================
res_df = pd.DataFrame(results)
col_order = [
    "Model",
    "CV_R2_mean", "CV_R2_sd", "CV_RMSE_mean", "CV_RMSE_sd", "CV_MAE_mean", "CV_MAE_sd",
    "Train_R2", "Train_RMSE", "Train_MAE",
    "Test_R2", "Test_RMSE", "Test_RMSE_CI95_low", "Test_RMSE_CI95_high",
    "Test_MAE", "Test_MAE_CI95_low", "Test_MAE_CI95_high",
    "BestParams"
]
res_df = res_df[col_order].sort_values("Test_RMSE", ascending=True).reset_index(drop=True)

csv_path = os.path.join(OUT_DIR, "model_performance_train_test_with_bootstrapCI.csv")
xlsx_path = os.path.join(OUT_DIR, "model_performance_train_test_with_bootstrapCI.xlsx")
res_df.to_csv(csv_path, index=False)
res_df.to_excel(xlsx_path, index=False)

print("\nSaved metrics table:")
print(" -", csv_path)
print(" -", xlsx_path)
print("\nTop rows (sorted by Test_RMSE):")
print(
    res_df[[
        "Model", "Test_R2", "Test_RMSE", "Test_MAE",
        "Test_RMSE_CI95_low", "Test_RMSE_CI95_high",
        "Test_MAE_CI95_low", "Test_MAE_CI95_high"
    ]].head(10)
)


# =========================
# 7) Plot Figure (a)
# =========================
fig_a, axes_a = plt.subplots(3, 2, figsize=(11.5, 12.5), dpi=300)
axes_a = axes_a.flatten()

model_order = [m[0] for m in models]
for i, name in enumerate(model_order):
    ax = axes_a[i]
    yptr = fitted_models[name]["y_pred_train"]
    ypte = fitted_models[name]["y_pred_test"]

    ax.scatter(y_train, yptr, s=POINT_SIZE, c=[TRAIN_COLOR], alpha=POINT_ALPHA,
               edgecolors="none", label="Training set")
    ax.scatter(y_test, ypte, s=POINT_SIZE, c=[TEST_COLOR], alpha=POINT_ALPHA,
               edgecolors="none", label="Test set")

    all_y = np.concatenate([y_train, y_test, yptr, ypte])
    y_min, y_max = np.min(all_y), np.max(all_y)
    pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.1
    lo, hi = y_min - pad, y_max + pad

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=LINE_WIDTH)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Observed pEC50")
    ax.set_ylabel("Predicted pEC50")

    put_model_label(ax, name)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    set_axes_spines(ax)

fig_a.tight_layout()
fig_a_path = os.path.join(OUT_DIR, "Figure_a_observed_vs_predicted_6models.png")
fig_a.savefig(fig_a_path, bbox_inches="tight")
print("\nSaved figure (a):", fig_a_path)


# =========================
# 8) Plot Figure (b)
# =========================
h_all = calc_leverage_hat(X_train=X_train, X_all=X)
h_train = h_all[train_mask]
h_test = h_all[test_mask]

p = X_train.shape[1]
n = X_train.shape[0]
h_star = 3.0 * (p + 1) / n

fig_b, axes_b = plt.subplots(3, 2, figsize=(11.5, 12.5), dpi=300)
axes_b = axes_b.flatten()

for i, name in enumerate(model_order):
    ax = axes_b[i]
    yptr = fitted_models[name]["y_pred_train"]
    ypte = fitted_models[name]["y_pred_test"]

    resid_train = y_train - yptr
    resid_test = y_test - ypte

    s = np.std(resid_train, ddof=1)
    if s == 0:
        s = 1e-12

    std_resid_train = resid_train / s
    std_resid_test = resid_test / s

    ax.scatter(h_train, std_resid_train, s=POINT_SIZE, c=[TRAIN_COLOR], alpha=POINT_ALPHA,
               edgecolors="none", label="Training set")
    ax.scatter(h_test, std_resid_test, s=POINT_SIZE, c=[TEST_COLOR], alpha=POINT_ALPHA,
               edgecolors="none", label="Test set")

    ax.axhline(3, color="k", linestyle="--", linewidth=LINE_WIDTH)
    ax.axhline(-3, color="k", linestyle="--", linewidth=LINE_WIDTH)
    ax.axvline(h_star, color="k", linestyle="--", linewidth=LINE_WIDTH)

    ax.set_xlabel("HAT")
    ax.set_ylabel("Std. residuals")

    put_model_label(ax, name)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    set_axes_spines(ax)

    ax.set_xlim(left=0, right=max(0.32, float(np.max(h_all) * 1.05)))
    ax.set_ylim(-4.2, 4.2)

fig_b.tight_layout()
fig_b_path = os.path.join(OUT_DIR, "Figure_b_williams_6models.png")
fig_b.savefig(fig_b_path, bbox_inches="tight")
print("Saved figure (b):", fig_b_path)

print("\nAll outputs saved in:", os.path.abspath(OUT_DIR))


# =========================
# 9) Export deployable SVR bundle (with exact mass)
# =========================
DEPLOY_DIR = "./deploy_bundle"
os.makedirs(DEPLOY_DIR, exist_ok=True)

FINAL_MODEL_NAME = "SVR_RBF"
final_estimator = fitted_models[FINAL_MODEL_NAME]["estimator"]

final_fp_features = feature_cols
if len(final_fp_features) != EXPECTED_N_FP:
    raise ValueError(
        f"当前导出的指纹特征数为 {len(final_fp_features)}，不是目标的 {EXPECTED_N_FP}。"
    )

# 最终模型输入顺序：60个fp + exact mass
model_input_order = final_fp_features + [actual_exact_mass_col]

# 1) 保存模型
model_path = os.path.join(DEPLOY_DIR, "svr_model.joblib")
joblib.dump(final_estimator, model_path)

# 2) 保存部署配置
feature_config = {
    "model_name": FINAL_MODEL_NAME,
    "target_column": TARGET_COL,
    "feature_prefix": FEATURE_PREFIX,

    # 指纹部分
    "n_fp_features": len(final_fp_features),
    "feature_order": final_fp_features,

    # exact mass 部分
    "include_exact_mass": True,
    "exact_mass_col": actual_exact_mass_col,

    # 最终输入
    "n_features_total": len(model_input_order),
    "model_input_order": model_input_order
}

feature_path = os.path.join(DEPLOY_DIR, "feature_config.json")
with open(feature_path, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, ensure_ascii=False, indent=2)

# 3) 元信息
model_meta = {
    "sheet_name": SHEET_NAME,
    "set_column": SET_COL,
    "target_column": TARGET_COL,
    "random_state": RANDOM_STATE,
    "cv_splits": N_SPLITS,
    "cv_repeats": N_REPEATS,
    "bootstrap_B": BOOTSTRAP_B,
    "mass_feature_type": "exact_mass",
    "exact_mass_col_used": actual_exact_mass_col,
    "notes": "Deployable SVR model for batch pEC50 prediction with 60 fingerprints + exact mass"
}

meta_path = os.path.join(DEPLOY_DIR, "model_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(model_meta, f, ensure_ascii=False, indent=2)

print("\n[OK] Deploy bundle exported:")
print(" -", model_path)
print(" -", feature_path)
print(" -", meta_path)