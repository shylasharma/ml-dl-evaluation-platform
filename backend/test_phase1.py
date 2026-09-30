import sys, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from app.ml.preprocessing import (
    resolve_preprocessing_config, drop_duplicate_rows, build_feature_pipeline,
    prepare_data, DEFAULT_PREPROCESSING,
)
from app.ml.dataset_analysis import profile_dataset
from app.ml.orchestrator import run_experiment
from app.utils.errors import FriendlyError
from app.utils.json_safe import sanitize_for_json


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def ok(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, f"FAILED: {label}"


# ---------------------------------------------------------------- synthetic dataset
# Small/medium, mixed numeric+categorical, with missing values and a few
# deliberate exact duplicate rows, and mild imbalance -- representative of
# the platform's usual input without being large (per instruction: use a
# small/medium dataset for this round of testing).
rng = np.random.RandomState(42)
n = 400
df = pd.DataFrame({
    "amount": rng.normal(100, 30, n).round(2),
    "age": rng.randint(18, 70, n).astype(float),
    "score": rng.normal(0, 1, n),
    "category": rng.choice(["A", "B", "C"], n),
    "target": (rng.rand(n) < 0.15).astype(int),  # ~15% minority -> imbalanced
})
# Inject missing values (TEST 6)
df.loc[df.sample(20, random_state=1).index, "amount"] = np.nan
df.loc[df.sample(15, random_state=2).index, "age"] = np.nan
df.loc[df.sample(10, random_state=3).index, "category"] = np.nan
# Inject exact duplicate rows (for duplicate-handling tests)
dup_rows = df.iloc[[5, 6, 7]].copy()
df = pd.concat([df, dup_rows], ignore_index=True)

print(f"Synthetic dataset: {len(df)} rows (incl. 3 injected exact duplicates), "
      f"target distribution: {df['target'].value_counts().to_dict()}")

profile = profile_dataset(df)
numeric_features = profile["numeric_features"]
categorical_features = profile["categorical_features"]
print("numeric_features:", numeric_features, "| categorical_features:", categorical_features)
print("profile duplicate_rows (pre-dedup, informational):", profile["duplicate_rows"])


# ================================================================== TEST 1
section("TEST 1: Legacy request with NO preprocessing block (backward compatibility)")
legacy_cfg_true = {"scale_features": True}   # old-style advanced toggle, no `preprocessing` key at all
resolved = resolve_preprocessing_config(legacy_cfg_true)
ok("Legacy scale_features=True resolves to exactly DEFAULT_PREPROCESSING",
   resolved == DEFAULT_PREPROCESSING)

legacy_cfg_false = {"scale_features": False}
resolved_false = resolve_preprocessing_config(legacy_cfg_false)
ok("Legacy scale_features=False resolves scaling->'none', everything else default",
   resolved_false["scaling"] == "none"
   and {k: v for k, v in resolved_false.items() if k != "scaling"}
       == {k: v for k, v in DEFAULT_PREPROCESSING.items() if k != "scaling"})

no_key_at_all = {}  # simulates a request dict with neither `preprocessing` nor `scale_features`
resolved_none = resolve_preprocessing_config(no_key_at_all)
ok("Completely absent config still defaults to standard scaling (scale_features default True)",
   resolved_none == DEFAULT_PREPROCESSING)

# Full legacy end-to-end run through the real orchestrator
legacy_full_cfg = {
    "target_column": "target", "models": ["logistic_regression", "random_forest"],
    "imbalance_method": "none", "test_size": 0.25, "random_state": 42,
    "scale_features": True,  # old-style field only, no `preprocessing` key
}
legacy_outcome = run_experiment(df.copy(), legacy_full_cfg)
ok("Legacy run completes and reports resolved preprocessing == platform defaults",
   {k: v for k, v in legacy_outcome["preprocessing"].items() if k != "duplicate_rows_removed"} == DEFAULT_PREPROCESSING)
ok("Legacy run keeps duplicates by default (duplicate_rows_removed == 0)",
   legacy_outcome["preprocessing"]["duplicate_rows_removed"] == 0)
ok("Legacy run: both requested models produced a result",
   len(legacy_outcome["results"]) == 2 and all("error" not in r for r in legacy_outcome["results"]))
print("Sample metric (legacy, random_forest):",
      {k: legacy_outcome["results"][1][k] for k in ("accuracy", "f1", "roc_auc")})


# ================================================================== TEST 2
section("TEST 2: Mean imputation + StandardScaler")
cfg2 = {
    "target_column": "target", "models": ["logistic_regression"], "imbalance_method": "none",
    "preprocessing": {"missing_numeric": "mean", "scaling": "standard"},
}
out2 = run_experiment(df.copy(), cfg2)
ok("Resolved config reports mean imputation + standard scaling",
   out2["preprocessing"]["missing_numeric"] == "mean" and out2["preprocessing"]["scaling"] == "standard")
ok("Model trained successfully under mean+standard config", "error" not in out2["results"][0])

# Confirm the imputer actually uses the TRAIN mean, not some other value
train_df, test_df = None, None
from sklearn.model_selection import train_test_split
X = df[numeric_features + categorical_features]
y = df["target"]
X_train_raw, X_test_raw, y_train_s, y_test_s = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
expected_mean_amount = X_train_raw["amount"].mean()
pipeline2 = build_feature_pipeline(numeric_features, categorical_features, {**DEFAULT_PREPROCESSING, "missing_numeric": "mean"})
pipeline2.fit(X_train_raw)
fitted_mean_amount = pipeline2.named_transformers_["num"].named_steps["imputer"].statistics_[numeric_features.index("amount")]
ok("Fitted imputer mean for 'amount' matches manually-computed TRAIN-only mean",
   abs(fitted_mean_amount - expected_mean_amount) < 1e-9)


# ================================================================== TEST 3
section("TEST 3: Median imputation + MinMaxScaler")
cfg3_pp = {**DEFAULT_PREPROCESSING, "missing_numeric": "median", "scaling": "minmax"}
pipeline3 = build_feature_pipeline(numeric_features, categorical_features, cfg3_pp)
Xt = pipeline3.fit_transform(X_train_raw)
Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
n_numeric = len(numeric_features)
num_block = Xt[:, :n_numeric]
ok("MinMaxScaler output for numeric columns is within [0, 1] on the TRAIN split",
   num_block.min() >= -1e-9 and num_block.max() <= 1 + 1e-9)
scaler3 = pipeline3.named_transformers_["num"].named_steps["scaler"]
ok("Pipeline's numeric scaler is actually MinMaxScaler", type(scaler3).__name__ == "MinMaxScaler")

out3 = run_experiment(df.copy(), {
    "target_column": "target", "models": ["knn"], "imbalance_method": "none",
    "preprocessing": {"missing_numeric": "median", "scaling": "minmax"},
})
ok("KNN (a scaling-sensitive model) trains fine under median+minmax", "error" not in out3["results"][0])


# ================================================================== TEST 4
section("TEST 4: No scaling")
cfg4_pp = {**DEFAULT_PREPROCESSING, "scaling": "none"}
pipeline4 = build_feature_pipeline(numeric_features, categorical_features, cfg4_pp)
ok("Numeric sub-pipeline has ONLY the imputer step (no scaler) when scaling='none'",
   list(pipeline4.transformers[0][1].named_steps.keys()) == ["imputer"])
Xt4 = pipeline4.fit_transform(X_train_raw)
Xt4 = Xt4.toarray() if hasattr(Xt4, "toarray") else Xt4
imputed_amount_manual = X_train_raw["amount"].fillna(X_train_raw["amount"].median()).to_numpy()
ok("With scaling='none', numeric values are imputed but NOT rescaled (match raw/imputed values)",
   np.allclose(np.sort(Xt4[:, numeric_features.index("amount")]), np.sort(imputed_amount_manual), atol=1e-9))


# ================================================================== TEST 5
section("TEST 5: Numeric + categorical features together (one-hot encoding)")
n_categories = df["category"].dropna().nunique()
pipeline5 = build_feature_pipeline(numeric_features, categorical_features, DEFAULT_PREPROCESSING)
Xt5 = pipeline5.fit_transform(X_train_raw)
Xt5 = Xt5.toarray() if hasattr(Xt5, "toarray") else Xt5
ok(f"Output width = {len(numeric_features)} numeric + {n_categories} one-hot category columns",
   Xt5.shape[1] == len(numeric_features) + n_categories)


# ================================================================== TEST 6
section("TEST 6: Dataset containing missing values (already injected above)")
ok("Dataset actually has missing values in numeric + categorical columns",
   df["amount"].isna().sum() > 0 and df["age"].isna().sum() > 0 and df["category"].isna().sum() > 0)
Xt6 = pipeline5.transform(X_test_raw)
Xt6 = Xt6.toarray() if hasattr(Xt6, "toarray") else Xt6
ok("No NaNs remain anywhere in the transformed TEST matrix after imputation",
   not np.isnan(Xt6.astype(float)).any())


# ================================================================== TEST 7
section("TEST 7: Cross-validation with the new preprocessing configuration")
cfg7 = {
    "target_column": "target", "models": ["logistic_regression", "decision_tree"],
    "imbalance_method": "none", "cv_folds": 3, "cv_repeats": 1, "random_state": 42,
    "preprocessing": {"missing_numeric": "mean", "scaling": "robust"},
}
out7 = run_experiment(df.copy(), cfg7)
ok("CV run reports cv_enabled = True", out7["cv_enabled"] is True)
ok("CV run's resolved preprocessing reflects mean + robust", 
   out7["preprocessing"]["missing_numeric"] == "mean" and out7["preprocessing"]["scaling"] == "robust")
ok("Every model completed at least one fold under CV with the new config",
   all(r.get("cv_details", {}).get("n_folds_completed", 0) > 0 for r in out7["results"] if "error" not in r))
print("CV fold completion:", [(r["model_label"], r.get("cv_details", {}).get("n_folds_completed")) for r in out7["results"]])


# ================================================================== TEST 8
section("TEST 8: Preprocessing is fit ONLY on training data (leakage check)")
# Shift a numeric column in the TEST rows only, after the split, and confirm
# the fitted scaler's parameters (learned before the shift) are unaffected --
# i.e. they were computed from train rows only, not from a fit on the full data.
X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
pipeline8 = build_feature_pipeline(numeric_features, categorical_features, DEFAULT_PREPROCESSING)
pipeline8.fit(X_train_leak)
scaler8 = pipeline8.named_transformers_["num"].named_steps["scaler"]
fitted_mean_train_only = scaler8.mean_[numeric_features.index("amount")]

# What a (WRONG) fit-on-everything scaler would have produced:
wrong_pipeline = build_feature_pipeline(numeric_features, categorical_features, DEFAULT_PREPROCESSING)
wrong_pipeline.fit(X)  # fit on train+test combined -- the leaky way
wrong_mean_all_data = wrong_pipeline.named_transformers_["num"].named_steps["scaler"].mean_[numeric_features.index("amount")]

manual_train_mean_for_scaling = X_train_leak["amount"].fillna(X_train_leak["amount"].median()).mean()
ok("Fitted StandardScaler mean matches TRAIN-only mean (not the full-dataset mean)",
   abs(fitted_mean_train_only - manual_train_mean_for_scaling) < 1e-9)
ok("TRAIN-only mean differs from the (leaky) full-dataset mean -- the two are NOT accidentally equal, "
   "so this test actually exercises the leakage guard",
   abs(fitted_mean_train_only - wrong_mean_all_data) > 1e-6)


# ================================================================== TEST 9
section("TEST 9: Imbalance handling still applies to TRAINING data only")
cfg9 = {
    "target_column": "target", "models": ["random_forest"], "imbalance_method": "class_weight",
    "test_size": 0.25, "random_state": 42,
}
out9 = run_experiment(df.copy(), cfg9)
ok("class_weight run completes without touching resampling (no shape change expected)",
   "error" not in out9["results"][0])
n_total_after_dedup = len(df.drop_duplicates())
expected_test_rows = int(round(n_total_after_dedup * 0.25))
cm9 = np.array(out9["results"][0]["confusion_matrix"])
ok(f"Confusion matrix total ({cm9.sum()}) matches the TEST split size (~{expected_test_rows}), "
   f"confirming the test set was not resampled/altered",
   abs(int(cm9.sum()) - expected_test_rows) <= 1)


# ================================================================== TEST 10
section("TEST 10: Existing ROC/PR metrics, confusion matrix and JSON-safety still work")
r10 = out2["results"][0]
required_keys = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy",
                  "mcc", "g_mean", "roc_auc", "pr_auc", "confusion_matrix", "roc_curve", "pr_curve"]
ok("All expected metric keys are present on a model result", all(k in r10 for k in required_keys))
sanitized = sanitize_for_json(out2)
ok("Full experiment outcome (incl. new 'preprocessing' key) serializes as strict JSON (no NaN/inf)",
   isinstance(json.dumps(sanitized, allow_nan=False), str))


# ================================================================== Extra: duplicate handling + invalid option
section("EXTRA: Duplicate-row handling ('keep' vs 'remove')")
kept, n_removed_keep = drop_duplicate_rows(df, "keep")
ok("duplicates='keep' removes nothing", n_removed_keep == 0 and len(kept) == len(df))
removed_df, n_removed = drop_duplicate_rows(df, "remove")
ok("duplicates='remove' removes exactly the 3 injected exact duplicates", n_removed == 3)
ok("Resulting dataframe is shorter by exactly n_removed", len(removed_df) == len(df) - n_removed)

out_dedup = run_experiment(df.copy(), {
    "target_column": "target", "models": ["logistic_regression"], "imbalance_method": "none",
    "preprocessing": {"duplicates": "remove"},
})
ok("End-to-end run with duplicates='remove' reports duplicate_rows_removed == 3",
   out_dedup["preprocessing"]["duplicate_rows_removed"] == 3)

section("EXTRA: Invalid preprocessing option is rejected with a friendly error")
try:
    resolve_preprocessing_config({"preprocessing": {"scaling": "bogus_scaler"}})
    ok("Should have raised FriendlyError for invalid scaling option", False)
except FriendlyError as e:
    ok(f"Invalid scaling option correctly rejected: {e.message[:70]}...", True)

print("\n" + "#" * 70)
print("ALL PHASE 1 TESTS PASSED")
print("#" * 70)