import React, { useMemo } from "react";

const ML_MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  svm: "Support Vector Machine",
  knn: "K-Nearest Neighbors",
  naive_bayes: "Naive Bayes",
  gradient_boosting: "Gradient Boosting",
  adaboost: "AdaBoost",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
};

const HYBRID_METHODS = [
  {
    value: "hard_voting",
    label: "Hard Voting",
    description:
      "Each base model votes for a class. The class receiving the most votes becomes the final prediction.",
  },
  {
    value: "soft_voting",
    label: "Soft Voting",
    description:
      "Combines class probabilities from the base models to make the final prediction.",
  },
  {
    value: "stacking",
    label: "Stacking",
    description:
      "Uses predictions from multiple base models as inputs to a final meta-classifier.",
  },
  {
    value: "blending",
    label: "Blending",
    description:
      "Combines probability predictions from multiple base models using configurable weights.",
  },
];

const ML_MODEL_KEYS = Object.keys(ML_MODEL_LABELS);

export default function HybridizationPanel({
  value,
  onChange,
  selectedModels = [],
  cvFolds = 0,
}) {
  const config = {
    enabled: false,
    method: "hard_voting",
    base_models: [],
    weights: null,
    ...value,
  };

  const availableModels = useMemo(
    () =>
      selectedModels.filter((model) =>
        ML_MODEL_KEYS.includes(model)
      ),
    [selectedModels]
  );

  const selectedBaseModels = config.base_models || [];

  const update = (patch) => {
    onChange({
      ...config,
      ...patch,
    });
  };

  const toggleEnabled = (enabled) => {
    if (!enabled) {
      update({
        enabled: false,
        base_models: [],
        weights: null,
      });
      return;
    }

    const defaults = availableModels.slice(0, 3);

    update({
      enabled: true,
      method: config.method || "hard_voting",
      base_models:
        selectedBaseModels.length >= 2
          ? selectedBaseModels
          : defaults,
      weights: null,
    });
  };

  const toggleBaseModel = (modelKey) => {
    const exists = selectedBaseModels.includes(modelKey);

    const next = exists
      ? selectedBaseModels.filter(
          (key) => key !== modelKey
        )
      : [...selectedBaseModels, modelKey];

    update({
      base_models: next,
      weights: null,
    });
  };

  const methodInfo =
    HYBRID_METHODS.find(
      (method) => method.value === config.method
    ) || HYBRID_METHODS[0];

  const insufficientModels =
    config.enabled &&
    selectedBaseModels.length < 2;

  const noMlModels =
    config.enabled &&
    availableModels.length < 2;

  const cvWarning =
    config.enabled && cvFolds > 0;

  return (
    <div className="mt-6 rounded-2xl border border-border bg-panel p-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <p className="font-semibold text-lg">
            Model Hybridization
          </p>

          <p className="text-sm text-slate-500 mt-1 max-w-3xl">
            Combine multiple ML classifiers into a single
            hybrid prediction system using voting,
            stacking, or blending.
          </p>
        </div>

        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={Boolean(config.enabled)}
            onChange={(e) =>
              toggleEnabled(e.target.checked)
            }
            className="rounded border-slate-300"
          />

          <span className="font-medium text-sm">
            Enable Hybridization
          </span>
        </label>
      </div>

      {!config.enabled && (
        <div className="mt-4 rounded-xl border border-border bg-slate-50 p-4">
          <p className="text-sm font-medium text-slate-700">
            Hybridization is currently disabled.
          </p>

          <p className="text-xs text-slate-500 mt-1">
            The selected models will be trained and
            evaluated individually.
          </p>
        </div>
      )}

      {config.enabled && (
        <div className="mt-5 space-y-5">
          {/* CV warning */}
          {cvWarning && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
              <p className="text-sm font-semibold text-amber-800">
                Cross-validation hybridization is not
                available yet.
              </p>

              <p className="text-xs text-amber-700 mt-1">
                For the first hybrid experiment, use
                single train/test split mode. This
                prevents incorrect or leakage-prone
                hybrid fitting across folds.
              </p>
            </div>
          )}

          {/* Method */}
          <div>
            <label className="text-xs font-semibold text-slate-500">
              Hybridization Method
            </label>

            <select
              value={config.method}
              onChange={(e) =>
                update({
                  method: e.target.value,
                })
              }
              className="input mt-1 w-full"
            >
              {HYBRID_METHODS.map((method) => (
                <option
                  key={method.value}
                  value={method.value}
                >
                  {method.label}
                </option>
              ))}
            </select>

            <div className="mt-2 rounded-xl bg-primary-50/50 border border-primary-100 p-3">
              <p className="text-sm font-medium text-primary-800">
                {methodInfo.label}
              </p>

              <p className="text-xs text-slate-600 mt-1">
                {methodInfo.description}
              </p>
            </div>
          </div>

          {/* Base models */}
          <div>
            <div className="flex items-center justify-between gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-500">
                  Base Models
                </label>

                <p className="text-xs text-slate-500 mt-1">
                  Select at least two ML models to combine.
                </p>
              </div>

              <span className="chip">
                {selectedBaseModels.length} selected
              </span>
            </div>

            {availableModels.length === 0 ? (
              <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-4">
                <p className="text-sm font-semibold text-amber-800">
                  No ML base models available
                </p>

                <p className="text-xs text-amber-700 mt-1">
                  Select at least two ML classifiers in
                  the Model Selection step.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
                {availableModels.map((modelKey) => {
                  const checked =
                    selectedBaseModels.includes(
                      modelKey
                    );

                  return (
                    <label
                      key={modelKey}
                      className={`flex items-start gap-3 rounded-xl border p-3 cursor-pointer transition ${
                        checked
                          ? "border-primary-300 bg-primary-50"
                          : "border-border bg-panel"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() =>
                          toggleBaseModel(modelKey)
                        }
                        className="rounded border-slate-300 mt-0.5"
                      />

                      <div>
                        <p className="text-sm font-semibold">
                          {ML_MODEL_LABELS[modelKey]}
                        </p>

                        <p className="text-xs text-slate-500 mt-0.5">
                          {modelKey}
                        </p>
                      </div>
                    </label>
                  );
                })}
              </div>
            )}
          </div>

          {/* Validation */}
          {insufficientModels && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
              <p className="text-xs font-medium text-amber-800">
                Select at least two base models.
              </p>
            </div>
          )}

          {noMlModels && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-3">
              <p className="text-xs font-medium text-red-700">
                Hybridization requires at least two
                selected ML classifiers.
              </p>
            </div>
          )}

          {/* Configuration summary */}
          <div className="rounded-xl border border-border bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Hybrid Configuration
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
              <div>
                <p className="text-xs text-slate-500">
                  Method
                </p>

                <p className="text-sm font-semibold mt-1">
                  {methodInfo.label}
                </p>
              </div>

              <div>
                <p className="text-xs text-slate-500">
                  Base Models
                </p>

                <p className="text-sm font-semibold mt-1">
                  {selectedBaseModels.length}
                </p>
              </div>

              <div>
                <p className="text-xs text-slate-500">
                  Validation
                </p>

                <p className="text-sm font-semibold mt-1">
                  {cvFolds > 0
                    ? "CV restricted"
                    : "Train/Test Split"}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}