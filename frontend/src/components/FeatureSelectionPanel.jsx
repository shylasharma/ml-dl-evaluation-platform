import React from "react";


const METHODS = [
  {
    value: "correlation",
    label: "Correlation",
    category: "Filter",
    help: "Removes highly correlated features so redundant information is reduced.",
  },
  {
    value: "chi2",
    label: "Chi-Square",
    category: "Filter",
    help: "Ranks features by their association with the categorical target. Suitable for non-negative feature values.",
  },
  {
    value: "anova",
    label: "ANOVA / F-test",
    category: "Filter",
    help: "Ranks features according to differences between class groups.",
  },
  {
    value: "mutual_information",
    label: "Mutual Information",
    category: "Filter",
    help: "Measures how much information a feature provides about the target, including nonlinear relationships.",
  },
  {
    value: "rfe",
    label: "RFE",
    category: "Wrapper",
    help: "Recursively removes less useful features using a model-based ranking.",
  },
  {
    value: "sequential",
    label: "Sequential Feature Selection",
    category: "Wrapper",
    help: "Adds or removes features sequentially according to model performance.",
  },
  {
    value: "l1",
    label: "L1 Regularization",
    category: "Embedded",
    help: "Uses L1 regularization to shrink less useful feature coefficients toward zero.",
  },
  {
    value: "tree_importance",
    label: "Tree-Based Importance",
    category: "Embedded",
    help: "Uses tree-model feature importance to retain informative features.",
  },
];

const DEFAULT_CONFIG = {
  enabled: false,
  method: "none",
  k: null,
  threshold: null,
  correlation_threshold: 0.9,
  scoring: "f1_weighted",
  direction: "forward",
  step: 1,
  max_features: null,
};

export default function FeatureSelectionPanel({ value, onChange }) {
  const cfg = {
    ...DEFAULT_CONFIG,
    ...(value || {}),
  };

  const update = (key, val) => {
    onChange({
      ...cfg,
      [key]: val,
    });
  };

  const enableSelection = (enabled) => {
    if (!enabled) {
      onChange({
        ...DEFAULT_CONFIG,
        enabled: false,
      });
      return;
    }

    onChange({
      ...cfg,
      enabled: true,
      method:
        cfg.method === "none"
          ? "mutual_information"
          : cfg.method,
    });
  };

  const methodInfo = METHODS.find(
    (method) => method.value === cfg.method
  );

  const usesK =
    [
      "chi2",
      "anova",
      "mutual_information",
      "rfe",
      "sequential",
    ].includes(cfg.method);

  const usesCorrelationThreshold =
    cfg.method === "correlation";

  const usesEmbeddedThreshold =
    cfg.method === "l1" ||
    cfg.method === "tree_importance";

  const usesSequentialOptions =
    cfg.method === "sequential";

  return (
    <div className="card p-5">

      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-5">
        <div>
          <p className="font-semibold text-lg">
            Feature Selection
          </p>

          <p className="text-sm text-slate-500 mt-1 max-w-3xl">
            Select informative features before model training.
            Feature selection is performed using training data
            only and the same selection is then applied to
            validation/test data.
          </p>
        </div>

        <span className="chip">
          Optional
        </span>
      </div>

      {/* Main question */}
      <div className="rounded-xl border border-border bg-panel p-4">

        <p className="font-medium text-sm">
          Would you like to perform feature selection?
        </p>

        <div className="flex flex-wrap gap-3 mt-3">

          <label
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border cursor-pointer transition ${
              !cfg.enabled
                ? "border-primary-400 bg-primary-50"
                : "border-border"
            }`}
          >
            <input
              type="radio"
              name="feature-selection-enabled"
              checked={!cfg.enabled}
              onChange={() => enableSelection(false)}
            />

            <span className="text-sm">
              No
            </span>
          </label>

          <label
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border cursor-pointer transition ${
              cfg.enabled
                ? "border-primary-400 bg-primary-50"
                : "border-border"
            }`}
          >
            <input
              type="radio"
              name="feature-selection-enabled"
              checked={cfg.enabled}
              onChange={() => enableSelection(true)}
            />

            <span className="text-sm">
              Yes
            </span>
          </label>

        </div>
      </div>

      {/* Configuration */}
      {cfg.enabled && (
        <div className="mt-5 space-y-4">

          {/* Method */}
          <div>
            <label className="text-xs font-medium text-slate-500 flex items-center gap-1">
              Selection Method

              <span
  className="text-slate-400 cursor-help"
  title="Choose a filter, wrapper, or embedded feature-selection technique."
>
  ⓘ
</span>
            </label>

            <select
              value={cfg.method}
              onChange={(e) =>
                update("method", e.target.value)
              }
              className="input mt-1"
            >
              {METHODS.map((method) => (
                <option
                  key={method.value}
                  value={method.value}
                >
                  {method.label} — {method.category}
                </option>
              ))}
            </select>

            {methodInfo && (
              <p className="text-xs text-slate-500 mt-2">
                {methodInfo.help}
              </p>
            )}
          </div>

          {/* K */}
          {usesK && (
            <div>
              <label className="text-xs font-medium text-slate-500">
                Number of Features (K)
              </label>

              <input
                type="number"
                min="1"
                value={cfg.k ?? ""}
                onChange={(e) =>
                  update(
                    "k",
                    e.target.value
                      ? Number(e.target.value)
                      : null
                  )
                }
                placeholder="Example: 10"
                className="input mt-1"
              />

              <p className="text-xs text-slate-400 mt-1">
                Number of features to retain.
              </p>
            </div>
          )}

          {/* Correlation threshold */}
          {usesCorrelationThreshold && (
            <div>
              <label className="text-xs font-medium text-slate-500">
                Correlation Threshold
              </label>

              <input
                type="number"
                min="0.01"
                max="0.99"
                step="0.01"
                value={cfg.correlation_threshold}
                onChange={(e) =>
                  update(
                    "correlation_threshold",
                    Number(e.target.value)
                  )
                }
                className="input mt-1"
              />

              <p className="text-xs text-slate-400 mt-1">
                Features with correlation above this threshold
                are treated as redundant.
              </p>
            </div>
          )}

          {/* Embedded threshold */}
          {usesEmbeddedThreshold && (
            <div>
              <label className="text-xs font-medium text-slate-500">
                Importance Threshold
              </label>

              <select
                value={cfg.threshold ?? "mean"}
                onChange={(e) =>
                  update(
                    "threshold",
                    e.target.value
                  )
                }
                className="input mt-1"
              >
                <option value="mean">
                  Mean Importance
                </option>

                <option value="median">
                  Median Importance
                </option>
              </select>
            </div>
          )}

          {/* Sequential controls */}
          {usesSequentialOptions && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">

              <div>
                <label className="text-xs font-medium text-slate-500">
                  Direction
                </label>

                <select
                  value={cfg.direction}
                  onChange={(e) =>
                    update(
                      "direction",
                      e.target.value
                    )
                  }
                  className="input mt-1"
                >
                  <option value="forward">
                    Forward
                  </option>

                  <option value="backward">
                    Backward
                  </option>
                </select>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-500">
                  Step
                </label>

                <input
                  type="number"
                  min="1"
                  value={cfg.step}
                  onChange={(e) =>
                    update(
                      "step",
                      Number(e.target.value)
                    )
                  }
                  className="input mt-1"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-slate-500">
                  Maximum Features
                </label>

                <input
                  type="number"
                  min="1"
                  value={cfg.max_features ?? ""}
                  onChange={(e) =>
                    update(
                      "max_features",
                      e.target.value
                        ? Number(e.target.value)
                        : null
                    )
                  }
                  placeholder="Optional"
                  className="input mt-1"
                />
              </div>

            </div>
          )}

          {/* Scoring */}
          {(cfg.method === "rfe" ||
            cfg.method === "sequential" ||
            cfg.method === "l1" ||
            cfg.method === "tree_importance") && (
            <div>
              <label className="text-xs font-medium text-slate-500">
                Selection Scoring
              </label>

              <select
                value={cfg.scoring}
                onChange={(e) =>
                  update(
                    "scoring",
                    e.target.value
                  )
                }
                className="input mt-1"
              >
                <option value="f1_weighted">
                  F1 Weighted
                </option>

                <option value="f1_macro">
                  F1 Macro
                </option>

                <option value="balanced_accuracy">
                  Balanced Accuracy
                </option>

                <option value="accuracy">
                  Accuracy
                </option>
              </select>
            </div>
          )}

          {/* Explanation */}
          <div className="rounded-xl bg-slate-50 border border-border p-4">

            <p className="text-sm font-medium">
              Research note
            </p>

            <p className="text-xs text-slate-500 mt-1">
              Feature selection is different from PCA.
              Feature selection keeps the original features
              and removes less useful ones. PCA creates new
              transformed components.
            </p>

          </div>

        </div>
      )}

    </div>
  );
}