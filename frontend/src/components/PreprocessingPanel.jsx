import React, { useState } from "react";
import Tooltip from "./common/Tooltip.jsx";

// Must mirror backend/app/ml/preprocessing.py: DEFAULT_PREPROCESSING exactly,
// so "Use Recommended Defaults" reproduces the platform's original behaviour.
export const DEFAULT_PREPROCESSING = {
  missing_numeric: "median",
  missing_categorical: "most_frequent",
  scaling: "standard",
  encoding: "onehot",
  duplicates: "keep",
};

const OPTIONS = {
  missing_numeric: [
    {
      value: "median",
      label: "Median",
      help: "Fills missing numeric values with the column's median. Robust to outliers; the platform's default.",
    },
    {
      value: "mean",
      label: "Mean",
      help: "Fills missing numeric values with the column's average. Simple, but sensitive to outliers.",
    },
  ],

  missing_categorical: [
    {
      value: "most_frequent",
      label: "Most Frequent",
      help: "Fills missing categorical values with the most common category in that column.",
    },
  ],

  scaling: [
    {
      value: "none",
      label: "No Scaling",
      help: "Leaves numeric values as-is. Fine for tree-based models (Random Forest, XGBoost, LightGBM, Decision Tree, boosting).",
    },
    {
      value: "standard",
      label: "StandardScaler",
      help: "Standardizes numeric features using the mean and standard deviation (mean 0, unit variance). Often useful for distance-based and gradient-based models. The platform's default.",
    },
    {
      value: "minmax",
      label: "MinMaxScaler",
      help: "Rescales numeric features to a fixed 0–1 range. Useful when you want bounded feature values, e.g. for neural networks.",
    },
    {
      value: "robust",
      label: "RobustScaler",
      help: "Scales using the median and interquartile range instead of mean/std, so it resists being skewed by outliers.",
    },
  ],

  encoding: [
    {
      value: "onehot",
      label: "One-Hot Encoding",
      help: "Turns each category into its own 0/1 column. Currently the only supported encoding — no ordering is implied between categories.",
    },
  ],

  duplicates: [
    {
      value: "keep",
      label: "Keep",
      help: "Leaves duplicate rows in the dataset. The platform's default.",
    },
    {
      value: "remove",
      label: "Remove Duplicates",
      help: "Removes exact duplicate rows (across all columns) before splitting into train/test. This is done before the split specifically because it is safer: an exact duplicate pair could otherwise land on both sides of the split.",
    },
  ],
};

const FIELDS = [
  ["missing_numeric", "Missing Numerical Values"],
  ["missing_categorical", "Missing Categorical Values"],
  ["scaling", "Feature Scaling"],
  ["encoding", "Categorical Encoding"],
  ["duplicates", "Duplicate Rows"],
];

export default function PreprocessingPanel({ value, onChange }) {
  const [defaultsApplied, setDefaultsApplied] = useState(false);

  const cfg = {
    ...DEFAULT_PREPROCESSING,
    ...value,
  };

  const set = (key, val) => {
    setDefaultsApplied(false);
    onChange({
      ...cfg,
      [key]: val,
    });
  };

  const applyRecommendedDefaults = () => {
    onChange({
      ...DEFAULT_PREPROCESSING,
    });

    setDefaultsApplied(true);

    // Remove the temporary confirmation after a short delay.
    window.setTimeout(() => {
      setDefaultsApplied(false);
    }, 1500);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs text-slate-500">
          Controls how the dataset is cleaned and scaled before splitting.
          Missing-value filling and scaling are always fit on the training
          rows only and simply applied to the test rows, so these choices
          never leak information from the test set.
        </p>

        <button
          type="button"
          onClick={applyRecommendedDefaults}
          className="shrink-0 ml-4 text-xs font-medium text-primary-600 hover:text-primary-700 hover:underline transition-colors"
        >
          {defaultsApplied
            ? "✓ Recommended Defaults Applied"
            : "Use Recommended Defaults"}
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {FIELDS.map(([key, label]) => (
          <div key={key}>
            <label className="text-xs font-medium text-slate-500 flex items-center gap-1">
              {label}

              <Tooltip
                text={
                  OPTIONS[key].find(
                    (o) => o.value === cfg[key]
                  )?.help || ""
                }
              >
                <span className="text-slate-400 cursor-help">
                  ⓘ
                </span>
              </Tooltip>
            </label>

            <select
              value={cfg[key]}
              onChange={(e) => set(key, e.target.value)}
              disabled={OPTIONS[key].length === 1}
              className="input mt-1 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {OPTIONS[key].map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
}