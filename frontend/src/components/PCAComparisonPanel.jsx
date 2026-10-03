import React from "react";

const METRICS = [
  ["accuracy", "Accuracy"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["specificity", "Specificity"],
  ["f1", "F1"],
  ["roc_auc", "ROC-AUC"],
  ["pr_auc", "PR-AUC"],
  ["mcc", "MCC"],
  ["balanced_accuracy", "Balanced Accuracy"],
  ["g_mean", "G-Mean"],
  ["training_time_sec", "Training Time (s)"],
];

function formatValue(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function formatDelta(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${Number(value).toFixed(3)}`;
}

export default function PCAComparisonPanel({ value, onChange }) {
  const comparison = value || {};
  const models = comparison.models || [];

  return (
    <div className="mt-5 rounded-2xl border border-border bg-panel/40 p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-semibold text-lg">PCA vs No-PCA Comparison</p>
          <p className="text-sm text-slate-500 mt-1">
            Run the same experiment twice and change only PCA so its effect can
            be measured instead of assumed.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onChange(!comparison.enabled)}
          className={`px-4 py-2 rounded-xl text-sm font-medium border ${
            comparison.enabled
              ? "bg-primary-50 text-primary-700 border-primary-200"
              : "bg-white text-slate-600 border-border"
          }`}
        >
          {comparison.enabled ? "Comparison On" : "Comparison Off"}
        </button>
      </div>

      {comparison.enabled && (
        <>
          <div className="mt-4 rounded-xl bg-white border border-border p-4 text-sm text-slate-600">
            <strong>Research design:</strong> both runs keep the same dataset,
            preprocessing, feature engineering, feature selection, imbalance
            method, models and random state. Only PCA changes.
          </div>

          {comparison.pca_config && (
            <p className="text-xs text-slate-500 mt-3">
              PCA configuration: {comparison.pca_config.mode === "components"
                ? `${comparison.pca_config.n_components} components`
                : `${Number(comparison.pca_config.variance ?? 0.95) * 100}% variance`}
            </p>
          )}

          {models.length > 0 && (
            <div className="mt-4 overflow-x-auto">
              <table className="text-sm w-full border-collapse min-w-[980px]">
                <thead>
                  <tr className="text-left text-xs text-slate-500 border-b border-border">
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">Metric</th>
                    <th className="py-2 pr-4">No PCA</th>
                    <th className="py-2 pr-4">With PCA</th>
                    <th className="py-2 pr-4">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {models.flatMap((model) =>
                    METRICS.map(([key, label]) => {
                      const metric = model.metrics?.[key] || {};
                      return (
                        <tr key={`${model.model_key}-${key}`} className="border-b border-border/60">
                          <td className="py-2 pr-4 font-medium">{model.model_label}</td>
                          <td className="py-2 pr-4">{label}</td>
                          <td className="py-2 pr-4">{formatValue(metric.no_pca)}</td>
                          <td className="py-2 pr-4">{formatValue(metric.pca)}</td>
                          <td className="py-2 pr-4">{formatDelta(metric.delta)}</td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          )}

          {comparison.comparison_note && (
            <p className="text-xs text-slate-400 mt-4">
              {comparison.comparison_note}
            </p>
          )}
        </>
      )}
    </div>
  );
}
