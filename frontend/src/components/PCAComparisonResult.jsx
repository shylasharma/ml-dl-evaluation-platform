import React, { useState } from "react";

const METRICS = [
  ["f1", "F1"],
  ["mcc", "MCC"],
  ["pr_auc", "PR-AUC"],
  ["recall", "Recall"],
  ["accuracy", "Accuracy"],
  ["training_time_sec", "Training Time (s)"],
];

function fmt(v) {
  return v == null ? "—" : Number(v).toFixed(3);
}

export default function PCAComparisonResults({ comparison }) {
  const [selectedModel, setSelectedModel] = useState(0);
  if (!comparison?.enabled || !comparison.models?.length) return null;

  const model = comparison.models[selectedModel] || comparison.models[0];

  return (
    <div className="card p-5 space-y-4">
      <div>
        <p className="font-semibold">🔬 PCA vs No-PCA Experimental Comparison</p>
        <p className="text-sm text-slate-500 mt-1">
          Both configurations use the same experiment settings; only PCA is changed.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {comparison.models.map((item, index) => (
          <button
            key={item.model_key}
            type="button"
            onClick={() => setSelectedModel(index)}
            className={`chip ${index === selectedModel ? "chip-active" : ""}`}
          >
            {item.model_label}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="text-sm w-full border-collapse min-w-[620px]">
          <thead>
            <tr className="text-left text-xs text-slate-500 border-b border-border">
              <th className="py-2 pr-4">Metric</th>
              <th className="py-2 pr-4">Without PCA</th>
              <th className="py-2 pr-4">With PCA</th>
              <th className="py-2 pr-4">Δ</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map(([key, label]) => {
              const metric = model.metrics?.[key] || {};
              const delta = metric.delta;
              return (
                <tr key={key} className="border-b border-border/60">
                  <td className="py-2 pr-4 font-medium">{label}</td>
                  <td className="py-2 pr-4">{fmt(metric.no_pca)}</td>
                  <td className="py-2 pr-4">{fmt(metric.pca)}</td>
                  <td className="py-2 pr-4 font-medium">
                    {delta == null ? "—" : `${delta >= 0 ? "+" : ""}${Number(delta).toFixed(3)}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-slate-400">
        This comparison is descriptive: it reports the observed difference for this dataset and configuration rather than assuming PCA is universally beneficial.
      </p>
    </div>
  );
}
