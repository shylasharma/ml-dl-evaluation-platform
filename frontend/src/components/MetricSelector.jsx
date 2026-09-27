import React from "react";
import { useApp } from "../context/AppContext.jsx";

export default function MetricSelector({ value, onChange, compact = false }) {
  const { metricsCatalog } = useApp();
  return (
    <div className={compact ? "" : "flex items-center gap-2"}>
      {!compact && <label className="text-sm font-medium text-slate-500">Metric Focus</label>}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-xl border border-border px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-primary-300 bg-white"
      >
        {(metricsCatalog.length ? metricsCatalog : DEFAULT_METRICS).map((m) => (
          <option key={m.key} value={m.key}>
            {m.label}{m.imbalance_relevant ? " ★" : ""}
          </option>
        ))}
      </select>
    </div>
  );
}

const DEFAULT_METRICS = [
  { key: "f1", label: "F1 Score" },
  { key: "recall", label: "Recall" },
  { key: "pr_auc", label: "PR-AUC" },
  { key: "mcc", label: "MCC" },
  { key: "balanced_accuracy", label: "Balanced Accuracy" },
  { key: "accuracy", label: "Accuracy" },
];
