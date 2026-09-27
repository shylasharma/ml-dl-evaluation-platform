import React, { useState } from "react";
import { Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import BeforeAfterChart from "../components/charts/BeforeAfterChart.jsx";

const METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "g_mean"];

export default function ImbalanceImpact() {
  const { activeExperiment } = useApp();
  const [selectedModel, setSelectedModel] = useState(null);

  if (!activeExperiment || activeExperiment.status !== "completed") {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">⚖️</div>
        <p className="font-semibold">No experiment results yet</p>
        <p className="text-sm text-slate-500 mt-1">Run an experiment first to see imbalance impact analysis.</p>
        <Link to="/experiments/new" className="btn-primary mt-5 inline-flex">Run New Experiment →</Link>
      </div>
    );
  }

  const r = activeExperiment.results;
  const beforeAfter = r.before_after;

  if (!beforeAfter) {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">⚖️</div>
        <p className="font-semibold">No before/after comparison available for this experiment</p>
        <p className="text-sm text-slate-500 mt-2">
          This section needs an experiment run with an imbalance-handling technique (not "No Balancing")
          and the "compare before/after" option enabled — and isn't available in cross-validation mode.
        </p>
        <Link to="/experiments/new" className="btn-primary mt-5 inline-flex">Configure a New Experiment →</Link>
      </div>
    );
  }

  const modelKeys = Object.keys(beforeAfter);
  const activeModel = selectedModel || modelKeys[0];
  const comp = beforeAfter[activeModel];

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="card p-5">
        <p className="font-semibold mb-1">⚖️ Imbalance Impact — Before vs After {r.imbalance_method_label}</p>
        <p className="text-sm text-slate-500 mb-4">
          Every model below was evaluated twice: once with no balancing, once with{" "}
          {r.imbalance_method_label}. This isolates exactly what the balancing technique changed.
        </p>
        <div className="flex flex-wrap gap-2">
          {modelKeys.map((key) => (
            <button
              key={key}
              onClick={() => setSelectedModel(key)}
              className={`chip ${activeModel === key ? "chip-active" : ""}`}
            >
              {beforeAfter[key].model_label}
            </button>
          ))}
        </div>
      </div>

      <div className="card p-5">
        <p className="font-semibold mb-3">{comp.model_label} — Full Metric Comparison</p>
        <div className="overflow-x-auto scroll-thin">
          <table className="text-sm w-full border-collapse min-w-[560px]">
            <thead>
              <tr className="text-left text-xs text-slate-500 border-b border-border">
                <th className="py-2 pr-4">Metric</th>
                <th className="py-2 pr-4">Before (No Balancing)</th>
                <th className="py-2 pr-4">After ({r.imbalance_method_label})</th>
                <th className="py-2 pr-4">Δ Change</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map((m) => {
                const before = comp.before?.[m];
                const after = comp.after?.[m];
                if (before == null || after == null) return null;
                const delta = after - before;
                return (
                  <tr key={m} className="border-b border-border/60">
                    <td className="py-2 pr-4 font-medium capitalize">{m.replace("_", " ")}</td>
                    <td className="py-2 pr-4">{before.toFixed(3)}</td>
                    <td className="py-2 pr-4">{after.toFixed(3)}</td>
                    <td className={`py-2 pr-4 font-semibold ${delta > 0 ? "text-emerald-600" : delta < 0 ? "text-accent-rose" : "text-slate-400"}`}>
                      {delta > 0 ? "▲" : delta < 0 ? "▼" : "—"} {delta >= 0 ? "+" : ""}{delta.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card p-5">
        <p className="font-semibold mb-3">Visual Comparison</p>
        <BeforeAfterChart beforeAfter={beforeAfter} modelKey={activeModel} />
      </div>

      <div className="flex justify-end">
        <Link to="/insights" className="btn-primary">View Insights & Conclusion →</Link>
      </div>
    </div>
  );
}