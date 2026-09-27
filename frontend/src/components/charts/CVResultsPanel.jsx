import React from "react";

const METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"];

export default function CVResultsPanel({ results, cvInfo }) {
  const successful = results.filter((r) => !r.error && r.cv_details);

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-1">
        <p className="font-semibold">Cross-Validation Results (mean ± std)</p>
        <span className="chip chip-active">
          {cvInfo?.n_splits}-fold × {cvInfo?.n_repeats} repeat(s) = {cvInfo?.n_folds_requested} folds
        </span>
      </div>
      <p className="text-xs text-slate-400 mb-4">
        Every fold refits preprocessing and the imbalance technique on that fold's training rows only.
      </p>

      <div className="overflow-x-auto scroll-thin">
        <table className="text-sm w-full border-collapse min-w-[760px]">
          <thead>
            <tr className="text-left text-xs text-slate-500 border-b border-border">
              <th className="py-2 pr-4">Model</th>
              {METRICS.map((m) => (
                <th key={m} className="py-2 pr-4">{m.replace("_", " ")}</th>
              ))}
              <th className="py-2 pr-4">Folds</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.model_key} className="border-b border-border/60">
                <td className="py-2 pr-4 font-medium">{r.model_label}</td>
                {r.error ? (
                  <td colSpan={METRICS.length + 1} className="py-2 pr-4 text-accent-rose">⚠ {r.error}</td>
                ) : (
                  <>
                    {METRICS.map((m) => {
                      const d = r.cv_details?.metrics?.[m];
                      return (
                        <td key={m} className="py-2 pr-4 whitespace-nowrap">
                          {d ? `${d.mean.toFixed(3)} ± ${d.std.toFixed(3)}` : "—"}
                        </td>
                      );
                    })}
                    <td className="py-2 pr-4 text-slate-500">
                      {r.cv_details.n_folds_completed}/{r.cv_details.n_folds_requested}
                    </td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {successful.some((r) => r.cv_details.fold_errors?.length > 0) && (
        <div className="mt-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-3">
          Some folds failed for one or more models (shown in their fold count above) — this is
          usually caused by too few minority-class samples for the chosen imbalance technique in
          that particular fold.
        </div>
      )}
    </div>
  );
}