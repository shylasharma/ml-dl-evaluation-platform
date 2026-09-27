import React from "react";

export default function SignificanceTests({ significance }) {
  if (!significance) return null;

  if (!significance.applicable) {
    return (
      <div className="card p-5">
        <p className="font-semibold mb-2">Statistical Significance</p>
        <p className="text-sm text-slate-400">{significance.reason}</p>
      </div>
    );
  }

  const { friedman, pairwise_wilcoxon: pairwise, metric } = significance;

  return (
    <div className="card p-5 space-y-5">
      <div>
        <p className="font-semibold">Statistical Significance — {metric?.toUpperCase()}</p>
        <p className="text-xs text-slate-400 mt-1">
          Friedman test (omnibus) + pairwise Wilcoxon signed-rank tests with Holm-Bonferroni
          correction, following Demšar (2006).
        </p>
      </div>

      <div className={`rounded-xl p-4 text-sm ${
        friedman.applicable
          ? friedman["significant_at_0.05"]
            ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
            : "bg-slate-50 text-slate-600 border border-border"
          : "bg-slate-50 text-slate-500 border border-border"
      }`}>
        <p className="font-medium mb-1">Friedman Test</p>
        <p>{friedman.applicable ? friedman.interpretation : friedman.reason}</p>
      </div>

      {pairwise?.comparisons?.length > 0 && (
        <div className="overflow-x-auto scroll-thin">
          <p className="font-medium text-sm mb-2">Pairwise Comparisons (Holm-Bonferroni corrected)</p>
          <table className="text-sm w-full border-collapse min-w-[640px]">
            <thead>
              <tr className="text-left text-xs text-slate-500 border-b border-border">
                <th className="py-2 pr-4">Model A</th>
                <th className="py-2 pr-4">Model B</th>
                <th className="py-2 pr-4">Mean A</th>
                <th className="py-2 pr-4">Mean B</th>
                <th className="py-2 pr-4">Adjusted p</th>
                <th className="py-2 pr-4">Significant?</th>
              </tr>
            </thead>
            <tbody>
              {pairwise.comparisons.map((c, i) => (
                <tr key={i} className="border-b border-border/60">
                  <td className="py-2 pr-4">{c.model_a}</td>
                  <td className="py-2 pr-4">{c.model_b}</td>
                  <td className="py-2 pr-4">{c.mean_a.toFixed(3)}</td>
                  <td className="py-2 pr-4">{c.mean_b.toFixed(3)}</td>
                  <td className="py-2 pr-4">{c.p_value_holm.toFixed(4)}</td>
                  <td className="py-2 pr-4">
                    {c["significant_at_0.05"] ? (
                      <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700">Yes</span>
                    ) : (
                      <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500">No</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-slate-400 mt-2">
            "Significant" means p &lt; 0.05 after correcting for multiple comparisons — a difference
            unlikely to be due to chance across folds, not necessarily a large or practically
            important difference.
          </p>
        </div>
      )}
    </div>
  );
}