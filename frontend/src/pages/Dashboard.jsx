import React, { useEffect, useState } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import api from "../api/client.js";
import KpiCard from "../components/common/KpiCard.jsx";
import MetricSelector from "../components/MetricSelector.jsx";
import BarChartMetric from "../components/charts/BarChartMetric.jsx";
import ConfusionMatrix from "../components/charts/ConfusionMatrix.jsx";
import RocCurve from "../components/charts/RocCurve.jsx";
import PrCurve from "../components/charts/PrCurve.jsx";
import ClassDistribution from "../components/charts/ClassDistribution.jsx";
import CVResultsPanel from "../components/charts/CVResultsPanel.jsx";
import SignificanceTests from "../components/charts/SignificanceTests.jsx";
import ResearchSummary from "../components/research/ResearchSummary.jsx";

const CURVE_COLORS = ["#4f46e5", "#0d9488", "#d97706", "#e11d48", "#0284c7", "#7c3aed", "#65a30d"];

export default function Dashboard() {
  const [params] = useSearchParams();
  const experimentId = params.get("experiment");
  const { activeExperiment, setActiveExperiment, notify } = useApp();
  const [loading, setLoading] = useState(false);
  const [metric, setMetric] = useState("f1");

  useEffect(() => {
    if (experimentId && (!activeExperiment || String(activeExperiment.id) !== experimentId)) {
      setLoading(true);
      api.getExperiment(experimentId)
        .then(setActiveExperiment)
        .catch((err) => notify(err.message, "error"))
        .finally(() => setLoading(false));
    }
  }, [experimentId]); // eslint-disable-line react-hooks/exhaustive-deps

  const exp = activeExperiment;

  if (loading) return <p className="text-center text-slate-400 py-16">Loading experiment…</p>;

  if (!exp || exp.status !== "completed") {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">📊</div>
        <p className="font-semibold">No results to show yet</p>
        <p className="text-sm text-slate-500 mt-1">Run an experiment to see the interactive dashboard.</p>
        <Link to="/experiments/new" className="btn-primary mt-5 inline-flex">Run New Experiment →</Link>
      </div>
    );
  }

  const r = exp.results;
  const results = r.results || [];
  const successful = results.filter((x) => !x.error);
  const profile = r.dataset_profile;

  const bestOf = (m) => successful.reduce((best, cur) => (cur[m] != null && (!best || cur[m] > best[m]) ? cur : best), null);
  const bestF1 = bestOf("f1");
  const bestRecall = bestOf("recall");
  const bestPrAuc = bestOf("pr_auc");
  const bestMcc = bestOf("mcc");

  const mlResults = successful.filter((x) => x.family === "ML");
  const dlResults = successful.filter((x) => x.family === "DL");

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard label="Models Tested" value={results.length} sub={`${successful.length} succeeded`} />
        <KpiCard label="Best F1" value={bestF1 ? bestF1.f1.toFixed(3) : "—"} sub={bestF1?.model_label} tone="primary" />
        <KpiCard label="Best Recall" value={bestRecall ? bestRecall.recall.toFixed(3) : "—"} sub={bestRecall?.model_label} tone="teal" />
        <KpiCard label="Best PR-AUC" value={bestPrAuc?.pr_auc != null ? bestPrAuc.pr_auc.toFixed(3) : "—"} sub={bestPrAuc?.model_label} tone="teal" />
        <KpiCard label="Imbalance Ratio" value={`${profile.imbalance_ratio}:1`} tone={profile.is_imbalanced ? "amber" : "default"} />
        <KpiCard label="Dataset Size" value={profile.rows.toLocaleString()} sub={`${profile.n_features} features`} />
        <KpiCard label="Balancing Used" value={r.imbalance_method_label} />
        <KpiCard label="Best MCC" value={bestMcc ? bestMcc.mcc.toFixed(3) : "—"} sub={bestMcc?.model_label} tone="amber" />
      </div>

      {r.cv_enabled && (
        <div className="rounded-xl px-4 py-3 text-sm font-medium bg-primary-50 text-primary-700 border border-primary-200">
          🔁 Cross-validated results — {r.cv_info?.n_splits}-fold × {r.cv_info?.n_repeats} repeat(s).
          All metrics below are means across folds; see the Cross-Validation panel for standard
          deviations and the Statistical Significance panel for pairwise tests.
        </div>
      )}

      {r.cv_enabled && <CVResultsPanel results={results} cvInfo={r.cv_info} />}
      {r.cv_enabled && <SignificanceTests significance={r.significance} />}
{r.research_summary && (
  <ResearchSummary research={r.research_summary} />
)}
      {r.pca_comparison?.enabled && (
        <PCAComparisonSection comparison={r.pca_comparison} />
      )}

      {/* Model comparison */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <p className="font-semibold">Model Performance Comparison</p>
          <MetricSelector value={metric} onChange={setMetric} compact />
        </div>
        <BarChartMetric data={results} metric={metric} />
      </div>

      {/* Comparison table */}
      <div className="card p-5 overflow-x-auto scroll-thin">
        <p className="font-semibold mb-3">Comparison Table {r.cv_enabled && <span className="text-xs font-normal text-slate-400">(mean across folds)</span>}</p>
        <table className="text-sm w-full border-collapse min-w-[720px]">
          <thead>
            <tr className="text-left text-xs text-slate-500 border-b border-border">
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4">Family</th>
              <th className="py-2 pr-4">Accuracy</th>
              <th className="py-2 pr-4">Precision</th>
              <th className="py-2 pr-4">Recall</th>
              <th className="py-2 pr-4">F1</th>
              <th className="py-2 pr-4">ROC-AUC</th>
              <th className="py-2 pr-4">PR-AUC</th>
              <th className="py-2 pr-4">MCC</th>
              <th className="py-2 pr-4">Bal. Acc.</th>
            </tr>
          </thead>
          <tbody>
            {results.map((row) => (
              <tr key={row.model_key} className="border-b border-border/60 hover:bg-panel/60">
                <td className="py-2 pr-4 font-medium">
                  {row.error ? row.model_label : (
                    <Link to={`/models/${row.model_key}?experiment=${exp.id}`} className="text-primary-600">{row.model_label}</Link>
                  )}
                </td>
                <td className="py-2 pr-4">{row.family}</td>
                {row.error ? (
                  <td className="py-2 pr-4 text-accent-rose" colSpan={8}>⚠ {row.error}</td>
                ) : (
                  <>
                    {["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"].map((k) => (
                      <td key={k} className="py-2 pr-4">{row[k] != null ? row[k].toFixed(3) : "—"}</td>
                    ))}
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="card p-5">
          <p className="font-semibold mb-3">Class Distribution</p>
          <ClassDistribution classCounts={profile.class_counts} />
        </div>
        <div className="card p-5">
          <p className="font-semibold mb-3">ML vs DL Comparison ({metric.toUpperCase()})</p>
          {mlResults.length > 0 && dlResults.length > 0 ? (
            <BarChartMetric
              data={[
                { model_label: "ML Average", family: "ML", [metric]: avg(mlResults, metric) },
                { model_label: "DL Average", family: "DL", [metric]: avg(dlResults, metric) },
                ...results,
              ].filter((x, i, arr) => i < 2 || true)}
              metric={metric}
            />
          ) : (
            <p className="text-sm text-slate-400 py-8 text-center">Select at least one ML and one DL model to compare.</p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="card p-5">
          <p className="font-semibold mb-3">ROC Curves</p>
          <RocCurve curves={successful.map((m, i) => ({ label: m.model_label, points: m.roc_curve, color: CURVE_COLORS[i % CURVE_COLORS.length] }))} />
        </div>
        <div className="card p-5">
          <p className="font-semibold mb-3">Precision-Recall Curves</p>
          <PrCurve curves={successful.map((m, i) => ({ label: m.model_label, points: m.pr_curve, color: CURVE_COLORS[i % CURVE_COLORS.length] }))} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {successful.slice(0, 4).map((m) => (
          <div key={m.model_key} className="card p-5">
            <p className="font-semibold mb-3">{m.model_label} — Confusion Matrix</p>
            <ConfusionMatrix matrix={m.confusion_matrix} labels={r.class_labels} />
          </div>
        ))}
      </div>

      {r.before_after && (
        <div className="card p-5 flex items-center justify-between">
          <div>
            <p className="font-semibold">⚖️ Imbalance Impact Available</p>
            <p className="text-sm text-slate-500 mt-1">
              See the full before/after breakdown for every model on the dedicated Imbalance Impact page.
            </p>
          </div>
          <Link to="/imbalance-impact" className="btn-primary shrink-0">View Imbalance Impact →</Link>
        </div>
      )}

      <div className="flex justify-end gap-3">
        <Link to="/compare" className="btn-secondary">Compare Models in Detail →</Link>
        <Link to="/insights" className="btn-primary">View Insights & Conclusion →</Link>
      </div>
    </div>
  );
}


function PCAComparisonSection({ comparison }) {
  const [metric, setMetric] = useState("f1");

  const models = comparison?.models || [];
  const pcaMeta = comparison?.pca?.pca_metadata || {};
  const metrics = [
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
  ];

  const format = (value) =>
    typeof value === "number" && Number.isFinite(value)
      ? value.toFixed(3)
      : "—";

  const formatDelta = (value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
  };

  return (
    <div className="card p-5 space-y-5">
      <div>
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-primary-600">
              Feature Selection &amp; Dimensionality Reduction
            </p>
            <h2 className="font-semibold text-lg mt-1">
              PCA vs No-PCA Experimental Comparison
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              The same experiment was evaluated with and without PCA. Only the PCA stage changes.
            </p>
          </div>
          <span className="chip chip-active">Paired comparison</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MiniPCAStat label="Original features" value={pcaMeta.original_feature_count ?? "—"} />
        <MiniPCAStat label="PCA components" value={pcaMeta.component_count ?? "—"} />
        <MiniPCAStat
          label="Target variance"
          value={
            comparison?.pca_config?.mode === "components"
              ? "Custom components"
              : `${Math.round(Number(comparison?.pca_config?.variance ?? 0.95) * 100)}%`
          }
        />
        <MiniPCAStat
          label="Variance retained"
          value={
            typeof pcaMeta.variance_retained === "number"
              ? `${(pcaMeta.variance_retained * 100).toFixed(2)}%`
              : "—"
          }
        />
      </div>

      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="font-semibold">Metric-wise comparison</p>
          <p className="text-xs text-slate-500 mt-1">Positive Δ means PCA minus No-PCA.</p>
        </div>
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="border border-slate-300 rounded-lg px-3 py-2 bg-white text-sm"
        >
          {metrics.map(([key, label]) => (
            <option key={key} value={key}>{label}</option>
          ))}
        </select>
      </div>

      <div className="overflow-x-auto">
        <table className="text-sm w-full border-collapse min-w-[720px]">
          <thead>
            <tr className="text-left text-xs text-slate-500 border-b border-border">
              <th className="py-3 pr-4">Model</th>
              <th className="py-3 pr-4">Family</th>
              <th className="py-3 pr-4">No PCA</th>
              <th className="py-3 pr-4">PCA</th>
              <th className="py-3 pr-4">Δ</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => {
              const values = model.metrics?.[metric] || {};
              return (
                <tr key={model.model_key} className="border-b border-border/60">
                  <td className="py-3 pr-4 font-medium">{model.model_label}</td>
                  <td className="py-3 pr-4">{model.family || "—"}</td>
                  <td className="py-3 pr-4">{format(values.no_pca)}</td>
                  <td className="py-3 pr-4">{format(values.pca)}</td>
                  <td className={`py-3 pr-4 font-semibold ${typeof values.delta === "number" ? (values.delta > 0 ? "text-emerald-600" : values.delta < 0 ? "text-rose-600" : "text-slate-500") : "text-slate-400"}`}>
                    {formatDelta(values.delta)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-primary-100 bg-primary-50 p-4 text-sm text-primary-900">
        <p className="font-medium">Research note</p>
        <p className="mt-1 text-primary-800">
          {comparison.comparison_note ||
            "PCA and No-PCA use matched experimental settings so the observed metric differences can be measured rather than assumed."}
        </p>
      </div>
    </div>
  );
}

function MiniPCAStat({ label, value }) {
  return (
    <div className="rounded-xl border border-border/70 bg-panel/40 p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="font-semibold text-slate-900 mt-1">{value}</p>
    </div>
  );
}

function avg(arr, key) {
  const vals = arr.map((x) => x[key]).filter((v) => v != null);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}