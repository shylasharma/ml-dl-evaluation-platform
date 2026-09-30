import React from "react";
import { Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";

export default function Insights() {
  const { activeExperiment } = useApp();

  if (!activeExperiment || activeExperiment.status !== "completed") {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">💡</div>

        <p className="font-semibold">
          No insights yet
        </p>

        <p className="text-sm text-slate-500 mt-1">
          Run an experiment first to generate data-driven insights.
        </p>

        <Link
          to="/experiments/new"
          className="btn-primary mt-5 inline-flex"
        >
          Run New Experiment →
        </Link>
      </div>
    );
  }

  const r = activeExperiment.results || {};

  const insights =
    r.insights ||
    activeExperiment.insights ||
    [];

  const conclusion =
    r.conclusion ||
    activeExperiment.conclusion ||
    "";

  const significance = r.significance;
  const cvInfo = r.cv_info;
  const researchSummary = r.research_summary;
  const beforeAfter = r.before_after;

  return (
    <div className="max-w-5xl mx-auto space-y-6">

      {/* =====================================================
          PAGE HEADER
      ===================================================== */}
      <div className="card p-6">

        <div className="flex items-start justify-between gap-4">

          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-primary-600">
              Research Analysis
            </p>

            <h1 className="text-2xl font-bold mt-1">
              Insights & Conclusion
            </h1>

            <p className="text-sm text-slate-500 mt-2">
              Data-driven observations and conclusions generated from
              the completed experiment.
            </p>
          </div>

          <span className="chip chip-active">
            🔬 Research
          </span>

        </div>

      </div>


      {/* =====================================================
          AUTOMATIC INSIGHTS
      ===================================================== */}
      <div className="card p-6">

        <div className="mb-5">
          <p className="font-semibold text-lg">
            💡 Automatic Insights
          </p>

          <p className="text-sm text-slate-500 mt-1">
            These observations are generated directly from the
            experiment results.
          </p>
        </div>

        {insights.length > 0 ? (
          <div className="space-y-3">

            {insights.map((insight, index) => (
              <div
                key={index}
                className="flex items-start gap-3 rounded-xl border border-border bg-panel/50 p-4"
              >

                <span className="mt-0.5 w-7 h-7 shrink-0 rounded-full bg-primary-50 text-primary-600 grid place-items-center text-xs font-bold">
                  {index + 1}
                </span>

                <p className="text-sm text-slate-700 leading-6">
                  {insight}
                </p>

              </div>
            ))}

          </div>
        ) : (
          <p className="text-sm text-slate-500">
            No automatic insights were generated for this experiment.
          </p>
        )}

      </div>


      {/* =====================================================
          EXPERIMENT OVERVIEW
      ===================================================== */}
      {researchSummary?.experiment_summary && (
        <div className="card p-6">

          <p className="font-semibold text-lg">
            📊 Experiment Overview
          </p>

          <p className="text-sm text-slate-500 mt-1">
            Configuration and execution information used for
            this experiment.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">

            <InfoCard
              label="Models Requested"
              value={
                researchSummary.experiment_summary
                  .models_requested ?? "—"
              }
            />

            <InfoCard
              label="Models Succeeded"
              value={
                researchSummary.experiment_summary
                  .models_succeeded ?? "—"
              }
            />

            <InfoCard
              label="Models Failed"
              value={
                researchSummary.experiment_summary
                  .models_failed ?? "—"
              }
            />

            <InfoCard
              label="Random State"
              value={
                researchSummary.experiment_summary
                  .random_state ?? "—"
              }
            />

          </div>

        </div>
      )}


      {/* =====================================================
          CROSS VALIDATION
      ===================================================== */}
      {cvInfo && (
        <div className="card p-6">

          <p className="font-semibold text-lg">
            🔁 Cross-Validation Analysis
          </p>

          <p className="text-sm text-slate-500 mt-1">
            Evaluation was performed using repeated stratified
            cross-validation.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4">

            <InfoCard
              label="Folds"
              value={cvInfo.n_splits ?? "—"}
            />

            <InfoCard
              label="Repeats"
              value={cvInfo.n_repeats ?? "—"}
            />

            <InfoCard
              label="Total Fold Runs"
              value={cvInfo.n_folds_requested ?? "—"}
            />

          </div>

        </div>
      )}


      {/* =====================================================
          STATISTICAL SIGNIFICANCE
      ===================================================== */}
      {significance && (
        <div className="card p-6">

          <p className="font-semibold text-lg">
            📐 Statistical Significance
          </p>

          <p className="text-sm text-slate-500 mt-1">
            Statistical comparison results generated from the
            cross-validation experiment.
          </p>

          <div className="mt-4 rounded-xl border border-border bg-panel/50 p-4">

            <pre className="text-xs whitespace-pre-wrap overflow-auto">
              {JSON.stringify(significance, null, 2)}
            </pre>

          </div>

          <p className="text-xs text-slate-400 mt-3">
            Detailed statistical results are also available in the
            Cross-Validation and Statistical Significance panels.
          </p>

        </div>
      )}


      {/* =====================================================
          IMBALANCE IMPACT
      ===================================================== */}
      {beforeAfter?.available && (
        <div className="card p-6">

          <div className="flex items-start justify-between gap-4">

            <div>
              <p className="font-semibold text-lg">
                ⚖️ Imbalance Handling Impact
              </p>

              <p className="text-sm text-slate-500 mt-1">
                Before/after analysis is available for this experiment.
              </p>
            </div>

            <Link
              to="/imbalance-impact"
              className="btn-secondary shrink-0"
            >
              View Detailed Analysis →
            </Link>

          </div>

        </div>
      )}


      {/* =====================================================
          CONCLUSION
      ===================================================== */}
      <div className="card p-6 bg-gradient-to-br from-slate-900 to-slate-800 text-white">

        <div className="flex items-center gap-2 mb-3">
          <span className="text-xl">📝</span>

          <p className="font-semibold text-lg">
            Experimental Conclusion
          </p>
        </div>

        {conclusion ? (
          <p className="text-sm text-slate-200 leading-7">
            {conclusion}
          </p>
        ) : (
          <p className="text-sm text-slate-400">
            No automatic conclusion was generated for this experiment.
          </p>
        )}

      </div>


      {/* =====================================================
          ACTIONS
      ===================================================== */}
      <div className="flex justify-end gap-3">

        <Link
          to="/dashboard"
          className="btn-secondary"
        >
          ← Back to Dashboard
        </Link>

        <Link
          to="/reports"
          className="btn-primary"
        >
          Export Full Report →
        </Link>

      </div>

    </div>
  );
}


/* =========================================================
   SMALL COMPONENT
========================================================= */

function InfoCard({ label, value }) {
  return (
    <div className="rounded-xl border border-border bg-panel/50 p-3">

      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p className="font-bold mt-1">
        {value}
      </p>

    </div>
  );
}