import React from "react";
import { Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import api from "../api/client.js";

const FORMATS = [
  {
    key: "pdf",
    label: "PDF Research Report",
    icon: "📄",
    desc: "Formatted research report containing experiment context, model results, insights and conclusion.",
  },
  {
    key: "excel",
    label: "Excel Research Workbook",
    icon: "📊",
    desc: "Structured workbook for further analysis of experiment results and metrics.",
  },
  {
    key: "csv",
    label: "CSV Results",
    icon: "📑",
    desc: "Model-level comparison data for external statistical or spreadsheet analysis.",
  },
  {
    key: "json",
    label: "JSON Experiment Record",
    icon: "🧩",
    desc: "Machine-readable experiment configuration and results for reproducibility.",
  },
];

export default function Reports() {
  const { activeExperiment } = useApp();

  if (!activeExperiment || activeExperiment.status !== "completed") {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">📄</div>

        <p className="font-semibold">
          No experiment selected
        </p>

        <p className="text-sm text-slate-500 mt-1">
          Run or open a completed experiment to export its research report.
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

  const results = activeExperiment.results || {};

  const modelResults = results.results || [];

  const successfulModels = modelResults.filter(
    (model) => !model.error
  );

  const researchSummary =
    results.research_summary || {};

  const experimentSummary =
    researchSummary.experiment_summary || {};

  const beforeAfter =
    results.before_after;

  const cvEnabled =
    Boolean(results.cv_enabled);

  const significance =
    results.significance;

  const insights =
    results.insights ||
    activeExperiment.insights ||
    [];

  const conclusion =
    results.conclusion ||
    activeExperiment.conclusion ||
    "";

  return (
    <div className="max-w-5xl mx-auto space-y-6">

      {/* =====================================================
          HEADER
      ===================================================== */}
      <div className="card p-6">

        <div className="flex items-start justify-between gap-4">

          <div>

            <p className="text-xs font-semibold uppercase tracking-wide text-primary-600">
              Research Reporting
            </p>

            <h1 className="text-2xl font-bold mt-1">
              Experiment #{activeExperiment.id} Reports
            </h1>

            <p className="text-sm text-slate-500 mt-2">
              Export the completed experiment in a format suitable
              for research analysis, documentation and reproducibility.
            </p>

          </div>

          <Link
            to="/dashboard"
            className="btn-secondary shrink-0"
          >
            ← Dashboard
          </Link>

        </div>

      </div>


      {/* =====================================================
          EXPERIMENT SNAPSHOT
      ===================================================== */}
      <div className="card p-5">

        <p className="font-semibold text-lg">
          📊 Experiment Snapshot
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">

          <InfoCard
            label="Dataset"
            value={activeExperiment.dataset_name || "—"}
          />

          <InfoCard
            label="Models"
            value={successfulModels.length}
          />

          <InfoCard
            label="Imbalance Method"
            value={
              results.imbalance_method_label || "—"
            }
          />

          <InfoCard
            label="Random State"
            value={
              experimentSummary.random_state ??
              "—"
            }
          />

          <InfoCard
            label="CV"
            value={
              cvEnabled
                ? `${results.cv_info?.n_splits || "—"} folds × ${
                    results.cv_info?.n_repeats || "—"
                  }`
                : "Disabled"
            }
          />

          <InfoCard
            label="Before / After"
            value={
              beforeAfter?.available
                ? "Available"
                : "Not available"
            }
          />

          <InfoCard
            label="Insights"
            value={insights.length}
          />

          <InfoCard
            label="Conclusion"
            value={conclusion ? "Available" : "Not available"}
          />

        </div>

      </div>


      {/* =====================================================
          EXPORT OPTIONS
      ===================================================== */}
      <div>

        <div className="mb-4">

          <p className="font-semibold text-lg">
            📥 Export Formats
          </p>

          <p className="text-sm text-slate-500 mt-1">
            Choose the format that best fits your research workflow.
          </p>

        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">

          {FORMATS.map((format) => (

            <a
              key={format.key}
              href={api.exportUrl(
                activeExperiment.id,
                format.key
              )}
              className="card p-5 hover:border-primary-300 hover:shadow-sm transition-all"
            >

              <div className="text-3xl mb-3">
                {format.icon}
              </div>

              <p className="font-semibold">
                {format.label}
              </p>

              <p className="text-sm text-slate-500 mt-1">
                {format.desc}
              </p>

              <p className="text-sm text-primary-600 font-medium mt-4">
                Download →
              </p>

            </a>

          ))}

        </div>

      </div>


      {/* =====================================================
          REPORT CONTENT
      ===================================================== */}
      <div className="card p-5">

        <p className="font-semibold text-lg">
          📋 Research Report Contents
        </p>

        <p className="text-sm text-slate-500 mt-1">
          The experiment record contains the information required
          to interpret and reproduce the analysis.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-5">

          <ReportItem
            icon="🗂️"
            title="Dataset Information"
            text="Dataset size, features and class-distribution context."
          />

          <ReportItem
            icon="⚖️"
            title="Imbalance Handling"
            text="Selected imbalance technique and its experimental context."
          />

          <ReportItem
            icon="🤖"
            title="Model Evaluation"
            text="Model-by-model results across the evaluation metrics."
          />

          <ReportItem
            icon="📈"
            title="Performance Metrics"
            text="Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, MCC, Balanced Accuracy and G-Mean."
          />

          <ReportItem
            icon="⚖️"
            title="Before / After Analysis"
            text="Metric changes associated with the selected imbalance-handling method."
          />

          <ReportItem
            icon="🔁"
            title="Cross-Validation"
            text="Cross-validation configuration and aggregated results when enabled."
          />

          <ReportItem
            icon="📐"
            title="Statistical Analysis"
            text="Statistical significance results when cross-validation is enabled."
          />

          <ReportItem
            icon="💡"
            title="Automatic Insights"
            text="Observations generated from the actual experiment results."
          />

          <ReportItem
            icon="📝"
            title="Experimental Conclusion"
            text="Automatically generated conclusion with experiment-specific limitations."
          />

          <ReportItem
            icon="🔬"
            title="Reproducibility Information"
            text="Configuration and random-state information required to reproduce the experiment."
          />

        </div>

      </div>


      {/* =====================================================
          QUICK NAVIGATION
      ===================================================== */}
      <div className="flex flex-wrap justify-end gap-3">

        <Link
          to="/history"
          className="btn-secondary"
        >
          Experiment History →
        </Link>

        <Link
          to="/insights"
          className="btn-secondary"
        >
          Insights & Conclusion →
        </Link>

        <Link
          to="/dashboard"
          className="btn-primary"
        >
          View Full Dashboard →
        </Link>

      </div>

    </div>
  );
}


/* =========================================================
   SMALL COMPONENTS
========================================================= */

function InfoCard({ label, value }) {
  return (
    <div className="rounded-xl border border-border bg-panel/50 p-3">

      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p className="font-semibold mt-1 break-words">
        {value}
      </p>

    </div>
  );
}


function ReportItem({ icon, title, text }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-border bg-panel/40 p-4">

      <div className="text-xl shrink-0">
        {icon}
      </div>

      <div>
        <p className="font-medium">
          {title}
        </p>

        <p className="text-sm text-slate-500 mt-1 leading-5">
          {text}
        </p>
      </div>

    </div>
  );
}