import React from "react";

const METRICS = [
  "accuracy",
  "precision",
  "recall",
  "specificity",
  "f1",
  "roc_auc",
  "pr_auc",
  "mcc",
  "balanced_accuracy",
  "g_mean",
];

export default function ResearchSummary({ research }) {
  if (!research) return null;

  const experiment = research.experiment_summary || {};
  const dataset = research.dataset || {};
  const modelComparison = research.model_comparison || {};
  const metricSummary = research.metric_summary || {};
  const beforeAfter = research.before_after || {};

  const models = modelComparison.models || [];

  return (
    <div className="space-y-5">

      {/* =====================================================
          RESEARCH OVERVIEW
      ===================================================== */}
      <div className="card p-5">
        <div className="flex items-start justify-between gap-4">

          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-primary-600">
              Research Analysis
            </p>

            <h2 className="text-xl font-bold mt-1">
              Experiment Research Summary
            </h2>

            <p className="text-sm text-slate-500 mt-1">
              Reproducible analysis of model performance,
              imbalance handling and experimental settings.
            </p>
          </div>

          <span className="chip chip-active">
            🔬 Research
          </span>

        </div>

        {/* Experiment information */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">

          <ResearchStat
            label="Models Requested"
            value={experiment.models_requested ?? "—"}
          />

          <ResearchStat
            label="Models Succeeded"
            value={experiment.models_succeeded ?? "—"}
          />

          <ResearchStat
            label="Models Failed"
            value={experiment.models_failed ?? "—"}
          />

          <ResearchStat
            label="Random State"
            value={experiment.random_state ?? "—"}
          />

          <ResearchStat
            label="Imbalance Method"
            value={experiment.imbalance_method ?? "—"}
          />

          <ResearchStat
            label="Dataset Rows"
            value={
              dataset.rows != null
                ? Number(dataset.rows).toLocaleString()
                : "—"
            }
          />

          <ResearchStat
            label="Features"
            value={dataset.features ?? "—"}
          />

          <ResearchStat
            label="Imbalance Ratio"
            value={
              dataset.imbalance_ratio != null
                ? `${dataset.imbalance_ratio}:1`
                : "—"
            }
          />

        </div>
      </div>


      {/* =====================================================
          MODEL COMPARISON
      ===================================================== */}
      {models.length > 0 && (
        <div className="card p-5 overflow-x-auto scroll-thin">

          <div className="mb-4">
            <p className="font-semibold text-lg">
              Model Comparison Summary
            </p>

            <p className="text-sm text-slate-500 mt-1">
              Research metrics produced by the evaluated models.
            </p>
          </div>

          <table className="text-sm w-full border-collapse min-w-[850px]">

            <thead>
              <tr className="text-left text-xs text-slate-500 border-b border-border">

                <th className="py-2 pr-4">
                  Model
                </th>

                <th className="py-2 pr-4">
                  Family
                </th>

                <th className="py-2 pr-4">
                  Accuracy
                </th>

                <th className="py-2 pr-4">
                  Precision
                </th>

                <th className="py-2 pr-4">
                  Recall
                </th>

                <th className="py-2 pr-4">
                  F1
                </th>

                <th className="py-2 pr-4">
                  ROC-AUC
                </th>

                <th className="py-2 pr-4">
                  PR-AUC
                </th>

                <th className="py-2 pr-4">
                  MCC
                </th>

              </tr>
            </thead>

            <tbody>

              {models.map((model, index) => (
                <tr
                  key={model.model_key ?? index}
                  className="border-b border-border/60 hover:bg-panel/60"
                >

                  <td className="py-2 pr-4 font-medium">
                    {model.model_label ?? model.model_key ?? "—"}
                  </td>

                  <td className="py-2 pr-4">
                    {model.family ?? "—"}
                  </td>

                  <MetricCell value={model.accuracy} />
                  <MetricCell value={model.precision} />
                  <MetricCell value={model.recall} />
                  <MetricCell value={model.f1} />
                  <MetricCell value={model.roc_auc} />
                  <MetricCell value={model.pr_auc} />
                  <MetricCell value={model.mcc} />

                </tr>
              ))}

            </tbody>

          </table>

        </div>
      )}


      {/* =====================================================
          METRIC-WISE ANALYSIS
      ===================================================== */}
      <div className="card p-5">

        <div className="mb-4">
          <p className="font-semibold text-lg">
            Metric-wise Analysis
          </p>

          <p className="text-sm text-slate-500 mt-1">
            Performance values reported for each evaluation metric.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">

          {METRICS.map((metric) => {

            const values = metricSummary[metric] || [];

            return (
              <div
                key={metric}
                className="rounded-xl border border-border bg-panel/50 p-3"
              >

                <p className="text-xs text-slate-500">
                  {formatLabel(metric)}
                </p>

                {values.length === 0 ? (
                  <p className="font-bold mt-1">
                    —
                  </p>
                ) : (
                  <div className="mt-2 space-y-1">

                    {values.map((item, index) => (
                      <div
                        key={`${item.model_key ?? index}-${metric}`}
                        className="flex items-center justify-between gap-2"
                      >

                        <span
                          className="text-xs truncate"
                          title={item.model_label}
                        >
                          {item.model_label}
                        </span>

                        <span className="font-semibold text-sm">
                          {formatNumber(item.value)}
                        </span>

                      </div>
                    ))}

                  </div>
                )}

              </div>
            );
          })}

        </div>

      </div>


      {/* =====================================================
          BEFORE / AFTER ANALYSIS
      ===================================================== */}
      {beforeAfter.available && (
        <BeforeAfterSection beforeAfter={beforeAfter} />
      )}

    </div>
  );
}


/* =========================================================
   BEFORE / AFTER SECTION
========================================================= */

function BeforeAfterSection({ beforeAfter }) {

  const models = beforeAfter.models || {};

  return (
    <div className="card p-5">

      <div className="mb-4">

        <p className="font-semibold text-lg">
          ⚖️ Before / After Imbalance Analysis
        </p>

        <p className="text-sm text-slate-500 mt-1">
          Change in model performance after applying the selected
          imbalance-handling technique.
        </p>

      </div>

      <div className="space-y-5">

        {Object.entries(models).map(
          ([modelKey, model]) => {

            if (!model.available) {
              return (
                <div
                  key={modelKey}
                  className="rounded-xl border border-border p-4"
                >
                  <p className="font-medium">
                    {model.model_label || modelKey}
                  </p>

                  <p className="text-sm text-slate-500 mt-1">
                    Before/after comparison was not available
                    for this model.
                  </p>
                </div>
              );
            }

            const metrics =
              model.comparison?.metrics || {};

            return (
              <div
                key={modelKey}
                className="rounded-xl border border-border overflow-hidden"
              >

                <div className="px-4 py-3 bg-panel border-b border-border">

                  <p className="font-semibold">
                    {model.model_label || modelKey}
                  </p>

                  <p className="text-xs text-slate-500 mt-1">
                    Performance change after imbalance handling
                  </p>

                </div>

                <div className="overflow-x-auto">

                  <table className="text-sm w-full border-collapse min-w-[650px]">

                    <thead>
                      <tr className="text-left text-xs text-slate-500 border-b border-border">

                        <th className="py-2 px-4">
                          Metric
                        </th>

                        <th className="py-2 px-4">
                          Before
                        </th>

                        <th className="py-2 px-4">
                          After
                        </th>

                        <th className="py-2 px-4">
                          Delta
                        </th>

                        <th className="py-2 px-4">
                          Change %
                        </th>

                      </tr>
                    </thead>

                    <tbody>

                      {Object.entries(metrics).map(
                        ([metric, values]) => (

                          <tr
                            key={metric}
                            className="border-b border-border/60"
                          >

                            <td className="py-2 px-4 font-medium">
                              {formatLabel(metric)}
                            </td>

                            <td className="py-2 px-4">
                              {formatNumber(values.before)}
                            </td>

                            <td className="py-2 px-4">
                              {formatNumber(values.after)}
                            </td>

                            <td className="py-2 px-4">
                              <DeltaValue
                                value={values.delta}
                              />
                            </td>

                            <td className="py-2 px-4">
                              <DeltaValue
                                value={values.percentage_change}
                                percentage
                              />
                            </td>

                          </tr>

                        )
                      )}

                    </tbody>

                  </table>

                </div>

                {/* Summary */}
                {model.comparison?.summary && (
                  <div className="grid grid-cols-3 md:grid-cols-4 gap-3 p-4 bg-panel/40">

                    <ResearchStat
                      label="Improved"
                      value={
                        model.comparison.summary
                          .metrics_improved
                      }
                    />

                    <ResearchStat
                      label="Declined"
                      value={
                        model.comparison.summary
                          .metrics_declined
                      }
                    />

                    <ResearchStat
                      label="Unchanged"
                      value={
                        model.comparison.summary
                          .metrics_unchanged
                      }
                    />

                    <ResearchStat
                      label="Evaluated"
                      value={
                        model.comparison.summary
                          .metrics_evaluated
                      }
                    />

                  </div>
                )}

              </div>
            );
          }
        )}

      </div>

    </div>
  );
}


/* =========================================================
   SMALL COMPONENTS
========================================================= */

function ResearchStat({ label, value }) {

  return (
    <div className="rounded-xl border border-border bg-panel/50 p-3">

      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p className="font-bold mt-1 break-words">
        {value}
      </p>

    </div>
  );
}


function MetricCell({ value }) {

  return (
    <td className="py-2 pr-4">
      {formatNumber(value)}
    </td>
  );
}


function DeltaValue({ value, percentage = false }) {

  if (
    value == null ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  const number = Number(value);

  return (
    <span className="font-medium">
      {number >= 0 ? "+" : ""}
      {number.toFixed(percentage ? 2 : 3)}
      {percentage ? "%" : ""}
    </span>
  );
}


function formatNumber(value) {

  if (
    value == null ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return Number(value).toFixed(3);
}


function formatLabel(value) {

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}