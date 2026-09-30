import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import api from "../api/client.js";

export default function History() {
  const {
    activeExperiment,
    setActiveExperiment,
    notify,
  } = useApp();

  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionId, setActionId] = useState(null);

  const loadHistory = async () => {
    setLoading(true);

    try {
      const data = await api.listExperiments();

      // Support either a direct array or { experiments: [...] }
      const items = Array.isArray(data)
        ? data
        : data?.experiments || [];

      setExperiments(items);
    } catch (err) {
      notify(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const openExperiment = async (id) => {
    setActionId(id);

    try {
      const experiment = await api.getExperiment(id);

      setActiveExperiment(experiment);
    } catch (err) {
      notify(err.message, "error");
    } finally {
      setActionId(null);
    }
  };

  const deleteExperiment = async (id) => {
    const confirmed = window.confirm(
      "Delete this experiment? This action cannot be undone."
    );

    if (!confirmed) return;

    setActionId(id);

    try {
      await api.deleteExperiment(id);

      if (
        activeExperiment &&
        String(activeExperiment.id) === String(id)
      ) {
        setActiveExperiment(null);
      }

      notify("Experiment deleted.", "success");

      await loadHistory();
    } catch (err) {
      notify(err.message, "error");
    } finally {
      setActionId(null);
    }
  };

  const rerunExperiment = async (id) => {
    setActionId(id);

    try {
      const result = await api.rerunExperiment(id);

      setActiveExperiment(result);

      notify(
        "Experiment re-run completed successfully.",
        "success"
      );

      await loadHistory();
    } catch (err) {
      notify(err.message, "error");
    } finally {
      setActionId(null);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">

      {/* =====================================================
          HEADER
      ===================================================== */}
      <div className="card p-6">

        <div className="flex items-start justify-between gap-4">

          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-primary-600">
              Research Management
            </p>

            <h1 className="text-2xl font-bold mt-1">
              Experiment History
            </h1>

            <p className="text-sm text-slate-500 mt-2">
              Revisit completed experiments, inspect their
              configurations and reproduce previous analyses.
            </p>
          </div>

          <Link
            to="/experiments/new"
            className="btn-primary shrink-0"
          >
            + New Experiment
          </Link>

        </div>

      </div>


      {/* =====================================================
          LOADING
      ===================================================== */}
      {loading && (
        <div className="card p-10 text-center">
          <div className="text-2xl mb-2">⏳</div>

          <p className="font-medium">
            Loading experiment history…
          </p>

        </div>
      )}


      {/* =====================================================
          EMPTY STATE
      ===================================================== */}
      {!loading && experiments.length === 0 && (
        <div className="card p-10 text-center">

          <div className="text-4xl mb-3">
            🧪
          </div>

          <p className="font-semibold">
            No experiments yet
          </p>

          <p className="text-sm text-slate-500 mt-1">
            Run your first experiment to start building
            your research history.
          </p>

          <Link
            to="/experiments/new"
            className="btn-primary mt-5 inline-flex"
          >
            Run New Experiment →
          </Link>

        </div>
      )}


      {/* =====================================================
          EXPERIMENT LIST
      ===================================================== */}
      {!loading && experiments.length > 0 && (
        <div className="space-y-4">

          {experiments.map((experiment) => (
            <ExperimentCard
              key={experiment.id}
              experiment={experiment}
              busy={actionId === experiment.id}
              onOpen={openExperiment}
              onRerun={rerunExperiment}
              onDelete={deleteExperiment}
            />
          ))}

        </div>
      )}

    </div>
  );
}


/* =========================================================
   EXPERIMENT CARD
========================================================= */

function ExperimentCard({
  experiment,
  busy,
  onOpen,
  onRerun,
  onDelete,
}) {
  const config = experiment.config || {};
  const results = experiment.results || {};

  const models =
    config.models ||
    config.selected_models ||
    [];

  const modelCount =
    Array.isArray(models)
      ? models.length
      : results.results?.length || "—";

  const imbalanceMethod =
    config.imbalance_method ||
    results.imbalance_method_label ||
    "—";

  const primaryMetric =
    config.primary_metric ||
    "f1";

  const randomState =
    config.random_state ??
    "—";

  const cvFolds =
    config.cv_folds ??
    0;

  const cvRepeats =
    config.cv_repeats ??
    1;

  const status =
    experiment.status ||
    "unknown";

  return (
    <div className="card p-5">

      {/* Header */}
      <div className="flex items-start justify-between gap-4">

        <div>

          <div className="flex items-center gap-2">

            <h2 className="font-semibold">
              Experiment #{experiment.id}
            </h2>

            <StatusBadge status={status} />

          </div>

          <p className="text-sm text-slate-500 mt-1">
            {experiment.dataset_name ||
              config.dataset_name ||
              `Dataset ${experiment.dataset_id ?? "—"}`}
          </p>

        </div>

        <span className="text-xs text-slate-400">
          {formatDate(
            experiment.created_at ||
            experiment.created
          )}
        </span>

      </div>


      {/* Reproducibility information */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">

        <InfoCard
          label="Models"
          value={modelCount}
        />

        <InfoCard
          label="Imbalance"
          value={formatLabel(imbalanceMethod)}
        />

        <InfoCard
          label="Primary Metric"
          value={formatLabel(primaryMetric)}
        />

        <InfoCard
          label="Random State"
          value={randomState}
        />

        <InfoCard
          label="CV Folds"
          value={cvFolds || "Off"}
        />

        <InfoCard
          label="CV Repeats"
          value={cvFolds ? cvRepeats : "—"}
        />

        <InfoCard
          label="Before / After"
          value={
            config.compare_before_after
              ? "Enabled"
              : "Disabled"
          }
        />

        <InfoCard
          label="Mode"
          value={formatLabel(config.mode || "—")}
        />

      </div>


      {/* Actions */}
      <div className="flex flex-wrap justify-end gap-2 mt-5 pt-4 border-t border-border">

        <button
          type="button"
          className="btn-secondary"
          disabled={busy}
          onClick={() => onOpen(experiment.id)}
        >
          {busy ? "Loading…" : "View Experiment"}
        </button>

        <button
          type="button"
          className="btn-secondary"
          disabled={busy}
          onClick={() => onRerun(experiment.id)}
        >
          {busy ? "Running…" : "↻ Re-run"}
        </button>

        <button
          type="button"
          className="btn-secondary text-accent-rose"
          disabled={busy}
          onClick={() => onDelete(experiment.id)}
        >
          Delete
        </button>

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


function StatusBadge({ status }) {
  const normalized = String(status).toLowerCase();

  const isCompleted = normalized === "completed";
  const isFailed = normalized === "failed";

  return (
    <span
      className={
        isCompleted
          ? "text-xs px-2 py-1 rounded-full bg-emerald-50 text-emerald-700"
          : isFailed
          ? "text-xs px-2 py-1 rounded-full bg-rose-50 text-rose-700"
          : "text-xs px-2 py-1 rounded-full bg-amber-50 text-amber-700"
      }
    >
      {formatLabel(status)}
    </span>
  );
}


function formatLabel(value) {
  if (value == null) return "—";

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}


function formatDate(value) {
  if (!value) return "Date unavailable";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString();
}