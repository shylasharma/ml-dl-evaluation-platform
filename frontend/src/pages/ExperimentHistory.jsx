import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import api from "../api/client.js";

export default function ExperimentHistory() {
  const { experimentHistory, refreshHistory, notify, setActiveExperiment } = useApp();
  const navigate = useNavigate();

  const view = async (id) => {
    try {
      const exp = await api.getExperiment(id);
      setActiveExperiment(exp);
      navigate(`/dashboard?experiment=${id}`);
    } catch (err) {
      notify(err.message, "error");
    }
  };

  const rerun = async (id) => {
    try {
      notify("Re-running experiment…", "info");
      const exp = await api.rerunExperiment(id);
      setActiveExperiment(exp);
      refreshHistory();
      navigate(`/dashboard?experiment=${exp.id}`);
    } catch (err) {
      notify(err.message, "error");
    }
  };

  const remove = async (id) => {
    try {
      await api.deleteExperiment(id);
      refreshHistory();
      notify("Experiment deleted.", "success");
    } catch (err) {
      notify(err.message, "error");
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      {experimentHistory.length === 0 && (
        <div className="card p-10 text-center">
          <p className="font-semibold">No experiments yet</p>
          <Link to="/experiments/new" className="btn-primary mt-5 inline-flex">Run Your First Experiment →</Link>
        </div>
      )}

      {experimentHistory.map((e) => (
        <div key={e.id} className="card p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <p className="font-semibold">Experiment #{e.id}</p>
              <StatusBadge status={e.status} />
            </div>
            <p className="text-sm text-slate-500 mt-1">
              {e.dataset_name} · {e.config?.models?.length ?? 0} model(s) ·{" "}
              {e.config?.imbalance_method === "none" ? "No Balancing" : e.config?.imbalance_method}
            </p>
            {e.summary && (
              <p className="text-xs text-slate-400 mt-1">
                Best F1: {e.summary.best_f1 ?? "—"} · {e.summary.n_successful}/{e.summary.n_models} models succeeded
              </p>
            )}
            <p className="text-xs text-slate-400 mt-1">{new Date(e.created_at).toLocaleString()}</p>
          </div>
          <div className="flex gap-2 flex-wrap">
            <button onClick={() => view(e.id)} className="btn-secondary text-sm px-3 py-1.5">View</button>
            <button onClick={() => rerun(e.id)} className="btn-secondary text-sm px-3 py-1.5">Re-run</button>
            <a href={api.exportUrl(e.id, "json")} className="btn-secondary text-sm px-3 py-1.5">Export</a>
            <button onClick={() => remove(e.id)} className="btn-secondary text-sm px-3 py-1.5 text-accent-rose">Delete</button>
          </div>
        </div>
      ))}
    </div>
  );
}

function StatusBadge({ status }) {
  const styles = {
    completed: "bg-emerald-50 text-emerald-700",
    failed: "bg-rose-50 text-rose-700",
    running: "bg-amber-50 text-amber-700",
    pending: "bg-slate-100 text-slate-500",
  };
  return <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${styles[status] || styles.pending}`}>{status}</span>;
}
