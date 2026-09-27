import React, { useEffect, useState } from "react";
import { useParams, useSearchParams, Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import api from "../api/client.js";
import KpiCard from "../components/common/KpiCard.jsx";
import ConfusionMatrix from "../components/charts/ConfusionMatrix.jsx";
import RocCurve from "../components/charts/RocCurve.jsx";
import PrCurve from "../components/charts/PrCurve.jsx";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend,
} from "recharts";

export default function ModelDetail() {
  const { modelKey } = useParams();
  const [params] = useSearchParams();
  const experimentId = params.get("experiment");
  const { activeExperiment, setActiveExperiment } = useApp();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (experimentId && (!activeExperiment || String(activeExperiment.id) !== experimentId)) {
      setLoading(true);
      api.getExperiment(experimentId).then(setActiveExperiment).finally(() => setLoading(false));
    }
  }, [experimentId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) return <p className="text-center text-slate-400 py-16">Loading…</p>;

  const exp = activeExperiment;
  const model = exp?.results?.results?.find((r) => r.model_key === modelKey);

  if (!exp || !model) {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <p className="font-semibold">Model result not found</p>
        <Link to="/dashboard" className="btn-primary mt-5 inline-flex">Back to Dashboard →</Link>
      </div>
    );
  }

  if (model.error) {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">⚠️</div>
        <p className="font-semibold">{model.model_label} could not be trained</p>
        <p className="text-sm text-slate-500 mt-2">{model.error}</p>
        <Link to="/dashboard" className="btn-primary mt-5 inline-flex">Back to Dashboard →</Link>
      </div>
    );
  }

  const importanceData = model.feature_importance
    ? model.feature_importance.map((v, i) => ({ feature: `F${i + 1}`, importance: v })).sort((a, b) => b.importance - a.importance).slice(0, 15)
    : null;

  const historyData = model.training_history
    ? model.training_history.loss.map((v, i) => ({
        epoch: i + 1, loss: v, val_loss: model.training_history.val_loss[i],
      }))
    : null;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${model.family === "DL" ? "bg-teal-50 text-accent-teal" : "bg-indigo-50 text-primary-600"}`}>
            {model.family}
          </span>
          <h2 className="text-xl font-bold mt-1">{model.model_label}</h2>
        </div>
        <Link to="/dashboard" className="btn-secondary">← Back to Dashboard</Link>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KpiCard label="Accuracy" value={model.accuracy.toFixed(3)} />
        <KpiCard label="Precision" value={model.precision.toFixed(3)} />
        <KpiCard label="Recall" value={model.recall.toFixed(3)} tone="primary" />
        <KpiCard label="F1 Score" value={model.f1.toFixed(3)} tone="primary" />
        <KpiCard label="Specificity" value={model.specificity.toFixed(3)} />
        <KpiCard label="ROC-AUC" value={model.roc_auc != null ? model.roc_auc.toFixed(3) : "—"} />
        <KpiCard label="PR-AUC" value={model.pr_auc != null ? model.pr_auc.toFixed(3) : "—"} tone="teal" />
        <KpiCard label="MCC" value={model.mcc.toFixed(3)} tone="amber" />
        <KpiCard label="Balanced Acc." value={model.balanced_accuracy.toFixed(3)} />
        <KpiCard label="G-Mean" value={model.g_mean.toFixed(3)} />
      </div>

      <div className="card p-4 flex flex-wrap gap-4 text-sm text-slate-500">
        <span>⏱ Training time: <strong className="text-ink">{model.training_time_sec}s</strong></span>
        {model.epochs_run != null && <span>🔁 Epochs run: <strong className="text-ink">{model.epochs_run}</strong></span>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="card p-5">
          <p className="font-semibold mb-3">Confusion Matrix</p>
          <ConfusionMatrix matrix={model.confusion_matrix} labels={exp.results.class_labels} />
        </div>
        <div className="card p-5">
          <p className="font-semibold mb-3">ROC Curve</p>
          <RocCurve curves={[{ label: model.model_label, points: model.roc_curve, color: "#4f46e5" }]} />
        </div>
      </div>

      <div className="card p-5">
        <p className="font-semibold mb-3">Precision-Recall Curve</p>
        <PrCurve curves={[{ label: model.model_label, points: model.pr_curve, color: "#0d9488" }]} />
      </div>

      {importanceData && (
        <div className="card p-5">
          <p className="font-semibold mb-3">Feature Importance (top 15)</p>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={importanceData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} width={50} />
              <Tooltip />
              <Bar dataKey="importance" fill="#4f46e5" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {historyData && (
        <div className="card p-5">
          <p className="font-semibold mb-3">Training Curve (Loss)</p>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={historyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="loss" stroke="#4f46e5" dot={false} strokeWidth={2} name="Training Loss" />
              <Line type="monotone" dataKey="val_loss" stroke="#e11d48" dot={false} strokeWidth={2} name="Validation Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
