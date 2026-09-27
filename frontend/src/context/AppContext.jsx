import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import api from "../api/client.js";

const AppContext = createContext(null);

export function AppProvider({ children }) {
    const [dataset, setDataset] = useState(null); // full profile object
  const [modelsCatalog, setModelsCatalog] = useState({ ml_models: [], dl_models: [] });
  const [imbalanceMethods, setImbalanceMethods] = useState([]);
  const [metricsCatalog, setMetricsCatalog] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]); // shared across Models & Experiment Lab
  const [activeExperiment, setActiveExperiment] = useState(null); // full experiment result
  const [experimentHistory, setExperimentHistory] = useState([]);
  const [toast, setToast] = useState(null);

  const notify = useCallback((message, kind = "info") => {
    setToast({ message, kind, id: Date.now() });
  }, []);

  useEffect(() => {
    api.listModels().then(setModelsCatalog).catch(() => {});
    api.listImbalanceMethods().then(setImbalanceMethods).catch(() => {});
    api.listMetrics().then(setMetricsCatalog).catch(() => {});
  }, []);

  const refreshHistory = useCallback(() => {
    api.listExperiments().then(setExperimentHistory).catch(() => {});
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(t);
  }, [toast]);

    const value = {
    dataset, setDataset,
    modelsCatalog,
    imbalanceMethods,
    metricsCatalog,
    selectedModels, setSelectedModels,
    activeExperiment, setActiveExperiment,
    experimentHistory, refreshHistory,
    notify, toast,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
