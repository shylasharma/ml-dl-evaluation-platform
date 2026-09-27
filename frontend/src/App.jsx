import React from "react";
import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Layout/Sidebar.jsx";
import Topbar from "./components/Layout/Topbar.jsx";

import Home from "./pages/Home.jsx";
import DatasetPage from "./pages/DatasetPage.jsx";
import ModelsPage from "./pages/ModelsPage.jsx";
import NewExperiment from "./pages/NewExperiment.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Compare from "./pages/Compare.jsx";
import ImbalanceImpact from "./pages/ImbalanceImpact.jsx";
import ModelDetail from "./pages/ModelDetail.jsx";
import Insights from "./pages/Insights.jsx";
import ExperimentHistory from "./pages/ExperimentHistory.jsx";
import Reports from "./pages/Reports.jsx";

export default function App() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 min-w-0">
        <Topbar />
        <main className="p-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/models/:modelKey" element={<ModelDetail />} />
            <Route path="/experiments/new" element={<NewExperiment />} />
            <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/compare" element={<Compare />} />
            <Route path="/imbalance-impact" element={<ImbalanceImpact />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/history" element={<ExperimentHistory />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
