import React from "react";
import { useNavigate } from "react-router-dom";
import DatasetUploader, { DatasetProfileSummary } from "../components/DatasetUploader.jsx";
import { useApp } from "../context/AppContext.jsx";

export default function DatasetPage() {
  const { dataset } = useApp();
  const navigate = useNavigate();

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <DatasetUploader onLoaded={() => {}} />
      {dataset && (
        <>
          <DatasetProfileSummary profile={dataset} />
          <div className="flex justify-end">
            <button onClick={() => navigate("/experiments/new")} className="btn-primary">
              Continue to Model Selection →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
