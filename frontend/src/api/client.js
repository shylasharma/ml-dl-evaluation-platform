import axios from "axios";

const client = axios.create({ baseURL: "/api" });

// Normalize backend errors into a plain, user-facing message string.
client.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err?.response?.data?.detail ||
      err?.message ||
      "Something went wrong talking to the server.";
    return Promise.reject(new Error(message));
  }
);

export const api = {
  // Dataset
  uploadDataset: (file, targetColumn) => {
    const form = new FormData();
    form.append("file", file);
    if (targetColumn) form.append("target_column", targetColumn);
    return client.post("/dataset/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data);
  },
  listSampleDatasets: () => client.get("/dataset/samples").then((r) => r.data),
  loadSampleDataset: (filename) =>
    client.post(`/dataset/samples/${filename}/load`).then((r) => r.data),
  getDataset: (id) => client.get(`/dataset/${id}`).then((r) => r.data),
  previewDataset: (id, rows = 25) =>
    client.get(`/dataset/${id}/preview`, { params: { rows } }).then((r) => r.data),

  // Models & metrics
  listModels: () => client.get("/models").then((r) => r.data),
  listImbalanceMethods: () => client.get("/imbalance-methods").then((r) => r.data),
  listMetrics: () => client.get("/metrics").then((r) => r.data),

  // Experiments
  runExperiment: (config) => client.post("/experiments/run", config).then((r) => r.data),
  listExperiments: () => client.get("/experiments").then((r) => r.data),
  getExperiment: (id) => client.get(`/experiments/${id}`).then((r) => r.data),
  deleteExperiment: (id) => client.delete(`/experiments/${id}`).then((r) => r.data),
  rerunExperiment: (id) => client.post(`/experiments/${id}/rerun`).then((r) => r.data),

  // Reports
  exportUrl: (id, format) => `/api/experiments/${id}/export?format=${format}`,
};

export default api;
