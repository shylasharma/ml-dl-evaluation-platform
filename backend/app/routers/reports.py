import io
import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import Experiment

router = APIRouter(prefix="/api/experiments", tags=["reports"])


def _get_experiment_or_404(experiment_id: int, db: Session) -> Experiment:
    e = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not e:
        raise HTTPException(404, "Experiment not found.")
    if e.status != "completed":
        raise HTTPException(400, "This experiment has not completed successfully yet.")
    return e


def _results_dataframe(e: Experiment) -> pd.DataFrame:
    results = e.results_json.get("results", [])
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"model": r.get("model_label", r.get("model_key")), "family": r.get("family"),
                         "status": "failed", "error": r["error"]})
        else:
            rows.append({
                "model": r["model_label"], "family": r["family"], "status": "ok",
                "accuracy": r.get("accuracy"), "precision": r.get("precision"),
                "recall": r.get("recall"), "specificity": r.get("specificity"),
                "f1": r.get("f1"), "roc_auc": r.get("roc_auc"), "pr_auc": r.get("pr_auc"),
                "mcc": r.get("mcc"), "balanced_accuracy": r.get("balanced_accuracy"),
                "g_mean": r.get("g_mean"), "training_time_sec": r.get("training_time_sec"),
            })
    return pd.DataFrame(rows)


@router.get("/{experiment_id}/export")
def export_experiment(experiment_id: int, format: str = "json", db: Session = Depends(get_db)):
    e = _get_experiment_or_404(experiment_id, db)

    if format == "json":
        payload = {
            "experiment_id": e.id,
            "dataset_name": e.dataset_name,
            "config": e.config_json,
            "results": e.results_json,
            "insights": e.insights_json,
            "conclusion": e.conclusion_text,
        }
        buf = io.BytesIO(json.dumps(payload, indent=2, default=str).encode("utf-8"))
        return StreamingResponse(buf, media_type="application/json", headers={
            "Content-Disposition": f"attachment; filename=experiment_{e.id}.json"
        })

    if format == "csv":
        df = _results_dataframe(e)
        buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        return StreamingResponse(buf, media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename=experiment_{e.id}_results.csv"
        })

    if format == "excel":
        df = _results_dataframe(e)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)
            meta = pd.DataFrame([
                {"key": "Dataset", "value": e.dataset_name},
                {"key": "Imbalance Method", "value": e.results_json.get("imbalance_method_label")},
                {"key": "Conclusion", "value": e.conclusion_text},
            ])
            meta.to_excel(writer, sheet_name="Summary", index=False)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=experiment_{e.id}_report.xlsx"},
        )

    if format == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
        except ImportError:
            raise HTTPException(500, "PDF export requires the 'reportlab' package. Run `pip install reportlab`.")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"Experiment Report — {e.dataset_name}", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Imbalance technique: {e.results_json.get('imbalance_method_label')}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("Conclusion", styles["Heading2"]),
            Paragraph(e.conclusion_text or "", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("Insights", styles["Heading2"]),
        ]
        for insight in (e.insights_json or []):
            story.append(Paragraph(f"• {insight}", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Model Comparison", styles["Heading2"]))

        df = _results_dataframe(e)
        table_data = [list(df.columns)] + df.round(3).astype(str).values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(table)

        doc.build(story)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/pdf", headers={
            "Content-Disposition": f"attachment; filename=experiment_{e.id}_report.pdf"
        })

    raise HTTPException(400, f"Unsupported export format '{format}'. Use json, csv, excel, or pdf.")
