import datetime as dt

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import Dataset, Experiment
from app.schemas import ExperimentConfig
from app.ml.orchestrator import run_experiment
from app.utils.errors import friendly_message, FriendlyError

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post("/run")
def run_new_experiment(cfg: ExperimentConfig, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == cfg.dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found. Upload or select a dataset first.")

    experiment = Experiment(
        name=f"Experiment on {dataset.name}",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        config_json=cfg.dict(),
        status="running",
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    try:
        df = pd.read_csv(dataset.file_path)
    except Exception as exc:
        experiment.status = "failed"
        experiment.error_message = friendly_message(exc)
        db.commit()
        raise HTTPException(400, experiment.error_message)

    config_dict = cfg.dict()
    config_dict["target_column"] = cfg.target_column or dataset.target_column

    try:
        outcome = run_experiment(df, config_dict)
    except FriendlyError as exc:
        experiment.status = "failed"
        experiment.error_message = exc.message
        db.commit()
        raise HTTPException(exc.status_code, exc.message)
    except Exception as exc:
        experiment.status = "failed"
        experiment.error_message = friendly_message(exc)
        db.commit()
        raise HTTPException(400, experiment.error_message)

    experiment.results_json = outcome
    experiment.insights_json = outcome["insights"]
    experiment.conclusion_text = outcome["conclusion"]
    experiment.status = "completed"
    experiment.completed_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(experiment)

    return _serialize_experiment(experiment)


@router.get("")
def list_experiments(db: Session = Depends(get_db)):
    experiments = db.query(Experiment).order_by(Experiment.id.desc()).all()
    return [
        {
            "id": e.id,
            "name": e.name,
            "dataset_name": e.dataset_name,
            "status": e.status,
            "created_at": e.created_at.isoformat() if e.created_at else None,
            "config": e.config_json,
            "summary": _quick_summary(e),
        }
        for e in experiments
    ]


@router.get("/{experiment_id}")
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    e = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not e:
        raise HTTPException(404, "Experiment not found.")
    return _serialize_experiment(e)


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    e = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not e:
        raise HTTPException(404, "Experiment not found.")
    db.delete(e)
    db.commit()
    return {"deleted": True}


@router.post("/{experiment_id}/rerun")
def rerun_experiment(experiment_id: int, db: Session = Depends(get_db)):
    e = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not e:
        raise HTTPException(404, "Experiment not found.")
    cfg = ExperimentConfig(**e.config_json)
    return run_new_experiment(cfg, db)


def _quick_summary(e: Experiment):
    if not e.results_json:
        return None
    results = e.results_json.get("results", [])
    ok = [r for r in results if "error" not in r]
    best_f1 = max((r["f1"] for r in ok if r.get("f1") is not None), default=None)
    return {
        "n_models": len(results),
        "n_successful": len(ok),
        "best_f1": round(best_f1, 3) if best_f1 is not None else None,
        "imbalance_method_label": e.results_json.get("imbalance_method_label"),
    }


def _serialize_experiment(e: Experiment):
    return {
        "id": e.id,
        "name": e.name,
        "dataset_name": e.dataset_name,
        "status": e.status,
        "config": e.config_json,
        "results": e.results_json,
        "insights": e.insights_json,
        "conclusion": e.conclusion_text,
        "error_message": e.error_message,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }
