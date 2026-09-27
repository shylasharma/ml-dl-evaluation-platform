import os
import shutil
import uuid

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import DATASETS_DIR
from app.database import get_db
from app.db_models import Dataset
from app.ml.dataset_analysis import profile_dataset
from app.utils.errors import friendly_message, FriendlyError

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

ALLOWED_EXTENSIONS = {".csv"}


def _read_csv_safely(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise FriendlyError(f"This file could not be read as a CSV. Details: {exc}")

    if df.shape[0] < 20:
        raise FriendlyError("The dataset has too few rows (minimum 20) for a meaningful train/test split.")
    if df.shape[1] < 2:
        raise FriendlyError("The dataset needs at least one feature column plus a target column.")
    return df


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(None),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only CSV files are supported.")

    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(DATASETS_DIR, saved_name)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        df = _read_csv_safely(saved_path)
        profile = profile_dataset(df, target_hint=target_column)
    except FriendlyError as exc:
        os.remove(saved_path)
        raise HTTPException(exc.status_code, exc.message)
    except Exception as exc:
        os.remove(saved_path)
        raise HTTPException(400, friendly_message(exc))

    dataset = Dataset(
        name=file.filename,
        file_path=saved_path,
        target_column=profile["target_column"],
        profile_json=profile,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {"dataset_id": dataset.id, "name": dataset.name, **profile}


@router.get("/samples")
def list_sample_datasets():
    """Bundled example datasets a user can pick without uploading anything."""
    from app.config import BASE_DIR
    sample_dir = os.path.join(BASE_DIR, "sample_data")
    samples = []
    if os.path.isdir(sample_dir):
        for fname in os.listdir(sample_dir):
            if fname.endswith(".csv"):
                samples.append({"filename": fname, "label": fname.replace("_", " ").replace(".csv", "").title()})
    return samples


@router.post("/samples/{filename}/load")
def load_sample_dataset(filename: str, target_column: str = None, db: Session = Depends(get_db)):
    from app.config import BASE_DIR
    src_path = os.path.join(BASE_DIR, "sample_data", filename)
    if not os.path.isfile(src_path):
        raise HTTPException(404, "Sample dataset not found.")

    saved_name = f"{uuid.uuid4().hex}.csv"
    saved_path = os.path.join(DATASETS_DIR, saved_name)
    shutil.copyfile(src_path, saved_path)

    try:
        df = _read_csv_safely(saved_path)
        profile = profile_dataset(df, target_hint=target_column)
    except FriendlyError as exc:
        os.remove(saved_path)
        raise HTTPException(exc.status_code, exc.message)

    dataset = Dataset(name=filename, file_path=saved_path,
                       target_column=profile["target_column"], profile_json=profile)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {"dataset_id": dataset.id, "name": dataset.name, **profile}


@router.get("/{dataset_id}")
def get_dataset_profile(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    return {"dataset_id": dataset.id, "name": dataset.name, **dataset.profile_json}


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: int, rows: int = 25, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    try:
        df = pd.read_csv(dataset.file_path)
    except Exception as exc:
        raise HTTPException(400, friendly_message(exc))
    preview = df.head(rows).where(pd.notnull(df.head(rows)), None).to_dict(orient="records")
    return {"columns": list(df.columns), "rows": preview}
