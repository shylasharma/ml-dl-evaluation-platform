import datetime as dt

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON

from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    target_column = Column(String, nullable=True)
    profile_json = Column(JSON, nullable=True)  # cached profiling result
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    dataset_id = Column(Integer, nullable=False)
    dataset_name = Column(String, nullable=False)

    config_json = Column(JSON, nullable=False)       # request configuration
    results_json = Column(JSON, nullable=True)        # per-model metrics + curves
    insights_json = Column(JSON, nullable=True)       # list of generated insight strings
    conclusion_text = Column(Text, nullable=True)

    status = Column(String, default="pending")         # pending | running | completed | failed
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=dt.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
