"""
Central configuration for the ML/DL Evaluation Platform backend.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")
EXPORTS_DIR = os.path.join(STORAGE_DIR, "exports")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", f"sqlite:///{os.path.join(STORAGE_DIR, 'app.db')}"
)

# Safety limits so a slow/careless request can't hang the server forever.
MAX_ROWS_FOR_QUICK_ANALYSIS = 200_000
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.25
DL_DEFAULT_EPOCHS = 30
DL_DEFAULT_BATCH_SIZE = 32
DL_MAX_EPOCHS_ADVANCED = 200

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
