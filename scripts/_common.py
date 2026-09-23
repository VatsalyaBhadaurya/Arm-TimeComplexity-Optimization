import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ["bimaual_dataset_new_1", "deksha_data_330_1"]
REPORTS = ROOT / "reports"
OPTIMIZED = ROOT / "optimized"
