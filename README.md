# Arm Time-Complexity Optimization

Analysis and cycle-time reduction for two bimanual robot-arm datasets (LeRobot v2.1
format, "gen2" robot: two 7-joint arms + grippers, 30 fps):

- [`aiengineer56/bimaual_dataset_new_1`](https://huggingface.co/datasets/aiengineer56/bimaual_dataset_new_1): 201 episodes, "pick the sprite bottle from the right box and place it in the left box"
- [`aiengineer56/deksha_data_330_1`](https://huggingface.co/datasets/aiengineer56/deksha_data_330_1): 330 episodes, "pick the bottles and place it in the box"

**Full results: [reports/REPORT.md](reports/REPORT.md)**

## Results in brief

| | bimaual_dataset_new_1 | deksha_data_330_1 |
|---|---|---|
| Mean episode now | 31.7 s | 40.0 s |
| `safe` (idle & pauses removed, motion at demo speed) | 27.9 s (−12.0%) | 34.8 s (−12.9%) |
| **`balanced`** (motion up to 2×, within demo p99 joint speed/accel) | **26.4 s (−16.7%)** | **32.4 s (−19.1%)** |
| `fast` (motion up to 3×) | 24.6 s (−22.5%) | 30.2 s (−24.5%) |

The recorded joint path is never changed; only the timing along it is. No episode gets
slower. The biggest remaining opportunity is that the two arms mostly take turns (both
move together only 2–3 s per episode). See section 5 of the report.

## Layout

| Path | What |
|---|---|
| `bimaual_dataset_new_1/`, `deksha_data_330_1/` | source datasets (`data/` parquet + `meta/`; videos not included) |
| `armopt/` | library: dataset I/O, motion analysis, time-optimal retiming |
| `scripts/analyze.py` | per-episode motion/idle analysis → `reports/<dataset>/` |
| `scripts/optimize.py` | retime all episodes under 3 profiles → `reports/<dataset>/`, `optimized/<dataset>/` |
| `scripts/report.py` | builds `reports/REPORT.md` and `reports/figures/` |
| `optimized/<dataset>/` | retimed dataset (balanced profile), LeRobot layout, with a `source_frame` column for retiming videos |
| `tests/` | unit tests |
| `.github/workflows/fetch-datasets.yml` | manual workflow that re-downloads the datasets from Hugging Face |

## Run

```bash
pip install -r requirements.txt
python scripts/analyze.py
python scripts/optimize.py            # --write-profile safe|balanced|fast
python scripts/report.py
python -m pytest
```

The whole pipeline takes about 30 s on a laptop CPU.
