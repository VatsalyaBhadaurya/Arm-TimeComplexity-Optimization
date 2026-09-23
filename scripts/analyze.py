"""Per-episode motion analysis of each dataset.

Writes reports/<dataset>/episode_metrics.csv and reports/<dataset>/summary.json.
Usage: python scripts/analyze.py
"""
import json

import numpy as np
import pandas as pd
from _common import DATASETS, REPORTS, ROOT

from armopt.analysis import ActivityParams, action_state_lag, analyze_episode
from armopt.io import load_dataset


def main():
    params = ActivityParams()
    for name in DATASETS:
        ds = load_dataset(ROOT / name)
        rows, lags = [], []
        for ep in ds.episodes():
            rows.append(analyze_episode(ep, params))
            lags.append(action_state_lag(ep))
        df = pd.DataFrame(rows)
        df["action_lag_frames"] = lags
        out = REPORTS / name
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "episode_metrics.csv", index=False, float_format="%.4f")

        lengths = df["duration_s"].round(2).value_counts()
        summary = {
            "episodes_parquet": len(df),
            "episodes_in_meta": len(ds.episodes_meta),
            "frames": int(round(df["duration_s"].sum() * ds.fps)),
            "frames_in_meta": ds.info["total_frames"],
            "fps": ds.fps,
            "tasks": [t["task"] for t in ds.tasks],
            "joint_names": ds.joint_names,
            "duration_s": df["duration_s"].describe().round(2).to_dict(),
            "common_durations_s": {str(k): int(v) for k, v in lengths.head(5).items()},
            "mean_s": {c: round(float(df[c].mean()), 3) for c in [
                "duration_s", "idle_start_s", "pause_s", "short_idle_s", "active_s", "idle_end_s",
                "left_active_s", "right_active_s", "both_active_s"]},
            "total_hours": round(df["duration_s"].sum() / 3600, 3),
            "idle_fraction": round(float(df["idle_s"].sum() / df["duration_s"].sum()), 4),
            "pauses_per_episode": round(float(df["n_pauses"].mean()), 2),
            "reversals_per_episode": round(float((df["left_reversals"] + df["right_reversals"]).mean()), 2),
            "grip_events_per_episode": round(float((df["left_grip_events"] + df["right_grip_events"]).mean()), 2),
            "action_lag_frames_median": float(np.median(lags)),
            "activity_params": params.__dict__,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{name}: {len(df)} episodes, idle {summary['idle_fraction']:.1%}, "
              f"mean {summary['mean_s']['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
