"""Load and write LeRobot v2.1 datasets (parquet + meta, no videos)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LEFT_ARM = slice(0, 7)
LEFT_GRIPPER = 7
RIGHT_ARM = slice(8, 15)
RIGHT_GRIPPER = 15


@dataclass
class Episode:
    index: int
    state: np.ndarray  # (T, 16) measured joint positions
    action: np.ndarray  # (T, 16) commanded joint positions
    task_index: int
    fps: float
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.state) / self.fps


@dataclass
class Dataset:
    name: str
    root: Path
    info: dict
    tasks: list[dict]
    episodes_meta: list[dict]

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def joint_names(self) -> list[str]:
        return list(self.info["features"]["observation.state"]["names"])

    def episode_files(self) -> list[Path]:
        return sorted(self.root.glob("data/chunk-*/episode_*.parquet"))

    def load_episode(self, path: Path) -> Episode:
        df = pd.read_parquet(path)
        return Episode(
            index=int(df["episode_index"].iloc[0]),
            state=np.stack(df["observation.state"].to_numpy()).astype(np.float64),
            action=np.stack(df["action"].to_numpy()).astype(np.float64),
            task_index=int(df["task_index"].iloc[0]),
            fps=self.fps,
        )

    def episodes(self):
        for path in self.episode_files():
            yield self.load_episode(path)


def load_dataset(root: str | Path) -> Dataset:
    root = Path(root)
    read_jsonl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    return Dataset(
        name=root.name,
        root=root,
        info=json.loads((root / "meta" / "info.json").read_text()),
        tasks=read_jsonl(root / "meta" / "tasks.jsonl"),
        episodes_meta=read_jsonl(root / "meta" / "episodes.jsonl"),
    )


def write_episode(out_root: Path, chunk: int, ep: Episode, source_frame: np.ndarray, global_start: int):
    """Write one episode in LeRobot layout. `source_frame` maps each output frame to a
    (fractional) frame of the original recording, so videos can be resampled later."""
    T = len(ep.state)
    df = pd.DataFrame({
        "observation.state": list(ep.state.astype(np.float32)),
        "action": list(ep.action.astype(np.float32)),
        "timestamp": (np.arange(T) / ep.fps).astype(np.float32),
        "frame_index": np.arange(T, dtype=np.int64),
        "episode_index": np.full(T, ep.index, dtype=np.int64),
        "index": np.arange(global_start, global_start + T, dtype=np.int64),
        "task_index": np.full(T, ep.task_index, dtype=np.int64),
        "source_frame": source_frame.astype(np.float32),
    })
    path = out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep.index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _stats(x: np.ndarray) -> dict:
    x = x.reshape(len(x), -1).astype(np.float64)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def episode_stats(ep: Episode, global_start: int) -> dict:
    T = len(ep.state)
    return {"episode_index": ep.index, "stats": {
        "timestamp": _stats(np.arange(T) / ep.fps),
        "frame_index": _stats(np.arange(T)),
        "episode_index": _stats(np.full(T, ep.index)),
        "index": _stats(np.arange(global_start, global_start + T)),
        "task_index": _stats(np.full(T, ep.task_index)),
        "observation.state": _stats(ep.state),
        "action": _stats(ep.action),
    }}


def write_meta(out_root: Path, ds: Dataset, episodes: list[Episode], stats: list[dict], note: dict):
    """Meta for an optimized, state/action-only copy of `ds`. Video features are dropped
    (the videos are not retimed here); `source_frame` records where each frame came from."""
    meta = out_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = json.loads(json.dumps(ds.info))
    info["features"] = {k: v for k, v in info["features"].items() if v.get("dtype") != "video"}
    info["features"]["source_frame"] = {"dtype": "float32", "shape": [1], "names": None}
    info["video_path"] = None
    info["total_videos"] = 0
    info["total_episodes"] = len(episodes)
    info["total_frames"] = int(sum(len(e.state) for e in episodes))
    info["splits"] = {"train": f"0:{len(episodes)}"}
    info["optimization"] = note
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    tasks = {t["task_index"]: t["task"] for t in ds.tasks}
    with open(meta / "tasks.jsonl", "w") as f:
        for t in ds.tasks:
            f.write(json.dumps(t) + "\n")
    with open(meta / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps({"episode_index": e.index, "tasks": [tasks[e.task_index]], "length": len(e.state)}) + "\n")
    with open(meta / "episodes_stats.jsonl", "w") as f:
        for s in stats:
            f.write(json.dumps(s) + "\n")
