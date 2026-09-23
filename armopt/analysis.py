"""Per-episode motion analysis: where does the cycle time go?"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .io import LEFT_ARM, LEFT_GRIPPER, RIGHT_ARM, RIGHT_GRIPPER, Episode


@dataclass
class ActivityParams:
    arm_speed_thr: float = 0.05  # rad/s, joint-space speed norm of one arm
    gripper_speed_thr: float = 0.01  # m/s
    fill_gap_s: float = 0.2  # idle gaps shorter than this are treated as motion
    min_motion_s: float = 0.1  # motion bursts shorter than this are treated as noise
    pause_min_s: float = 0.3  # an idle run at least this long counts as a pause
    reversal_amp: float = 0.02  # rad, hysteresis for counting direction reversals
    smooth_window: int = 9  # Savitzky-Golay window (frames) for derivatives


def smooth(q: np.ndarray, window: int) -> np.ndarray:
    if len(q) < window:
        return q.copy()
    return savgol_filter(q, window, 2, axis=0, mode="interp")


def derivatives(q: np.ndarray, fps: float, window: int):
    """Smoothed position, velocity and acceleration (per second)."""
    qs = smooth(q, window)
    v = np.gradient(qs, axis=0) * fps
    a = np.gradient(v, axis=0) * fps
    return qs, v, a


def runs(mask: np.ndarray):
    """(start, end_exclusive, value) for each run of equal values."""
    if len(mask) == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(mask)]
    return [(int(s), int(e), bool(mask[s])) for s, e in zip(starts, ends)]


def clean_mask(mask: np.ndarray, fps: float, p: ActivityParams) -> np.ndarray:
    m = mask.copy()
    fill = int(round(p.fill_gap_s * fps))
    for s, e, val in runs(m):
        if not val and s > 0 and e < len(m) and e - s < fill:
            m[s:e] = True
    drop = int(round(p.min_motion_s * fps))
    for s, e, val in runs(m):
        if val and e - s < drop:
            m[s:e] = False
    return m


def activity_masks(ep: Episode, p: ActivityParams):
    """Boolean per-frame masks: left arm active, right arm active."""
    _, v, _ = derivatives(ep.state, ep.fps, p.smooth_window)
    masks = []
    for arm, grip in ((LEFT_ARM, LEFT_GRIPPER), (RIGHT_ARM, RIGHT_GRIPPER)):
        moving = (np.linalg.norm(v[:, arm], axis=1) > p.arm_speed_thr) | (np.abs(v[:, grip]) > p.gripper_speed_thr)
        masks.append(clean_mask(moving, ep.fps, p))
    return masks[0], masks[1]


def count_reversals(x: np.ndarray, amp: float) -> int:
    """Direction reversals of a 1-D signal, ignoring wiggles smaller than `amp`."""
    n, direction, extreme = 0, 0, x[0]
    for val in x[1:]:
        if direction == 0:
            if abs(val - extreme) > amp:
                direction, extreme = (1 if val > extreme else -1), val
        elif direction * (val - extreme) > 0:
            extreme = val  # still moving the same way
        elif abs(val - extreme) > amp:
            n += 1
            direction, extreme = -direction, val
    return n


def gripper_events(g: np.ndarray, hysteresis: float = 0.3) -> int:
    """Number of open/close transitions of a gripper signal."""
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    if hi - lo < 0.005:
        return 0
    x = (g - lo) / (hi - lo)
    state, n = x[0] > 0.5, 0
    for val in x[1:]:
        if state and val < 0.5 - hysteresis / 2:
            state, n = False, n + 1
        elif not state and val > 0.5 + hysteresis / 2:
            state, n = True, n + 1
    return n


def analyze_episode(ep: Episode, p: ActivityParams | None = None) -> dict:
    p = p or ActivityParams()
    fps = ep.fps
    left, right = activity_masks(ep, p)
    anyarm = left | right
    T = len(anyarm)
    active_idx = np.flatnonzero(anyarm)

    if len(active_idx):
        first, last = active_idx[0], active_idx[-1] + 1
    else:
        first, last = T, T
    idle_start = first / fps
    idle_end = (T - last) / fps if len(active_idx) else 0.0

    pause_frames, n_pauses = 0, 0
    for s, e, val in runs(anyarm[first:last]):
        if not val and (e - s) / fps >= p.pause_min_s:
            pause_frames += e - s
            n_pauses += 1
    inner_idle = int((~anyarm[first:last]).sum())

    qs = smooth(ep.state, p.smooth_window)
    path = np.abs(np.diff(qs, axis=0))
    row = {
        "episode": ep.index,
        "duration_s": T / fps,
        "idle_start_s": idle_start,
        "idle_end_s": idle_end,
        "pause_s": pause_frames / fps,
        "n_pauses": n_pauses,
        "short_idle_s": (inner_idle - pause_frames) / fps,
        "active_s": anyarm.sum() / fps,
        "left_active_s": left.sum() / fps,
        "right_active_s": right.sum() / fps,
        "both_active_s": (left & right).sum() / fps,
        "left_path_rad": path[:, LEFT_ARM].sum(),
        "right_path_rad": path[:, RIGHT_ARM].sum(),
        "left_reversals": sum(count_reversals(qs[:, j], p.reversal_amp) for j in range(0, 7)),
        "right_reversals": sum(count_reversals(qs[:, j], p.reversal_amp) for j in range(8, 15)),
        "left_grip_events": gripper_events(ep.state[:, LEFT_GRIPPER]),
        "right_grip_events": gripper_events(ep.state[:, RIGHT_GRIPPER]),
    }
    row["idle_s"] = row["duration_s"] - row["active_s"]
    row["idle_frac"] = row["idle_s"] / row["duration_s"]
    return row


def action_state_lag(ep: Episode, max_lag: int = 15) -> int:
    """Frames by which measured state trails the commanded action (arm joints only)."""
    joints = np.r_[0:7, 8:15]
    a, s = ep.action[:, joints], ep.state[:, joints]
    errs = [np.mean(np.abs(a[: len(a) - k] - s[k:])) for k in range(max_lag)]
    return int(np.argmin(errs))
