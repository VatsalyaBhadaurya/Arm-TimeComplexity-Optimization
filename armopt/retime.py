"""Cycle-time optimization by time re-parameterization.

The recorded joint-space path is kept exactly; only *when* the arm is at each point
changes. Three things are removed or shortened:

1. idle time before the first and after the last motion,
2. pauses between motions (kept as a short dwell, longer after gripper actions),
3. slow motion: each motion segment is retimed as fast as per-joint velocity and
   acceleration limits allow (TOPP-style forward/backward pass on s-dot^2),
   capped by a maximum speed-up factor.

Both arms share one time map, so bimanual coordination is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import ActivityParams, activity_masks, derivatives, runs
from .io import LEFT_GRIPPER, RIGHT_GRIPPER, Episode

GRIPPERS = (LEFT_GRIPPER, RIGHT_GRIPPER)


@dataclass
class Limits:
    vmax: np.ndarray  # (J,) per-joint velocity limit
    amax: np.ndarray  # (J,) per-joint acceleration limit

    def scaled(self, v: float = 1.0, a: float = 1.0) -> "Limits":
        return Limits(self.vmax * v, self.amax * a)


@dataclass
class RetimeParams:
    max_speedup: float = 2.0  # cap on playback speed of motion segments vs. the demo
    pause_max_speedup: float = 20.0  # cap while crossing (drifting) pauses
    dwell_s: float = 0.1  # minimum time kept for a pause
    grip_dwell_s: float = 0.3  # minimum pause kept right after a gripper action
    pad_s: float = 0.1  # idle kept before the first / after the last motion
    curvature_share: float = 0.5  # fraction of the acceleration budget for path curvature
    limit_percentile: float = 99.0  # percentile of demo |vel|/|acc| used as joint limits


PROFILES = {
    # Only idle and pauses are removed; every motion plays at the demo's own speed.
    "safe": RetimeParams(max_speedup=1.0),
    # Motion sped up to at most 2x, within the p99 joint speed/acceleration of the demos.
    "balanced": RetimeParams(max_speedup=2.0, curvature_share=0.5),
    # Up to 3x, more of the acceleration budget spent on following the path at speed.
    "fast": RetimeParams(max_speedup=3.0, curvature_share=0.7),
}


def estimate_limits(episodes, percentile: float, act: ActivityParams | None = None) -> Limits:
    """Per-joint limits = `percentile` of |velocity| / |acceleration| seen while that arm
    was moving. At 99 the optimized motion never exceeds what operators routinely did."""
    act = act or ActivityParams()
    vs, as_ = [], []
    for ep in episodes:
        left, right = activity_masks(ep, act)
        _, v, a = derivatives(ep.state, ep.fps, act.smooth_window)
        for mask, cols in ((left, np.r_[0:8]), (right, np.r_[8:16])):
            v_m = np.full_like(v, np.nan)
            a_m = np.full_like(a, np.nan)
            v_m[np.ix_(mask, cols)] = v[np.ix_(mask, cols)]
            a_m[np.ix_(mask, cols)] = a[np.ix_(mask, cols)]
            vs.append(v_m[mask])
            as_.append(a_m[mask])
    V = np.abs(np.concatenate(vs))
    A = np.abs(np.concatenate(as_))
    return Limits(np.nanpercentile(V, percentile, axis=0), np.nanpercentile(A, percentile, axis=0))


def topp(q: np.ndarray, ds: float, lim: Limits, max_sdot: float, curvature_share: float = 0.5, return_sdot: bool = False):
    """Time-optimal timing of path q (N, J) sampled every `ds` seconds of original time,
    starting and ending at rest. Returns the new time stamp of each sample.

    Path speed r = s-dot is relative to the demo (1.0 = original speed). Joint velocity is
    q' r and joint acceleration is q'' r^2 + q' r-dot. Each joint's acceleration budget is
    split between the two terms (share c for curvature), which keeps every sample feasible
    (exact TOPP chatters on the encoder noise of slow recorded motion):

        r       <= vmax / |q'|              (velocity)
        r^2     <= c amax / |q''|           (path curvature)
        |r-dot| <= (1 - c) amax / |q'|      (speeding up / slowing down)

    The last one is enforced with a backward then forward pass on x = r^2, where
    x_{i+1} - x_i = 2 r-dot ds.
    """
    N = len(q)
    if N < 3:
        return (np.arange(N) * ds, np.ones(N)) if return_sdot else np.arange(N) * ds
    tiny = 1e-9
    dq = np.abs(np.gradient(q, ds, axis=0))
    ddq = np.abs(np.gradient(np.gradient(q, ds, axis=0), ds, axis=0))
    x_max = np.minimum.reduce([
        np.min((lim.vmax / np.maximum(dq, tiny)) ** 2, axis=1),
        np.min(curvature_share * lim.amax / np.maximum(ddq, tiny), axis=1),
        np.full(N, max_sdot**2),
    ])
    r_dot = np.min((1 - curvature_share) * lim.amax / np.maximum(dq, tiny), axis=1)

    x = x_max.copy()
    x[-1] = 0.0
    for i in range(N - 2, -1, -1):  # must be able to stop at the end
        x[i] = min(x[i], x[i + 1] + 2 * ds * r_dot[i])
    x[0] = 0.0
    for i in range(N - 1):  # start from rest
        x[i + 1] = min(x[i + 1], x[i] + 2 * ds * r_dot[i])

    sdot = np.sqrt(x)
    dt = 2 * ds / np.maximum(sdot[:-1] + sdot[1:], 1e-6)
    times = np.r_[0.0, np.cumsum(dt)]
    return (times, sdot) if return_sdot else times


def _segments(anyarm: np.ndarray, fps: float, act: ActivityParams):
    """Split [first motion, last motion) into alternating ('move'|'pause', start, end)."""
    idx = np.flatnonzero(anyarm)
    if len(idx) == 0:
        return []
    first, last = idx[0], idx[-1] + 1
    mask = anyarm[first:last].copy()
    min_pause = int(round(act.pause_min_s * fps))
    for s, e, val in runs(mask):
        if not val and e - s < min_pause:
            mask[s:e] = True  # short hesitations are retimed as part of the motion
    return [("move" if val else "pause", first + s, first + e) for s, e, val in runs(mask)]


def retime_episode(ep: Episode, lim: Limits, rp: RetimeParams | None = None, act: ActivityParams | None = None):
    """Return (optimized Episode, source_frame per output frame, summary dict)."""
    rp = rp or RetimeParams()
    act = act or ActivityParams()
    fps, ds = ep.fps, 1.0 / ep.fps
    left, right = activity_masks(ep, act)
    qs, v, _ = derivatives(ep.state, fps, act.smooth_window)
    grip_moving = np.abs(v[:, list(GRIPPERS)]).max(axis=1) > act.gripper_speed_thr

    segs = _segments(left | right, fps, act)
    T = len(ep.state)
    if not segs:
        segs = [("pause", 0, T)]
    pad = int(round(rp.pad_s * fps))
    start = max(segs[0][1] - pad, 0)
    end = min(segs[-1][2] + pad, T)

    demo_speed = np.ones(T, dtype=bool)  # frames played back at the original timing
    knots_t, knots_src = [0.0], [float(start)]
    t = 0.0
    if segs[0][1] > start:  # lead-in pad played at original speed
        t += (segs[0][1] - start) * ds
        knots_t.append(t)
        knots_src.append(float(segs[0][1]))
    for kind, s, e in segs:
        e_incl = min(e, T - 1)
        if e_incl <= s:
            continue
        cap = rp.max_speedup if kind == "move" else rp.pause_max_speedup
        times = topp(qs[s : e_incl + 1], ds, lim, cap, curvature_share=rp.curvature_share)
        if times[-1] > (e_incl - s) * ds:  # never slower than the demo, which the robot already executed
            times = np.arange(e_incl - s + 1) * ds
        else:
            demo_speed[s : e_incl + 1] = False
        if kind == "pause":
            tail = grip_moving[max(s - int(0.5 * fps), 0) : s].any()
            floor = min((e_incl - s) * ds, rp.grip_dwell_s if tail else rp.dwell_s)
            if times[-1] < floor:
                times = times * (floor / max(times[-1], 1e-9))
        knots_t.extend(t + times[1:])
        knots_src.extend(np.arange(s + 1, e_incl + 1, dtype=float))
        t += times[-1]
    last_src = knots_src[-1]
    if end - 1 > last_src:  # lead-out pad at original speed
        t += (end - 1 - last_src) * ds
        knots_t.append(t)
        knots_src.append(float(end - 1))

    knots_t = np.asarray(knots_t)
    knots_src = np.asarray(knots_src)
    n_out = int(np.floor(knots_t[-1] * fps)) + 1
    src = np.interp(np.arange(n_out) / fps, knots_t, knots_src)

    def sample(arr):
        i0 = np.clip(np.floor(src).astype(int), 0, T - 1)
        i1 = np.clip(i0 + 1, 0, T - 1)
        w = (src - i0)[:, None]
        return arr[i0] * (1 - w) + arr[i1] * w

    out = Episode(ep.index, sample(ep.state), sample(ep.action), ep.task_index, fps)
    # Measure the output exactly as the limits were measured on the demos.
    _, v_new, a_new = derivatives(out.state, fps, act.smooth_window)
    pv = np.abs(v_new) / lim.vmax
    pa = np.abs(a_new) / lim.amax
    _, v_old, a_old = derivatives(ep.state, fps, act.smooth_window)
    # Limits are only claimed for retimed frames; demo-speed frames reproduce the recording.
    checked = ~demo_speed[np.clip(np.round(src).astype(int), 0, T - 1)]
    pv, pa = (pv[checked], pa[checked]) if checked.any() else (np.zeros((1, 1)), np.zeros((1, 1)))
    summary = {
        "episode": ep.index,
        "orig_s": T / fps,
        "trimmed_s": (end - start) / fps,
        "new_s": n_out / fps,
        "speedup": (T / fps) / (n_out / fps),
        "n_segments": sum(k == "move" for k, _, _ in segs),
        "n_pauses": sum(k == "pause" for k, _, _ in segs),
        "retimed_frac": float(checked.mean()),
        "vel_ratio_max": float(pv.max()),
        "acc_ratio_max": float(pa.max()),
        "vel_ratio_p99": float(np.percentile(pv, 99)),
        "acc_ratio_p99": float(np.percentile(pa, 99)),
        "demo_vel_ratio_max": float((np.abs(v_old) / lim.vmax).max()),
        "demo_acc_ratio_max": float((np.abs(a_old) / lim.amax).max()),
    }
    return out, src, summary
