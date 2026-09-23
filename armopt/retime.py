"""Cycle-time optimization by time re-parameterization.

The recorded joint-space path is kept exactly; only *when* the arm is at each point
changes. Three steps, each usable on its own:

1. `step1_trim_idle`: drop idle time before the first and after the last motion,
2. `step2_compress_pauses`: shorten pauses between motions (kept as a short dwell,
   longer after gripper actions),
3. `step3_speed_up_motion`: retime each motion segment as fast as per-joint velocity
   and acceleration limits allow (TOPP-style forward/backward pass on s-dot^2),
   capped by a maximum speed-up factor.

`render` turns the resulting plan into 30 fps frames; `retime_episode` runs it all.

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


@dataclass
class Piece:
    """A stretch [s, e] of the recording (inclusive frame indices) and the new time stamp
    of each of its frames, relative to the start of the piece."""

    kind: str  # "pad", "move" or "pause"
    s: int
    e: int
    times: np.ndarray
    retimed: bool = False  # False: played at the demo's own timing

    @property
    def duration(self) -> float:
        return float(self.times[-1])


@dataclass
class Plan:
    """An episode's timing plan: the pieces kept, in order, each with its own time stamps."""

    ep: Episode
    qs: np.ndarray  # smoothed joint path, used for the limits
    grip_moving: np.ndarray  # per frame: is a gripper opening/closing
    pieces: list[Piece]

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.pieces)


def _demo_times(n_frames: int, ds: float) -> np.ndarray:
    return np.arange(n_frames) * ds


def _retime_piece(plan: Plan, piece: Piece, lim: Limits, cap: float, rp: RetimeParams) -> Piece:
    """Time-optimal timing for one piece, but never slower than the demo (which the robot
    has already executed safely)."""
    ds = 1.0 / plan.ep.fps
    times = topp(plan.qs[piece.s : piece.e + 1], ds, lim, cap, curvature_share=rp.curvature_share)
    if times[-1] > (piece.e - piece.s) * ds:
        return Piece(piece.kind, piece.s, piece.e, _demo_times(piece.e - piece.s + 1, ds), False)
    return Piece(piece.kind, piece.s, piece.e, times, True)


def step1_trim_idle(ep: Episode, rp: RetimeParams | None = None, act: ActivityParams | None = None) -> Plan:
    """Step 1: drop the idle time before the first and after the last motion (keeping
    `pad_s` on each side). Everything kept still plays at the demo's own timing, split
    into motion and pause pieces for the next steps."""
    rp = rp or RetimeParams()
    act = act or ActivityParams()
    fps, ds, T = ep.fps, 1.0 / ep.fps, len(ep.state)
    left, right = activity_masks(ep, act)
    qs, v, _ = derivatives(ep.state, fps, act.smooth_window)
    grip_moving = np.abs(v[:, list(GRIPPERS)]).max(axis=1) > act.gripper_speed_thr

    segs = _segments(left | right, fps, act) or [("pause", 0, T)]
    pad = int(round(rp.pad_s * fps))
    start = max(segs[0][1] - pad, 0)
    end = min(segs[-1][2] + pad, T)

    bounds = [("pad", start, segs[0][1])]
    bounds += [(kind, s, min(e, T - 1)) for kind, s, e in segs]
    bounds.append(("pad", bounds[-1][2], end - 1))
    pieces = [Piece(kind, s, e, _demo_times(e - s + 1, ds)) for kind, s, e in bounds if e > s]
    return Plan(ep, qs, grip_moving, pieces)


def step2_compress_pauses(plan: Plan, lim: Limits, rp: RetimeParams | None = None) -> Plan:
    """Step 2: cross each mid-task pause as fast as the limits allow (the arm drifts a
    little while "still"), keeping at least `dwell_s`, or `grip_dwell_s` right after a
    gripper action so the grasp can settle."""
    rp = rp or RetimeParams()
    fps = plan.ep.fps
    pieces = []
    for p in plan.pieces:
        if p.kind == "pause":
            p = _retime_piece(plan, p, lim, rp.pause_max_speedup, rp)
            after_grip = plan.grip_moving[max(p.s - int(0.5 * fps), 0) : p.s].any()
            floor = min((p.e - p.s) / fps, rp.grip_dwell_s if after_grip else rp.dwell_s)
            if p.duration < floor:
                p = Piece(p.kind, p.s, p.e, p.times * (floor / max(p.duration, 1e-9)), p.retimed)
        pieces.append(p)
    return Plan(plan.ep, plan.qs, plan.grip_moving, pieces)


def step3_speed_up_motion(plan: Plan, lim: Limits, rp: RetimeParams | None = None) -> Plan:
    """Step 3: retime each motion piece time-optimally within the joint limits, at most
    `max_speedup` times the demo speed."""
    rp = rp or RetimeParams()
    pieces = [_retime_piece(plan, p, lim, rp.max_speedup, rp) if p.kind == "move" else p for p in plan.pieces]
    return Plan(plan.ep, plan.qs, plan.grip_moving, pieces)


def render(plan: Plan, lim: Limits, act: ActivityParams | None = None):
    """Resample a plan at the episode's frame rate.
    Returns (optimized Episode, source_frame per output frame, summary dict)."""
    act = act or ActivityParams()
    ep = plan.ep
    fps, T = ep.fps, len(ep.state)
    first = plan.pieces[0]
    knots_t, knots_src, t = [0.0], [float(first.s)], 0.0
    retimed = np.zeros(T, dtype=bool)
    for p in plan.pieces:
        knots_t.extend(t + p.times[1:])
        knots_src.extend(np.arange(p.s + 1, p.e + 1, dtype=float))
        t += p.duration
        if p.retimed:
            retimed[p.s : p.e + 1] = True
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
    checked = retimed[np.clip(np.round(src).astype(int), 0, T - 1)]
    pv, pa = (pv[checked], pa[checked]) if checked.any() else (np.zeros((1, 1)), np.zeros((1, 1)))
    summary = {
        "episode": ep.index,
        "orig_s": T / fps,
        "trimmed_s": (plan.pieces[-1].e + 1 - first.s) / fps,
        "new_s": n_out / fps,
        "speedup": (T / fps) / (n_out / fps),
        "n_segments": sum(p.kind == "move" for p in plan.pieces),
        "n_pauses": sum(p.kind == "pause" for p in plan.pieces),
        "retimed_frac": float(checked.mean()),
        "vel_ratio_max": float(pv.max()),
        "acc_ratio_max": float(pa.max()),
        "vel_ratio_p99": float(np.percentile(pv, 99)),
        "acc_ratio_p99": float(np.percentile(pa, 99)),
        "demo_vel_ratio_max": float((np.abs(v_old) / lim.vmax).max()),
        "demo_acc_ratio_max": float((np.abs(a_old) / lim.amax).max()),
    }
    return out, src, summary


def retime_episode(ep: Episode, lim: Limits, rp: RetimeParams | None = None,
                   act: ActivityParams | None = None, steps: int = 3):
    """Run steps 1..`steps` and render. Returns (optimized Episode, source_frame, summary)."""
    rp = rp or RetimeParams()
    act = act or ActivityParams()
    plan = step1_trim_idle(ep, rp, act)
    if steps >= 2:
        plan = step2_compress_pauses(plan, lim, rp)
    if steps >= 3:
        plan = step3_speed_up_motion(plan, lim, rp)
    out, src, summary = render(plan, lim, act)
    summary["steps"] = steps
    return out, src, summary
