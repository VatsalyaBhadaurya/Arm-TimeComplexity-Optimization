import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from armopt.analysis import ActivityParams, analyze_episode, count_reversals, gripper_events, runs
from armopt.io import Episode
from armopt.retime import Limits, RetimeParams, retime_episode, topp

FPS = 30.0
J = 16


def limits(v=1.0, a=4.0):
    return Limits(np.full(J, v), np.full(J, a))


def synthetic_episode(idle_start=2.0, move=3.0, pause=2.0, idle_end=3.0, amp=0.6):
    """Idle, a slow smooth move of the left arm, a pause, a move back, idle."""
    def ramp(n, a, b):
        s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        return a + (b - a) * s

    n = lambda sec: int(round(sec * FPS))
    q1 = np.r_[np.zeros(n(idle_start)), ramp(n(move), 0, amp), np.full(n(pause), amp),
               ramp(n(move), amp, 0), np.zeros(n(idle_end))]
    state = np.zeros((len(q1), J))
    state[:, 0] = q1
    state[:, 3] = -0.5 * q1
    return Episode(0, state, state.copy(), 0, FPS)


def test_runs():
    assert runs(np.array([1, 1, 0, 0, 0, 1], bool)) == [(0, 2, True), (2, 5, False), (5, 6, True)]


def test_count_reversals_ignores_small_wiggles():
    t = np.linspace(0, 4 * np.pi, 400)  # extrema at pi/2, 3pi/2, 5pi/2, 7pi/2
    assert count_reversals(np.sin(t), 0.1) == 4
    assert count_reversals(0.001 * np.sin(t), 0.1) == 0


def test_gripper_events():
    g = np.r_[np.zeros(10), np.full(10, 0.04), np.zeros(10)]
    assert gripper_events(g) == 2


def test_topp_respects_limits_and_rests_at_ends():
    s = np.linspace(0, 1, 301)
    q = np.zeros((301, J))
    q[:, 0] = np.sin(np.pi * s) * 0.5  # 10 s of slow motion in the demo
    lim = limits(v=0.8, a=3.0)
    times, sdot = topp(q, 1 / FPS, lim, max_sdot=10.0, return_sdot=True)
    assert sdot[0] == 0 and sdot[-1] == 0
    assert np.all(np.diff(times) > 0)
    assert times[-1] < 10.0  # faster than the demo
    t_uniform = np.arange(0, times[-1], 1 / 200)
    spline = CubicSpline(times, q[:, 0])  # smooth, as a robot controller would track it
    assert np.abs(spline(t_uniform, 1)).max() <= 0.8 * 1.02
    assert np.abs(spline(t_uniform, 2)).max() <= 3.0 * 1.05


def test_topp_speedup_cap():
    q = np.zeros((301, J))
    q[:, 0] = np.linspace(0, 0.01, 301)  # nearly still: only the cap binds
    times = topp(q, 1 / FPS, limits(), max_sdot=2.0)
    assert times[-1] >= 10.0 / 2.0


def test_analysis_finds_idle_and_pause():
    ep = synthetic_episode()
    row = analyze_episode(ep, ActivityParams())
    assert row["idle_start_s"] == pytest.approx(2.0, abs=0.4)
    assert row["idle_end_s"] == pytest.approx(3.0, abs=0.4)
    assert row["n_pauses"] == 1
    assert row["right_active_s"] == 0


@pytest.mark.parametrize("profile", [RetimeParams(max_speedup=1.0), RetimeParams(max_speedup=3.0)])
def test_retime_episode_is_a_pure_time_warp(profile):
    ep = synthetic_episode()
    new, src, summary = retime_episode(ep, limits(), profile)
    assert summary["new_s"] < summary["orig_s"]
    assert np.all(np.diff(src) >= -1e-9)  # never plays backwards
    # every output pose lies on the recorded path (interpolated between neighbours)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(ep.state) - 1)
    between = (new.state[:, 0] >= np.minimum(ep.state[lo, 0], ep.state[hi, 0]) - 1e-9) & (
        new.state[:, 0] <= np.maximum(ep.state[lo, 0], ep.state[hi, 0]) + 1e-9)
    assert between.all()
    # the motion itself is preserved end to end
    assert new.state[:, 0].max() == pytest.approx(0.6, abs=1e-3)
    # trimming stops at the idle threshold, so only a sub-threshold creep is cut off
    assert new.state[-1, 0] == pytest.approx(0.0, abs=5e-3)
    assert summary["vel_ratio_max"] <= 1.1


def test_retime_never_slower_than_demo():
    ep = synthetic_episode(amp=2.0, move=1.0)  # demo faster than the limits
    _, _, summary = retime_episode(ep, limits(v=0.5, a=1.0), RetimeParams(max_speedup=3.0))
    assert summary["new_s"] <= summary["orig_s"]
