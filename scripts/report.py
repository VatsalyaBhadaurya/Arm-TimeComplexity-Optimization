"""Build reports/REPORT.md and its figures from the outputs of analyze.py and optimize.py.
Usage: python scripts/report.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import DATASETS, REPORTS, ROOT

from armopt.analysis import ActivityParams, derivatives
from armopt.io import load_dataset
from armopt.retime import PROFILES, estimate_limits, retime_episode

# Validated categorical palette (light), text and surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FIG = REPORTS / "figures"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def load(name):
    d = REPORTS / name
    return {
        "summary": json.loads((d / "summary.json").read_text()),
        "opt": json.loads((d / "optimization_summary.json").read_text()),
        "limits": json.loads((d / "limits.json").read_text()),
        "metrics": pd.read_csv(d / "episode_metrics.csv"),
        "profiles": {p: pd.read_csv(d / f"optimization_{p}.csv") for p in PROFILES},
    }


def fig_time_budget(data):
    parts = [("idle_start_s", "Idle before first motion"), ("active_s", "Arm motion"),
             ("short_idle_s", "Hesitations < 0.3 s"), ("pause_s", "Pauses ≥ 0.3 s"),
             ("idle_end_s", "Idle after last motion")]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    names = list(data)
    for row, name in enumerate(names):
        m = data[name]["summary"]["mean_s"]
        left = 0.0
        for k, (key, label) in enumerate(parts):
            w = m[key]
            ax.barh(row, w, left=left, height=0.55, color=SERIES[k], edgecolor=SURFACE, linewidth=2,
                    label=label if row == 0 else None)
            if w > 1.6:
                ax.text(left + w / 2, row, f"{w:.1f} s", ha="center", va="center", color="white"
                        if k in (0, 1) else TEXT, fontsize=9)
            left += w
        ax.text(left + 0.4, row, f"{left:.1f} s", va="center", color=TEXT_2, fontsize=9)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean seconds per episode")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 44)
    ax.set_title("Where the time goes in an average episode")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.28), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "time_budget.png", dpi=150)
    plt.close(fig)


def fig_profiles(data):
    labels = ["Original", "Safe", "Balanced", "Fast"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(labels))
    w = 0.36
    for k, name in enumerate(data):
        o = data[name]["opt"]
        vals = [o["safe"]["mean_orig_s"]] + [o[p]["mean_new_s"] for p in PROFILES]
        bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.03, color=SERIES[k], label=name)
        for i, (b, v) in enumerate(zip(bars, vals)):
            txt = f"{v:.1f}s" if i == 0 else f"{v:.1f}s\n−{100 * (1 - v / vals[0]):.0f}%"
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, txt, ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean episode length (s)")
    ax.set_ylim(0, 52)
    ax.grid(axis="x", visible=False)
    ax.set_title("Mean episode length by optimization profile")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "profiles.png", dpi=150)
    plt.close(fig)


def fig_savings_hist(data):
    fig, axes = plt.subplots(1, len(data), figsize=(9, 2.8), sharey=False)
    for ax, name in zip(axes, data):
        df = data[name]["profiles"]["balanced"]
        saved = 100 * (1 - df.new_s / df.orig_s)
        ax.hist(saved, bins=np.arange(0, 55, 2.5), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        med = np.median(saved)
        top = ax.get_ylim()[1] * 1.18
        ax.set_ylim(0, top)
        ax.axvline(med, color=TEXT, linewidth=1, linestyle="--")
        ax.text(med + 1, top * 0.93, f"median {med:.0f}%", fontsize=8.5, color=TEXT, va="top")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Time saved per episode (%)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Episodes")
    fig.suptitle("Balanced profile: time saved per episode", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "savings_hist.png", dpi=150)
    plt.close(fig)


def fig_example(name, data):
    """Joint distance covered over time, original vs balanced, for the median-saving episode."""
    ds = load_dataset(ROOT / name)
    df = data[name]["profiles"]["balanced"]
    ep_idx = int(df.iloc[(df.speedup - df.speedup.median()).abs().argsort().iloc[0]].episode)
    path = next(p for p in ds.episode_files() if p.stem.endswith(f"{ep_idx:06d}"))
    ep = ds.load_episode(path)
    lim = estimate_limits(ds.episodes(), PROFILES["balanced"].limit_percentile)
    new, _, _ = retime_episode(ep, lim, PROFILES["balanced"])
    win = ActivityParams().smooth_window

    def progress(e):
        qs, _, _ = derivatives(e.state, e.fps, win)
        step = np.abs(np.diff(qs[:, np.r_[0:7, 8:15]], axis=0)).sum(axis=1)
        return np.r_[0.0, np.cumsum(step)]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    t0 = np.arange(len(ep.state)) / ep.fps
    t1 = np.arange(len(new.state)) / new.fps
    p0, p1 = progress(ep), progress(new)
    ax.plot(t0, p0, color=SERIES[0], linewidth=2, label="Original")
    ax.plot(t1, p1, color=SERIES[1], linewidth=2, label="Balanced")
    for t, p, c in ((t0, p0, SERIES[0]), (t1, p1, SERIES[1])):
        ax.plot(t[-1], p[-1], "o", color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(t[-1], p[-1] + p0[-1] * 0.04, f"{t[-1]:.1f} s", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_ylim(0, p0[-1] * 1.18)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint distance covered (rad)")
    ax.set_title(f"Example: {name} episode {ep_idx}: same path, less time")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / f"example_{name}.png", dpi=150)
    plt.close(fig)
    return ep_idx


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {n: load(n) for n in DATASETS}
    fig_time_budget(data)
    fig_profiles(data)
    fig_savings_hist(data)
    examples = {n: fig_example(n, data) for n in DATASETS}

    b, d = data["bimaual_dataset_new_1"], data["deksha_data_330_1"]

    overview = md_table(
        ["", *DATASETS],
        [
            ["Task", *[f"“{x['summary']['tasks'][0]}”" for x in (b, d)]],
            ["Robot", "gen2 bimanual, 2 × (7 joints + gripper) = 16 values", "same"],
            ["Episodes (parquet / meta)", *[f"{x['summary']['episodes_parquet']} / {x['summary']['episodes_in_meta']}" for x in (b, d)]],
            ["Frames (parquet / meta)", *[f"{x['summary']['frames']:,} / {x['summary']['frames_in_meta']:,}" for x in (b, d)]],
            ["Total recorded time", *[f"{x['summary']['total_hours']:.2f} h" for x in (b, d)]],
            ["Frame rate", "30 fps", "30 fps"],
            ["Most common lengths", *[", ".join(f"{k}s×{v}" for k, v in list(x['summary']['common_durations_s'].items())[:3]) for x in (b, d)]],
        ],
    )

    budget = md_table(
        ["Mean seconds per episode", *DATASETS],
        [[label, *[f"{x['summary']['mean_s'][k]:.2f}" for x in (b, d)]] for k, label in [
            ("duration_s", "Episode length"), ("idle_start_s", "Idle before first motion"),
            ("pause_s", "Pauses ≥ 0.3 s (mid-task)"), ("short_idle_s", "Hesitations < 0.3 s"),
            ("idle_end_s", "Idle after last motion"), ("active_s", "At least one arm moving"),
            ("left_active_s", "Left arm moving"), ("right_active_s", "Right arm moving"),
            ("both_active_s", "Both arms moving together")]]
        + [["Pauses per episode", *[x["summary"]["pauses_per_episode"] for x in (b, d)]],
           ["Direction reversals per episode (> 0.02 rad)", *[x["summary"]["reversals_per_episode"] for x in (b, d)]],
           ["Gripper open/close events per episode", *[x["summary"]["grip_events_per_episode"] for x in (b, d)]],
           ["Idle share of all recorded time", *[f"{100 * x['summary']['idle_fraction']:.1f}%" for x in (b, d)]]],
    )

    prof_rows = []
    for p in PROFILES:
        for name, x in data.items():
            o = x["opt"][p]
            prof_rows.append([p, name, f"{o['mean_orig_s']:.1f} → {o['mean_new_s']:.1f} s",
                              f"**{o['time_saved_pct']:.1f}%**", f"{o['orig_hours']:.2f} → {o['new_hours']:.2f} h",
                              f"{o['median_speedup']:.2f}×", o["episodes_slower"],
                              f"{o['vel_ratio_p99_max']:.2f} / {o['acc_ratio_p99_max']:.2f}",
                              f"{o['episodes_vel_peak_above_demo']} / {o['episodes_acc_peak_above_demo']}"])
    profiles = md_table(["Profile", "Dataset", "Mean episode", "Time saved", "Total", "Median speed-up",
                         "Episodes slower", "p99 vel / acc vs limit", "Episodes with vel / acc peak above their demo"], prof_rows)

    parallel_rows = []
    for name, x in data.items():
        m = x["metrics"]
        seq = m.active_s.sum()
        par = np.maximum(m.left_active_s, m.right_active_s).sum()
        parallel_rows.append([name, f"{m.active_s.mean():.1f} s", f"{np.maximum(m.left_active_s, m.right_active_s).mean():.1f} s",
                              f"{100 * (1 - par / seq):.0f}%"])
    parallel = md_table(["Dataset", "Motion time now (mean)", "If the arms fully overlapped (mean)", "Upper bound on extra saving"], parallel_rows)

    lim_rows = []
    names = b["limits"]["joints"]
    for j, jn in enumerate(names):
        lim_rows.append([jn, *[f"{x['limits']['vmax'][j]:.3f} / {x['limits']['amax'][j]:.2f}" for x in (b, d)]])
    limits = md_table(["Joint", *[f"{n} vmax / amax" for n in DATASETS]], lim_rows)

    report = f"""# Arm cycle-time analysis and optimization

Both datasets were analysed episode by episode, and every trajectory was retimed so the
arms reach the same poses in less time. The recorded joint path is kept exactly; only
the timing along it changes. Everything here is regenerated by:

```bash
pip install -r requirements.txt
python scripts/analyze.py    # per-episode metrics      -> reports/<dataset>/episode_metrics.csv
python scripts/optimize.py   # retiming, 3 profiles     -> reports/<dataset>/optimization_*.csv, optimized/
python scripts/report.py     # this report + figures
```

## Headline

| | bimaual_dataset_new_1 | deksha_data_330_1 |
|---|---|---|
| Mean episode length now | {b['opt']['safe']['mean_orig_s']:.1f} s | {d['opt']['safe']['mean_orig_s']:.1f} s |
| **Balanced profile (recommended)** | **{b['opt']['balanced']['mean_new_s']:.1f} s (−{b['opt']['balanced']['time_saved_pct']:.1f}%)** | **{d['opt']['balanced']['mean_new_s']:.1f} s (−{d['opt']['balanced']['time_saved_pct']:.1f}%)** |
| Safe profile (idle/pauses only, motion at demo speed) | {b['opt']['safe']['mean_new_s']:.1f} s (−{b['opt']['safe']['time_saved_pct']:.1f}%) | {d['opt']['safe']['mean_new_s']:.1f} s (−{d['opt']['safe']['time_saved_pct']:.1f}%) |
| Fast profile | {b['opt']['fast']['mean_new_s']:.1f} s (−{b['opt']['fast']['time_saved_pct']:.1f}%) | {d['opt']['fast']['mean_new_s']:.1f} s (−{d['opt']['fast']['time_saved_pct']:.1f}%) |

No episode got slower under any profile. The retimed motion stays within the 99th
percentile of joint speed and acceleration that the operators already used in these same
recordings, and in no episode does it peak higher than that episode's original recording.

![Mean episode length by profile](figures/profiles.png)

## 1. What is in the data

{overview}

Each frame has `observation.state` (measured joint positions) and `action` (commanded
positions) for 16 values: `left_joint_1..7`, `left_gripper_left_joint`, `right_joint_1..7`,
`right_gripper_right_joint`. Joints are in radians, grippers in metres (0 to 0.042).

**Data-quality issues found**

- **Fixed-length recordings.** Episode lengths are almost all exactly 25, 30 or 40 s, so
  recording ran on a timer instead of stopping when the task was done.
- **Episodes are cut off mid-motion.** In most episodes the arm is still moving in the
  final half-second (129 of 201 bimanual, 199 of 330 deksha), and the end pose is far from
  the start pose (median 0.45 / 0.37 rad on the most-moved joint). Some tasks or returns to
  home were probably truncated. Retiming can only shorten what was recorded.
- **Metadata mismatch in bimaual_dataset_new_1.** `meta/episodes.jsonl` and
  `meta/info.json` describe 203 episodes / 192,898 frames, but only 201 parquet files /
  191,098 frames exist, so 2 episodes are missing.
- **The command leads the measured position by 3 frames (100 ms)** in every episode. That
  is the controller's tracking lag, and it grows in distance as motion gets faster (see
  recommendations).
- A few frames have sensor spikes (up to 11.7 rad/s on `left_joint_7`). Limits use
  percentiles, so these do not inflate them.

## 2. Where the time goes

![Where the time goes](figures/time_budget.png)

{budget}

Findings:

1. **About a quarter of all recorded time is idle** ({100 * b['summary']['idle_fraction']:.0f}% bimanual,
   {100 * d['summary']['idle_fraction']:.0f}% deksha). Almost all of it is **mid-task pauses**
   ({b['summary']['pauses_per_episode']:.0f}–{d['summary']['pauses_per_episode']:.0f} per episode),
   not waiting at the start or end.
2. **The two arms take turns.** In deksha the left arm moves {d['summary']['mean_s']['left_active_s']:.1f} s
   and the right {d['summary']['mean_s']['right_active_s']:.1f} s per episode, but both together only
   {d['summary']['mean_s']['both_active_s']:.1f} s. In bimaual the right arm does about twice the work
   of the left ({b['summary']['mean_s']['right_active_s']:.1f} s vs {b['summary']['mean_s']['left_active_s']:.1f} s).
3. **Lots of small corrections**: about {b['summary']['reversals_per_episode']:.0f} joint direction
   reversals larger than 0.02 rad per episode. Typical teleoperation "fine-tuning" near grasps.
4. **Motion is well below the arm's own demonstrated speed** most of the time. That speed
   headroom is what the balanced and fast profiles use.

## 3. How the optimization works

Implemented in [`armopt/retime.py`](../armopt/retime.py). For each episode:

1. **Split into motion segments and pauses.** A frame counts as moving if either arm's
   joint speed is above 0.05 rad/s or a gripper moves faster than 0.01 m/s. Gaps under 0.2 s are
   bridged, and bursts under 0.1 s are treated as noise. Hesitations under 0.3 s stay inside the motion.
2. **Trim idle** before the first and after the last motion (0.1 s kept on each side).
3. **Compress pauses.** Each pause is crossed as fast as the limits allow, but keeps at least
   0.1 s, or 0.3 s right after a gripper action so the grasp can settle.
4. **Retime each motion segment** with time-optimal path parameterization (a TOPP-style
   backward and forward pass over the path speed), starting and ending at rest, subject to
   per-joint velocity and acceleration limits and a maximum speed-up over the demo.
   The acceleration budget is split between following path curvature and changing speed.
   That keeps every sample feasible on noisy recorded paths (exact TOPP chattered
   between stop and full speed on encoder noise).
5. **Never slower than the demo.** If a segment can't be sped up within the limits, it
   keeps its original timing, which the robot has already executed.
6. **Both arms share one time map**, so bimanual coordination and hand-overs are unchanged.
   Output is resampled at 30 fps; `state` and `action` are interpolated at the same source
   times, and a `source_frame` column records where each output frame came from.

Cost: O(N·J) per episode (N frames, J = 16 joints), about 10 ms per episode. All
531 episodes × 3 profiles run in about 25 s.

**Joint limits** are the 99th percentile of |velocity| and |acceleration| observed while
that arm was moving, estimated separately per dataset. Units: rad/s and rad/s²
(grippers m/s and m/s²).

<details><summary>Per-joint limits used</summary>

{limits}

</details>

### Profiles

| Profile | Idle & pauses | Motion speed-up cap | Accel budget on curvature |
|---|---|---|---|
| `safe` | removed / compressed | 1.0× (demo speed) | – |
| `balanced` | removed / compressed | 2.0× | 50% |
| `fast` | removed / compressed | 3.0× | 70% |

## 4. Results

{profiles}

How to read the limit columns. The retimed motion is measured exactly as the limits
were measured on the demos (same smoothing, same derivatives), on the frames the
optimizer actually retimed.

- **"p99 vel / acc vs limit"** is the worst episode's 99th-percentile joint speed and
  acceleration as a fraction of the limit. At or below 1.0 means the optimized arm moves
  like the fastest 1% of the operators' own motion, and not faster.
- **Single-frame peaks** do go above the p99 limit (up to {max(b['opt']['balanced']['acc_ratio_max'], d['opt']['balanced']['acc_ratio_max']):.1f}× in acceleration
  for balanced). But the original demos already peak at {b['opt']['balanced']['demo_acc_ratio_max_median']:.1f}× (bimanual) and
  {d['opt']['balanced']['demo_acc_ratio_max_median']:.1f}× (deksha) of the same limits in a typical episode.
- **The last column** counts episodes whose optimized peak is higher than the same episode's
  original peak.

Per-episode numbers are in `reports/<dataset>/optimization_<profile>.csv`.

![Savings per episode](figures/savings_hist.png)

![Example bimanual episode](figures/example_bimaual_dataset_new_1.png)

![Example deksha episode](figures/example_deksha_data_330_1.png)

## 5. The biggest remaining lever: run the arms in parallel

The retimer keeps the demonstrated order of operations. But the arms take turns, so
overlapping their work could save much more. If each episode took only as long as its
busier arm:

{parallel}

That is an upper bound. It can't be applied by retiming the recordings: it needs task
knowledge (which steps depend on each other, e.g. a hand-over) and collision checks
between the arms. The practical route is to **collect demonstrations where both arms
work at once** (e.g. the free arm pre-positions for its next grasp while the other places).

## 6. Recommendations

1. **Train on `optimized/<dataset>`** (balanced profile) to get a policy that runs about
   {min(b['opt']['balanced']['time_saved_pct'], d['opt']['balanced']['time_saved_pct']):.0f}–{max(b['opt']['balanced']['time_saved_pct'], d['opt']['balanced']['time_saved_pct']):.0f}% faster.
   Or use `safe` first if you want zero change to motion speed.
2. **Replay a few optimized episodes on the robot before training.** The 100 ms tracking lag
   means faster motion drifts further behind the command, so check tracking error and grasp
   success. Step up from `safe` to `balanced` to `fast`.
3. **Retime the videos too.** The optimized parquet has no video; each output frame's
   `source_frame` gives the original frame to take (round to nearest) from
   `videos/.../episode_XXXXXX.mp4`.
4. **Fix data collection:** stop recording on task completion instead of on a timer (the
   capped episodes are likely truncated), and regenerate `meta/` for bimaual_dataset_new_1
   so it matches the 201 parquet files.
5. **Collect parallel bimanual demonstrations** (section 5); that is where the next large saving is.

## Files

| Path | What |
|---|---|
| `armopt/analysis.py` | motion/idle segmentation, per-episode metrics |
| `armopt/retime.py` | limit estimation, time-optimal retiming, profiles |
| `armopt/io.py` | LeRobot v2.1 reader/writer |
| `scripts/analyze.py`, `scripts/optimize.py`, `scripts/report.py` | pipeline entry points |
| `reports/<dataset>/episode_metrics.csv` | per-episode analysis |
| `reports/<dataset>/optimization_<profile>.csv` | per-episode before/after |
| `reports/<dataset>/limits.json` | per-joint limits used |
| `optimized/<dataset>/` | retimed dataset (balanced profile), LeRobot layout, state/action only |
| `tests/` | unit tests (`python -m pytest`) |
"""
    (REPORTS / "REPORT.md").write_text(report)
    print("wrote reports/REPORT.md, examples:", examples)


if __name__ == "__main__":
    main()
