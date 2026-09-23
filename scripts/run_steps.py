"""Walk one episode through the optimization, one step at a time.

Usage: python scripts/run_steps.py [--dataset deksha_data_330_1] [--episode 0] [--profile balanced]
"""
import argparse

from _common import DATASETS, ROOT

from armopt.analysis import ActivityParams
from armopt.io import load_dataset
from armopt.retime import (PROFILES, estimate_limits, render, step1_trim_idle, step2_compress_pauses,
                           step3_speed_up_motion)


def describe(title, plan, lim, act, orig_s):
    _, _, summary = render(plan, lim, act)
    kinds = {k: sum(p.duration for p in plan.pieces if p.kind == k) for k in ("pad", "move", "pause")}
    print(f"{title:28s} {summary['new_s']:6.2f} s  (-{100 * (1 - summary['new_s'] / orig_s):4.1f}%)   "
          f"motion {kinds['move']:5.2f} s  pauses {kinds['pause']:5.2f} s  pads {kinds['pad']:4.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASETS[1], choices=DATASETS)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--profile", default="balanced", choices=list(PROFILES))
    args = ap.parse_args()

    ds = load_dataset(ROOT / args.dataset)
    rp, act = PROFILES[args.profile], ActivityParams()
    print(f"Estimating joint limits (p{rp.limit_percentile:g}) from {args.dataset} ...")
    lim = estimate_limits(ds.episodes(), rp.limit_percentile, act)
    path = next(p for p in ds.episode_files() if p.stem.endswith(f"{args.episode:06d}"))
    ep = ds.load_episode(path)
    orig_s = len(ep.state) / ep.fps

    print(f"\nEpisode {args.episode}, profile '{args.profile}'")
    print(f"{'Original recording':28s} {orig_s:6.2f} s")
    plan = step1_trim_idle(ep, rp, act)
    describe("Step 1: trim idle", plan, lim, act, orig_s)
    plan = step2_compress_pauses(plan, lim, rp)
    describe("Step 2: + compress pauses", plan, lim, act, orig_s)
    plan = step3_speed_up_motion(plan, lim, rp)
    describe("Step 3: + speed up motion", plan, lim, act, orig_s)
    retimed = [p for p in plan.pieces if p.kind == "move" and p.retimed]
    moves = [p for p in plan.pieces if p.kind == "move"]
    print(f"\n{len(retimed)} of {len(moves)} motion segments sped up; the rest were already at the limits "
          f"and keep the demo's timing.")


if __name__ == "__main__":
    main()
