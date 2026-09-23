"""Retime every episode under each profile and write the optimized dataset.

Writes
  reports/<dataset>/limits.json                  per-joint velocity/acceleration limits
  reports/<dataset>/optimization_<profile>.csv   per-episode before/after durations
  reports/<dataset>/optimization_summary.json
  optimized/<dataset>/                            LeRobot copy retimed with --write-profile
Usage: python scripts/optimize.py [--write-profile balanced]
"""
import argparse
import json
import shutil
from dataclasses import asdict

import pandas as pd
from _common import DATASETS, OPTIMIZED, REPORTS, ROOT

from armopt.io import episode_stats, load_dataset, write_episode, write_meta
from armopt.retime import PROFILES, estimate_limits, retime_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-profile", default="balanced", choices=list(PROFILES))
    args = ap.parse_args()

    for name in DATASETS:
        ds = load_dataset(ROOT / name)
        episodes = list(ds.episodes())
        pct = PROFILES[args.write_profile].limit_percentile
        lim = estimate_limits(episodes, pct)
        out = REPORTS / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "limits.json").write_text(json.dumps({
            "percentile": pct,
            "joints": ds.joint_names,
            "vmax": lim.vmax.round(4).tolist(),
            "amax": lim.amax.round(4).tolist(),
        }, indent=2))

        summary = {}
        for prof, rp in PROFILES.items():
            write = prof == args.write_profile
            dest = OPTIMIZED / name
            if write and dest.exists():
                shutil.rmtree(dest)
            rows, new_eps, stats, gidx = [], [], [], 0
            for ep in episodes:
                new, src, row = retime_episode(ep, lim, rp)
                rows.append(row)
                if write:
                    chunk = ep.index // ds.info["chunks_size"]
                    write_episode(dest, chunk, new, src, gidx)
                    stats.append(episode_stats(new, gidx))
                    new_eps.append(new)
                    gidx += len(new.state)
            df = pd.DataFrame(rows)
            df.to_csv(out / f"optimization_{prof}.csv", index=False, float_format="%.4f")
            summary[prof] = {
                "params": asdict(rp),
                "orig_hours": round(df.orig_s.sum() / 3600, 3),
                "new_hours": round(df.new_s.sum() / 3600, 3),
                "time_saved_pct": round(100 * (1 - df.new_s.sum() / df.orig_s.sum()), 2),
                "mean_orig_s": round(df.orig_s.mean(), 2),
                "mean_new_s": round(df.new_s.mean(), 2),
                "median_speedup": round(df.speedup.median(), 3),
                "episodes_slower": int((df.new_s > df.orig_s + 1e-6).sum()),
                "retimed_frac_mean": round(df.retimed_frac.mean(), 3),
                "vel_ratio_p99_max": round(df.vel_ratio_p99.max(), 3),
                "acc_ratio_p99_max": round(df.acc_ratio_p99.max(), 3),
                "vel_ratio_max": round(df.vel_ratio_max.max(), 3),
                "acc_ratio_max": round(df.acc_ratio_max.max(), 3),
                "demo_vel_ratio_max_median": round(df.demo_vel_ratio_max.median(), 3),
                "demo_acc_ratio_max_median": round(df.demo_acc_ratio_max.median(), 3),
                "episodes_vel_peak_above_demo": int((df.vel_ratio_max > df.demo_vel_ratio_max).sum()),
                "episodes_acc_peak_above_demo": int((df.acc_ratio_max > df.demo_acc_ratio_max).sum()),
            }
            if write:
                write_meta(dest, ds, new_eps, stats, {"profile": prof, **summary[prof], "limits": "see reports"})
            print(f"{name} {prof:9s} saved {summary[prof]['time_saved_pct']:5.1f}%  "
                  f"mean {summary[prof]['mean_orig_s']:.1f}s -> {summary[prof]['mean_new_s']:.1f}s")
        (out / "optimization_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
