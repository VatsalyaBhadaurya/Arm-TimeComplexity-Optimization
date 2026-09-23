# Arm Time-Complexity Optimization

Reduce the time complexity / execution time of robot-arm trajectories using two
collected datasets:

- [`aiengineer56/bimaual_dataset_new_1`](https://huggingface.co/datasets/aiengineer56/bimaual_dataset_new_1) — bimanual arm episodes
- [`aiengineer56/deksha_data_330_1`](https://huggingface.co/datasets/aiengineer56/deksha_data_330_1) — single-arm episodes

## Status

This container's network egress policy currently blocks `huggingface.co`, so
the datasets have not been pulled into the repo yet (`git clone` and API
access both return `403`). To unblock:

1. Open the environment menu in the session title bar → **Edit** → **Network
   access**, and either allow `huggingface.co` or broaden the access level.
2. Re-run:
   ```bash
   git clone https://huggingface.co/datasets/aiengineer56/bimaual_dataset_new_1 data/bimaual_dataset_new_1
   git clone https://huggingface.co/datasets/aiengineer56/deksha_data_330_1 data/deksha_data_330_1
   ```

## Plan once data is available

1. **Analyze** — episode lengths, frame rates, per-joint velocity/acceleration
   profiles, idle time at episode start/end and mid-episode pauses, jitter and
   back-and-forth corrections, and (for the bimanual set) how much each arm
   contributes to total task time.
2. **Optimize** — trim idle/dead time, smooth jittery corrections, and
   re-time each trajectory as fast as the joint speed/acceleration limits
   observed in the data allow, producing a faster version of each episode.
3. **Report** — quantify time saved per episode and in aggregate.

## Layout

- `data/` — cloned datasets (git-ignored until pulled)
- `scripts/` — analysis and optimization scripts
- `reports/` — generated analysis/optimization reports
