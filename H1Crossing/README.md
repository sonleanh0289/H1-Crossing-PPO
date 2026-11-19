# H1 Crossing

H1 Crossing is a simulation research project on humanoid road crossing with incomplete traffic observations. It runs in Isaac Sim and Isaac Lab. The project combines RGB-D vehicle tracking, radar velocity estimates, and a PPO policy that selects a forward, stop, or reverse command. A frozen, fine-tuned H1 locomotion policy executes the command.

The main research question is whether simulator ground truth can improve training while the crossing policy relies on sensor observations at inference.

## How it works

1. Three RGB-D cameras estimate vehicle position. Two radar heads estimate velocity, with camera-based velocity as a fallback when valid.
2. The crossing actor reads 23 features from perception, odometry, and the mapped goal. During training, the critic sees the same features plus six ground-truth vehicle features. At inference, only the actor runs.
3. The actor runs at 10 Hz and outputs a bounded action mapped to a speed command between -1 and 3 m/s. The command guard, heading controller, and smoothing feed the locomotion policy at 50 Hz.

| Component | Entry points |
| --- | --- |
| Training and inference | [Train](scripts/train_crossing_behavior.py), [play](scripts/play_crossing_behavior.py), [PPO workflow](scripts/behavior_training.py) |
| Environment and reward | [Crossing environment](scripts/behavior_environment.py), [reward configuration](configs/crossing_rewards.yaml) |
| Sensor and critic observations | [Camera perception](scripts/camera_perception.py), [radar observations](scripts/radar_observations.py), [privileged critic features](scripts/asymmetric_observations.py) |
| Command and evaluation | [Command guard](scripts/crossing_command_guard.py), [trajectory recording](scripts/behavior_trajectory.py), [ground-truth diagnostics](scripts/ground_truth_evaluation.py) |

## Selected simulation result

The asymmetric PPO checkpoint at iteration 1,000 was evaluated deterministically over five seeds (42â€“46), with 1,000 decisions per seed and the original guard. The counts below cover 28 completed episodes. Five episodes still in progress at the evaluation cutoff are excluded.

| Reached goal | Collided | Timed out | Fell | Left bounds |
| ---: | ---: | ---: | ---: | ---: |
| 14 | 4 | 10 | 0 | 0 |

All four collisions involved cars that did not yield. The actor requested forward motion throughout this evaluation and relied heavily on the guard for stopping. This experiment measures performance with the guard enabled. A matched experiment using a sensor-only critic is needed to isolate the effect of privileged information. The default [asymmetric configuration](configs/crossing_behavior_asymmetric.yaml) now uses `dual_loss_ttc_v1`, a different guard from the one used in this result.

## Run the project

Keep [H1FlatFineTune](../H1FlatFineTune/README.md) next to `H1Crossing/`. Activate a Python environment that can already run Isaac Lab. See the [Python dependency list](requirements.txt) for setup.

Provide the `checkpoints/h1_flat_stability_v2_0500/` locomotion bundle or point `--bundle` to a validated copy. The loader checks the policy, source checkpoint, evaluation, and metadata together. See the [locomotion bundle guide](README_FINETUNED.md) for details.

From `H1Crossing/`, run training with the default asymmetric configuration:

```bash
python -B scripts/train_crossing_behavior.py \
  --observation_source camera_radar --asymmetric_critic --fresh_behavior \
  --config configs/crossing_behavior_asymmetric.yaml \
  --perception_config configs/crossing_perception_multicolor_960x720.yaml \
  --radar_config configs/crossing_radar.yaml --vehicle_palette multicolor \
  --num_envs 8 --max_iterations 1000 --seed 42 --headless
```

The run writes numbered checkpoints every 200 iterations. Run `python -B scripts/play_crossing_behavior.py --help` for playback options.

