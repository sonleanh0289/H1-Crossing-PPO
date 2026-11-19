# Stability v2: directly reduce torso angular acceleration

Code preparation only: no simulator, training or runtime smoke was launched.
CPU tests validate reward math, selective resets, config routing and checkpoint
warm start. Actual learning improvement and simulator integration remain to be
tested. Crossing and the existing passing policies are not replaced.

## What changes

Source: the passing stability_v1 run02 checkpoint at iteration 1000.
Use a NEW warm start from its actor/critic weights, NOT a full resume:
iteration restarts at 0, optimizer is new, exploration std resets to 0.15.
The learned movement is retained in the weights, not randomly initialized.

| Setting | stability_v1 | stability_v2 |
|---|---:|---:|
| Fixed learning rate | 5e-5 | 2.5e-5 |
| Action-rate penalty | -0.01 | -0.015 |
| Torso XY angular-acceleration penalty | absent | -0.05 |
| Angular-acceleration normalization | none | 10 rad/s^2 |

Other v1 settings stay unchanged: 5 PPO epochs, entropy coefficient 0.001,
fresh std 0.15, base angular-velocity penalty -0.10, torso angular-velocity
penalty -0.05. Native tracking/orientation rewards, actuators, observations,
actions, flat scene and Stage E velocity range [-1, 3] m/s stay unchanged.
Keep turn_focus command sampling to retain stand/reverse/both-turn practice.
The camera is not used or filtered by this experiment.

The new unweighted reward term is:

    alpha_xy = (omega_world_xy[t] - omega_world_xy[t-1]) / 0.02
    penalty = (alpha_x / 10)^2 + (alpha_y / 10)^2

The reward manager applies weight -0.05 and the policy timestep 0.02 s.
World Z/yaw acceleration is excluded. The first sample after each selected env's
reset contributes zero; normal stopping, reversal and turning are NOT excluded.
There is no low-pass filtering or clipping of the acceleration signal.
The scale 10 is a normalization constant, NOT a hard limit or claimed safety
threshold. This matches the finite-difference definition used in evaluation.

Settings: `h1_finetune/training_profiles.py`.
Stateful reward: `h1_finetune/stability_rewards.py`.
v1 remains unchanged for old checkpoints. Full resume rejects profile/setting
changes; start a new warm-start run to change an objective.

## CMD: start a 1000-iteration experiment

Only the following command starts training. Use a new run suffix if output exists.
Do not delete old runs to reuse a name. 128 envs, save every 500 iterations, train
continuously, then full Stage E evaluation with stability metrics at the cap.

```bat
cd /d D:\IssacSim\H1FlatFineTune
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --stage E --init_checkpoint "D:\IssacSim\H1FlatFineTune\logs\stability_v1_e4000_run02\E_1000\checkpoint.pt" --training_profile stability_v2 --command_profile turn_focus --num_envs 128 --max_iterations 1000 --headless --output "D:\IssacSim\H1FlatFineTune\logs\stability_v2_from_v1_1000_run01"
```

Outputs go to `logs/stability_v2_from_v1_1000_run01/E_1000/`, including
`checkpoint_0500.pt`, `checkpoint_1000.pt`, `checkpoint.pt`, TensorBoard,
resolved reward/PPO configs, source/checkpoint checksums and `evaluation/`.
The end-of-run full evaluation may take considerably longer than the last
training update. It is not a second training job.

## Compare against the existing v1 evaluation

The v1 baseline already contains the required torso metrics. Do not compare raw
training reward across v1/v2 because their reward definitions differ.

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\compare_stability.py --baseline logs\stability_v1_e4000_run02\E_1000\evaluation\evaluation.json --candidate logs\stability_v2_from_v1_1000_run01\E_1000\evaluation\evaluation.json --output reports\stability_v2_1000_vs_v1_1000.json
```

For checkpoint 500:

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\evaluate.py --stage E --checkpoint logs\stability_v2_from_v1_1000_run01\E_1000\checkpoint_0500.pt --record_stability_metrics --headless --output reports\stability_v2_0500_eval
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\compare_stability.py --baseline logs\stability_v1_e4000_run02\E_1000\evaluation\evaluation.json --candidate reports\stability_v2_0500_eval\evaluation.json --output reports\stability_v2_0500_vs_v1_1000.json
```

Compare torso angular-acceleration RMS at 3 m/s first, then all speeds and both
turns, torso angular-speed RMS, action smoothness, velocity tracking, survival,
foot slip and torque saturation. Lower acceleration alone is not sufficient:
reject candidates that lose the full Stage E locomotion gate or introduce a
material regression elsewhere. Do not automatically select the newest checkpoint.
Tilt RMS includes sustained lean; it is not a +/- oscillation amplitude.
Per-scenario RMS is computed on surviving full trials, so always inspect survival.
No new numerical stability acceptance threshold is silently introduced.

## Play, or resume a selected v2 checkpoint

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\play.py --stage E --checkpoint logs\stability_v2_from_v1_1000_run01\E_1000\checkpoint.pt --num_envs 1 --keyboard
```

Resume only after examining results; this example adds 1000 iterations to a v2
checkpoint at 1000. It inherits the v2 profile, optimizer, learned std and env
count. `--max_iterations` is a total cap, not an additional count.

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --stage E --resume logs\stability_v2_from_v1_1000_run01\E_1000\checkpoint.pt --max_iterations 2000 --headless --output logs\stability_v2_from_v1_1000_run02
```

No bundle is exported or switched into crossing automatically.
