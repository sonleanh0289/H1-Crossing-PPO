# H1 stability refinement (code preparation only)

This opt-in experiment starts from the passing E4000 weights already used by
H1Crossing. It does not replace that checkpoint, change the camera, or train until
you explicitly run a command below. No simulator/training was launched while
preparing this feature; CPU unit tests use tensors and mocked runtime objects.
Effectiveness is unverified until training and paired evaluation.

## Objective and isolation

Use the existing H1FlatFineTune project with a NEW run directory. Baseline training
commands and crossing playback remain unchanged. The flat asset, free root,
69 observations, 19 actions, action scale, actuators, physics/policy rates,
Stage E range [-1, 3] m/s, yaw limits and command slew are unchanged.
Do not overwrite the source bundle or copy a new policy into crossing yet.

Training profile: `--training_profile stability_v1`.
Settings live in `h1_finetune/training_profiles.py`:

| Setting | Baseline | stability_v1 |
|---|---:|---:|
| Fixed learning rate | 1e-4 | 5e-5 |
| PPO epochs/update | 5 | 5 |
| Entropy coefficient | 0.005 | 0.001 |
| Fresh warm-start exploration std | 0.30 | 0.15 |
| Base angular XY penalty weight | -0.05 | -0.10 |
| Action-rate penalty weight | -0.005 | -0.01 |
| Torso world-XY angular penalty | absent | -0.05 |

Torso penalty ignores world Z (intentional yaw). Flat orientation stays -1.0;
feet-air-time stays 0.25 with threshold 0.4 s. Tracking rewards and all other
terms remain unchanged. The torso penalty complements the base penalty: neither
freezes the body. These are conservative experimental settings, not proven optima.
Do not compare raw training rewards across these different reward objectives.

The new warm start loads BOTH actor and critic weights from E4000, but starts
iteration 0 with a fresh optimizer and std 0.15; it is NOT training from random
weights and is NOT a full resume of E4000. Reduced exploration may help preserve
the gait, but could also slow adaptation. Evaluate instead of assuming improvement.

Keep `--command_profile turn_focus` as used by the source E4000 (50% forward,
20% reverse, 10% stand, 20% turn); do not discard previously learned turn behavior.
No camera/rendering is required for this flat headless training.

## CMD: measure baseline (no training)

Run once to get the new torso metrics on the old passing checkpoint. The old
bundle evaluation has locomotion metrics but not the new torso measurements.
This is the full Stage E evaluation, not a short smoke test.

```bat
cd /d D:\IssacSim\H1FlatFineTune
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\evaluate.py --stage E --checkpoint "D:\IssacSim\H1Crossing\checkpoints\h1_flat_stage_e_4000\checkpoint.pt" --record_stability_metrics --headless --output reports\stability_e4000_baseline
```

## CMD: start a 1000-iteration pilot

This command DOES start training. 128 envs, 24 rollout steps, 5 PPO epochs,
checkpoint every 500 iterations. Continuous training; full Stage E evaluation
with stability metrics only after the requested cap. Start small and compare
500/1000 instead of assuming the final checkpoint is best.

```bat
cd /d D:\IssacSim\H1FlatFineTune
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --stage E --init_checkpoint "D:\IssacSim\H1Crossing\checkpoints\h1_flat_stage_e_4000\checkpoint.pt" --training_profile stability_v1 --command_profile turn_focus --num_envs 128 --max_iterations 1000 --headless --output logs\stability_v1_e4000_run01
```

Outputs: `logs/stability_v1_e4000_run01/E_1000/`:

- `checkpoint_0500.pt`, `checkpoint_1000.pt`, `checkpoint.pt`.
- `tensorboard/`, training loss/reward JSON, resolved configs, source hashes.
- `evaluation/evaluation.json` and `trials.csv` including torso RMS metrics.

Use a new run suffix if this directory already contains a run. Do not delete
or overwrite old training results. If VRAM is insufficient, start a NEW warm-start
run with fewer environments; a full resume requires the original env count.

## CMD: evaluate checkpoint 500 and compare

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\evaluate.py --stage E --checkpoint logs\stability_v1_e4000_run01\E_1000\checkpoint_0500.pt --record_stability_metrics --headless --output reports\stability_v1_0500
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\compare_stability.py --baseline reports\stability_e4000_baseline\evaluation.json --candidate reports\stability_v1_0500\evaluation.json --output reports\stability_v1_0500_comparison.json
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\compare_stability.py --baseline reports\stability_e4000_baseline\evaluation.json --candidate logs\stability_v1_e4000_run01\E_1000\evaluation\evaluation.json --output reports\stability_v1_1000_comparison.json
```

Measured on torso_link at 50 Hz after 5 s warmup, over the next 10 s:
tilt RMS (rad), horizontal angular-speed RMS (rad/s), horizontal angular-
acceleration RMS (rad/s^2, finite difference), and vertical-speed RMS (m/s).
Tilt includes sustained lean, so smaller tilt alone does not prove less shaking.
Angular speed/acceleration and action-rate help distinguish oscillation from lean.
Only surviving full trials are aggregated; always inspect survival too.
New metrics cover fixed-command scenarios, not the stop/reverse transition series.

Comparison reports percentage changes per scenario, not a single averaged score.
Negative means lower. It NEVER turns lower oscillation into an automatic pass.
Keep the full original Stage E gates (stand, both turns, all speed milestones,
transition sequence), then compare torso motion/action smoothness at 1/2/3 m/s,
foot slip, torque saturation, and tracking error. Check both GUI motion and
metrics; camera filtering is not a balance improvement.

## CMD: resume this experiment later

Only use after deciding further training is justified. A new profile cannot be
silently resumed from a baseline checkpoint: use the warm-start command above.
A stability resume restores optimizer, learned std, LR, profile and iterations;
changes to profile settings are rejected.

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --stage E --resume logs\stability_v1_e4000_run01\E_1000\checkpoint.pt --max_iterations 2000 --headless --output logs\stability_v1_e4000_run02
```

2000 is the TOTAL cap: a checkpoint at 1000 gets 1000 additional iterations.
Profile/128-env setting are inherited; do not supply a conflicting profile.
To play a candidate in the isolated flat simulator (not crossing):

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\play.py --stage E --checkpoint logs\stability_v1_e4000_run01\E_1000\checkpoint.pt --keyboard
```

CPU tests only (does not train):

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B -m unittest discover -s tests -v
```
