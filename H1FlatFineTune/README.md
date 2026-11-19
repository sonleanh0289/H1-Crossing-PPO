# H1 Flat Fine-Tune

This project fine-tunes the published Isaac Lab H1 Flat locomotion policy for forward motion, reversing, standing, and turning on level ground. It runs as an external Isaac Lab task. The adjacent H1Crossing project uses an exported locomotion bundle from this work.

## Setup and code checks

Run the following commands from `H1FlatFineTune/` with your Isaac Lab Python environment active. `scripts/train.py --help` displays the available training modes. The code checks use CPU tests, contract mocks, and checksums.

```powershell
python -B scripts/check_code.py
python -B scripts/train.py --help
```

The reference base checkpoint is `checkpoints/base/checkpoint.pt`, a copy of the published H1 Flat model. Its SHA-256 is:

```text
8800eb8fc6c161949f9bca903e976ad7c5ef993b6b47b3a806acad4658047f66
```

| Path | Purpose |
| --- | --- |
| `h1_finetune/` | External tasks, command profiles, PPO worker, evaluation, and export |
| `scripts/` | Training, evaluation, playback, export, and code check entry points |
| `tests/` | CPU unit tests and runtime contract mocks |
| `logs/` | Training runs and checkpoints organized by stage |
| `reports/` | Evaluation evidence and code checks |
| `exports/` | Bundles produced from checkpoints that pass evaluation |

## Locomotion contract

The project registers `H1-Velocity-Flat-RunRetreat-v0` and `H1-Velocity-Flat-RunRetreat-Play-v0` when `h1_finetune.tasks` is imported after Isaac Lab's AppLauncher starts. Both inherit `H1FlatEnvCfg` and use the free-root `H1_MINIMAL_CFG` asset on a plane.

The 69 observations are ordered as linear velocity (3), angular velocity (3), projected gravity (3), filtered command (3), relative joint positions (19), relative joint velocities (19), and previous action (19). The 19 actions are joint position offsets with scale 0.5, added to the default joint positions. The project preserves the native joint order and actuator configuration.

The actor and critic use three 128-unit hidden layers with ELU activations. Observation normalization is disabled. Physics uses a 0.005 s step and four physics steps per policy update, giving a 50 Hz policy rate. Runtime checks cover input and output dimensions, joint and observation order, action scaling and offsets, actuators, and the flat observation configuration. Fine-tuned checkpoints record the resolved contract for resume, playback, and evaluation.

## Training

The default setup uses 128 environments, 24 rollout steps per environment, five PPO epochs, and four minibatches. This produces 3,072 transitions and 20 optimizer updates per iteration. PPO uses a fixed learning rate of 1e-4, entropy coefficient 0.005, clipping 0.2, gamma 0.99, lambda 0.95, and maximum gradient norm 1.0.

| Stage | Forward velocity command range (m/s) |
| --- | --- |
| A | [-0.50, 2.00] |
| A1 | [-0.30, 1.25] |
| A2 | [-0.35, 1.50] |
| A3 | [-0.40, 1.50] |
| A4 | [-0.45, 1.75] |
| A5 | [-0.50, 2.00] |
| B | [-0.625, 2.25] |
| C | [-0.75, 2.50] |
| D | [-0.875, 2.75] |
| E | [-1.00, 3.00] |

New runs can progress through A1–A5 and then B–E. Each promotion requires a passing evaluation. Stage A remains available for older runs. See the [128-environment curriculum commands](FRESH_A1_128.md) and the [historical Stage A resume notes](CONTINUE_STAGE_A.md).

The default sampler allocates 60% of commands to forward motion, 25% to reverse, 10% to standing, and 5% to turning in place. Among forward commands, 60% are drawn from the upper half of the speed range. Among moving commands, 30% include yaw. The lateral command is zero. Commands are resampled every 4–8 s and ramp at up to 2 m/s² and 1 rad/s². The yaw-rate limit is 0.5 rad/s at speeds up to 1.5 m/s and 0.25 rad/s above that speed. During acceleration, the controller waits at the speed threshold if yaw still exceeds the faster-motion limit.

Episodes last 20 s and start from a standing pose with zero root velocity and random yaw. The default training setup retains observation noise and material/mass randomization and disables pushes and external wrenches. Its feet-air-time reward has a weight of 0.25 and a 0.4 s threshold. Evaluation disables observation noise and startup material/mass randomization.

```powershell
# Short PPO smoke test with 32 environments and five iterations
python -B scripts/train.py --smoke_test --headless

# Standalone Stage A run
python -B scripts/train.py --stage A --num_envs 64 --max_iterations 3000 --seed 42 --headless

# Resume a compatible Stage A checkpoint up to a total of 6,000 iterations
python -B scripts/train.py --resume "path/to/checkpoint.pt" --stage A --max_iterations 6000 --seed 42 --headless
```

`--init_checkpoint PATH` starts a new run with the checkpoint's actor and critic weights, a fresh optimizer, action standard deviation 0.3, and iteration 0. `--resume` restores the optimizer, learned standard deviation, learning rate, iteration counters, and Torch/Python RNG state. Resume requires matching PPO settings and environment count and starts new episodes. Changing the range within the same stage requires `--allow_command_range_change`. Moving from A5 to B requires a passing `--promotion_report` for the source checkpoint. For less GPU memory use, start a new warm-start run with fewer environments.

The worker saves a checkpoint every 500 iterations and evaluates at the end of the stage in a separate process. `--max_iterations` sets the total cap for the stage. `--stage` runs one stage, and `--curriculum` advances through the stages that pass evaluation.

Optional profiles are described in [turn_focus](TURN_FOCUS.md), [stability_v1](STABILITY_FINETUNE.md), and [stability_v2](STABILITY_V2.md). The latter adds an angular-acceleration penalty at 50 Hz.

## Playback

```powershell
# Fixed forward command in headless mode
python -B scripts/play.py --headless --steps 300 --vx 0.5

# Windows GUI playback with a fine-tuned checkpoint
python -B scripts/play.py --stage E --checkpoint "path/to/checkpoint.pt" --num_envs 1 --keyboard
```

Focus the simulator viewport and release the control keys before starting. **W/S** requests 1 m/s forward or 0.5 m/s in reverse by default. **A/D** turns. **Shift+W/S** uses the stage limits, which are +3 and -1 m/s at Stage E. Change the default speeds with positive values for `--keyboard_speed` and `--keyboard_reverse_speed`. **Space** latches a stop, **Enter** resumes after the keys are released, and **R** resets the scene. Moving focus away requests zero speed. Commands still ramp down through the acceleration limiter. Use keyboard mode in the Windows GUI on its own.

For Windows joystick playback, replace `--keyboard` with `--joystick`. Center the stick to arm it. Use the vertical axis for forward and reverse and the horizontal axis for turning. Buttons 1, 2, and 3 resume, latch a stop, and reset, respectively. Disconnecting the joystick requests zero speed. To adjust the mapping, use `--forward_axis y --turn_axis x --forward_sign -1 --turn_sign -1` and select the device with `--joystick_id 0`.

## Evaluation and export

```powershell
python -B scripts/evaluate.py --checkpoint "path/to/checkpoint.pt" --stage C --headless
python -B scripts/export.py --checkpoint "path/to/checkpoint.pt" --report "path/to/evaluation.json" --output "exports/validated_C"
```

Evaluation runs 50 rollouts per scenario using 10 environments and five fixed seeds (42, 142, 242, 342, and 442). It covers standing, turning at ±0.5 rad/s, moving at 0.5 m/s, and speed milestones from the selected and prior stages. For fixed commands, it skips the first 5 s of the ramp and measures the next 10 s. Linear speed is measured in a yaw-aligned frame, and yaw rate is measured about world Z.

The fixed-command gate requires finite values and at least 95% fall-free rollouts. Forward velocity MAE must be at most 0.20 m/s at speeds up to 1 m/s and at most 0.30 m/s above 1 m/s. Lateral velocity MAE must be at most 0.15 m/s, yaw-rate MAE at most 0.20 rad/s, and mean planar speed while standing at most 0.10 m/s. Falls include termination, root height below 0.65 m, or tilt over 60°. Evaluation logs also record tilt, foot slip during contact, joint effort near the configured limit, and action-rate MSE.

The 20 s sequence alternates four-second phases of forward, stop, reverse, stop, and forward. Each stop must reach a planar speed below 0.15 m/s within 3 s and hold it for at least 1 s. The moving phases must track the requested speed during their final second. At least 95% of rollouts must pass the sequence. Reports include JSON and per-rollout CSV outputs.

Export runs on CPU and requires a complete passing report for the same checkpoint and contract. The bundle includes a deterministic TorchScript policy, its source checkpoint, resolved configs, the evaluation report, and a checksum manifest. The manifest lists the evaluated speed milestones on flat ground. See the [crossing integration guide](../H1Crossing/README_FINETUNED.md) for the Stability v2 bundle used in H1Crossing.

Each training run records the seed, software versions, source hashes, resolved configuration, and input checkpoint hash. The historical code-only handoff documented CPU checks separately in `reports/code_checks.json`. Use the evaluation report associated with each exported bundle when citing locomotion results.
