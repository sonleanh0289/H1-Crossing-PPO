# Focused turning experiment (no change to reward/PPO or evaluation gates)

Start a separate branch from E checkpoint 3500. Do not overwrite old checkpoints.
The experiment uses 128 environments, inherited optimizer/std/LR and 5 PPO epochs.
The stage range stays [-1, 3] m/s; linear/yaw slew limits stay 2 m/s^2 and 1 rad/s^2.

`--command_profile turn_focus` samples 50% forward, 20% reverse, 10% stand, 20% in-place turns.
Turn signs are independently balanced in expectation. Of in-place turns, 60% have
|yaw| uniformly in [0.4, 0.5] rad/s, and 40% uniformly in [0, 0.4]. Moving yaw and
forward upper-half sampling are unchanged. The baseline profile remains 60/25/10/5.
Omitting the profile on resume inherits the checkpoint profile; legacy checkpoints
default to baseline. An explicit profile switches the sampler and records history
in the new checkpoint, without resetting the optimizer or learned exploration std.

## CMD: train 1000 additional iterations, from 3500 to 4500

```bat
cd /d D:\IssacSim\H1FlatFineTune
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --stage E --resume "D:\IssacSim\H1FlatFineTune\logs\20260917T042757_895748Z_train\E_4500\checkpoint_3500.pt" --command_profile turn_focus --num_envs 128 --max_iterations 4500 --seed 42 --headless --output "D:\IssacSim\H1FlatFineTune\logs\turn_focus_E3500_to4500"
```

Use a new output folder for each experiment. Outputs here go under
`logs\turn_focus_E3500_to4500\E_4500`, including checkpoints 4000 and 4500.
Training is continuous; full evaluation runs only at the end. For a shorter 500-iteration
experiment, use cap 4000 and a different output directory.

## CMD: test both new checkpoints, including yaw traces

```bat
for %i in (4000 4500) do D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\evaluate.py --stage E --checkpoint "D:\IssacSim\H1FlatFineTune\logs\turn_focus_E3500_to4500\E_4500\checkpoint_%i.pt" --record_yaw_trace --headless --output "D:\IssacSim\H1FlatFineTune\reports\turn_focus_E_checkpoint_tests\iter_%i"
```

In a .bat file, use %%i instead of %i. Keep the full evaluation suite and original
pass thresholds. Do not select a model on turn_right alone.

## Optional baseline diagnostic before training

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\evaluate.py --stage E --checkpoint "D:\IssacSim\H1FlatFineTune\logs\20260917T042757_895748Z_train\E_4500\checkpoint_3500.pt" --record_yaw_trace --headless --output "D:\IssacSim\H1FlatFineTune\reports\E_checkpoint_tests\iter_3500_yaw_trace"
```

`yaw_trace.csv` records both turns at 50 Hz for all 50 trials/scenario: target,
post-step smoothed command, actual yaw rate, signed error (actual minus target),
seed, trial, phase and alive flag. This is diagnostic only, not a new gate.
The first 250 steps are ramp; evaluation measures the next 500 steps.
Ignore rows with alive=False when interpreting valid motion: the simulator may
have reset that robot. A partial CSV after interruption is not a passing evaluation.
On turn_right, positive signed bias indicates slower/right-to-left-biased rotation;
negative bias indicates excess rightward rotation. Use the time series to distinguish
steady bias from oscillation. No guarantee that additional training will pass.

The final automatic evaluation of a turn_focus run also records yaw_trace.csv.
No training or simulator execution is part of the code-only implementation checks.
