# Crossing behavior PPO — frozen H1 locomotion

This is a separate high-level trainer in H1Crossing. It does NOT fine-tune the
19-joint locomotion policy or change keyboard/joystick/manual preview.
Uses the installed env_isaaclab, adjacent H1FlatFineTune helper package and
RSL-RL PPO; no package installation or Isaac Lab core edits.

Reward update 2026-09-25: timeout -1 with no bonus; signed goal progress +0.5/m
closer and -0.5/m farther. Start a fresh behavior run with the command below.
Do not resume a checkpoint trained under the old timeout-bonus reward contract.
Frozen locomotion remains unchanged; this is not training H1 locomotion from scratch.

## CMD: train a new behavior policy

```bat
cd /d D:\IssacSim\H1Crossing
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train_crossing_behavior.py --num_envs 32 --max_iterations 1000 --headless
```

32 environments is a conservative starting configuration, not a measured 8GB
capacity limit. Increase only after checking VRAM and throughput. An iteration is
32 behavior transitions per environment, followed by 5 PPO optimization epochs;
it is not an episode. No simulator starts without --max_iterations or --smoke_test.

Each run creates a new timestamped logs/*_behavior_train directory. No overwrite.
Checkpoints are saved every 500 iterations, without stopping/reloading the scene,
and checkpoint.pt at the end. Metrics: metrics.jsonl and tensorboard/.
The final training_result.json records completion and checkpoint integrity;
failure.json means the run did not finish successfully.

## Resume (behavior checkpoint, NOT locomotion checkpoint)

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train_crossing_behavior.py --resume "D:\IssacSim\H1Crossing\logs\YOUR_RUN\checkpoint_000500.pt" --max_iterations 1000 --headless
```

Replace YOUR_RUN with the actual folder printed at startup.
--max_iterations is the TOTAL target: from 500 to 1000 means 500 additional updates.
Resume restores actor, critic, optimizer, learning rate, iteration and RNG states.
It inherits environment count and seed, and refuses changed reward/config/schema
or frozen locomotion bundle. Physics and rollout buffers are not serialized:
resuming starts fresh episodes and is not bit-identical to uninterrupted training.
Outputs go to a NEW run folder; the source checkpoint is preserved.

## Play the learned crossing behavior in Isaac Lab

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\play_crossing_behavior.py --checkpoint "D:\IssacSim\H1Crossing\logs\YOUR_RUN\checkpoint.pt" --num_envs 1 --steps -1
```

This runs the deterministic behavior actor, not keyboard control.
Add --headless --steps 210 for a bounded non-GUI check.
play_result.json records events but is NOT a multi-seed acceptance evaluation.
Smoke checkpoints contain only a few updates and are not competent crossing models.

## Configuration and control

- configs/crossing_behavior.yaml: PPO, episode length, heading and contact settings.
- configs/crossing_rewards.yaml: requested rewards; see REWARDS.md.
- scripts/behavior_environment.py: physics stepping, frozen action adapter,
  collision/fall/goal/timeout, selective reset.
- scripts/behavior_training.py: RSL-RL PPO loop, logging, save/resume/play.
- scripts/behavior_control.py: action mapping, observation schema and traffic.

One action at 10 Hz: clamp to [-1,1], deadband +/-0.05 for stop, then rescale.
Negative actions command retreat down to -1 m/s; positive actions forward up to
3 m/s. Heading is automatically steered toward the goal, not learned. Yaw command
is limited to 0.5 rad/s, or 0.25 rad/s above 1.5 m/s. Existing command smoothing
limits linear acceleration to 2 m/s² and yaw-rate change to 1 rad/s².
Frozen locomotion runs at 50 Hz; PhysX at 200 Hz. No camera observations.

The default frozen bundle is checkpoints/h1_flat_stability_v2_0500. Its 69D
locomotion observations, 19 actions, joints/scales/actuators stay unchanged.
Only a NEW behavior actor/critic learns: 19 inputs, hidden [128,128], ELU;
actor has one Gaussian action, critic one value. No observation normalization.
PPO: rollout32, LR3e-4 fixed, 5 epochs, 4 minibatches, entropy0.005,
initial std0.5; remaining PPO settings are copied from the validated bundle and
saved explicitly in training_contract.json.

## Behavior observation: 19 values

The raw crossing group remains 8D: relative car XY, relative goal XY, car world
velocity XY, collision TTC and TTC validity. For the behavior actor these are
scaled by [50,10,10,10,15,15,10,1]. Additional inputs:
H1 world planar velocity/3 (2), cos/sin yaw (2), current smoothed velocity/yaw
command divided by [3,1,0.5] (3), stationary anchor XY offset/0.1 (2),
stationary time/episode duration (1), remaining episode fraction (1).
All are clipped to [-10,10]. No H1 joints or camera images enter this actor.
The exact ordered schema is stored in every checkpoint.

## Scene, reward and episode lifecycle

### Variable-speed vehicles (2026-09-26)

Train/play now default to `--traffic_mode variable`. Each vehicle samples a
target speed uniformly in 30–50 km/h on reset and every 1–3 seconds while moving.
Speed approaches the target with acceleration <=2 m/s² and braking <=3 m/s²;
position integrates that ramp without teleporting. Target changes do not reset
the episode. Speed can jump only on a spawn/reset. Cars still move along +X.
Target timers are checked at simulation-step boundaries, not at PPO iterations.
Defaults live in `scripts/vehicle_motion.py:TrafficConfig`, shared by batched
behavior and keyboard/joystick play. No speed-history logging was added.

Existing 19D observations use current car velocity, not target velocity.
TTC is recomputed using current relative velocity (constant-velocity prediction,
not an acceleration-aware forecast). No acceleration observation was added.

Use `--traffic_mode constant --yield_probability 0` to reproduce constant speed between respawns.
Old behavior checkpoints may be played under variable traffic with a warning;
their previous success metrics do not certify performance under acceleration.
Resume rejects changed traffic dynamics: legacy checkpoints require
`--traffic_mode constant --yield_probability 0`; new traffic training starts without `--resume`.
Reward, network and frozen locomotion contracts remain unchanged.

### Probabilistic yielding (default 50%)

`--yield_probability 0.5` independently selects a yielding driver once per spawn
or episode reset, not once per timestep. It is a probability, not an exact 50/50
quota or a reward weight. `0` disables yielding; `1` makes every driver willing
to yield. Both behavior train/play and keyboard/joystick play use this setting.

Yielding drivers react to simulator ground-truth H1 position in env-local XY:
|X|<=2.5m and Y from -5.5m to the far edge of the vehicle lane plus H1 radius and
0.4m clearance (Y=-0.1m in the default lane). This includes H1 waiting on the
near sidewalk; it is not learned camera perception or a prediction of intent.
They approach a center stop line X=-5.5m with a 0.25m buffer, slow using a
stopping-distance envelope, and may stop at 0m/s until the conflict region has
been clear for 0.5s. They then accelerate back toward the current cruise target.
Non-yielding drivers retain random cruise behavior; cars already clear of the
crossing do not brake for H1 behind them.

Normal deceleration is 3m/s²; late yielding may use up to 8m/s² emergency braking.
No speed/position teleport is used to guarantee a stop. A fast vehicle spawned
too near the crossing, or H1 entering late, may still collide. The 30–50km/h
range describes cruise targets, not actual speed during yielding or recovery.
No driver-type or acceleration observation and no extra event log was added.
TTC remains a current-velocity approximation and does not forecast braking.

Traffic/yield parameters are checkpoint-contract metadata. Checkpoint 1000 can
be played with the new default traffic with a warning, but is not certified for
it. Resume with changed traffic is rejected. To resume earlier variable-speed
checkpoints without yielding, pass `--yield_probability 0`.

- Flat road, flat sidewalks; H1 starts (0,-4.6,1.2), yaw +90 degrees.
  Goal (0,+4.6); all coordinates offset by each environment origin.
- Independent cars: +X, lane Y=-1.75, sampled 30–50 km/h, spawn X in [-50,-15].
  Respawn after X>25. Environment spacing120m separates 110m road layouts.
- Goal: root within0.6m XY, +1 once; safe goal end adds +1 => +2.
- Natural20s timeout costs -1, with NO safe bonus. Safe goal still earns +2.
- Signed progress = 0.5 * (previous_goal_distance - current_goal_distance).
  No extra dt. No positive progress on a collision/fall substep; zero while stationary.
- Actual filtered PhysX H1-car normal contact >1N: -2 and terminal.
  Filters include each robot rigid body separately, excluding ground/other envs.
  Contacts are checked every physics substep, not inferred from TTC.
- Fall: root height<0.55m or root up-axis Z<0.5 (tilt>60degrees).
  Terminal, no safe bonus; default extra fall penalty remains0.
- Out-of-bounds: local |X|>55 or |Y|>8. Terminal, no safe bonus, no extra penalty.
- Stationary cost: existing integral ramp after2s; no second multiplication by dt.
  See REWARDS.md for coefficients and known incentive limitations.

The wrapper calculates rewards from pre-reset physics state, latches terminal
events, then resets only finished environments at the 10Hz boundary.
The inherited native locomotion reward table may appear during setup but its
rewards/terminations are NOT used by this PPO loop.
Timeout is a true finite-horizon terminal (remaining time is observed).
RSL-RL timeout bootstrapping is intentionally disabled for this task.

Standing, oscillating to reset the stationary timer, or falling to avoid future
costs can be reward exploits. Monitor
success/collision/fall/timeout counts rather than accepting reward alone.
This implementation is a baseline, not a guarantee of safe learned behavior.

## Verification

2026-09-25 reward update: 68 CPU tests and the physics-only reward smoke passed
(4envs, zero PPO updates). See REWARDS.md. The training/play runs below are
historical checks of the earlier reward contract.

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B -m unittest discover -s tests -v
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train_crossing_behavior.py --smoke_test --headless
```

Smoke uses4envs x2updates only. Before PPO, it deliberately creates goal, fall,
timeout, out-of-bounds and a real H1-car collision in isolated environments.
It verifies reward amounts, failure precedence, resets, no ground/cross-env
false collision, finite tensors, actor update, frozen policy integrity and
safe weights_only checkpoint round-trip.

Verified on2026-09-24:
- 61 CPU tests passed.
- Physical event smoke +2/-2 and4env x2PPO updates passed:
  logs/20260924T143439_939679Z_behavior_train.
- Separate resume from2 to3 updates passed:
  logs/20260924T143551_714226Z_behavior_train.
- Deterministic headless play210steps (21 simulated seconds) passed, including
  one natural timeout/reset, no NaN/Inf:
  logs/20260924T143701_571930Z_behavior_play. GUI appearance not verified this turn.
- No long training; independent crossing performance evaluation remains required.
