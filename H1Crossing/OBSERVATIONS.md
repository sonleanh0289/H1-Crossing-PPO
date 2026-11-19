# State-based crossing observations (v1)

Implemented in `scripts/crossing_observations.py`, installed by
`scripts/play_crossing_finetuned.py:build_config`. No camera input, training,
checkpoint edits, or Isaac Lab core changes.

## Public observation groups

`obs["policy"]`: unchanged 69D locomotion observation. Only this group is passed
to the frozen checkpoint-500 actor, which still outputs 19 joint actions.

`obs["crossing"]`: a float tensor `(num_envs, 8)`, raw SI units, no noise,
normalization, image tensors, joints or history. Updated at 50 Hz in current play.

| Index | Field | Meaning |
|---|---|---|
| 0:2 | vehicle_relative_x_m, vehicle_relative_y_m | Vehicle centre XY minus H1 root XY, metres |
| 2:4 | goal_relative_x_m, goal_relative_y_m | Goal centre XY minus H1 root XY, metres |
| 4:6 | vehicle_vx_mps, vehicle_vy_mps | Scripted vehicle WORLD velocity, m/s |
| 6 | ttc_s | Predicted time to footprint intersection, seconds |
| 7 | ttc_valid | 1 for intersection within 10 s; otherwise 0 |

All XY vectors are world-aligned: +X is along the road/traffic, +Y is across the
road. They do NOT rotate with H1. The goal uses `env_origin + goal_position_local`;
relative positions therefore remain invariant under translation of an environment.
Both position and velocity tensor helpers accept a batch of environments. The
current preview and its traffic controller still use one environment; this does
not yet implement vectorized traffic lifecycle management.

## TTC definition and limitations

Relative position is `vehicle_position - h1_position`; relative velocity is
`vehicle_velocity - h1_velocity`, using H1's measured world root velocity.
The car collision proxy is an X-aligned 4.2 x 1.8 m rectangle. Add a configurable
H1 planar radius (default 0.35 m) to each half-extent and solve the intersection
time interval along both axes. The earliest common nonnegative time is TTC.
Vehicle size is shared with `CrossingConfig.vehicle_size`, not duplicated geometry.

- Existing footprint overlap or boundary touch: `ttc=0, valid=1`.
- Receding, parallel nonintersecting, stationary separated, or intersection beyond
  the 10-second horizon: `ttc=10, valid=0`. No NaN/Inf is fed into observations.
- An intersection exactly at 10 seconds has `valid=1`; always use the validity bit.
- Standing on the sidewalk normally has no predicted intersection, even when a
  car is approaching the crossing. TTC is NOT simply distance / car speed, nor
  time until the car reaches the zebra crossing.
- Assumes constant velocities; no anticipation of H1's next command, acceleration,
  braking, lane changes, or another vehicle. The inflated rectangle is conservative
  at its corners and does not model the robot's changing limb footprint or exact
  PhysX contact. It is a behavior feature, not a collision/safety certificate.

Car velocity comes from the current `VehicleMotion.speed_mps` (or `BatchedTraffic.speed`
in behavior train/play), including acceleration/braking toward sampled targets.
TTC uses that current velocity; it does not predict future target changes or acceleration.
Do not infer velocity from positions: wrapping/spawn is a
teleport. `place_vehicle()` publishes the speed each update, including resets.
After an automatic reset, observations are recomputed AFTER publishing the new
traffic pass. During manager construction only, the shape probe uses zero car
velocity; the first externally returned sample has the actual spawn and speed.

## Inspect or record

```bat
cd /d D:\IssacSim\H1Crossing
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\play_crossing_finetuned.py --keyboard --keyboard_reverse_speed 1.0 --record_crossing_observations
```

Add `--h1_camera` if wanted for display; it is still not a policy observation.
Each run saves `crossing_observation_schema.json` and initial/final crossing
samples in `play_result.json`. With the recording flag, `crossing_observations.csv`
contains step 0 plus one sample after each step, with simulation time, reset count
and traffic pass number. Rows are post-step/reset states, NOT a complete transition
dataset (no behavior actions/rewards/dones). Manual reset starts are represented by
the updated reset count on the next recorded step. The console prints the latest
named values every 100 steps when recording is enabled.

## Training boundary

These features now feed a separate high-level behavior PPO trainer; see
[TRAINING_BEHAVIOR.md](TRAINING_BEHAVIOR.md). Its 19D `behavior` input consists of
the scaled 8D crossing features plus H1 planar velocity, heading, current smoothed
command, stationary anchor offset/timer and remaining episode time. It has a
connected crossing reward and filtered vehicle-contact termination.
The current `--auto`/keyboard/joystick still sends velocity commands and does not
consume `crossing` or avoid vehicles. Do not train the 19-joint PPO task against
these observations or append them to its 69D actor input.

The behavior environment maps one signed action to forward/stop/retreat velocity
at 10 Hz, assists heading toward the goal, and runs the frozen locomotion actor at
50 Hz. It owns reward, termination and selective reset synchronization. No camera
or joint states enter the behavior actor. No claim of a complete Markov observation
is made: internal locomotion dynamics are only partially observed.

## Verification commands

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B -m unittest discover -s tests -v
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\play_crossing_finetuned.py --headless --auto --walk_speed 0 --vehicle_seed 42 --steps 650 --record_crossing_observations
D:\IssacSim\env_isaaclab\Scripts\python.exe -B tests\smoke_crossing_observations.py
```

The explicit lifecycle smoke uses short episodes only inside that test, checking
the actual simulator's observations at initial and automatic resets. It does not
change normal play configuration or run training.
